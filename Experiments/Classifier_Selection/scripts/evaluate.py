# =============================================================================
# scripts/evaluate.py
# =============================================================================
# Evaluación de clasificación para el experimento few-shot
#
# RESPONSABILIDADES:
#   1. Calcular métricas de clasificación en un dataset
#   2. Generar matriz de confusión
#   3. Calcular precision, recall, f1 por clase
#
# DISEÑO:
#   - compute_classification_metrics(y_true, y_pred, class_names): NÚCLEO en numpy
#     puro (sin torch) -> métricas a partir de etiquetas/predicciones. Testeable
#     de forma aislada.
#   - evaluate_classification(model, data_loader, device): corre la inferencia
#     (torch) sobre el loader y delega el cálculo en la función anterior.
#
# FUNCIÓN PRINCIPAL:
#   evaluate_classification(model, data_loader, device) -> dict
#
# RETORNO:
#   {
#     "accuracy": float,
#     "f1_macro": float,
#     "f1_per_class": {"normal": float, "glaucoma": float},
#     "precision_per_class": {"normal": float, "glaucoma": float},
#     "recall_per_class": {"normal": float, "glaucoma": float},
#     "confusion_matrix": [[TN, FP], [FN, TP]],   # filas=verdad, cols=predicho
#     "total_samples": int
#   }
#
# NOTAS:
#   - Clases por índice (orden de class_names): normal=0, glaucoma=1. Con ese orden
#     la matriz binaria queda [[TN, FP], [FN, TP]] (negativo=normal, positivo=glaucoma).
#   - F1-macro es la métrica CRÍTICA por el imbalance 10:1.
#   - Accuracy engaña con imbalance (da ~90% solo prediciendo normal).
#   - Las muestras sin label (label < 0, p.ej. split test) se ignoran.
#
# USO:
#   from scripts.evaluate import evaluate_classification
#   metrics = evaluate_classification(model, val_loader, device="cuda")
# =============================================================================

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Nombres de clase por índice. DEBE casar con CLASS_TO_IDX de data_module.py
# (normal=0, glaucoma=1). Se usa si el modelo no expone su propio class_names.
DEFAULT_CLASS_NAMES = ["normal", "glaucoma"]


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    """
    Calcula métricas de clasificación a partir de etiquetas y predicciones.

    Núcleo en numpy puro (sin torch), para poder testearlo de forma aislada.

    Args:
        y_true: Índices de clase verdaderos (enteros 0..C-1).
        y_pred: Índices de clase predichos (enteros 0..C-1).
        class_names: Nombres por índice (ej. ["normal", "glaucoma"]).

    Returns:
        dict con accuracy, f1_macro, *_per_class, confusion_matrix y total_samples.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    num_classes = len(class_names)

    # Matriz de confusión: filas = clase verdadera, columnas = clase predicha.
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_idx, pred_idx in zip(y_true, y_pred):
        confusion[true_idx, pred_idx] += 1

    total = int(y_true.shape[0])
    accuracy = float((y_true == y_pred).mean()) if total > 0 else 0.0

    precision_per_class: dict[str, float] = {}
    recall_per_class: dict[str, float] = {}
    f1_per_class: dict[str, float] = {}

    for c, name in enumerate(class_names):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)  # predichas c pero verdad != c
        fn = int(confusion[c, :].sum() - tp)  # verdad c pero predichas != c
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precision_per_class[name] = precision
        recall_per_class[name] = recall
        f1_per_class[name] = f1

    f1_macro = float(np.mean(list(f1_per_class.values()))) if num_classes > 0 else 0.0

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "confusion_matrix": confusion.tolist(),
        "total_samples": total,
    }


def _resolve_module_and_device(
    model, device: str | torch.device | None
) -> tuple[nn.Module, torch.device]:
    """Obtiene el nn.Module subyacente y el device a usar.

    Acepta un CNNClassifier (expone .model / .device) o un nn.Module directo.
    """
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        net = model.model
    elif isinstance(model, nn.Module):
        net = model
    else:
        raise TypeError("model debe ser un CNNClassifier o un nn.Module.")

    if device is None:
        device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
    return net, torch.device(device)


def evaluate_classification(model, data_loader, device: str | torch.device | None = None) -> dict:
    """
    Evalúa la clasificación de un modelo sobre un DataLoader.

    Args:
        model: CNNClassifier (o nn.Module) a evaluar.
        data_loader: DataLoader cuyos batches son dicts con "image" y "label"
            (formato de M1 DataModule).
        device: Device de inferencia; por defecto el del modelo o cuda/cpu.

    Returns:
        dict de métricas (ver compute_classification_metrics).
    """
    net, device = _resolve_module_and_device(model, device)
    class_names = list(getattr(model, "class_names", DEFAULT_CLASS_NAMES))

    net.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    n_skipped = 0

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

            logits = net(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            # Ignorar muestras sin label (label < 0, p.ej. split test).
            labeled = labels >= 0
            n_skipped += int((~labeled).sum())
            all_true.append(labels[labeled])
            all_pred.append(preds[labeled])

    if n_skipped:
        logger.warning("Se ignoraron %d muestras sin label (label < 0).", n_skipped)

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
    return compute_classification_metrics(y_true, y_pred, class_names)
