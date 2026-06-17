# =============================================================================
# modules/cnn_classifier.py
# =============================================================================
# M2: CNNClassifier - 3 backbones (ResNet-18 / EfficientNet-B0 / DenseNet-121)
#     + clasificación + Grad-CAM + training loop con early stopping.
#
# DECISIÓN DE DISEÑO:
#   Se implementa UNA sola clase pública `CNNClassifier` (no una por backbone),
#   porque así la instancian el notebook y few_shot.py:
#       CNNClassifier({"backbone": "resnet18", "num_classes": 2, ...})
#   El backbone concreto lo construye `modules/backbones.build_raw_backbone`
#   (factory compartida con el few-shot por prototipos), según config["backbone"].
#   Todos los backbones provienen de torchvision (no timm): torchvision tiene los
#   tres y las capas objetivo de Grad-CAM documentadas (layer4, features) asumen
#   la estructura de torchvision; además evita una dependencia extra.
#
# MÉTODOS PÚBLICOS:
#   - __init__(config: dict)
#   - train(train_loader, val_loader, ...) -> dict
#   - predict(image: Tensor) -> {"prediction", "distribution", "predicted_index"}
#   - get_gradcam(image: Tensor) -> ndarray (H, W) en [0, 1]  (heatmap CONTINUO)
#   - save(path) / load(path)
#
# GRAD-CAM:
#   - Capa objetivo: última conv del backbone (resnet=layer4, dense/effnet=features)
#   - Devuelve el heatmap CONTINUO normalizado a [0, 1]. La binarización (percentil
#     95) y el IoU vs máscara GT son responsabilidad de scripts/extract_gradcam.py.
#
# FREEZE STRATEGY (apropiada para few-shot: pocos parámetros entrenables):
#   - Se congela todo el backbone excepto el último bloque conv + la cabeza FC.
#
# CONFIG ESPERADA:
#   {
#       "backbone": "resnet18" | "efficientnet_b0" | "densenet121",
#       "num_classes": 2,
#       "pretrained": true,
#       "seed": 42,
#       "dropout": 0.5,                 # opcional
#       "class_names": ["normal","glaucoma"]  # opcional (debe casar con M1)
#   }
#
# NOTA: mixup (config.augmentations.mixup) NO se aplica todavía en train(); está
#       marcado como extensión futura (mezcla labels -> requiere soft-target loss).
# =============================================================================

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from modules.backbones import SUPPORTED_BACKBONES, build_raw_backbone

logger = logging.getLogger(__name__)

# Nombres de clase por índice. DEBE casar con CLASS_TO_IDX de data_module.py
# (normal=0, glaucoma=1). Se define local para mantener M2 independiente de M1.
DEFAULT_CLASS_NAMES = ["normal", "glaucoma"]


def set_global_seed(seed: int) -> None:
    """Fija semillas globales (random, numpy, torch) para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNNClassifier:
    """
    Clasificador CNN para glaucoma vs normal con transfer learning.

    Arquitectura: backbone pre-entrenado (ImageNet) -> GAP -> Dropout -> Linear.
    Solo se afina (fine-tune) el último bloque conv y la cabeza FC.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: {backbone, num_classes, pretrained, seed, [dropout], [class_names]}.
        """
        self.backbone_name: str = config["backbone"]
        if self.backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Backbone '{self.backbone_name}' no soportado. "
                f"Opciones: {SUPPORTED_BACKBONES}"
            )

        self.num_classes: int = config.get("num_classes", 2)
        self.pretrained: bool = config.get("pretrained", True)
        self.seed: int = config.get("seed", 42)
        self.dropout: float = config.get("dropout", 0.5)
        self.class_names: list[str] = config.get(
            "class_names",
            DEFAULT_CLASS_NAMES
            if self.num_classes == 2
            else [f"class_{i}" for i in range(self.num_classes)],
        )

        set_global_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.target_layer es el módulo conv final usado por Grad-CAM;
        # self.target_layer_path es su ruta string (para el módulo compartido).
        self.model, self.target_layer, self.target_layer_path = self._build_model()
        self.model.to(self.device)

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "CNNClassifier(%s) | params entrenables: %d / %d (%.1f%%) | device=%s",
            self.backbone_name,
            n_trainable,
            n_total,
            100.0 * n_trainable / n_total,
            self.device,
        )

    # ---- Construcción del modelo --------------------------------------------

    def _make_head(self, in_features: int) -> nn.Sequential:
        """Cabeza de clasificación: Dropout + Linear (reemplaza la FC de ImageNet)."""
        return nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features, self.num_classes),
        )

    def _build_model(self) -> tuple[nn.Module, nn.Module, str]:
        """
        Construye el backbone (via build_raw_backbone), reemplaza la cabeza por
        una entrenable (Dropout + Linear) y aplica la freeze strategy.

        Returns:
            (model, target_layer, target_path) — target_layer es la última capa
            conv (para Grad-CAM) y target_path su ruta string.
        """
        parts = build_raw_backbone(self.backbone_name, pretrained=self.pretrained)
        head = self._make_head(parts.feat_dim)
        setattr(parts.model, parts.head_attr, head)
        self._apply_freeze(parts.model, parts.last_block, head)
        return parts.model, parts.target_layer, parts.target_path

    @staticmethod
    def _apply_freeze(model: nn.Module, last_block: nn.Module, head: nn.Module) -> None:
        """Congela todo y descongela solo el último bloque conv + la cabeza FC."""
        for param in model.parameters():
            param.requires_grad = False
        for param in last_block.parameters():
            param.requires_grad = True
        for param in head.parameters():
            param.requires_grad = True

    # ---- Entrenamiento -------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        lr: float = 1e-4,
        patience: int = 5,
        weight_decay: float = 1e-5,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 3,
    ) -> dict:
        """
        Fine-tune con early stopping sobre val_loss; restaura los mejores pesos.

        Returns:
            {
              "best_val_acc": float, "best_val_f1": float, "best_val_loss": float,
              "epochs_trained": int, "history": [ {epoch, val_loss, val_acc, val_f1, lr} ]
            }
        """
        criterion = nn.CrossEntropyLoss()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
        )

        best_val_loss = float("inf")
        best_acc, best_f1 = 0.0, 0.0
        best_state: dict | None = None
        epochs_no_improve = 0
        history: list[dict] = []

        for epoch in range(epochs):
            self._run_train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, val_f1 = self._evaluate_loader(val_loader, criterion)
            scheduler.step(val_loss)

            history.append(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            logger.info(
                "[%s] epoch %d/%d | val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                self.backbone_name,
                epoch + 1,
                epochs,
                val_loss,
                val_acc,
                val_f1,
            )

            # Early stopping: mejora si val_loss baja de forma apreciable.
            if val_loss < best_val_loss - 1e-4:
                best_val_loss, best_acc, best_f1 = val_loss, val_acc, val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("Early stopping en epoch %d (patience=%d)", epoch + 1, patience)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "best_val_acc": best_acc,
            "best_val_f1": best_f1,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history),
            "history": history,
        }

    def _run_train_epoch(
        self, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        """Una época de entrenamiento (actualiza pesos)."""
        self.model.train()
        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            optimizer.zero_grad()
            logits = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    def _evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float, float]:
        """Evalúa un loader y devuelve (loss_media, accuracy, f1_macro)."""
        self.model.eval()
        total_loss, n_samples = 0.0, 0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = self.model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                n_samples += images.size(0)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        accuracy = float((preds == labels).mean())
        f1_macro = self._f1_macro(labels, preds, self.num_classes)
        return total_loss / max(n_samples, 1), accuracy, f1_macro

    @staticmethod
    def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
        """F1-macro sin sklearn (promedio del F1 por clase). Robusto a clases vacías."""
        f1_scores = []
        for c in range(num_classes):
            tp = int(((y_pred == c) & (y_true == c)).sum())
            fp = int(((y_pred == c) & (y_true != c)).sum())
            fn = int(((y_pred != c) & (y_true == c)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f1_scores.append(2 * precision * recall / (precision + recall))
            else:
                f1_scores.append(0.0)
        return float(np.mean(f1_scores))

    # ---- Inferencia ----------------------------------------------------------

    def predict(self, image: torch.Tensor) -> dict:
        """
        Clasifica una imagen ya normalizada.

        Args:
            image: Tensor (3, H, W) o (1, 3, H, W).

        Returns:
            {"prediction": str, "distribution": {clase: prob}, "predicted_index": int}
        """
        self.model.eval()
        batch = self._ensure_batch(image)
        with torch.no_grad():
            logits = self.model(batch.to(self.device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_index = int(probs.argmax())
        distribution = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        return {
            "prediction": self.class_names[predicted_index],
            "distribution": distribution,
            "predicted_index": predicted_index,
        }

    def get_gradcam(self, image: torch.Tensor) -> np.ndarray:
        """
        Genera el Grad-CAM (heatmap continuo) de la clase predicha.

        Args:
            image: Tensor (3, H, W) o (1, 3, H, W) normalizado.

        Returns:
            ndarray (H, W) en [0, 1]. NO binarizado (eso lo hace el evaluador).
        """
        self.model.eval()
        batch = self._ensure_batch(image).to(self.device)

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        # Capturamos activaciones y gradiente con UN forward hook que, además,
        # registra un hook sobre el tensor de salida para obtener su gradiente.
        # NO usamos register_full_backward_hook: este crea una función de autograd
        # cuya salida es una "view", y densenet aplica F.relu(..., inplace=True)
        # justo sobre la salida de `features` -> modificar in-place esa view está
        # prohibido (RuntimeError). El patrón forward-hook + tensor.register_hook
        # es el estándar de Grad-CAM y funciona con los 3 backbones.
        def forward_hook(_module, _inp, output: torch.Tensor) -> None:
            activations["value"] = output
            if output.requires_grad:
                output.register_hook(
                    lambda grad: gradients.__setitem__("value", grad.detach())
                )

        handle = self.target_layer.register_forward_hook(forward_hook)
        try:
            logits = self.model(batch)
            class_idx = int(logits.argmax(dim=1).item())
            self.model.zero_grad()
            logits[0, class_idx].backward()

            acts = activations["value"].detach()  # (1, C, h, w)
            grads = gradients["value"]  # (1, C, h, w) ya detached en el hook
            # Peso de cada canal = promedio espacial de su gradiente.
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1, 1, h, w)
            cam = F.interpolate(
                cam, size=batch.shape[-2:], mode="bilinear", align_corners=False
            )[0, 0]
            # Normalizar a [0, 1].
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            return cam.detach().cpu().numpy()
        finally:
            handle.remove()

    @staticmethod
    def _ensure_batch(image: torch.Tensor) -> torch.Tensor:
        """Asegura forma (1, 3, H, W) a partir de (3, H, W) o (1, 3, H, W)."""
        if image.dim() == 3:
            return image.unsqueeze(0)
        if image.dim() == 4:
            return image
        raise ValueError(f"Imagen con forma inesperada: {tuple(image.shape)}")

    # ---- Persistencia --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Guarda los pesos del modelo (+ metadatos) en disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "backbone": self.backbone_name,
                "num_classes": self.num_classes,
                "class_names": self.class_names,
            },
            path,
        )
        logger.info("Modelo guardado en %s", path)

    def load(self, path: str | Path) -> None:
        """Carga pesos desde disco en el modelo ya construido."""
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        logger.info("Modelo cargado desde %s", path)
