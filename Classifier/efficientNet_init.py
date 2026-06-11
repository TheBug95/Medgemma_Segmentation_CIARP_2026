# =============================================================================
# Classifier/efficientNet_init.py
# =============================================================================
# Setup del clasificador de produccion: EfficientNet-B0 para glaucoma vs normal.
#
# Es la version "de produccion" del backbone GANADOR del experimento
# Experiments/Classifier_Selection (efficientnet_b0, score 0.439). Porta la
# logica ya validada de modules/cnn_classifier.py pero enfocada SOLO en
# EfficientNet-B0 (sin las ramas de resnet/densenet).
#
# CLASE PUBLICA: EfficientNetClassifier
#   - __init__(config: dict)
#   - train(train_loader, val_loader, ...) -> dict
#   - predict(image: Tensor) -> {"prediction", "distribution", "predicted_index"}
#   - get_gradcam(image: Tensor, target_class=None) -> ndarray (H, W) en [0, 1]
#   - save(path) / load(path) / close()
#
# UTILIDADES DE MODULO (compartidas por training e inferencia):
#   - set_global_seed(seed)
#   - build_eval_transform()        -> Resize 448 + ToTensor + Normalize ImageNet
#   - build_train_transform(aug)    -> + augmentations (solo train)
#   - constantes: IMAGENET_MEAN/STD, IMAGE_SIZE, CLASS_TO_IDX, IDX_TO_CLASS
#
# GRAD-CAM:
#   get_gradcam DELEGA en el modulo compartido modules/gradcam_module.py
#   (GradCAMExtractorModule, target_layer="features[-1]"). Una sola
#   implementacion de Grad-CAM en todo el proyecto. El modulo de la raiz se
#   carga por RUTA (importlib) para evitar la colision con el paquete `modules`
#   del experimento Classifier_Selection (mismo patron que usa su notebook).
#
# ARQUITECTURA:
#   Input (3, 448, 448) ImageNet-norm -> EfficientNet-B0 features -> GAP ->
#   Dropout(0.5) -> Linear(1280, num_classes) -> Softmax
#
# FREEZE STRATEGY (few-shot: pocos parametros entrenables):
#   Se congela todo salvo el ultimo bloque conv (features[-1]) + la cabeza FC.
# =============================================================================

from __future__ import annotations

import importlib.util
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constantes fijas del modelo (NO dependen del dataset; compartidas train/infer)
# -----------------------------------------------------------------------------

# Normalizacion ImageNet (EfficientNet-B0 pre-entrenado la espera).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Tamano de entrada de la CNN.
IMAGE_SIZE: tuple[int, int] = (448, 448)

# Mapeo clase textual <-> indice. glaucoma=1 (clase de interes), normal=0.
# DEBE casar con read_label() de training/data_interface.py.
CLASS_TO_IDX: dict[str, int] = {"normal": 0, "glaucoma": 1}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
DEFAULT_CLASS_NAMES: list[str] = ["normal", "glaucoma"]

# Capa objetivo de Grad-CAM para EfficientNet (ultimo bloque conv de `features`).
GRADCAM_TARGET_LAYER: str = "features[-1]"

# Rutas para localizar el modulo compartido de Grad-CAM en la raiz del repo.
_CLASSIFIER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CLASSIFIER_DIR.parent
_GRADCAM_MODULE_PATH = _REPO_ROOT / "modules" / "gradcam_module.py"

# Cache del modulo de Grad-CAM ya cargado (se carga una sola vez por proceso).
_gradcam_module = None


# =============================================================================
# Utilidades de modulo
# =============================================================================


def set_global_seed(seed: int) -> None:
    """Fija las semillas globales (random, numpy, torch) para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_eval_transform(image_size: tuple[int, int] = IMAGE_SIZE) -> T.Compose:
    """
    Transform de evaluacion/inferencia: Resize + ToTensor + Normalize ImageNet.

    Recibe una imagen PIL RGB cruda y devuelve el tensor (3, H, W) listo para la
    CNN. Sin augmentation (deterministico).
    """
    return T.Compose(
        [
            T.Resize(image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_train_transform(
    augmentations: dict | None = None, image_size: tuple[int, int] = IMAGE_SIZE
) -> T.Compose:
    """
    Transform de entrenamiento: Resize + augmentations + ToTensor + Normalize.

    Args:
        augmentations: dict con flags de augmentation (seccion `augmentations` del
            config). Si es None, no se aplica ninguna (equivale al eval transform).
        image_size: tamano destino (alto, ancho).
    """
    aug = augmentations or {}
    steps: list = [T.Resize(image_size, antialias=True)]

    if aug.get("horizontal_flip"):
        steps.append(T.RandomHorizontalFlip(p=0.5))
    if aug.get("rotation_degrees"):
        steps.append(T.RandomRotation(degrees=aug["rotation_degrees"]))
    if aug.get("color_jitter_brightness") or aug.get("color_jitter_contrast"):
        steps.append(
            T.ColorJitter(
                brightness=aug.get("color_jitter_brightness", 0.0),
                contrast=aug.get("color_jitter_contrast", 0.0),
            )
        )
    steps.append(T.ToTensor())
    steps.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    if aug.get("random_erasing"):
        steps.append(T.RandomErasing(p=aug.get("random_erasing_p", 0.2)))

    return T.Compose(steps)


def _load_gradcam_module():
    """
    Carga modules/gradcam_module.py de la raiz del repo por ruta (importlib).

    Se carga por ruta y NO con `import modules.gradcam_module` para evitar la
    colision con el paquete `modules` del experimento Classifier_Selection
    (mismo patron que experiment_orchestrator.ipynb). Cacheado por proceso.
    """
    global _gradcam_module
    if _gradcam_module is not None:
        return _gradcam_module

    if not _GRADCAM_MODULE_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontro el modulo de Grad-CAM en {_GRADCAM_MODULE_PATH}. "
            f"Se esperaba <repo>/modules/gradcam_module.py."
        )

    spec = importlib.util.spec_from_file_location(
        "gradcam_module_root", _GRADCAM_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # Registrar en sys.modules ANTES de ejecutar: los @dataclass del modulo
    # resuelven sus type hints via sys.modules[cls.__module__] (en Python 3.13
    # falla con AttributeError si el modulo no esta registrado). Mismo patron
    # que usa experiment_orchestrator.ipynb del experimento.
    sys.modules["gradcam_module_root"] = module
    spec.loader.exec_module(module)
    _gradcam_module = module
    return module


# =============================================================================
# Clasificador
# =============================================================================


class EfficientNetClassifier:
    """
    Clasificador EfficientNet-B0 para glaucoma vs normal (transfer learning).

    Arquitectura: backbone EfficientNet-B0 pre-entrenado (ImageNet) -> GAP ->
    Dropout -> Linear. Solo se afina (fine-tune) el ultimo bloque conv + la cabeza.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: {num_classes, pretrained, seed, [dropout], [class_names],
                     [gradcam]}. La seccion `gradcam` (opcional) puede traer
                     {percentile, use_external} para el extractor de Grad-CAM.
        """
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
        self._gradcam_cfg: dict = config.get("gradcam", {})

        set_global_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model()
        self.model.to(self.device)

        # Extractor de Grad-CAM (lazy: se crea la primera vez que se llama get_gradcam).
        self._gradcam_extractor = None

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "EfficientNetClassifier | params entrenables: %d / %d (%.1f%%) | device=%s",
            n_trainable,
            n_total,
            100.0 * n_trainable / max(n_total, 1),
            self.device,
        )

    # ---- Construccion del modelo --------------------------------------------

    def _make_head(self, in_features: int) -> nn.Sequential:
        """Cabeza de clasificacion: Dropout + Linear (reemplaza la FC de ImageNet)."""
        return nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features, self.num_classes),
        )

    def _build_model(self) -> nn.Module:
        """Construye EfficientNet-B0, reemplaza la cabeza y aplica la freeze strategy."""
        weights = "DEFAULT" if self.pretrained else None
        model = tvm.efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features  # 1280
        head = self._make_head(in_features)
        model.classifier = head

        # Freeze: congelar todo y descongelar solo el ultimo bloque conv + la cabeza.
        last_block = model.features[-1]
        for param in model.parameters():
            param.requires_grad = False
        for param in last_block.parameters():
            param.requires_grad = True
        for param in head.parameters():
            param.requires_grad = True

        return model

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
                "epoch %d/%d | val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                epoch + 1,
                epochs,
                val_loss,
                val_acc,
                val_f1,
            )

            # Early stopping: mejora si val_loss baja de forma apreciable.
            if val_loss < best_val_loss - 1e-4:
                best_val_loss, best_acc, best_f1 = val_loss, val_acc, val_f1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
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
        """Una epoca de entrenamiento (actualiza pesos)."""
        self.model.train()
        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            optimizer.zero_grad()
            logits = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    def _evaluate_loader(
        self, loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float, float]:
        """Evalua un loader y devuelve (loss_media, accuracy, f1_macro)."""
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

        preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
        labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=int)
        accuracy = float((preds == labels).mean()) if preds.size else 0.0
        f1_macro = self._f1_macro(labels, preds, self.num_classes)
        return total_loss / max(n_samples, 1), accuracy, f1_macro

    @staticmethod
    def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
        """F1-macro sin sklearn (promedio del F1 por clase). Robusto a clases vacias."""
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
        return float(np.mean(f1_scores)) if f1_scores else 0.0

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

    def get_gradcam(
        self, image: torch.Tensor, target_class: int | None = None
    ) -> np.ndarray:
        """
        Genera el Grad-CAM (heatmap continuo) delegando en modules/gradcam_module.py.

        Args:
            image: Tensor (3, H, W) o (1, 3, H, W) normalizado.
            target_class: clase contra la cual derivar el Grad-CAM. None = clase
                predicha por el modelo (en el uso normal solo se llama cuando la
                prediccion es glaucoma, asi que None ya apunta a glaucoma).

        Returns:
            ndarray (H, W) en [0, 1]. NO binarizado.
        """
        extractor = self._get_gradcam_extractor()
        batch = self._ensure_batch(image)
        return extractor.extract(batch[0], target_class=target_class)

    def _get_gradcam_extractor(self):
        """Crea (lazy) y cachea el GradCAMExtractorModule sobre el modelo actual."""
        if self._gradcam_extractor is not None:
            return self._gradcam_extractor

        gc_mod = _load_gradcam_module()
        gradcam_config = gc_mod.GradCAMConfig(
            target_layer=GRADCAM_TARGET_LAYER,
            percentile=self._gradcam_cfg.get("percentile", 95),
            use_external=self._gradcam_cfg.get("use_external", True),
            image_size=IMAGE_SIZE,
        )
        self._gradcam_extractor = gc_mod.GradCAMExtractorModule(
            model=self.model, config=gradcam_config, device=self.device
        )
        return self._gradcam_extractor

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
                "backbone": "efficientnet_b0",
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
        # Invalidar el extractor de Grad-CAM cacheado (re-creado en el proximo uso).
        self.close()
        logger.info("Modelo cargado desde %s", path)

    def close(self) -> None:
        """Libera los hooks del extractor de Grad-CAM (si se creo)."""
        if self._gradcam_extractor is not None:
            self._gradcam_extractor.cleanup()
            self._gradcam_extractor = None

    def __del__(self) -> None:
        """Limpieza de hooks al destruir el objeto (silenciosa en GC)."""
        try:
            self.close()
        except Exception:
            pass
