# =============================================================================
# modules/gradcam_extractor.py
# =============================================================================
# Módulo de extracción de Grad-CAM usando ResNet18 pre-entrenado
#
# RESPONSABILIDADES:
#   1. Cargar dataset REFUGE y aplicar preprocesamiento
#   2. Entrenar clasificador ResNet18 (glaucoma vs normal)
#   3. Extraer Grad-CAM de la última capa convolucional
#   4. Calcular métricas: IoU, SSIM, pointing accuracy
#   5. Generar visualizaciones comparativas
#
# CLASES PROVEÍDAS:
#   - GradCAMExtractor: ResNet18 con Grad-CAM
#   - SSIMCalculator: Cálculo manual de SSIM
#   - GradCAMDataModule: Carga de datos REFUGE
#
# MÉTODOS PÚBLICOS:
#   - GradCAMExtractor.__init__(config)
#   - GradCAMExtractor.train(train_loader, val_loader) -> dict
#   - GradCAMExtractor.predict(image) -> dict
#   - GradCAMExtractor.get_gradcam(image) -> ndarray
#   - GradCAMExtractor.save(path)
#   - GradCAMExtractor.load(path)
#   - GradCAMDataModule.get_train_loader() -> DataLoader
#   - GradCAMDataModule.get_test_loader() -> DataLoader
#
# GRAD-CAM PROCEDURE:
#   1. Forward pass con imagen (3, 448, 448)
#   2. Backward pass en la clase predicha
#   3. Pooled gradients × activations de layer4[-1].conv2
#   4. ReLU → heatmap sin normalizar
#   5. Normalizar a [0, 1]
#   6. Binarizar con percentil 95
#
# SSIM MANUAL:
#   - Ventana gaussiana σ=1.5, window_size=11
#   - C1 = 9e-4, C2 = 1.6e-2
#   - Padding: replicate
#
# CONFIG ESPERADA:
#   {
#       "backbone": "resnet18",
#       "num_classes": 2,
#       "pretrained": true,
#       "seed": 42,
#       "image_size": [448, 448],
#       "epochs": 30,
#       "lr": 1e-4,
#       "weight_decay": 1e-5,
#       "patience": 5,
#       "gradcam": {"target_layer": "layer4[-1].conv2", "percentile": 95}
#   }
# =============================================================================

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

SEED = 42


def set_global_seed(seed: int) -> None:
    """Establece semilla global para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# SSIM Calculator
# =============================================================================


class SSIMCalculator:
    """Calcula SSIM manualmente entre dos imágenes."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = (0.01 * 1) ** 2
        self.C2 = (0.03 * 1) ** 2
        self.window = self._create_gaussian_window()

    def _create_gaussian_window(self) -> torch.Tensor:
        """Crea ventana gaussiana 2D."""
        import math
        coords = torch.arange(self.window_size, dtype=torch.float32)
        coords -= (self.window_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        window_2d = g.outer(g)
        return window_2d.unsqueeze(0).unsqueeze(0)

    def compute_ssim(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> float:
        """Calcula SSIM entre dos imágenes (H, W) en [0, 1]."""
        img1_t = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0)
        img2_t = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0)

        if torch.cuda.is_available():
            img1_t = img1_t.cuda()
            img2_t = img2_t.cuda()
            window = self.window.cuda()
        else:
            window = self.window

        padding = self.window_size // 2

        mu1 = F.conv2d(img1_t, window, padding=padding, groups=1)
        mu2 = F.conv2d(img2_t, window, padding=padding, groups=1)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1_t ** 2, window, padding=padding, groups=1) - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2_t ** 2, window, padding=padding, groups=1) - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1_t * img2_t, window, padding=padding, groups=1)
            - mu1_mu2
        )

        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (
            mu1_sq + mu2_sq + self.C1
        ) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = numerator / (denominator + 1e-8)

        return float(ssim_map.mean())


# =============================================================================
# Dataset Classes
# =============================================================================


class RefugeDataset(Dataset):
    """Dataset para imágenes REFUGE con máscara GT."""

    def __init__(
        self,
        annotations: dict,
        data_dir: Path,
        image_size: tuple[int, int],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        self.transform = transform

        self.samples = []
        for img_id, info in annotations.items():
            if info.get("split") != split:
                continue
            mask_path = self.data_dir / info["mask_path"]
            if not mask_path.exists():
                logging.warning(f"Máscara no encontrada: {mask_path}")
                continue
            self.samples.append(
                {
                    "image_id": img_id,
                    "image_filename": info["image_filename"],
                    "label": info["label"],
                    "mask_path": info["mask_path"],
                    "split": info["split"],
                }
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Dataset vacío para split='{split}'. "
                f"Anotaciones totales={len(annotations)}, "
                f"data_dir={self.data_dir.resolve()}. "
                f"Verifica la ruta del dataset y las máscaras."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        sample = self.samples[idx]
        label = 1 if sample["label"] == "glaucoma" else 0

        image_path = self.data_dir / sample["split"] / "Images_Cropped" / sample["image_filename"]
        if not image_path.exists():
            image_path = self.data_dir / sample["split"] / "Images" / sample["image_filename"]

        mask_path = self.data_dir / sample["mask_path"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)
        mask_binary = (mask > 0).astype(np.float32)

        if self.transform:
            image_t = self.transform(image)
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1)

        mask_t = torch.from_numpy(mask_binary)

        return image_t, mask_t, label, sample["image_id"]


class RefugeDataModule:
    """Carga y prepara datos de REFUGE para el experimento."""

    def __init__(self, config: dict):
        self.config = config
        set_global_seed(config.get("seed", SEED))

        self.data_dir = self._resolve_data_dir(config["data"]["data_dir"])
        self.image_size = tuple(config["data"]["image_size"])
        self.batch_size = config["data"].get("batch_size", 16)
        self.num_workers = config["data"].get("num_workers", 2)
        self.test_split = config["data"].get("test_split", "test")

        logging.info(f"Usando data_dir: {self.data_dir}")
        self._load_annotations()
        self._setup_transforms()

    @staticmethod
    def _resolve_data_dir(data_dir: str) -> Path:
        """
        Resuelve la ruta al dataset probando múltiples estrategias.

        1. Si es absoluta y existe, usar directamente.
        2. Resolver relativa a CWD.
        3. Resolver relativa al directorio de este archivo (modules/).
        4. Resolver relativa al directorio del script que lo llama.
        5. Buscar en ubicaciones comunes de Colab/Drive.
        """
        path = Path(data_dir)

        # 1. Ruta absoluta existente
        if path.is_absolute() and path.exists():
            return path.resolve()

        candidates = []

        # 2. Relativa a CWD
        candidates.append(Path.cwd() / path)

        # 3. Relativa a este archivo (modules/gradcam_extractor.py → Experiments/gradcam.../)
        module_dir = Path(__file__).parent.parent
        candidates.append(module_dir / path)
        # También subir hasta el proyecto (desde Experiments/gradcam.../ → raíz)
        candidates.append(module_dir.parent.parent / path)

        # 4. Relativa al stack de llamadas (script que invoca)
        import inspect
        for frame in inspect.stack()[1:]:
            caller_dir = Path(frame.filename).parent
            candidates.append(caller_dir / path)
            # Subir uno más (por si el script está en scripts/)
            candidates.append(caller_dir.parent / path)

        # 5. Ubicaciones comunes en Google Colab
        candidates.append(Path("/content/drive/MyDrive/REFUGE"))
        candidates.append(Path("/content/drive/MyDrive/Datasets/REFUGE"))
        candidates.append(Path("/content/REFUGE"))
        candidates.append(Path("/content/Datasets/REFUGE"))

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists():
                logging.info(f"Dataset encontrado en: {resolved}")
                return resolved

        # 6. Búsqueda recursiva en Google Drive por cualquier carpeta 'REFUGE'
        drive_base = Path("/content/drive/MyDrive")
        if drive_base.exists():
            logging.info(f"Buscando 'REFUGE' recursivamente en {drive_base}...")
            for refuge_dir in drive_base.rglob("REFUGE"):
                if refuge_dir.is_dir():
                    # Verificar que parece un dataset REFUGE válido (tenga train/Images)
                    if (refuge_dir / "train" / "Images").exists() or (refuge_dir / "train" / "Images_Cropped").exists():
                        logging.info(f"Dataset REFUGE encontrado recursivamente en: {refuge_dir}")
                        return refuge_dir.resolve()

        # Si nada funciona, devolver la primera candidata para que falle con error descriptivo
        logging.warning(f"No se encontró el dataset en ninguna ubicación conocida. Usando: {candidates[0]}")
        return candidates[0]

    def _load_annotations(self) -> None:
        """Carga annotations.json y genera splits si no existen."""
        annot_path = self.data_dir / "annotations.json"
        if annot_path.exists():
            with open(annot_path) as f:
                self.annotations = json.load(f)
            return

        annot_path = self.data_dir.parent / "annotations.json"
        if annot_path.exists():
            with open(annot_path) as f:
                self.annotations = json.load(f)
            return

        logging.warning(
            "annotations.json no encontrado. Generando desde estructura de carpetas."
        )
        self.annotations = self._generate_annotations_from_structure()

    def _generate_annotations_from_structure(self) -> dict:
        """Genera annotations.json desde la estructura del dataset."""
        annotations = {}
        split_mapping = {
            "train": ["train", "Train", "Training", "training", "TRAIN"],
            "val": ["val", "Val", "Validation", "validation", "VAL", "dev"],
            "test": ["test", "Test", "Testing", "testing", "TEST"],
        }

        logging.info(f"Buscando dataset en: {self.data_dir.resolve()}")
        for split, alternatives in split_mapping.items():
            split_dir = None
            for alt in alternatives:
                candidate = self.data_dir / alt
                if candidate.exists():
                    split_dir = candidate
                    break

            if split_dir is None:
                logging.warning(f"No se encontró carpeta para split '{split}' en {self.data_dir}")
                continue

            logging.info(f"Split '{split}' → {split_dir}")

            images_dir = split_dir / "Images_Cropped"
            if not images_dir.exists():
                images_dir = split_dir / "Images"
            if not images_dir.exists():
                images_dir = split_dir / "images"

            masks_dir = split_dir / "Masks_Cropped"
            if not masks_dir.exists():
                masks_dir = split_dir / "Masks"
            if not masks_dir.exists():
                masks_dir = split_dir / "masks"

            if not images_dir.exists() or not masks_dir.exists():
                logging.warning(f"  No se encontraron imágenes/máscaras en {split_dir}")
                continue

            logging.info(f"  Images dir: {images_dir} (exists={images_dir.exists()})")
            logging.info(f"  Masks dir: {masks_dir} (exists={masks_dir.exists()})")

            # Buscar imágenes con varias extensiones
            img_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]:
                img_paths.extend(list(images_dir.glob(ext)))
            img_paths = sorted(set(img_paths))

            logging.info(f"  Encontradas {len(img_paths)} imágenes")

            for img_path in img_paths:
                img_id = img_path.stem
                # Detectar si es glaucoma por prefijo o por carpeta
                label = "glaucoma" if img_path.stem.startswith("g") else "normal"

                img_name = img_path.name
                # Inferir nombre de máscara (misma base, extensión .png)
                mask_name = img_path.stem + ".png"

                annotations[img_id] = {
                    "image_filename": img_name,
                    "label": label,
                    "mask_path": f"{split_dir.name}/{masks_dir.name}/{mask_name}",
                    "split": split,
                }

        if not annotations:
            raise RuntimeError(
                f"No se encontraron imágenes en {self.data_dir}. "
                f"Verifica que el dataset REFUGE esté en la ruta correcta."
            )

        logging.info(f"Total de anotaciones generadas: {len(annotations)}")
        return annotations

    def _setup_transforms(self) -> None:
        """Configura pipeline de augmentations para entrenamiento."""
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std),
            ]
        )

    def get_train_loader(self) -> DataLoader:
        """Retorna DataLoader de entrenamiento."""
        dataset = RefugeDataset(
            annotations=self.annotations,
            data_dir=self.data_dir,
            image_size=self.image_size,
            split=self.config["data"].get("train_split", "train"),
            transform=self.train_transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_val_loader(self) -> DataLoader:
        """Retorna DataLoader de validación."""
        dataset = RefugeDataset(
            annotations=self.annotations,
            data_dir=self.data_dir,
            image_size=self.image_size,
            split=self.config["data"].get("val_split", "val"),
            transform=self.val_transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self) -> DataLoader:
        """Retorna DataLoader de test."""
        dataset = RefugeDataset(
            annotations=self.annotations,
            data_dir=self.data_dir,
            image_size=self.image_size,
            split=self.test_split,
            transform=self.val_transform,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# =============================================================================
# Grad-CAM Extractor (ResNet18)
# =============================================================================


class GradCAMExtractor(nn.Module):
    """ResNet18 con extracción de Grad-CAM."""

    def __init__(self, config: dict):
        super().__init__()
        set_global_seed(config.get("seed", SEED))

        self.config = config
        self.image_size = tuple(config["data"]["image_size"])
        self.gradcam_percentile = config["gradcam"]["percentile"]

        self.model = resnet18(weights="DEFAULT" if config["classifier"]["pretrained"] else None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, config["classifier"]["num_classes"])

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.target_layer = self._get_target_layer()

        self._register_hooks()

        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        self.norm_mean = img_mean
        self.norm_std = img_std

        self.ssim_calc = SSIMCalculator()

    def _get_target_layer(self) -> nn.Module:
        """Retorna la capa objetivo para Grad-CAM."""
        return self.model.layer4[-1].conv2

    def _register_hooks(self) -> None:
        """Registra hooks para capturar gradientes y activaciones."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    @torch.no_grad()
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Convierte tensor normalizado a imagen [0,1] para visualización."""
        mean_t = torch.tensor(self.norm_mean, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
        std_t = torch.tensor(self.norm_std, dtype=torch.float32, device=tensor.device).view(-1, 1, 1)
        img = tensor * std_t + mean_t
        img = torch.clamp(img, 0, 1)
        return img.cpu().numpy()

    def get_gradcam(self, image: torch.Tensor) -> np.ndarray:
        """
        Extrae Grad-CAM para una imagen.

        Args:
            image: Tensor (3, H, W) normalizado con ImageNet stats

        Returns:
            heatmap: ndarray (H, W) con valores en [0, 1]
        """
        self.model.eval()

        device = next(self.model.parameters()).device
        image = image.unsqueeze(0).to(device)
        image.requires_grad = True

        output = self.model(image)
        pred_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (pooled_gradients * self.activations).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        heatmap = heatmap.squeeze().cpu().numpy()

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def get_gradcam_binary(self, image: torch.Tensor) -> np.ndarray:
        """
        Extrae Grad-CAM y lo binariza usando el percentil configurado.

        Args:
            image: Tensor (3, H, W) normalizado

        Returns:
            mask_binary: ndarray (H, W) con valores 0 o 1
        """
        heatmap = self.get_gradcam(image)
        threshold = np.percentile(heatmap, self.gradcam_percentile)
        mask_binary = (heatmap >= threshold).astype(np.float32)
        return mask_binary

    def predict(self, image: torch.Tensor) -> dict:
        """Predice clase y distribución."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            probs = F.softmax(output, dim=1)[0]
            pred_class = output.argmax(dim=1).item()
            dist = {"glaucoma": float(probs[1]), "normal": float(probs[0])}
        return {"prediction": "glaucoma" if pred_class == 1 else "normal", "distribution": dist}

    def compute_metrics(
        self, gradcam_mask: np.ndarray, gt_mask: np.ndarray
    ) -> dict:
        """
        Calcula IoU, SSIM y pointing accuracy entre Grad-CAM y máscara GT.

        Args:
            gradcam_mask: ndarray binario (H, W) del Grad-CAM binarizado
            gt_mask: ndarray binario (H, W) de la máscara GT (0=bg, 1=OD)

        Returns:
            dict con "iou", "ssim", "pointing_accuracy"
        """
        gt_binary = (gt_mask > 0).astype(np.float32)

        tp = np.sum((gradcam_mask == 1) & (gt_binary == 1))
        fp = np.sum((gradcam_mask == 1) & (gt_binary == 0))
        fn = np.sum((gradcam_mask == 0) & (gt_binary == 1))
        iou = tp / (tp + fp + fn + 1e-8)

        gradcam_normalized = gradcam_mask
        ssim = self.ssim_calc.compute_ssim(gradcam_normalized, gt_binary)

        max_pos = np.unravel_index(np.argmax(gradcam_mask), gradcam_mask.shape)
        pointing_inside = bool(gt_binary[max_pos] == 1)
        pointing_acc = 1.0 if pointing_inside else 0.0

        return {"iou": float(iou), "ssim": float(ssim), "pointing_accuracy": float(pointing_acc)}

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict:
        """Entrena el modelo con early stopping."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["classifier"]["lr"],
            weight_decay=self.config["classifier"].get("weight_decay", 1e-5),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.config["classifier"].get("scheduler_factor", 0.5),
            patience=self.config["classifier"].get("scheduler_patience", 3),
        )

        best_acc = 0.0
        patience_counter = 0
        best_state = None

        epochs = self.config["classifier"]["epochs"]
        patience = self.config["classifier"]["patience"]

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, masks, labels, _ in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_acc = train_correct / train_total

            self.model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, masks, labels, _ in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = val_correct / val_total
            scheduler.step(val_acc)

            logging.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss/len(train_loader):.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping en epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(device)

        return {"best_val_acc": best_acc, "epochs_trained": epoch + 1}

    def save(self, path: str) -> None:
        """Guarda el modelo en disco."""
        torch.save(self.model.state_dict(), path)
        logging.info(f"Modelo guardado en {path}")

    def load(self, path: str) -> None:
        """Carga el modelo desde disco."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model = self.model.to(device)
        logging.info(f"Modelo cargado desde {path}")
