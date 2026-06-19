# =============================================================================
# Classifier/private_data_module.py
# =============================================================================
# DataModule para el DATASET PRIVADO de glaucoma (formato distinto a REFUGE),
# pensado para reusar el pipeline del experimento de seleccion de backbone
# (Experiments/Classifier_Selection): expone la MISMA interfaz que su DataModule,
# asi `train_few_shot`, `PrototypeClassifier`, `evaluate_classification` y la
# evaluacion de Grad-CAM funcionan sin cambios.
#
# DIFERENCIAS DEL DATASET PRIVADO:
#   - Un split.json con los REGISTROS COMPLETOS por split:
#       {"train": [ {image_filename, label, ...}, ... ], "validation": [...], "test": [...]}
#     (la clave "validation" se normaliza a "val").
#   - label = "Pathological" / "Normal"  ->  se mapea a "glaucoma" / "normal".
#   - Imagenes .jpg en una carpeta PLANA (images_dir).
#   - Mascaras en .npy: <stem>_disc.npy y <stem>_cup.npy (masks_dir). El disco
#     optico GT (para el IoU del Grad-CAM) = union (disc | cup), binarizado.
#
# CONFIG ESPERADA (dict):
#   {
#     "split_json": ".../split.json",      # registros por split
#     "images_dir": ".../images",          # carpeta plana con los .jpg
#     "masks_dir":  ".../masks",           # carpeta plana con los .npy (_disc/_cup)
#     "image_size": [224, 224],
#     "batch_size": 16,
#     "num_workers": 2,
#     "seed": 42,                          # opcional
#     "augmentations": {...},              # opcional
#     "cache_images": true                 # opcional (recomendado: dataset chico)
#   }
#
# INTERFAZ PUBLICA (igual que modules/data_module.DataModule):
#   - splits, annotations
#   - get_glaucoma_indices(split) -> list[str]
#   - build_dataset(image_ids, augment) -> Dataset
#   - build_loader(image_ids, ...) -> DataLoader
#   - get_train_loader / get_val_loader / get_test_loader
# =============================================================================

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Mapeo clase textual -> indice. DEBE casar con CLASS_TO_IDX del experimento
# (normal=0, glaucoma=1) para que el pipeline reusado funcione.
CLASS_TO_IDX: dict[str, int] = {"normal": 0, "glaucoma": 1}
UNLABELED_IDX = -1

# Mapeo del label del dataset privado a la clase del proyecto.
LABEL_MAP: dict[str, str] = {"Pathological": "glaucoma", "Normal": "normal"}

# Normalizacion ImageNet (EfficientNet-B0 pre-entrenado la espera).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_AUGMENTATIONS: dict = {
    "horizontal_flip": True,
    "rotation_degrees": 15,
    "color_jitter_brightness": 0.1,
    "color_jitter_contrast": 0.1,
    "random_erasing": True,
    "random_erasing_p": 0.2,
}

# Normaliza los nombres de split del dataset privado a los del pipeline.
_SPLIT_NAME_MAP = {"train": "train", "validation": "val", "val": "val", "test": "test"}


def set_global_seed(seed: int) -> None:
    """Fija las semillas globales (random, numpy, torch) para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(_worker_id: int) -> None:
    """worker_init_fn: deriva la semilla de cada worker del estado de torch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_eval_transform() -> T.Compose:
    """Transform de evaluacion: ToTensor + Normalize (el Resize se hace al cargar)."""
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_train_transform(augmentations: dict) -> T.Compose:
    """Transform de train: augmentations + ToTensor + Normalize (sin Resize)."""
    aug = augmentations or {}
    steps: list = []
    if aug.get("horizontal_flip"):
        steps.append(T.RandomHorizontalFlip(p=0.5))
    if aug.get("rotation_degrees"):
        steps.append(T.RandomRotation(degrees=aug["rotation_degrees"]))
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


class PrivateDataset(Dataset):
    """
    Dataset del set privado: imagen (.jpg) + mascara de disco optico (disc | cup,
    desde .npy) + label. Mismo dict de salida que el RefugeDataset del experimento.

    Con cache_images=True guarda en RAM la imagen ya redimensionada y la mascara
    (el dataset es chico y el val/train se releen muchas veces).
    """

    def __init__(
        self,
        annotations: dict[str, dict],
        image_ids: list[str],
        images_dir: Path,
        masks_dir: Path,
        image_size: tuple[int, int],
        image_transform: T.Compose,
        cache_images: bool = False,
    ) -> None:
        self.annotations = annotations
        self.image_ids = list(image_ids)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.image_transform = image_transform
        self.cache_images = cache_images

        self._resize = T.Resize(image_size, antialias=True)
        self._image_cache: dict[str, Image.Image] = {}
        self._mask_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, filename: str) -> Image.Image:
        """Abre la imagen RGB de la carpeta plana y la redimensiona (cacheable)."""
        image = Image.open(self.images_dir / filename).convert("RGB")
        return self._resize(image)

    def _load_mask(self, stem: str) -> torch.Tensor:
        """
        Disco optico GT = union de <stem>_disc.npy y <stem>_cup.npy, binarizado y
        redimensionado (NEAREST) a image_size. Si no hay mascaras (faltan o el .npy
        esta vacio/corrupto/ilegible) -> todo ceros.
        """
        merged: np.ndarray | None = None
        for suffix in ("_disc.npy", "_cup.npy"):
            path = self.masks_dir / f"{stem}{suffix}"
            if not path.is_file():
                continue
            try:
                binary = np.load(path) > 0
            except (EOFError, OSError, ValueError) as exc:
                # .npy presente pero ilegible (0 bytes / truncado / no es npy): se ignora.
                logger.warning("Mascara ilegible %s (%s); se ignora.", path.name, exc)
                continue
            merged = binary if merged is None else (merged | binary)

        if merged is None:
            logger.warning("Sin mascara (_disc/_cup) para %s; se usa mascara vacia.", stem)
            mask = np.zeros(self.image_size, dtype=np.float32)
        else:
            # Redimensionar con NEAREST (PIL espera (ancho, alto)).
            mask_img = Image.fromarray((merged.astype(np.uint8)) * 255).resize(
                (self.image_size[1], self.image_size[0]), Image.NEAREST
            )
            mask = (np.asarray(mask_img) > 0).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        record = self.annotations[image_id]

        if self.cache_images and image_id in self._image_cache:
            image = self._image_cache[image_id]
            mask_tensor = self._mask_cache[image_id]
        else:
            image = self._load_image(record["image_filename"])
            mask_tensor = self._load_mask(image_id)
            if self.cache_images:
                self._image_cache[image_id] = image
                self._mask_cache[image_id] = mask_tensor

        image_tensor = self.image_transform(image)
        label_name = record.get("label") or "unknown"
        label = CLASS_TO_IDX.get(label_name, UNLABELED_IDX)

        return {
            "image_id": image_id,
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label,
            "label_name": label_name,
        }


class PrivateDataModule:
    """
    Capa de datos del set privado. Misma interfaz que el DataModule del experimento
    de seleccion, para reusar el pipeline (train_few_shot, PrototypeClassifier, ...).
    """

    def __init__(self, config: dict) -> None:
        self.seed: int = config.get("seed", 42)
        set_global_seed(self.seed)

        self.split_json: Path = Path(config["split_json"])
        self.images_dir: Path = Path(config["images_dir"])
        self.masks_dir: Path = Path(config["masks_dir"])
        self.image_size: tuple[int, int] = tuple(config.get("image_size", [224, 224]))
        self.batch_size: int = config.get("batch_size", 16)
        self.num_workers: int = config.get("num_workers", 2)
        self.cache_images: bool = config.get("cache_images", True)
        self.augmentations: dict = {**DEFAULT_AUGMENTATIONS, **config.get("augmentations", {})}

        self.annotations: dict[str, dict] = {}
        self.splits: dict[str, list[str]] = {}
        self._load_split_json()

        self._train_transform = build_train_transform(self.augmentations)
        self._eval_transform = build_eval_transform()

        logger.info(
            "PrivateDataModule listo | splits=%s | images_dir=%s | masks_dir=%s",
            {k: len(v) for k, v in self.splits.items()},
            self.images_dir,
            self.masks_dir,
        )

    # ---- Carga del split.json -----------------------------------------------

    def _load_split_json(self) -> None:
        """Lee el split.json privado, mapea labels y arma annotations + splits."""
        if not self.split_json.is_file():
            raise FileNotFoundError(f"No se encontro split.json en {self.split_json}.")
        with self.split_json.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        for raw_split, records in raw.items():
            split = _SPLIT_NAME_MAP.get(raw_split, raw_split)
            ids: list[str] = []
            for record in records:
                image_filename = record["image_filename"]
                image_id = Path(image_filename).stem  # "1209_right.jpg" -> "1209_right"
                raw_label = record.get("label")
                label = LABEL_MAP.get(raw_label)  # Pathological->glaucoma, Normal->normal
                if label is None and raw_label is not None:
                    logger.warning("Label desconocido (%r) en %s; queda sin label.", raw_label, image_id)
                self.annotations[image_id] = {
                    "image_filename": image_filename,
                    "label": label,
                    "split": split,
                }
                ids.append(image_id)
            self.splits[split] = ids

    # ---- Construccion de datasets / loaders ---------------------------------

    def build_dataset(self, image_ids: list[str], *, augment: bool) -> PrivateDataset:
        """Crea un PrivateDataset sobre image_ids (transform de train si augment)."""
        transform = self._train_transform if augment else self._eval_transform
        return PrivateDataset(
            self.annotations,
            image_ids,
            self.images_dir,
            self.masks_dir,
            self.image_size,
            transform,
            cache_images=self.cache_images,
        )

    def build_loader(
        self,
        image_ids: list[str],
        *,
        shuffle: bool,
        augment: bool,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> DataLoader:
        """DataLoader a partir de una lista de image_ids (mismo contrato que el experimento)."""
        dataset = self.build_dataset(image_ids, augment=augment)

        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed if seed is None else seed)

        num_workers = 0 if self.cache_images else self.num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=_seed_worker,
            pin_memory=torch.cuda.is_available(),
        )

    def get_train_loader(self) -> DataLoader:
        return self.build_loader(self.splits["train"], shuffle=True, augment=True)

    def get_val_loader(self) -> DataLoader:
        return self.build_loader(self.splits["val"], shuffle=False, augment=False)

    def get_test_loader(self) -> DataLoader:
        return self.build_loader(self.splits["test"], shuffle=False, augment=False)

    # ---- Utilidades ----------------------------------------------------------

    def get_glaucoma_indices(self, split: str = "train") -> list[str]:
        """image_id de glaucoma en un split (lo usa el muestreo del support set)."""
        return [
            image_id
            for image_id in self.splits[split]
            if self.annotations[image_id].get("label") == "glaucoma"
        ]

    def get_class_distribution(self) -> dict[str, dict[str, int]]:
        """Conteo de clases por split (verificacion)."""
        distribution: dict[str, dict[str, int]] = {}
        for split, image_ids in self.splits.items():
            counts = {"glaucoma": 0, "normal": 0, "unknown": 0}
            for image_id in image_ids:
                label = self.annotations[image_id].get("label") or "unknown"
                counts[label] = counts.get(label, 0) + 1
            distribution[split] = counts
        return distribution
