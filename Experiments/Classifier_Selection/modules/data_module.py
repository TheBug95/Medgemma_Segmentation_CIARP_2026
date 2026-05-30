# =============================================================================
# modules/data_module.py
# =============================================================================
# M1: DataModule - Carga y preprocesa datos REFUGE para el clasificador CNN
#
# RESPONSABILIDADES:
#   1. Cargar annotations.json y splits.json (NO regenerar splits)
#   2. Preparar cada imagen para la CNN (resize 448, normalización ImageNet)
#   3. Aplicar augmentations SOLO en train
#   4. Aplanar máscaras 3-clase (0=bg, 1=rim, 2=cup) a binaria (0=bg, 1=OD)
#   5. Exponer los DataLoaders y utilidades que consumen few_shot / extract_gradcam
#
# CONFIG ESPERADA (sección `data` de config.yaml; seed y augmentations opcionales):
#   {
#       "data_dir": "../../Datasets/REFUGE",  # raíz del dataset (rel. al dir del experimento)
#       "output_dir": "./data",               # donde viven annotations.json y splits.json
#       "image_size": [448, 448],
#       "batch_size": 16,
#       "num_workers": 2,
#       "seed": 42,                           # opcional (default 42)
#       "augmentations": {...}                # opcional (default = DEFAULT_AUGMENTATIONS)
#   }
#
# MÉTODOS PÚBLICOS:
#   - __init__(config: dict)
#   - get_train_loader() -> DataLoader        # con augmentations, shuffle
#   - get_val_loader()   -> DataLoader        # sin augmentations
#   - get_test_loader()  -> DataLoader        # sin augmentations (test no tiene label)
#   - get_glaucoma_indices(split) -> list[str]   # IDs de glaucoma de un split
#   - build_loader(image_ids, ...) -> DataLoader # loader desde una lista de IDs (few_shot)
#   - get_sample(index, split) -> dict        # una muestra (debug/inspección)
#   - get_class_distribution() -> dict        # conteo por clase y split (verificación)
#
# CADA MUESTRA (dict) CONTIENE:
#   - image_id   : str
#   - image      : Tensor (3, 448, 448) normalizada ImageNet
#   - mask       : Tensor (1, 448, 448) binaria float {0., 1.} (disco óptico)
#   - label      : int  -> normal=0, glaucoma=1, sin label (test)=-1
#   - label_name : str  -> "normal" | "glaucoma" | "unknown"
#
# NOTAS:
#   - El label viene del campo `label` de annotations.json (no del nombre del archivo).
#   - La máscara solo se redimensiona (NEAREST) + binariza; las augmentations geométricas
#     se aplican SOLO a la imagen. En este experimento la máscara únicamente se usa en
#     val/test (sin augmentation, para Grad-CAM IoU), así que no hay desalineación real.
#   - mixup (config.augmentations.mixup) es a nivel de batch y mezcla labels: se maneja
#     en el training loop de few_shot.py, NO aquí.
#   - Requiere torch/torchvision (entorno Colab/GPU). PIL/numpy para E/S de imagen.
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

# Mapeo clase textual -> índice. glaucoma=1 (clase de interés/minoritaria), normal=0.
# Coincide con el campo Label original de REFUGE (1=glaucoma, 0=normal).
CLASS_TO_IDX: dict[str, int] = {"normal": 0, "glaucoma": 1}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

# Índice para muestras sin label (split test).
UNLABELED_IDX = -1

# Normalización ImageNet (los backbones pre-entrenados la esperan).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Augmentations por defecto (espejo de la sección `augmentations` de config.yaml).
# Se usan si el config no provee su propia sección `augmentations`.
DEFAULT_AUGMENTATIONS: dict = {
    "horizontal_flip": True,
    "rotation_degrees": 15,
    "color_jitter_brightness": 0.1,
    "color_jitter_contrast": 0.1,
    "random_erasing": True,
    "random_erasing_p": 0.2,
    "mixup": True,
    "mixup_alpha": 0.2,
}

# Directorio del experimento (Classifier_Selection), usado para resolver rutas
# relativas del config de forma independiente del directorio de trabajo actual.
_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def set_global_seed(seed: int) -> None:
    """Fija las semillas globales (random, numpy, torch) para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """worker_init_fn: deriva la semilla de cada worker del estado de torch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _resolve_dir(path_str: str) -> Path:
    """Resuelve una ruta del config: absoluta tal cual, relativa al dir del experimento."""
    path = Path(path_str)
    return path if path.is_absolute() else (_EXPERIMENT_DIR / path).resolve()


class RefugeDataset(Dataset):
    """
    Dataset de PyTorch sobre REFUGE para clasificación binaria (glaucoma vs normal).

    Lee cada imagen y su máscara GT bajo demanda (no carga todo en memoria).
    """

    def __init__(
        self,
        annotations: dict[str, dict],
        image_ids: list[str],
        data_dir: Path,
        image_size: tuple[int, int],
        image_transform: T.Compose,
    ) -> None:
        """
        Args:
            annotations: dict global {image_id: record} de annotations.json.
            image_ids: IDs que componen este dataset (subconjunto de un split).
            data_dir: Raíz del dataset REFUGE (absoluta).
            image_size: (alto, ancho) destino, p.ej. (448, 448).
            image_transform: Transform de torchvision aplicado a la imagen RGB.
        """
        self.annotations = annotations
        self.image_ids = list(image_ids)
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_mask(self, mask_rel: str) -> torch.Tensor:
        """Carga la máscara 3-clase, la redimensiona (NEAREST) y la binariza a {0,1}."""
        mask_img = Image.open(self.data_dir / mask_rel).convert("L")
        # PIL.resize espera (ancho, alto); NEAREST preserva los bordes discretos.
        mask_img = mask_img.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        mask_arr = np.asarray(mask_img)
        # Disco óptico = rim (1) + copa (2) = cualquier píxel > 0.
        binary = (mask_arr > 0).astype(np.float32)
        return torch.from_numpy(binary).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        record = self.annotations[image_id]

        # Ruta de imagen: <data_dir>/<split>/Images/<image_filename>.
        image_path = self.data_dir / record["split"] / "Images" / record["image_filename"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        mask_tensor = self._load_mask(record["mask_path"])

        label_name = record.get("label", "unknown")
        label = CLASS_TO_IDX.get(label_name, UNLABELED_IDX)

        return {
            "image_id": image_id,
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label,
            "label_name": label_name,
        }


class DataModule:
    """
    M1: capa de datos del experimento de selección de clasificador.

    Carga annotations.json/splits.json (generados por convert_refuge_format.py),
    prepara las transformaciones y entrega DataLoaders listos para la CNN.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Sección `data` de config.yaml. Puede incluir `seed` y
                `augmentations`; si faltan, se usan 42 y DEFAULT_AUGMENTATIONS.
        """
        self.seed: int = config.get("seed", 42)
        set_global_seed(self.seed)

        self.data_dir: Path = _resolve_dir(config["data_dir"])
        self.output_dir: Path = _resolve_dir(config["output_dir"])
        self.image_size: tuple[int, int] = tuple(config.get("image_size", [448, 448]))
        self.batch_size: int = config.get("batch_size", 16)
        self.num_workers: int = config.get("num_workers", 2)
        self.augmentations: dict = {**DEFAULT_AUGMENTATIONS, **config.get("augmentations", {})}

        self.annotations: dict[str, dict] = self._load_json(self.output_dir / "annotations.json")
        self.splits: dict[str, list[str]] = self._load_json(self.output_dir / "splits.json")

        self._train_transform = self._build_train_transform()
        self._eval_transform = self._build_eval_transform()

        logger.info(
            "DataModule listo | data_dir=%s | splits=%s",
            self.data_dir,
            {k: len(v) for k, v in self.splits.items()},
        )

    # ---- Carga de archivos ---------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> dict:
        """Carga un JSON; error claro si no existe (hay que correr convert_refuge_format.py)."""
        if not path.is_file():
            raise FileNotFoundError(
                f"No se encontró {path}. Ejecuta antes scripts/convert_refuge_format.py."
            )
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ---- Transformaciones ----------------------------------------------------

    def _build_eval_transform(self) -> T.Compose:
        """Transform de evaluación: resize + normalización (sin augmentation)."""
        return T.Compose(
            [
                T.Resize(self.image_size, antialias=True),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _build_train_transform(self) -> T.Compose:
        """Transform de entrenamiento: resize + augmentations + normalización."""
        aug = self.augmentations
        steps: list = [T.Resize(self.image_size, antialias=True)]

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

        # NOTA: mixup NO va aquí (es a nivel de batch y mezcla labels) -> few_shot.py.
        return T.Compose(steps)

    # ---- Construcción de loaders ---------------------------------------------

    def build_loader(
        self,
        image_ids: list[str],
        *,
        shuffle: bool,
        augment: bool,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> DataLoader:
        """
        Construye un DataLoader a partir de una lista explícita de image_ids.

        Es el método reutilizable que usan get_*_loader() y el few-shot (donde
        image_ids es el support set de glaucoma muestreado con una seed dada).

        Args:
            image_ids: IDs a incluir en el loader.
            shuffle: Barajar cada época (True en train, False en val/test).
            augment: Aplicar augmentations (True solo en train/few-shot).
            batch_size: Tamaño de batch; por defecto el de config.
            seed: Semilla para el shuffle reproducible; por defecto self.seed.
        """
        transform = self._train_transform if augment else self._eval_transform
        dataset = RefugeDataset(
            self.annotations, image_ids, self.data_dir, self.image_size, transform
        )

        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed if seed is None else seed)

        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=_seed_worker,
            pin_memory=torch.cuda.is_available(),
        )

    def get_train_loader(self) -> DataLoader:
        """DataLoader del split train completo (con augmentation y shuffle)."""
        return self.build_loader(self.splits["train"], shuffle=True, augment=True)

    def get_val_loader(self) -> DataLoader:
        """DataLoader del split val completo (sin augmentation, orden fijo)."""
        return self.build_loader(self.splits["val"], shuffle=False, augment=False)

    def get_test_loader(self) -> DataLoader:
        """DataLoader del split test completo (sin augmentation; test no tiene label)."""
        return self.build_loader(self.splits["test"], shuffle=False, augment=False)

    # ---- Utilidades ----------------------------------------------------------

    def get_glaucoma_indices(self, split: str = "train") -> list[str]:
        """
        Devuelve los image_id de glaucoma en un split, en el orden del split.

        En REFUGE/train son 40 IDs; few_shot.py muestrea N de aquí.
        """
        return [
            image_id
            for image_id in self.splits[split]
            if self.annotations[image_id].get("label") == "glaucoma"
        ]

    def get_sample(self, index: int, split: str = "train") -> dict:
        """Devuelve la muestra `index` de `split` (transform de eval, para debug)."""
        image_ids = self.splits[split]
        dataset = RefugeDataset(
            self.annotations, image_ids, self.data_dir, self.image_size, self._eval_transform
        )
        return dataset[index]

    def get_class_distribution(self) -> dict[str, dict[str, int]]:
        """Conteo de clases por split (glaucoma / normal / unknown) para verificación."""
        distribution: dict[str, dict[str, int]] = {}
        for split, image_ids in self.splits.items():
            counts = {"glaucoma": 0, "normal": 0, "unknown": 0}
            for image_id in image_ids:
                label_name = self.annotations[image_id].get("label", "unknown")
                counts[label_name] = counts.get(label_name, 0) + 1
            distribution[split] = counts
        return distribution
