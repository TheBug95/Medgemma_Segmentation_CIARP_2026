# =============================================================================
# Classifier/training/data_interface.py
# =============================================================================
# Puente entre los archivos split del usuario (split_repetition_*.json) y el
# clasificador. Lee splits, deduce la etiqueta glaucoma/normal del .json de
# anotacion de cada imagen, arma las listas (ruta, etiqueta), submuestrea a N y
# construye los DataLoaders.
#
# FORMATO DE LOS SPLITS (split_repetition_i.json):
#   {
#     "train":      [ {"image": "1209/1209_right.jpg", "annotation": "1209/1209_right.json", ...}, ... ],
#     "validation": [ ... ],
#     "test":       [ ... ],
#     "metadata":   { "repetition": 1, "train_count": 57, ... }
#   }
#   Las rutas son RELATIVAS a data_root (que en Colab apunta al Drive montado).
#   La etiqueta NO viene en el split: se lee del .json de anotacion (campo `label`).
#
# DEPENDENCIAS:
#   Las funciones "puras" (load_split, read_label, build_samples, subsample) usan
#   solo la libreria estandar -> se pueden probar SIN torch/torchvision. Solo
#   build_loader/_build_dataset importan torch/torchvision (de forma perezosa).
# =============================================================================

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Permitir importar efficientNet_init (carpeta padre) desde build_loader.
_CLASSIFIER_DIR = Path(__file__).resolve().parents[1]
if str(_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSIFIER_DIR))

# Mapeo clase textual -> indice. DEBE casar con CLASS_TO_IDX de efficientNet_init.
# Se define local para que este modulo no dependa de torch (efficientNet_init si).
CLASS_TO_IDX: dict[str, int] = {"normal": 0, "glaucoma": 1}

# Mapeo por defecto del campo `label` del dataset -> clase del clasificador.
DEFAULT_LABEL_MAP: dict[str, str] = {"Pathological": "glaucoma", "Normal": "normal"}

# Secciones esperadas en cada split. 'train_fewshot' es el conjunto YA balanceado
# (15 glaucoma + 15 normal) que se usa para el entrenamiento few-shot del
# clasificador; 'train' es el conjunto grande (no se usa para este few-shot).
SPLIT_SECTIONS = ("train", "train_fewshot", "validation", "test")


# =============================================================================
# Lectura de splits y etiquetas (puro Python, sin torch)
# =============================================================================


def load_split(split_path: str | Path) -> dict:
    """
    Carga un split_repetition_i.json y valida que tenga las secciones esperadas.

    Returns:
        El dict del JSON (con claves train/validation/test y metadata).
    """
    split_path = Path(split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"No se encontro el split {split_path}.")
    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    missing = [s for s in SPLIT_SECTIONS if s not in split]
    if missing:
        logger.warning("El split %s no tiene las secciones %s.", split_path.name, missing)
    return split


def _normalize_label(raw: str, label_map: dict[str, str]) -> str:
    """Mapea el valor crudo del campo label a una clase del clasificador (tolerante)."""
    if raw in label_map:
        return label_map[raw]
    low = str(raw).strip().lower()
    # Coincidencia case-insensitive contra las claves del label_map.
    for key, value in label_map.items():
        if key.lower() == low:
            return value
    # Quizas el valor ya es una clase final (ej. "glaucoma"/"normal").
    if low in CLASS_TO_IDX:
        return low
    raise ValueError(
        f"Etiqueta '{raw}' no reconocida. label_map={label_map}, "
        f"clases validas={list(CLASS_TO_IDX)}."
    )


def read_label(
    annotation_path: str | Path,
    label_field: str = "label",
    label_map: dict[str, str] | None = None,
) -> int:
    """
    Lee la etiqueta glaucoma/normal del .json de anotacion de una imagen.

    El .json de anotacion es un ARRAY con un registro (a veces varios anotadores)
    por imagen; ejemplo real:
        [ { "id": 485, "image_filename": "1494_right.jpg", "label": "Normal",
            "locs_data": {}, ... } ]
    Se toma el primer registro. Tambien se acepta un objeto directo {...}.

    Args:
        annotation_path: ruta al .json de anotacion (ej. data_root/"1209/1209_right.json").
        label_field: campo del que sale la etiqueta (default "label").
        label_map: mapeo valor_dataset -> clase (default Pathological/Normal).

    Returns:
        Indice de clase: normal=0, glaucoma=1.
    """
    label_map = label_map or DEFAULT_LABEL_MAP
    annotation_path = Path(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as f:
        annotation = json.load(f)

    # La anotacion viene como lista [ {registro} ]; tomar el primer registro.
    if isinstance(annotation, list):
        if not annotation:
            raise ValueError(f"Anotacion vacia (lista sin registros): {annotation_path}.")
        if len(annotation) > 1:
            logger.warning(
                "Anotacion %s tiene %d registros; se usa el primero.",
                annotation_path.name,
                len(annotation),
            )
        annotation = annotation[0]

    if label_field not in annotation:
        raise KeyError(
            f"El campo '{label_field}' no esta en {annotation_path}. "
            f"Campos disponibles: {list(annotation)}."
        )
    class_name = _normalize_label(annotation[label_field], label_map)
    return CLASS_TO_IDX[class_name]


def build_samples(
    entries: list[dict],
    data_root: str | Path,
    label_field: str = "label",
    label_map: dict[str, str] | None = None,
    skip_missing: bool = False,
) -> list[tuple[Path, int]]:
    """
    Convierte las entradas de un split en una lista de (ruta_imagen, etiqueta).

    Args:
        entries: lista de dicts del split (cada uno con "image" y "annotation").
        data_root: raiz contra la que se resuelven las rutas relativas del split.
        label_field / label_map: como en read_label.
        skip_missing: si True, omite (con warning) las entradas cuya imagen o
            anotacion no exista; si False, lanza error.

    Returns:
        Lista de (Path imagen absoluta, indice de clase).
    """
    data_root = Path(data_root)
    label_map = label_map or DEFAULT_LABEL_MAP
    samples: list[tuple[Path, int]] = []

    for entry in entries:
        image_path = data_root / entry["image"]
        annotation_path = data_root / entry["annotation"]

        if skip_missing and not (image_path.is_file() and annotation_path.is_file()):
            logger.warning("Omitida (falta imagen o anotacion): %s", entry.get("image"))
            continue

        label = read_label(annotation_path, label_field=label_field, label_map=label_map)
        samples.append((image_path, label))

    return samples


def _count_by_class(samples: list[tuple[Path, int]]) -> dict[int, int]:
    """Cuenta cuantas muestras hay por clase."""
    counts: dict[int, int] = {}
    for _, label in samples:
        counts[label] = counts.get(label, 0) + 1
    return counts


def subsample(
    samples: list[tuple[Path, int]],
    size: int | None,
    mode: str = "balanced",
    seed: int = 42,
) -> list[tuple[Path, int]]:
    """
    Submuestrea la lista de muestras a N elementos de forma reproducible.

    Modos:
        - "balanced":   size/2 de cada clase (ej. 20 glaucoma + 20 normal).
        - "stratified": size manteniendo la proporcion de clases del conjunto.
        - "random":     size al azar, sin mirar la clase.

    Si size es None o >= len(samples), devuelve todas las muestras (barajadas).
    Usa un RNG propio sembrado con `seed` (independiente del estado global).
    """
    rng = random.Random(seed)
    n_total = len(samples)

    if size is None or size >= n_total:
        if size is not None and size > n_total:
            logger.warning(
                "subsample: size=%d > disponibles=%d; se usan todas las muestras.",
                size,
                n_total,
            )
        shuffled = list(samples)
        rng.shuffle(shuffled)
        return shuffled

    if mode == "random":
        return rng.sample(list(samples), size)

    # Separar por clase (orden estable para reproducibilidad).
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)
    classes = sorted(by_class)

    if mode == "balanced":
        per_class = size // len(classes)
        target = {c: per_class for c in classes}
        # Repartir el remanente (si size no es divisible) entre las primeras clases.
        for c in classes[: size - per_class * len(classes)]:
            target[c] += 1
    elif mode == "stratified":
        target = {c: round(size * len(by_class[c]) / n_total) for c in classes}
        # Ajustar para que la suma sea exactamente `size`.
        _adjust_to_total(target, classes, size)
    else:
        raise ValueError(f"mode '{mode}' invalido. Use balanced | stratified | random.")

    selected: list[tuple[Path, int]] = []
    for c in classes:
        available = by_class[c]
        want = target[c]
        if want > len(available):
            logger.warning(
                "subsample(%s): clase %d pide %d pero solo hay %d; se usan %d.",
                mode,
                c,
                want,
                len(available),
                len(available),
            )
            want = len(available)
        selected.extend(rng.sample(available, want))

    rng.shuffle(selected)
    return selected


def _adjust_to_total(target: dict[int, int], classes: list[int], size: int) -> None:
    """Ajusta el reparto por clase para que sume exactamente `size` (in-place)."""
    diff = size - sum(target.values())
    i = 0
    while diff != 0 and classes:
        c = classes[i % len(classes)]
        if diff > 0:
            target[c] += 1
            diff -= 1
        elif target[c] > 0:
            target[c] -= 1
            diff += 1
        i += 1


# =============================================================================
# Construccion de DataLoaders (requiere torch/torchvision; import perezoso)
# =============================================================================


def _build_dataset(samples: list[tuple[Path, int]], transform):
    """Crea un Dataset de PyTorch sobre la lista de (ruta, etiqueta). Import perezoso."""
    from PIL import Image
    from torch.utils.data import Dataset

    class SimpleImageDataset(Dataset):
        """Dataset minimo: abre la imagen, aplica transform y entrega dict M1-compatible."""

        def __init__(self, samples, transform):
            self.samples = list(samples)
            self.transform = transform

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            image_path, label = self.samples[idx]
            image = Image.open(image_path).convert("RGB")
            return {
                "image": self.transform(image),
                "label": label,
                "image_path": str(image_path),
            }

    return SimpleImageDataset(samples, transform)


def build_loader(
    samples: list[tuple[Path, int]],
    *,
    augment: bool,
    batch_size: int = 16,
    seed: int = 42,
    shuffle: bool = True,
    augmentations: dict | None = None,
    image_size: tuple[int, int] = (448, 448),
    num_workers: int = 0,
):
    """
    Construye un DataLoader a partir de una lista de (ruta_imagen, etiqueta).

    Args:
        samples: lista de (Path, int) producida por build_samples/subsample.
        augment: aplicar augmentations (True en train, False en val/test).
        batch_size, seed, shuffle, num_workers: parametros del DataLoader.
        augmentations: dict de la seccion `augmentations` del config (solo si augment).
        image_size: tamano destino de la imagen.

    Returns:
        torch.utils.data.DataLoader cuyos batches son dicts {"image","label","image_path"}.
    """
    import torch
    from torch.utils.data import DataLoader

    # Los transforms viven en efficientNet_init (preprocesamiento fijo del modelo).
    from efficientNet_init import build_eval_transform, build_train_transform

    transform = (
        build_train_transform(augmentations, image_size)
        if augment
        else build_eval_transform(image_size)
    )
    dataset = _build_dataset(samples, transform)

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
