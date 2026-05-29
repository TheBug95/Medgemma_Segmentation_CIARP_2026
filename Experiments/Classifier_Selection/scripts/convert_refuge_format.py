# =============================================================================
# scripts/convert_refuge_format.py
# =============================================================================
# Convierte el dataset REFUGE al formato del proyecto (annotations.json + splits.json)
#
# INPUT (un index.json por carpeta de split):
#   - <data_dir>/train/index.json
#   - <data_dir>/val/index.json
#   - <data_dir>/test/index.json
#
# OUTPUT:
#   - <output_dir>/annotations.json
#   - <output_dir>/splits.json
#
# NOTAS SOBRE EL FORMATO REAL DE REFUGE:
#   - index.json es un DICT indexado por "0".."399" (no una lista). Cada valor
#     trae ImgName y, en train/val, Label (1=glaucoma, 0=normal).
#   - El label se toma del campo Label, NO del prefijo del nombre: solo train
#     codifica la clase en el nombre (g####/n####); val usa V#### y test T####.
#   - El split test NO trae Label -> sus entradas se generan SIN el campo 'label'.
#
# ANNOTATIONS.JSON SCHEMA (mínimo):
#   {
#     "g0001": {
#       "image_filename": "g0001.jpg",
#       "label": "glaucoma",                  # "glaucoma" | "normal"; AUSENTE en test
#       "mask_path": "train/Masks/g0001.png", # relativo a data_dir
#       "split": "train"
#     },
#     ...
#   }
#
# SPLITS.JSON SCHEMA:
#   {
#     "train": ["g0001", ..., "n0360"],
#     "val":   ["V0001", ..., "V0400"],
#     "test":  ["T0001", ..., "T0400"]
#   }
#
# USO:
#   python scripts/convert_refuge_format.py
#   (Las rutas se resuelven respecto a config.yaml, no al directorio actual.)
# =============================================================================

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Mapeo del campo numérico Label de REFUGE a la etiqueta textual del proyecto.
LABEL_MAP = {1: "glaucoma", 0: "normal"}

# Splits nativos de REFUGE (cada uno es una carpeta con su propio index.json).
SPLITS = ("train", "val", "test")


def load_config(config_path: Path) -> dict:
    """Carga config.yaml y devuelve el diccionario de configuración."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_split_annotations(split: str, data_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """
    Construye las anotaciones de un split a partir de su index.json.

    El label se asigna desde el campo `Label` de index.json. Las entradas sin
    `Label` (split test) se generan sin la clave `label`.

    Args:
        split: Nombre del split ("train", "val" o "test").
        data_dir: Ruta absoluta a la raíz del dataset REFUGE.

    Returns:
        (annotations, ids):
            annotations: {image_id: {image_filename, [label], mask_path, split}}
            ids: lista de image_id en el orden original del index.json
    """
    index_path = data_dir / split / "index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index: dict = json.load(f)

    annotations: dict[str, dict] = {}
    ids: list[str] = []

    # Recorrer en orden numérico del index para que la salida sea determinista.
    for key in sorted(index, key=int):
        entry = index[key]
        image_filename = entry["ImgName"]
        image_id = Path(image_filename).stem  # "g0001.jpg" -> "g0001"

        # La máscara comparte basename con la imagen, en .png dentro de Masks/.
        mask_rel = f"{split}/Masks/{image_id}.png"
        if not (data_dir / mask_rel).is_file():
            logger.warning("Máscara no encontrada para %s: %s", image_id, mask_rel)

        record: dict = {"image_filename": image_filename}

        # test no trae Label -> se omite el campo (no se escribe label=null).
        label_value = entry.get("Label")
        if label_value is not None:
            if label_value in LABEL_MAP:
                record["label"] = LABEL_MAP[label_value]
            else:
                logger.warning(
                    "Label desconocido (%r) en %s; se omite el campo label",
                    label_value,
                    image_id,
                )

        record["mask_path"] = mask_rel
        record["split"] = split

        annotations[image_id] = record
        ids.append(image_id)

    return annotations, ids


def convert_refuge(data_dir: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """
    Convierte los tres splits de REFUGE al formato del proyecto.

    Args:
        data_dir: Ruta absoluta a la raíz del dataset REFUGE.

    Returns:
        (annotations, splits):
            annotations: dict global {image_id: record}
            splits: {"train": [...], "val": [...], "test": [...]}
    """
    annotations: dict[str, dict] = {}
    splits: dict[str, list[str]] = {}

    for split in SPLITS:
        split_annotations, ids = build_split_annotations(split, data_dir)
        annotations.update(split_annotations)
        splits[split] = ids
        _log_split_summary(split, split_annotations)

    return annotations, splits


def _log_split_summary(split: str, annotations: dict[str, dict]) -> None:
    """Registra conteos por clase de un split para verificación rápida."""
    n_total = len(annotations)
    n_glaucoma = sum(1 for r in annotations.values() if r.get("label") == "glaucoma")
    n_normal = sum(1 for r in annotations.values() if r.get("label") == "normal")
    n_unlabeled = n_total - n_glaucoma - n_normal
    logger.info(
        "Split %-5s: %d imágenes (glaucoma=%d, normal=%d, sin label=%d)",
        split,
        n_total,
        n_glaucoma,
        n_normal,
        n_unlabeled,
    )


def write_json(data: dict, path: Path) -> None:
    """Escribe un dict como JSON indentado, creando el directorio si hace falta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Lee config.yaml, convierte REFUGE y escribe annotations.json y splits.json."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Rutas resueltas respecto a este script (independiente del directorio actual).
    script_dir = Path(__file__).resolve().parent
    experiment_dir = script_dir.parent
    config_path = experiment_dir / "config.yaml"

    config = load_config(config_path)
    data_cfg = config["data"]

    # data_dir y output_dir del config son relativos al directorio del config.
    data_dir = (experiment_dir / data_cfg["data_dir"]).resolve()
    output_dir = (experiment_dir / data_cfg["output_dir"]).resolve()

    logger.info("data_dir   = %s", data_dir)
    logger.info("output_dir = %s", output_dir)

    annotations, splits = convert_refuge(data_dir)

    write_json(annotations, output_dir / "annotations.json")
    write_json(splits, output_dir / "splits.json")

    total = sum(len(ids) for ids in splits.values())
    logger.info("Conversión completa: %d imágenes -> %s", total, output_dir)


if __name__ == "__main__":
    main()
