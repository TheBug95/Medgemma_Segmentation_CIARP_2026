# =============================================================================
# Classifier/efficientNet_classification.py
# =============================================================================
# Inferencia de alto nivel del clasificador: es lo que se llama para CLASIFICAR
# una imagen de fondo de ojo.
#
# Comportamiento (decision del usuario):
#   - Si la prediccion es "glaucoma": devuelve el veredicto + distribucion + el
#     Grad-CAM como MAPA NUMERICO (H,W)[0,1] e IMAGEN PNG (heatmap superpuesto).
#   - Si la prediccion es "normal":   devuelve solo el veredicto + distribucion.
#
# FUNCIONES PUBLICAS:
#   - load_config(path) -> dict
#   - load_classifier(checkpoint_path, config) -> EfficientNetClassifier
#   - overlay_heatmap(image, heatmap, colormap, alpha) -> PIL.Image   (sin torch)
#   - classify_image(image, classifier, config, out_dir=None, image_id=None) -> dict
#
# USO (CLI):
#   python efficientNet_classification.py --image foto.jpg \
#       --checkpoint checkpoints/fs_weights_split_1.pth --out-dir results/infer
#
# NOTA: load_classifier/classify_image importan efficientNet_init (torch/torchvision)
# de forma perezosa, para que overlay_heatmap se pueda usar/probar sin torch.
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Permitir importar efficientNet_init (mismo directorio) desde cualquier CWD.
_CLASSIFIER_DIR = Path(__file__).resolve().parent
if str(_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSIFIER_DIR))


# =============================================================================
# Configuracion
# =============================================================================


def load_config(path: str | Path) -> dict:
    """Carga el config.yaml del clasificador."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _image_size(config: dict) -> tuple[int, int]:
    """Tamano de imagen del config (default 448x448)."""
    size = config.get("preprocessing", {}).get("image_size", [448, 448])
    return tuple(size)


# =============================================================================
# Carga del clasificador
# =============================================================================


def load_classifier(checkpoint_path: str | Path, config: dict):
    """
    Instancia un EfficientNetClassifier y carga los pesos del checkpoint.

    Args:
        checkpoint_path: ruta a un fs_weights_split_<i>.pth.
        config: config del clasificador (se usa la seccion `model` + `gradcam`).
    """
    from efficientNet_init import EfficientNetClassifier

    model_cfg = dict(config.get("model", {}))
    model_cfg.setdefault("pretrained", False)  # se cargan pesos propios, no ImageNet
    model_cfg["gradcam"] = config.get("gradcam", {})

    classifier = EfficientNetClassifier(model_cfg)
    classifier.load(checkpoint_path)
    return classifier


# =============================================================================
# Visualizacion del Grad-CAM (numpy + PIL + matplotlib; SIN torch)
# =============================================================================


def _get_colormap(name: str):
    """Devuelve un colormap de matplotlib (API nueva con fallback a la antigua)."""
    try:
        from matplotlib import colormaps

        return colormaps[name]
    except Exception:
        from matplotlib import cm

        return cm.get_cmap(name)


def overlay_heatmap(image, heatmap: np.ndarray, colormap: str = "jet", alpha: float = 0.4):
    """
    Superpone un heatmap (H,W)[0,1] sobre una imagen con un colormap y transparencia.

    Args:
        image: PIL.Image RGB (la foto original del ojo).
        heatmap: ndarray (h,w) en [0,1] (se reescala al tamano de la imagen).
        colormap: nombre del colormap de matplotlib (ej. "jet").
        alpha: peso del heatmap en la mezcla (0=solo foto, 1=solo heatmap).

    Returns:
        PIL.Image RGB con el heatmap superpuesto.
    """
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size

    # Reescalar el heatmap (normalmente 448x448) al tamano real de la imagen.
    heatmap_uint8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_resized = Image.fromarray(heatmap_uint8).resize((width, height), Image.BILINEAR)
    heatmap_norm = np.asarray(heatmap_resized).astype(np.float32) / 255.0

    # Aplicar colormap -> RGB en [0,1].
    cmap = _get_colormap(colormap)
    colored = cmap(heatmap_norm)[..., :3]

    # Mezcla alpha con la imagen base.
    base = np.asarray(image).astype(np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * colored
    blended_uint8 = (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(blended_uint8)


# =============================================================================
# Inferencia
# =============================================================================


def _load_pil_image(image):
    """Acepta ruta (str/Path) o PIL.Image y devuelve una PIL.Image RGB."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(
        "classify_image espera una ruta o PIL.Image (para poder generar el overlay)."
    )


def classify_image(
    image,
    classifier,
    config: dict,
    out_dir: str | Path | None = None,
    image_id: str | None = None,
) -> dict:
    """
    Clasifica una imagen y, si es glaucoma, genera el Grad-CAM (mapa + PNG overlay).

    Args:
        image: ruta o PIL.Image de la foto de fondo de ojo.
        classifier: EfficientNetClassifier ya cargado.
        config: config del clasificador (gradcam, inference).
        out_dir: si se da, guarda el overlay PNG y el mapa .npy ahi (solo glaucoma).
        image_id: identificador para los archivos de salida (default: nombre del archivo).

    Returns:
        {
          "image_id": str,
          "prediction": "glaucoma" | "normal",
          "distribution": {"normal": p, "glaucoma": p},
          "predicted_index": int,
          "gradcam": None  |  {"heatmap": ndarray, "overlay_path": str|None, "npy_path": str|None}
        }
    """
    pil_image = _load_pil_image(image)
    if image_id is None:
        image_id = Path(image).stem if isinstance(image, (str, Path)) else "image"

    # Preprocesar -> tensor normalizado y predecir.
    from efficientNet_init import build_eval_transform

    transform = build_eval_transform(_image_size(config))
    image_tensor = transform(pil_image)
    prediction = classifier.predict(image_tensor)

    result = {
        "image_id": image_id,
        "prediction": prediction["prediction"],
        "distribution": prediction["distribution"],
        "predicted_index": prediction["predicted_index"],
        "gradcam": None,
    }

    # Grad-CAM solo si la prediccion es glaucoma (o si se desactiva esa condicion).
    only_glaucoma = config.get("inference", {}).get("gradcam_on_glaucoma_only", True)
    if only_glaucoma and prediction["prediction"] != "glaucoma":
        return result

    heatmap = classifier.get_gradcam(image_tensor)
    gc_cfg = config.get("gradcam", {})
    overlay = overlay_heatmap(
        pil_image,
        heatmap,
        colormap=gc_cfg.get("colormap", "jet"),
        alpha=gc_cfg.get("overlay_alpha", 0.4),
    )

    overlay_path, npy_path = None, None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = str(out_dir / f"{image_id}_gradcam.png")
        npy_path = str(out_dir / f"{image_id}_gradcam.npy")
        overlay.save(overlay_path)
        np.save(npy_path, heatmap.astype(np.float32))

    result["gradcam"] = {
        "heatmap": heatmap,
        "overlay_image": overlay,
        "overlay_path": overlay_path,
        "npy_path": npy_path,
    }
    return result


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Clasifica una sola imagen desde la terminal."""
    parser = argparse.ArgumentParser(description="Clasifica una imagen (glaucoma/normal) + Grad-CAM.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen de fondo de ojo.")
    parser.add_argument("--checkpoint", required=True, help="Ruta al .pth del modelo entrenado.")
    parser.add_argument(
        "--config",
        default=str(_CLASSIFIER_DIR / "config.yaml"),
        help="Ruta al config.yaml (default: Classifier/config.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Carpeta donde guardar el overlay PNG y el mapa .npy (si es glaucoma).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config(args.config)
    classifier = load_classifier(args.checkpoint, config)
    result = classify_image(args.image, classifier, config, out_dir=args.out_dir)

    # Resumen legible (se evita imprimir el ndarray del heatmap).
    summary = {k: v for k, v in result.items() if k != "gradcam"}
    if result["gradcam"] is not None:
        summary["gradcam"] = {
            "overlay_path": result["gradcam"]["overlay_path"],
            "npy_path": result["gradcam"]["npy_path"],
        }
    logger.info("Resultado: %s", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
