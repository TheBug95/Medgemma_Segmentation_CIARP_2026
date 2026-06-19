# Code to inference EfficientNet on test images from the split
"""
Inferencia del mejor EfficientNet-B0 (meta-prototipos) sobre un split (TEST por defecto).

Carga el .pth (prototipos + backbone meta-entrenado) y corre `predict` sobre cada
imagen, devolviendo lo que consume metrics_generation.py:
  - y_true  : 0 = normal, 1 = glaucoma (del split.json)
  - y_proba : P(glaucoma) = softmax(distancias negativas)["glaucoma"]
  - y_pred  : 1 si P(glaucoma) > umbral (0.5 por defecto), si no 0
  - ids     : image_id de cada imagen (para la tabla por imagen)

Resuelve sus imports por la UBICACION del archivo (CWD-independiente): sirve tanto
importado desde el notebook como ejecutado directo:
    python inference_on_test.py --pth best.pth --dataset-dir DIR --split-json split.json
"""

import os
import sys

# --- sys.path por ubicacion del archivo (repo/Classifier/Inference/este_archivo) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (
    os.path.join(_REPO, "Experiments", "Classifier_Selection"),  # modules/
    os.path.join(_REPO, "Classifier"),                            # private_data_module
    _HERE,                                                        # metrics_generation
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from modules.prototype_classifier import PrototypeClassifier


def run_inference(
    pth_path,
    data_module,
    *,
    backbone: str = "efficientnet_b0",
    split: str = "test",
    threshold: float = 0.5,
    seed: int = 42,
):
    """
    Infiere sobre `split` con el modelo guardado en `pth_path`.

    Returns:
        (y_true, y_proba, y_pred, ids): los 3 primeros np.ndarray; ids = list[str].
    """
    # Backbone meta -> freeze_backbone=False. pretrained=False: load() restaura los pesos.
    clf = PrototypeClassifier({
        "backbone": backbone,
        "num_classes": 2,
        "pretrained": False,
        "freeze_backbone": False,
        "seed": seed,
    })
    clf.load(pth_path)
    clf.model.eval()

    ds = data_module.build_dataset(data_module.splits[split], augment=False)

    ids: list[str] = []
    y_true: list[int] = []
    y_proba: list[float] = []
    for k in range(len(ds)):
        item = ds[k]
        if int(item["label"]) < 0:
            raise ValueError(f"La imagen {item['image_id']} no tiene label en el split '{split}'.")
        out = clf.predict(item["image"])  # softmax sobre distancias negativas
        ids.append(item["image_id"])
        y_true.append(int(item["label"]))  # 0 = normal, 1 = glaucoma
        y_proba.append(float(out["distribution"]["glaucoma"]))

    y_true_arr = np.array(y_true, dtype=int)
    y_proba_arr = np.array(y_proba, dtype=float)
    y_pred_arr = (y_proba_arr > threshold).astype(int)  # > 0.5 -> glaucoma
    return y_true_arr, y_proba_arr, y_pred_arr, ids


if __name__ == "__main__":
    import argparse

    from private_data_module import PrivateDataModule
    from metrics_generation import compute_all_metrics, print_report

    ap = argparse.ArgumentParser(
        description="Inferencia en test + metricas (bootstrap estratificado + Clopper-Pearson)."
    )
    ap.add_argument("--pth", required=True, help="Ruta al .pth de la mejor config.")
    ap.add_argument("--dataset-dir", required=True, help="Carpeta plana con imagenes (+ mascaras).")
    ap.add_argument("--split-json", required=True, help="split.json del dataset privado.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    dm = PrivateDataModule({
        "split_json": args.split_json,
        "images_dir": args.dataset_dir,
        "masks_dir": args.dataset_dir,
        "image_size": [args.image_size, args.image_size],
        "batch_size": 16,
        "seed": args.seed,
        "cache_images": True,
    })
    y_true, y_proba, y_pred, ids = run_inference(
        args.pth, dm, split=args.split, threshold=0.5, seed=args.seed
    )
    print_report(
        compute_all_metrics(y_true, y_proba, y_pred, n_bootstrap=args.n_bootstrap, seed=args.seed)
    )
