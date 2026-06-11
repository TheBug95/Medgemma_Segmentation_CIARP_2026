# =============================================================================
# Classifier/training/few_shot.py
# =============================================================================
# Entrenamiento + testing del clasificador con el protocolo de 5 REPETICIONES.
#
# PROTOCOLO (una por archivo split):
#   Para la repeticion i (1..5):
#     1. Cargar split_repetition_i.json con seed_i.
#     2. Entrenar con su seccion 'train' (submuestreada a N=40) usando 'validation'
#        para early stopping.
#     3. Guardar los pesos en checkpoints/fs_weights_split_<i>.pth.
#     4. Testear con su seccion 'test': guardar predicciones por imagen + Grad-CAM
#        (mapa .npy + PNG overlay) SOLO de las predichas glaucoma.
#   Resultado: 5 modelos + 5 carpetas results/split_<i>/.
#
#   Las metricas (accuracy/F1) NO se calculan aqui: se derivan despues de las
#   predicciones guardadas en results/split_<i>/predictions.json.
#
# FUNCIONES PUBLICAS:
#   - train_one_repetition(split_path, seed, config, rep_index) -> dict
#   - test_one_repetition(classifier, test_samples, config, out_dir) -> dict
#   - run_all_repetitions(config) -> dict
#
# USO (CLI):  python -m training.few_shot --config config.yaml
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Permitir imports absolutos del paquete Classifier desde cualquier CWD.
_CLASSIFIER_DIR = Path(__file__).resolve().parents[1]
if str(_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(_CLASSIFIER_DIR))

from efficientNet_init import EfficientNetClassifier, set_global_seed  # noqa: E402
from efficientNet_classification import classify_image, load_config  # noqa: E402
from training.data_interface import (  # noqa: E402
    CLASS_TO_IDX,
    build_loader,
    build_samples,
    load_split,
    subsample,
)

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# =============================================================================
# Helpers de rutas/config
# =============================================================================


def _resolve(path_str: str | Path) -> Path:
    """Resuelve una ruta del config: absoluta tal cual, relativa al dir del Classifier."""
    path = Path(path_str)
    return path if path.is_absolute() else (_CLASSIFIER_DIR / path).resolve()


def _data_label_cfg(config: dict) -> tuple[Path, str, dict]:
    """Extrae (data_root, label_field, label_map) de la seccion `data` del config."""
    data_cfg = config["data"]
    data_root = _resolve(data_cfg["data_root"])
    label_field = data_cfg.get("label_field", "label")
    label_map = data_cfg.get("label_map")
    return data_root, label_field, label_map


# =============================================================================
# Entrenamiento de una repeticion
# =============================================================================


def train_one_repetition(
    split_path: str | Path, seed: int, config: dict, rep_index: int
) -> dict:
    """
    Entrena el clasificador para una repeticion (un archivo split + una seed).

    Returns:
        {"classifier", "checkpoint", "history", "split", "repetition", "seed"}.
    """
    set_global_seed(seed)
    split = load_split(split_path)

    data_root, label_field, label_map = _data_label_cfg(config)
    train_cfg = config["training"]
    sub_cfg = train_cfg.get("train_subsample", {})
    train_section = train_cfg.get("train_section", "train_fewshot")
    batch_size = train_cfg.get("batch_size", 16)
    image_size = tuple(config.get("preprocessing", {}).get("image_size", [448, 448]))
    augmentations = config.get("augmentations", {})

    # --- Support set few-shot: la seccion 'train_fewshot' del split (ya balanceada:
    #     15 glaucoma + 15 normal). El submuestreo es OPCIONAL: solo si se define
    #     training.train_subsample.size; por defecto se usa el conjunto completo. ---
    train_samples = build_samples(
        split.get(train_section, []), data_root, label_field, label_map
    )
    sub_size = sub_cfg.get("size")
    if sub_size:
        train_samples = subsample(
            train_samples, size=sub_size, mode=sub_cfg.get("mode", "balanced"), seed=seed
        )
    val_samples = build_samples(
        split.get("validation", []), data_root, label_field, label_map
    )
    logger.info(
        "[rep %d | seed %d] %s=%d | val=%d",
        rep_index,
        seed,
        train_section,
        len(train_samples),
        len(val_samples),
    )

    train_loader = build_loader(
        train_samples,
        augment=True,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        augmentations=augmentations,
        image_size=image_size,
    )
    val_loader = build_loader(
        val_samples,
        augment=False,
        batch_size=batch_size,
        seed=seed,
        shuffle=False,
        image_size=image_size,
    )

    # --- Modelo + entrenamiento ---
    model_cfg = dict(config.get("model", {}))
    model_cfg["seed"] = seed
    model_cfg["gradcam"] = config.get("gradcam", {})
    classifier = EfficientNetClassifier(model_cfg)

    history = classifier.train(
        train_loader,
        val_loader,
        epochs=train_cfg.get("epochs", 30),
        lr=train_cfg.get("lr", 1e-4),
        patience=train_cfg.get("patience", 5),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        scheduler_factor=train_cfg.get("scheduler_factor", 0.5),
        scheduler_patience=train_cfg.get("scheduler_patience", 3),
    )

    # --- Guardar pesos ---
    checkpoints_dir = _resolve(config["paths"]["checkpoints_dir"])
    checkpoint = checkpoints_dir / f"fs_weights_split_{rep_index}.pth"
    classifier.save(checkpoint)
    logger.info(
        "[rep %d | seed %d] entrenamiento listo | best_val_f1=%.4f | epochs=%d -> %s",
        rep_index,
        seed,
        history["best_val_f1"],
        history["epochs_trained"],
        checkpoint.name,
    )

    return {
        "classifier": classifier,
        "checkpoint": str(checkpoint),
        "history": history,
        "split": split,
        "repetition": rep_index,
        "seed": seed,
    }


# =============================================================================
# Testing de una repeticion
# =============================================================================


def test_one_repetition(
    classifier,
    test_samples: list[tuple[Path, int]],
    config: dict,
    out_dir: str | Path,
) -> dict:
    """
    Testea el clasificador con el 'test' de su split y guarda las predicciones.

    Por cada imagen guarda en predictions.json: id, ruta, etiqueta real, clase
    predicha y probabilidades. Para las predichas glaucoma, ademas guarda el
    Grad-CAM (PNG overlay + .npy) en out_dir/gradcam/. No calcula metricas.

    Returns:
        {"out_dir", "predictions_path", "n_test", "n_gradcam"}.
    """
    out_dir = Path(out_dir)
    gradcam_dir = out_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict] = []
    n_gradcam = 0

    for image_path, true_label in test_samples:
        image_id = Path(image_path).stem
        result = classify_image(
            image_path, classifier, config, out_dir=gradcam_dir, image_id=image_id
        )

        record = {
            "image_id": image_id,
            "image_path": str(image_path),
            "true_label": int(true_label),
            "true_class": IDX_TO_CLASS.get(int(true_label), "unknown"),
            "pred_index": result["predicted_index"],
            "pred_class": result["prediction"],
            "prob_normal": result["distribution"].get("normal"),
            "prob_glaucoma": result["distribution"].get("glaucoma"),
            "gradcam_overlay": None,
            "gradcam_npy": None,
        }
        if result["gradcam"] is not None:
            record["gradcam_overlay"] = result["gradcam"]["overlay_path"]
            record["gradcam_npy"] = result["gradcam"]["npy_path"]
            n_gradcam += 1

        predictions.append(record)

    predictions_path = out_dir / "predictions.json"
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.info(
        "Testing listo | %d imagenes | %d con Grad-CAM (glaucoma) -> %s",
        len(test_samples),
        n_gradcam,
        predictions_path,
    )
    return {
        "out_dir": str(out_dir),
        "predictions_path": str(predictions_path),
        "n_test": len(test_samples),
        "n_gradcam": n_gradcam,
    }


# =============================================================================
# Orquestacion de las 5 repeticiones
# =============================================================================


def run_all_repetitions(config: dict) -> dict:
    """
    Corre las 5 repeticiones: rep i usa split_files[i-1] con seeds[i-1].

    Para cada una: entrena (train submuestreado), guarda fs_weights_split_<i>.pth,
    testea con su 'test' y guarda predicciones + Grad-CAM.

    Returns:
        {"repetitions": [ {repetition, seed, split_file, checkpoint, best_val_f1,
                           epochs_trained, test:{...}}, ... ]}.
    """
    data_cfg = config["data"]
    splits_dir = _resolve(data_cfg["splits_dir"])
    split_files = data_cfg["split_files"]
    seeds = config["training"]["seeds"]
    results_dir = _resolve(config["paths"]["results_dir"])
    data_root, label_field, label_map = _data_label_cfg(config)

    if len(split_files) != len(seeds):
        logger.warning(
            "Hay %d split_files y %d seeds; se usaran %d repeticiones.",
            len(split_files),
            len(seeds),
            min(len(split_files), len(seeds)),
        )

    summary: list[dict] = []

    for rep_index, (split_file, seed) in enumerate(zip(split_files, seeds), start=1):
        split_path = splits_dir / split_file
        logger.info("=" * 70)
        logger.info("REPETICION %d | split=%s | seed=%d", rep_index, split_file, seed)
        logger.info("=" * 70)

        train_res = train_one_repetition(split_path, seed, config, rep_index)
        classifier = train_res["classifier"]

        test_samples = build_samples(
            train_res["split"].get("test", []), data_root, label_field, label_map
        )
        test_res = test_one_repetition(
            classifier, test_samples, config, results_dir / f"split_{rep_index}"
        )

        summary.append(
            {
                "repetition": rep_index,
                "seed": seed,
                "split_file": split_file,
                "checkpoint": train_res["checkpoint"],
                "best_val_f1": train_res["history"]["best_val_f1"],
                "best_val_acc": train_res["history"]["best_val_acc"],
                "epochs_trained": train_res["history"]["epochs_trained"],
                "test": test_res,
            }
        )

        # Liberar hooks/VRAM antes de la siguiente repeticion.
        classifier.close()
        del classifier

    # Guardar un resumen de rutas/metricas de validacion (no de test).
    summary_path = results_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"repetitions": summary}, f, ensure_ascii=False, indent=2)
    logger.info("Resumen guardado en %s", summary_path)

    return {"repetitions": summary}


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Corre las 5 repeticiones desde la terminal."""
    parser = argparse.ArgumentParser(description="Entrena las 5 repeticiones del clasificador.")
    parser.add_argument(
        "--config",
        default=str(_CLASSIFIER_DIR / "config.yaml"),
        help="Ruta al config.yaml (default: Classifier/config.yaml).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_config(args.config)
    run_all_repetitions(config)


if __name__ == "__main__":
    main()
