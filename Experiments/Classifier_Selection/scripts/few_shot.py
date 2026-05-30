# =============================================================================
# scripts/few_shot.py
# =============================================================================
# Fine-tune few-shot del clasificador — Estrategia Multi-Seed
#
# RESPONSABILIDADES:
#   1. Componer el support set y fine-tunear un CNNClassifier fresco con N muestras
#   2. Usar LAS MISMAS imágenes para todos los backbones dentro de cada iteración
#   3. Evaluar en el val set COMPLETO (40 glaucoma + 360 normal) con evaluate.py
#   4. Guardar cada modelo entrenado (lo carga después extract_gradcam.py)
#
# COMPOSICIÓN DEL SUPPORT SET (config.few_shot.support_mode):
#   El SPEC original pedía support "solo glaucoma", pero entrenar un clasificador
#   CrossEntropy con una sola clase es DEGENERADO (colapsa a predecir glaucoma).
#   Por eso el modo es configurable (PENDIENTE de confirmar por el equipo):
#     - "balanced"                 -> N glaucoma + N normal  (few-shot estándar)
#     - "glaucoma_plus_all_normal" -> N glaucoma + los 360 normales
#     - "glaucoma_only"            -> solo N glaucoma (literal al SPEC; degenerado)
#   Default temporal: "balanced".
#
#   En REFUGE/train hay 40 glaucoma -> N=25, 30, 35 (todos < 40).
#
# ESTRATEGIA DE SEEDS:
#   El experimento corre 5 iteraciones. En la iteración i, TODOS los backbones
#   reciben la misma seeds[i] -> mismo support set y mismas condiciones iniciales,
#   garantizando comparabilidad. Resultados finales = mean ± std sobre las 5 seeds.
#
#   Llamada desde el orquestador:
#     for seed in config['few_shot']['seeds']:      # [42, 123, 456, 789, 1024]
#         for backbone in config['backbones']:      # mismo seed para TODOS
#             result = train_few_shot(backbone, data_module, config, seed=seed)
#
# FIRMA PRINCIPAL:
#   train_few_shot(backbone_name: str, data_module, config: dict, seed: int) -> dict
#
# RETORNO:
#   {
#     "seed": int, "backbone": str,
#     "N25": {"f1_macro": float, "accuracy": float, "epochs_trained": int},
#     "N30": {...}, "N35": {...}
#   }
#
# USO:
#   from scripts.few_shot import train_few_shot
#   result = train_few_shot("resnet18", data_module, config, seed=42)
# =============================================================================

from __future__ import annotations

import logging
import random
from pathlib import Path

from modules.cnn_classifier import CNNClassifier, set_global_seed
from scripts.evaluate import evaluate_classification

logger = logging.getLogger(__name__)

# Directorio del experimento (para resolver results_dir sin depender del CWD).
_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

# Modos válidos de composición del support set.
SUPPORT_MODES = ("balanced", "glaucoma_plus_all_normal", "glaucoma_only")
DEFAULT_SUPPORT_MODE = "balanced"


def get_glaucoma_indices(data_module, split: str = "train") -> list[str]:
    """Delega en DataModule.get_glaucoma_indices (única fuente de verdad)."""
    return data_module.get_glaucoma_indices(split)


def _get_indices_by_label(data_module, label: str, split: str = "train") -> list[str]:
    """IDs de un split cuyo annotation tiene el label dado (en el orden del split)."""
    return [
        image_id
        for image_id in data_module.splits[split]
        if data_module.annotations[image_id].get("label") == label
    ]


def get_few_shot_indices(glaucoma_ids: list[str], n_samples: int, seed: int) -> list[str]:
    """
    Muestrea N IDs de glaucoma de forma determinista.

    Misma seed -> mismo subconjunto (usa un RNG propio, independiente del global).
    """
    if n_samples > len(glaucoma_ids):
        raise ValueError(
            f"n_samples={n_samples} > glaucoma disponibles ({len(glaucoma_ids)})."
        )
    return random.Random(seed).sample(list(glaucoma_ids), n_samples)


def build_support_indices(
    data_module, n_samples: int, seed: int, support_mode: str = DEFAULT_SUPPORT_MODE
) -> list[str]:
    """
    Compone el support set de entrenamiento según support_mode.

    Args:
        data_module: M1 DataModule (expone get_glaucoma_indices, splits, annotations).
        n_samples: N de glaucoma a usar (25, 30 o 35).
        seed: Semilla del muestreo (reproducible).
        support_mode: "balanced" | "glaucoma_plus_all_normal" | "glaucoma_only".

    Returns:
        Lista de image_id que componen el support set.
    """
    if support_mode not in SUPPORT_MODES:
        raise ValueError(f"support_mode '{support_mode}' inválido. Opciones: {SUPPORT_MODES}")

    glaucoma_ids = data_module.get_glaucoma_indices("train")
    support = get_few_shot_indices(glaucoma_ids, n_samples, seed)

    if support_mode == "glaucoma_only":
        return support

    normal_ids = _get_indices_by_label(data_module, "normal", "train")
    if support_mode == "balanced":
        if n_samples > len(normal_ids):
            raise ValueError(
                f"n_samples={n_samples} > normales disponibles ({len(normal_ids)})."
            )
        # Stream derivado (seed+1) para el subset normal, reproducible e independiente.
        normals = random.Random(seed + 1).sample(list(normal_ids), n_samples)
    else:  # glaucoma_plus_all_normal
        normals = list(normal_ids)

    return support + normals


def create_few_shot_loader(data_module, indices: list[str], batch_size: int, seed: int):
    """DataLoader del support set (con augmentation y shuffle); seed controla el shuffle."""
    return data_module.build_loader(
        indices, shuffle=True, augment=True, batch_size=batch_size, seed=seed
    )


def _resolve_results_dir(config: dict) -> Path:
    """results_dir del config, resuelto contra el dir del experimento si es relativo."""
    results_dir = Path(config.get("results_dir", "./results"))
    if results_dir.is_absolute():
        return results_dir
    return (_EXPERIMENT_DIR / results_dir).resolve()


def train_few_shot(backbone_name: str, data_module, config: dict, seed: int) -> dict:
    """
    Fine-tune few-shot del backbone para cada tamaño N, con una seed dada.

    Para cada N: compone el support set (según support_mode), entrena un
    CNNClassifier fresco, evalúa en el val set COMPLETO y guarda el modelo en
    results_dir/<backbone>/seed_<seed>/model_N<n>.pth.

    Returns:
        {"seed", "backbone", "N25": {f1_macro, accuracy, epochs_trained}, "N30": {...}, ...}
    """
    set_global_seed(seed)

    fs_cfg = config["few_shot"]
    clf_cfg = config["classifier"]
    n_samples_list = fs_cfg["n_samples"]
    batch_size = config["data"]["batch_size"]

    support_mode = fs_cfg.get("support_mode", DEFAULT_SUPPORT_MODE)
    if "support_mode" not in fs_cfg:
        logger.warning(
            "config.few_shot.support_mode no definido; usando '%s' por defecto "
            "(PENDIENTE de confirmar la composición del support set).",
            DEFAULT_SUPPORT_MODE,
        )

    val_loader = data_module.get_val_loader()
    results_dir = _resolve_results_dir(config)

    results: dict = {"seed": seed, "backbone": backbone_name}

    for n_samples in n_samples_list:
        key = f"N{n_samples}"
        support_ids = build_support_indices(data_module, n_samples, seed, support_mode)
        loader = create_few_shot_loader(data_module, support_ids, batch_size, seed)

        logger.info(
            "[%s | seed=%d | %s] support=%d imágenes (mode=%s)",
            backbone_name,
            seed,
            key,
            len(support_ids),
            support_mode,
        )

        # Modelo fresco por N (la seed reinicializa cabeza FC y estado global).
        classifier = CNNClassifier(
            {
                "backbone": backbone_name,
                "num_classes": clf_cfg.get("num_classes", 2),
                "pretrained": clf_cfg.get("pretrained", True),
                "seed": seed,
            }
        )
        history = classifier.train(
            loader,
            val_loader,
            epochs=clf_cfg.get("epochs", 30),
            lr=clf_cfg.get("lr", 1e-4),
            patience=clf_cfg.get("patience", 5),
            weight_decay=clf_cfg.get("weight_decay", 1e-5),
            scheduler_factor=clf_cfg.get("scheduler_factor", 0.5),
            scheduler_patience=clf_cfg.get("scheduler_patience", 3),
        )

        # Métricas oficiales en el val completo (vía evaluate.py).
        metrics = evaluate_classification(classifier, val_loader)

        checkpoint = results_dir / backbone_name / f"seed_{seed}" / f"model_{key}.pth"
        classifier.save(checkpoint)

        results[key] = {
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
            "epochs_trained": history["epochs_trained"],
        }
        logger.info(
            "[%s | seed=%d | %s] f1_macro=%.4f accuracy=%.4f epochs=%d",
            backbone_name,
            seed,
            key,
            metrics["f1_macro"],
            metrics["accuracy"],
            history["epochs_trained"],
        )

    return results
