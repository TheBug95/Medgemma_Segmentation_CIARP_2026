# =============================================================================
# scripts/few_shot.py
# =============================================================================
# Few-shot del clasificador — Estrategia Multi-Seed, con DISPATCH de metodo.
#
# METODO (config.few_shot.method):
#   - "euclidean" (default): few-shot por PROTOTIPOS Euclidianos (EasyFSL
#     PrototypicalNetworks). Backbone congelado; 'fit' calcula los 2 prototipos
#     (media por clase) del support set. NO hay entrenamiento (sin backprop).
#   - "finetune": fine-tune del backbone (CNNClassifier) con CrossEntropy y early
#     stopping. Se conserva como modulo reutilizable / baseline "metodo viejo".
#
# RESPONSABILIDADES:
#   1. Componer el support set BALANCED (N glaucoma + N normal) por cada N.
#   2. Usar LAS MISMAS imagenes para todos los backbones dentro de cada iteracion
#      (misma seed -> mismo support) para que la comparacion sea justa.
#   3. Ajustar/entrenar un clasificador fresco y evaluarlo en el val COMPLETO
#      (40 glaucoma + 360 normal) con evaluate.py.
#   4. Guardar cada modelo (lo carga despues el evaluador de Grad-CAM).
#
# SUPPORT SET (config.few_shot.support_mode):
#   "balanced" (default y recomendado) -> N glaucoma + N normal. Para prototipos
#   es lo correcto (se necesitan ambas clases para los 2 prototipos). En REFUGE
#   /train hay 40 glaucoma y 360 normal -> N en {3, 6, 9, 12}.
#   (Los modos "glaucoma_plus_all_normal" / "glaucoma_only" siguen disponibles
#   pero no se usan en el experimento de prototipos.)
#
# ESTRATEGIA DE SEEDS:
#   5 iteraciones; en la iteracion i, TODOS los backbones usan seeds[i] -> mismo
#   support y mismas condiciones. Resultados finales = mean ± std sobre las 5 seeds.
#
#   Llamada desde el orquestador (sin cambios de firma):
#     for seed in config['few_shot']['seeds']:
#         for backbone in config['backbones']:
#             result = train_few_shot(backbone, data_module, config, seed=seed)
#
# RETORNO:
#   {
#     "seed": int, "backbone": str, "method": str,
#     "N3":  {"f1_macro": float, "accuracy": float, <extra>},
#     "N6":  {...}, "N9": {...}, "N12": {...}
#   }
#   <extra> = {"epochs_trained": int} en finetune | {"n_support": int} en euclidean.
#
# USO:
#   from scripts.few_shot import train_few_shot
#   result = train_few_shot("resnet18", data_module, config, seed=42)
# =============================================================================

from __future__ import annotations

import logging
import random
from pathlib import Path

from modules.cnn_classifier import set_global_seed
from scripts.evaluate import evaluate_classification

logger = logging.getLogger(__name__)

# Directorio del experimento (para resolver results_dir sin depender del CWD).
_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

# Modos validos de composicion del support set.
SUPPORT_MODES = ("balanced", "glaucoma_plus_all_normal", "glaucoma_only")
DEFAULT_SUPPORT_MODE = "balanced"

# Metodos few-shot validos.
FEW_SHOT_METHODS = ("euclidean", "finetune", "meta")
DEFAULT_METHOD = "euclidean"


def get_glaucoma_indices(data_module, split: str = "train") -> list[str]:
    """Delega en DataModule.get_glaucoma_indices (unica fuente de verdad)."""
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
    Compone el support set segun support_mode.

    Args:
        data_module: M1 DataModule (expone get_glaucoma_indices, splits, annotations).
        n_samples: N de glaucoma a usar (3, 6, 9 o 12).
        seed: Semilla del muestreo (reproducible).
        support_mode: "balanced" | "glaucoma_plus_all_normal" | "glaucoma_only".

    Returns:
        Lista de image_id que componen el support set.
    """
    if support_mode not in SUPPORT_MODES:
        raise ValueError(f"support_mode '{support_mode}' invalido. Opciones: {SUPPORT_MODES}")

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


def create_few_shot_loader(
    data_module, indices: list[str], batch_size: int, seed: int, augment: bool = True
):
    """
    DataLoader del support set.

    `augment` y `shuffle` se activan juntos: True para el fine-tune (variedad +
    barajado) y False para prototipos (embeddings limpios; el orden no importa
    porque se promedian). seed controla el shuffle.
    """
    return data_module.build_loader(
        indices, shuffle=augment, augment=augment, batch_size=batch_size, seed=seed
    )


def _resolve_results_dir(config: dict) -> Path:
    """results_dir del config, resuelto contra el dir del experimento si es relativo."""
    results_dir = Path(config.get("results_dir", "./results"))
    if results_dir.is_absolute():
        return results_dir
    return (_EXPERIMENT_DIR / results_dir).resolve()


def _build_classifier(method: str, backbone_name: str, clf_cfg: dict, seed: int):
    """Crea un clasificador fresco segun el metodo few-shot (import perezoso)."""
    cfg = {
        "backbone": backbone_name,
        "num_classes": clf_cfg.get("num_classes", 2),
        "pretrained": clf_cfg.get("pretrained", True),
        "seed": seed,
    }
    if method == "finetune":
        from modules.cnn_classifier import CNNClassifier

        return CNNClassifier(cfg)
    if method in ("euclidean", "meta"):
        from modules.prototype_classifier import PrototypeClassifier

        # meta -> backbone entrenable (se meta-entrena); euclidean -> congelado.
        cfg["freeze_backbone"] = method == "euclidean"
        return PrototypeClassifier(cfg)
    raise ValueError(f"few_shot.method '{method}' invalido. Opciones: {FEW_SHOT_METHODS}.")


def train_few_shot(
    backbone_name: str, data_module, config: dict, seed: int, method: str | None = None
) -> dict:
    """
    Ajusta el clasificador few-shot del backbone para cada tamano N, con una seed.

    Despacha segun `method` (o config.few_shot.method si method=None):
      - "euclidean": prototipos Euclidianos, backbone congelado (sin entrenar).
      - "meta": prototipos con meta-training episodico del backbone (n_shot = el N
        del barrido; resto de params en config.meta_training).
      - "finetune": fine-tune del backbone (baseline).
    Para cada N: compone el support balanced, ajusta un clasificador fresco, evalua
    en el val COMPLETO y guarda el modelo en
    results_dir/<backbone>/<method>/seed_<seed>/model_N<n>.pth.

    Returns:
        {"seed", "backbone", "method", "N1": {...}, "N2": {...}, ...}.
    """
    set_global_seed(seed)

    fs_cfg = config["few_shot"]
    clf_cfg = config["classifier"]
    meta_cfg = config.get("meta_training", {})
    n_samples_list = fs_cfg["n_samples"]
    batch_size = config["data"]["batch_size"]
    support_mode = fs_cfg.get("support_mode", DEFAULT_SUPPORT_MODE)
    if method is None:
        method = fs_cfg.get("method", DEFAULT_METHOD)
    if method not in FEW_SHOT_METHODS:
        raise ValueError(f"few_shot.method '{method}' invalido. Opciones: {FEW_SHOT_METHODS}.")

    # Prototipos -> sin augmentation en el support de despliegue (embeddings
    # limpios); fine-tune -> con augmentation. (El meta-training tiene su propia
    # augmentation en los episodios, ver meta_training.augment.)
    augment = method == "finetune"

    val_loader = data_module.get_val_loader()
    results_dir = _resolve_results_dir(config)

    results: dict = {"seed": seed, "backbone": backbone_name, "method": method}

    for n_samples in n_samples_list:
        key = f"N{n_samples}"
        support_ids = build_support_indices(data_module, n_samples, seed, support_mode)
        loader = create_few_shot_loader(
            data_module, support_ids, batch_size, seed, augment=augment
        )

        logger.info(
            "[%s | seed=%d | %s | %s] support=%d imagenes (mode=%s)",
            backbone_name,
            seed,
            key,
            method,
            len(support_ids),
            support_mode,
        )

        # Clasificador fresco por N (la seed reinicializa el estado).
        classifier = _build_classifier(method, backbone_name, clf_cfg, seed)

        if method == "finetune":
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
            extra = {"epochs_trained": history["epochs_trained"]}
        else:  # euclidean | meta: prototipos
            if method == "meta":
                # Meta-training episodico del backbone con n_shot = el N del barrido.
                classifier.meta_train(
                    data_module,
                    n_way=meta_cfg.get("n_way", 2),
                    n_shot=n_samples,
                    n_query=meta_cfg.get("n_query", 10),
                    n_episodes=meta_cfg.get("n_episodes", 100),
                    lr=meta_cfg.get("lr", 1e-3),
                    augment=meta_cfg.get("augment", True),
                )
            # Prototipos de despliegue con el mismo support k-shot (frozen y meta
            # difieren solo en el backbone).
            fit_info = classifier.fit(loader)
            extra = {"n_support": fit_info["n_support"]}

        # Metricas oficiales en el val completo (via evaluate.py).
        metrics = evaluate_classification(classifier, val_loader)

        checkpoint = results_dir / backbone_name / method / f"seed_{seed}" / f"model_{key}.pth"
        classifier.save(checkpoint)

        results[key] = {
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
            **extra,
        }
        logger.info(
            "[%s | seed=%d | %s | %s] f1_macro=%.4f accuracy=%.4f",
            backbone_name,
            seed,
            key,
            method,
            metrics["f1_macro"],
            metrics["accuracy"],
        )

    return results
