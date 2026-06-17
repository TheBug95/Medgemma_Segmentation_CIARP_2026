#!/usr/bin/env python3
# =============================================================================
# run_experiment.py
# =============================================================================
# Version SCRIPT de notebook/experiment_orchestrator.ipynb, para correr el
# experimento de seleccion de backbone FUERA de Colab (en una maquina propia).
#
# 6 configuraciones = 3 backbones (resnet18 / efficientnet_b0 / densenet121) x
# 2 metodos (euclidean / meta). Barrido N en {1,2,3,4,5}, 5 seeds compartidas.
# Para cada config: F1-macro en val + IoU/pointing Grad-CAM + costo (VRAM) ->
# score SPEC 8 -> winner. Guarda results/selection_summary.json.
#
# RUTAS (esta maquina, NO Colab):
#   - El dataset REFUGE vive en --data-dir (default /data/dvargas), con la
#     estructura <data-dir>/{train,val,test}/{Images,Masks,index.json}.
#   - El repo se importa en /work/dvargas; este script vive dentro del repo y
#     resuelve sus modulos por su propia ubicacion (no depende del CWD).
#
# USO:
#   python run_experiment.py                          # dataset en /data/dvargas
#   python run_experiment.py --data-dir /ruta/REFUGE  # otra ubicacion del dataset
#   python run_experiment.py --skip-convert           # annotations.json ya generado
#
# Grad-CAM: se usa el get_gradcam propio de cada clasificador (patron forward-hook,
# seguro para densenet); NO el modulo compartido (que rompe con densenet).
#
# REQUIERE: torch, torchvision, easyfsl, numpy, pyyaml, scipy (entorno con GPU).
# =============================================================================

from __future__ import annotations

import argparse
import datetime
import gc
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# El script vive en Experiments/Classifier_Selection/ -> esa carpeta debe estar en
# sys.path para los imports `from modules...` / `from scripts...`.
_EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

import torch  # noqa: E402

from modules.cnn_classifier import CNNClassifier  # noqa: E402
from modules.data_module import DataModule  # noqa: E402
from modules.prototype_classifier import PrototypeClassifier  # noqa: E402
from scripts.benchmark_inference import run_benchmark  # noqa: E402
from scripts.convert_refuge_format import convert_refuge, write_json  # noqa: E402
from scripts.few_shot import train_few_shot  # noqa: E402


# =============================================================================
# Helpers de Grad-CAM (opcion Y: el get_gradcam propio de cada clasificador,
# patron forward-hook, seguro para los 3 backbones incluido densenet)
# =============================================================================


def load_classifier_for_gradcam(backbone, method, path, config, seed):
    """Reconstruye y carga el clasificador (segun metodo) para extraer Grad-CAM."""
    nc = config["classifier"]["num_classes"]
    if method == "finetune":
        clf = CNNClassifier(
            {"backbone": backbone, "num_classes": nc, "pretrained": False, "seed": seed}
        )
    else:  # euclidean | meta -> prototipos. pretrained=True: 'euclidean' usa ImageNet;
        # 'meta' lo sobreescribe con los pesos guardados al hacer load().
        clf = PrototypeClassifier(
            {"backbone": backbone, "num_classes": nc, "pretrained": True, "seed": seed}
        )
    clf.load(path)
    return clf


def _iou(pred_binary, gt_binary):
    """IoU entre dos mascaras binarias (H, W)."""
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else 0.0


def _pointing_hit(heatmap, gt_binary):
    """1.0 si el pixel mas caliente del heatmap cae dentro de la GT, 0.0 si no."""
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return 1.0 if gt_binary[y, x] > 0 else 0.0


def evaluate_gradcam_quality(model, data_loader, config, device):
    """IoU + pointing del Grad-CAM (clase glaucoma) vs la mascara GT del disco optico."""
    percentile = config["gradcam"].get("percentile", 95)
    glaucoma_idx = (
        model.class_names.index("glaucoma") if "glaucoma" in model.class_names else 1
    )
    ious, pointings = [], []
    for batch in data_loader:
        images, masks = batch["image"], batch["mask"]
        for i in range(images.size(0)):
            gt = masks[i, 0].numpy()
            heatmap = model.get_gradcam(images[i], target_class=glaucoma_idx)
            threshold = np.percentile(heatmap, percentile)
            binary = (heatmap >= threshold).astype(np.float32)
            ious.append(_iou(binary, gt))
            pointings.append(_pointing_hit(heatmap, gt))
    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_pointing_accuracy": float(np.mean(pointings)) if pointings else 0.0,
        "iou_per_sample": ious,
        "pointing_per_sample": pointings,
    }


# =============================================================================
# Conversion de datos (REFUGE -> annotations.json + splits.json)
# =============================================================================


def run_convert(config, data_dir: Path) -> None:
    """Genera annotations.json + splits.json en output_dir, leyendo REFUGE de data_dir."""
    output_dir = (_EXPERIMENT_DIR / config["data"]["output_dir"]).resolve()
    print(f"Convirtiendo REFUGE | data_dir={data_dir} | output_dir={output_dir}")
    annotations, splits = convert_refuge(data_dir)
    write_json(annotations, output_dir / "annotations.json")
    write_json(splits, output_dir / "splits.json")
    counts = {k: len(v) for k, v in splits.items()}
    print(f"  -> {len(annotations)} imagenes | splits: {counts}")


# =============================================================================
# Bucle del experimento (6 configs x 5 seeds) + benchmark
# =============================================================================


def run_experiment(config, data_module, device):
    """Corre las 6 configs x 5 seeds + benchmark. Devuelve (results, computational)."""
    seeds = config["few_shot"]["seeds"]
    n_samples = config["few_shot"]["n_samples"]
    backbones = config["backbones"]
    methods = config["few_shot"]["methods"]
    n_keys = [f"N{n}" for n in n_samples]
    largest_n_key = n_keys[-1]

    val_loader = data_module.get_val_loader()
    results = {b: {m: {} for m in methods} for b in backbones}
    results_dir = (_EXPERIMENT_DIR / config.get("results_dir", "./results")).resolve()

    for seed in seeds:
        print(f"\n{'=' * 65}\nITERACION seed={seed}  |  Todas las configs usan esta seed\n{'=' * 65}")
        for backbone in backbones:
            for method in methods:
                print(f"\n  -- {backbone} | {method} | seed={seed}")

                # [1/2] Few-shot (todos los N): ajusta/entrena, evalua en val, guarda.
                print(f"    [1/2] {method}: N={n_samples} ...")
                few_shot_result = train_few_shot(
                    backbone, data_module, config, seed=seed, method=method
                )

                # [2/2] Calidad del Grad-CAM con el modelo del N mas grande.
                print(f"    [2/2] Grad-CAM ({largest_n_key}) vs mascara GT ...")
                model_path = (
                    results_dir / backbone / method / f"seed_{seed}" / f"model_{largest_n_key}.pth"
                )
                model = load_classifier_for_gradcam(backbone, method, model_path, config, seed)
                gradcam_metrics = evaluate_gradcam_quality(model, val_loader, config, device)

                results[backbone][method][seed] = {
                    "few_shot": few_shot_result,
                    "gradcam": gradcam_metrics,
                }
                print(
                    f"    OK {backbone}/{method} seed={seed} | "
                    f"IoU={gradcam_metrics['mean_iou']:.3f} "
                    f"pointing={gradcam_metrics['mean_pointing_accuracy']:.3f}"
                )

                # Liberar memoria entre configs (evita acumulacion en GPU -> OOM).
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Benchmark por ARQUITECTURA (mismo costo para frozen y meta).
    print(f"\n{'=' * 65}\nBENCHMARK COMPUTACIONAL (una vez por backbone)\n{'=' * 65}")
    computational = {}
    for backbone in backbones:
        bench_model = CNNClassifier(
            {
                "backbone": backbone,
                "num_classes": config["classifier"]["num_classes"],
                "pretrained": False,
                "seed": 42,
            }
        )
        computational[backbone] = run_benchmark(
            backbone, bench_model, val_loader, device, num_runs=config["benchmark"]["num_runs"]
        )
        print(
            f"  OK {backbone}: {computational[backbone]['total_parameters'] / 1e6:.1f}M params, "
            f"{computational[backbone]['vram_batch_1_mb']:.0f} MB VRAM"
        )
        del bench_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, computational


# =============================================================================
# Agregacion: score de las 6 configs + winner + selection_summary.json
# =============================================================================


def aggregate_and_save(config, results, computational) -> str:
    """Calcula el score de las 6 configs, elige el winner y guarda selection_summary.json."""
    seeds = config["few_shot"]["seeds"]
    n_samples = config["few_shot"]["n_samples"]
    backbones = config["backbones"]
    methods = config["few_shot"]["methods"]
    n_keys = [f"N{n}" for n in n_samples]
    configs = [(b, m) for b in backbones for m in methods]

    summary: dict = {}
    for backbone, method in configs:
        cfg_key = f"{backbone}/{method}"
        seed_results = results[backbone][method]
        f1_by_n = {
            nk: [seed_results[s]["few_shot"][nk]["f1_macro"] for s in seeds] for nk in n_keys
        }
        iou_list = [seed_results[s]["gradcam"]["mean_iou"] for s in seeds]

        per_seed: dict = {}
        for s in seeds:
            per_seed[str(s)] = {"mean_iou_gradcam": seed_results[s]["gradcam"]["mean_iou"]}
            for nk in n_keys:
                per_seed[str(s)][f"f1_{nk}"] = seed_results[s]["few_shot"][nk]["f1_macro"]

        aggregated = {
            "mean_iou_mean": float(np.mean(iou_list)),
            "mean_iou_std": float(np.std(iou_list)),
        }
        for nk in n_keys:
            aggregated[f"f1_{nk}_mean"] = float(np.mean(f1_by_n[nk]))
            aggregated[f"f1_{nk}_std"] = float(np.std(f1_by_n[nk]))

        summary[cfg_key] = {
            "backbone": backbone,
            "method": method,
            "per_seed": per_seed,
            "aggregated": aggregated,
            "computational": computational[backbone],
        }

    # Score por config (VRAM normalizada entre las 6 configs).
    all_vrams = [summary[f"{b}/{m}"]["computational"]["vram_batch_1_mb"] for b, m in configs]
    for backbone, method in configs:
        cfg_key = f"{backbone}/{method}"
        agg = summary[cfg_key]["aggregated"]
        vram = summary[cfg_key]["computational"]["vram_batch_1_mb"]
        f1_mean = float(np.mean([agg[f"f1_{nk}_mean"] for nk in n_keys]))
        iou_mean = agg["mean_iou_mean"]
        vram_norm = (vram - min(all_vrams)) / (max(all_vrams) - min(all_vrams) + 1e-8)
        summary[cfg_key]["score"] = float(0.40 * f1_mean + 0.40 * iou_mean + 0.20 * (1 - vram_norm))

    winner = max(summary, key=lambda k: summary[k]["score"])

    # Tabla (una fila por config).
    print(f"\nRESULTADOS FINALES (mean +/- std sobre {len(seeds)} seeds):")
    header = f"{'Config':<28}" + "".join(f"  {nk:>11}" for nk in n_keys) + f"  {'IoU':>11}  {'Score':>8}"
    print(header)
    print("-" * len(header))
    for cfg_key in summary:
        a = summary[cfg_key]["aggregated"]
        row = f"{cfg_key:<28}"
        for nk in n_keys:
            row += f"  {a[f'f1_{nk}_mean']:.3f}+-{a[f'f1_{nk}_std']:.3f}"
        row += f"  {a['mean_iou_mean']:.3f}+-{a['mean_iou_std']:.3f}"
        row += f"  {summary[cfg_key]['score']:.4f}"
        print(row)
    print(f"\nWinner: {winner} (score={summary[winner]['score']:.4f})")

    results_dir = (_EXPERIMENT_DIR / config.get("results_dir", "./results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    final = {
        "configs": summary,
        "winner": winner,
        "seeds_used": seeds,
        "n_samples": n_samples,
        "methods": methods,
        "config_used": config,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    out_path = results_dir / "selection_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"\nGuardado: {out_path}")
    return winner


# =============================================================================
# main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimento de seleccion de backbone (few-shot por prototipos), version script."
    )
    parser.add_argument(
        "--data-dir",
        default="/data/dvargas/REFUGE",
        help="Raiz de REFUGE (con subcarpetas train/val/test). Default: /data/dvargas",
    )
    parser.add_argument(
        "--config",
        default=str(_EXPERIMENT_DIR / "config.yaml"),
        help="Ruta al config.yaml (default: el del experimento).",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="No regenerar annotations.json/splits.json (usar los existentes).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apuntar el dataset a la ubicacion de esta maquina (ruta absoluta).
    data_dir = Path(args.data_dir).resolve()
    config["data"]["data_dir"] = str(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_configs = len(config["backbones"]) * len(config["few_shot"]["methods"])
    print(f"Device: {device}")
    print(f"Dataset: {data_dir}")
    print(
        f"Backbones: {config['backbones']} | Metodos: {config['few_shot']['methods']} "
        f"-> {n_configs} configs"
    )
    print(
        f"N_samples: {config['few_shot']['n_samples']} | seeds: {config['few_shot']['seeds']}"
    )

    # 1) REFUGE -> annotations.json + splits.json
    if not args.skip_convert:
        run_convert(config, data_dir)

    # 2) DataModule
    data_cfg = {**config["data"], "seed": config["seed"], "augmentations": config["augmentations"]}
    data_module = DataModule(data_cfg)
    train_glaucoma = data_module.get_glaucoma_indices(split="train")
    print(f"Glaucoma en train: {len(train_glaucoma)} (max N factible = {len(train_glaucoma)})")
    assert len(train_glaucoma) >= max(config["few_shot"]["n_samples"]), (
        f"Solo hay {len(train_glaucoma)} glaucoma pero se pide N={max(config['few_shot']['n_samples'])}."
    )

    # 3) Experimento + 4) agregacion
    results, computational = run_experiment(config, data_module, device)
    aggregate_and_save(config, results, computational)


if __name__ == "__main__":
    main()
