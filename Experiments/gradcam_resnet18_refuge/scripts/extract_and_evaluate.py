# =============================================================================
# scripts/extract_and_evaluate.py
# =============================================================================
# Extracción de Grad-CAM, cálculo de métricas y visualización
#
# FLUJO:
#   1. Carga modelo entrenado (o usa config para reentrenar si no existe)
#   2. Itera sobre dataset de test
#   3. Para cada imagen:
#      - Extrae Grad-CAM usando layer4[-1].conv2
#      - Binariza con percentil 95
#      - Carga máscara GT
#      - Calcula IoU, SSIM, pointing accuracy
#   4. Genera visualizaciones 2x2
#   5. Guarda métricas en JSON
#
# MÉTRICAS:
#   - IoU: TP / (TP + FP + FN) entre Grad-CAM binario y GT binario
#   - SSIM: similitud estructural (manual, ventana 11x11)
#   - Pointing Accuracy: máximo de Grad-CAM cae dentro del OD
#
# OUTPUT:
#   output/metrics.json: métricas agregadas + timestamp + config
#   visualizations/: PNGs 2x2 por imagen
#
# USO:
#   python scripts/extract_and_evaluate.py
#   python scripts/extract_and_evaluate.py --no-train --model-path ./output/model.pth
# =============================================================================

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

# Instalar pytorch-grad-cam automáticamente si no está disponible
try:
    import pytorch_grad_cam
except ImportError:
    logging.warning("pytorch-grad-cam no instalado. Instalando ahora...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "grad-cam", "-q"])
    import pytorch_grad_cam
    logging.warning("pytorch-grad-cam instalado correctamente.")

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.gradcam_extractor import GradCAMExtractor, RefugeDataModule, set_global_seed
from scripts.visualize import plot_gradcam_vs_gt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
_logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", action="store_true", help="No entrenar, usar modelo existente")
    parser.add_argument("--model-path", type=str, default=None, help="Ruta al modelo pre-entrenado")
    parser.add_argument("--data-dir", type=str, default=None, help="Sobrescribir ruta al dataset REFUGE")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolver data_dir relativo a config.yaml si es una ruta relativa
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
        _logger.info(f"data_dir sobrescrito por argumento: {args.data_dir}")
    else:
        data_dir = Path(config["data"]["data_dir"])
        if not data_dir.is_absolute():
            data_dir = (config_path.parent / data_dir).resolve()
            config["data"]["data_dir"] = str(data_dir)
            _logger.info(f"data_dir resuelto relativo a config.yaml: {data_dir}")

    set_global_seed(config["seed"])
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(config["output"]["visualizations_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or str(output_dir / "model.pth")

    extractor = GradCAMExtractor(config)

    if args.no_train and Path(model_path).exists():
        _logger.info(f"Cargando modelo desde {model_path}")
        extractor.load(model_path)
    else:
        _logger.info("Entrenando modelo...")
        datamodule = RefugeDataModule(config)
        train_loader = datamodule.get_train_loader()
        val_loader = datamodule.get_val_loader()
        train_result = extractor.train(train_loader, val_loader)
        _logger.info(f"Best val acc: {train_result['best_val_acc']:.4f}")
        extractor.save(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.model = extractor.model.to(device)
    extractor.model.eval()

    _logger.info("Cargando dataset de test...")
    datamodule = RefugeDataModule(config)
    test_loader = datamodule.get_test_loader()
    _logger.info(f"Test samples: {len(test_loader.dataset)}")

    num_samples = config["visualization"].get("num_samples", 20)
    save_images = config["visualization"].get("save_images", True)
    show_images = config["visualization"].get("show_images", False)
    figure_size = tuple(config["visualization"].get("figure_size", [12, 10]))

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in config.items() if k != "output"},
        "seed": config["seed"],
        "test_size": len(test_loader.dataset),
        "metrics_per_sample": [],
        "iou_vs_percentile": {},
        "aggregation": {},
    }

    ious, ssims, pointings = [], [], []

    # Percentiles para evaluar sensibilidad del threshold (50-100 de 10 en 10)
    percentiles = [50, 60, 70, 80, 90, 100]
    iou_by_percentile = {p: [] for p in percentiles}

    _logger.info("Extrayendo Grad-CAM y calculando métricas...")
    for i, (images, gt_masks, labels, image_ids) in enumerate(test_loader):
        for j in range(images.size(0)):
            img_idx = i * test_loader.batch_size + j
            if img_idx >= len(test_loader.dataset):
                break

            image_id = image_ids[j]
            image = images[j].to(device)
            gt_mask = gt_masks[j].numpy()

            image_np = extractor.denormalize(image)

            gradcam = extractor.get_gradcam(image)
            gradcam_binary = (gradcam >= np.percentile(gradcam, config["gradcam"]["percentile"])).astype(np.float32)

            metrics = extractor.compute_metrics(gradcam_binary, gt_mask)
            ious.append(metrics["iou"])
            ssims.append(metrics["ssim"])
            pointings.append(metrics["pointing_accuracy"])

            # Calcular IoU para múltiples percentiles (curva de sensibilidad)
            for p in percentiles:
                gb = (gradcam >= np.percentile(gradcam, p)).astype(np.float32)
                m = extractor.compute_metrics(gb, gt_mask)
                iou_by_percentile[p].append(m["iou"])

            results["metrics_per_sample"].append(
                {
                    "image_id": image_id,
                    "iou": metrics["iou"],
                    "ssim": metrics["ssim"],
                    "pointing_accuracy": metrics["pointing_accuracy"],
                }
            )

            if save_images or show_images:
                plot_gradcam_vs_gt(
                    image_np.transpose(1, 2, 0),
                    gradcam,
                    gt_mask,
                    image_id,
                    save_path=str(vis_dir / f"{image_id}_gradcam_vs_gt.png") if save_images else None,
                    figure_size=figure_size,
                    show=show_images,
                )

            if (i * test_loader.batch_size + j + 1) % 50 == 0:
                _logger.info(f"Procesadas {(i * test_loader.batch_size + j + 1)}/{len(test_loader.dataset)} imágenes")

            if num_samples > 0 and len(ious) >= num_samples:
                break

        if num_samples > 0 and len(ious) >= num_samples:
            break

    results["aggregation"] = {
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "mean_ssim": float(np.mean(ssims)),
        "std_ssim": float(np.std(ssims)),
        "mean_pointing_accuracy": float(np.mean(pointings)),
        "std_pointing_accuracy": float(np.std(pointings)),
    }

    # Curva IoU vs percentil (para elegir threshold óptimo en Pipeline B)
    results["iou_vs_percentile"] = {
        str(p): {
            "mean": float(np.mean(iou_by_percentile[p])),
            "std": float(np.std(iou_by_percentile[p])),
        }
        for p in percentiles
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    _logger.info("=" * 50)
    _logger.info("RESULTADOS:")
    _logger.info(f"  IoU mean: {results['aggregation']['mean_iou']:.4f} ± {results['aggregation']['std_iou']:.4f}")
    _logger.info(f"  SSIM mean: {results['aggregation']['mean_ssim']:.4f} ± {results['aggregation']['std_ssim']:.4f}")
    _logger.info(f"  Pointing Acc mean: {results['aggregation']['mean_pointing_accuracy']:.4f} ± {results['aggregation']['std_pointing_accuracy']:.4f}")
    _logger.info("  IoU vs Percentil:")
    for p in percentiles:
        mean_p = results["iou_vs_percentile"][str(p)]["mean"]
        _logger.info(f"    p{p:02d}: {mean_p:.4f}")
    _logger.info(f"Métricas guardadas en {metrics_path}")
    _logger.info(f"Visualizaciones guardadas en {vis_dir}")

    return results


if __name__ == "__main__":
    main()
