#!/usr/bin/env python3
# =============================================================================
# scripts/evaluate_fewshot.py
# =============================================================================
# Evalúa Grad-CAM para los 3 modelos few-shot en el val split.
# Genera métricas (IoU, SSIM, pointing) y visualizaciones para subset.
#
# USO:
#   python scripts/evaluate_fewshot.py --config config.yaml
#
# OUTPUT:
#   output/fewshot_25/metrics.json
#   output/fewshot_25/visualizations/...
#   output/fewshot_summary_eval.json  # comparativa
# =============================================================================

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Añadir modules/ al path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from gradcam_extractor import GradCAMExtractor, RefugeDataModule, set_global_seed
from visualize import plot_gradcam_vs_gt


def setup_logging() -> None:
    """Configura logging básico."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def select_representative_subset(
    samples: list, n_per_class: int = 10, seed: int = 42
) -> list:
    """
    Selecciona subset representativo estratificado por clase.
    
    Args:
        samples: lista de dicts con campo 'label'
        n_per_class: número de muestras por clase
        seed: semilla para reproducibilidad
        
    Returns:
        lista de muestras seleccionadas
    """
    import random
    random.seed(seed)
    
    glaucoma = [s for s in samples if s.get("label") == "glaucoma"]
    normal = [s for s in samples if s.get("label") == "normal"]
    
    selected_g = random.sample(glaucoma, min(n_per_class, len(glaucoma)))
    selected_n = random.sample(normal, min(n_per_class, len(normal)))
    
    return selected_g + selected_n


def evaluate_model(
    config: dict,
    model_path: Path,
    output_dir: Path,
    n_viz_per_class: int = 10,
    percentiles: list[int] = None,
) -> dict:
    """
    Evalúa un modelo few-shot en val split.
    
    Args:
        config: configuración del experimento
        model_path: ruta al modelo .pth
        output_dir: directorio para guardar resultados
        n_viz_per_class: número de visualizaciones por clase
        percentiles: lista de percentiles a evaluar
        
    Returns:
        dict con métricas agregadas
    """
    if percentiles is None:
        percentiles = [50, 60, 70, 80, 90, 100]
    
    set_global_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar data y modelo
    data_module = RefugeDataModule(config)
    val_loader = data_module.get_val_loader()
    
    extractor = GradCAMExtractor(config)
    extractor.load(str(model_path))
    extractor.model = extractor.model.to(device)
    extractor.model.eval()
    
    logging.info(f"Evaluando modelo: {model_path.name}")
    logging.info(f"Val set: {len(val_loader.dataset)} imágenes")
    
    # Métricas acumuladoras
    results = {
        "model_path": str(model_path),
        "percentiles": percentiles,
        "metrics_per_sample": [],
        "iou_by_percentile": {p: [] for p in percentiles},
        "dice_by_percentile": {p: [] for p in percentiles},
        "ssim_by_percentile": {p: [] for p in percentiles},
        "pointing_by_percentile": {p: [] for p in percentiles},
    }
    img_count = 0
    total_images = len(val_loader.dataset)
    
    # Para subset de visualización
    all_samples = []
    sample_metrics = []  # (image_id, label, iou_p95)
    
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks, labels, image_ids) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            for j in range(images.size(0)):
                image = images[j]
                mask = masks[j]
                label = labels[j].item()
                image_id = image_ids[j]
                
                # Extraer Grad-CAM (clase predicha)
                gradcam = extractor.get_gradcam(image)
                
                # Ground truth mask
                gt_mask = mask.cpu().numpy()
                
                # Para cada percentil, calcular métricas
                sample_result = {
                    "image_id": image_id,
                    "true_label": "glaucoma" if label == 1 else "normal",
                }
                
                best_iou = 0
                best_p = 95
                
                for p in percentiles:
                    gradcam_binary = (gradcam >= np.percentile(gradcam, p)).astype(np.float32)
                    # Pasar heatmap continuo para SSIM y pointing accuracy
                    metrics = extractor.compute_metrics(
                        gradcam_binary, gt_mask, gradcam_heatmap=gradcam
                    )
                    
                    sample_result[f"iou_p{p}"] = metrics["iou"]
                    sample_result[f"dice_p{p}"] = metrics["dice"]
                    sample_result[f"ssim_p{p}"] = metrics["ssim"]
                    sample_result[f"pointing_p{p}"] = metrics["pointing_accuracy"]
                    
                    results["iou_by_percentile"][p].append(metrics["iou"])
                    results["dice_by_percentile"][p].append(metrics["dice"])
                    results["ssim_by_percentile"][p].append(metrics["ssim"])
                    results["pointing_by_percentile"][p].append(metrics["pointing_accuracy"])
                    
                    if metrics["iou"] > best_iou:
                        best_iou = metrics["iou"]
                        best_p = p
                
                sample_result["best_iou"] = best_iou
                sample_result["best_percentile"] = best_p
                
                results["metrics_per_sample"].append(sample_result)
                sample_metrics.append((image_id, label, best_iou, gradcam, gt_mask))
                
                img_count += 1
                if img_count % 50 == 0:
                    logging.info(f"  Procesadas {img_count}/{total_images} imágenes...")
    
    # Calcular estadísticas agregadas
    summary = {}
    for p in percentiles:
        ious = results["iou_by_percentile"][p]
        dices = results["dice_by_percentile"][p]
        ssims = results["ssim_by_percentile"][p]
        pointings = results["pointing_by_percentile"][p]
        
        summary[f"p{p}"] = {
            "iou_mean": float(np.mean(ious)),
            "iou_std": float(np.std(ious)),
            "dice_mean": float(np.mean(dices)),
            "dice_std": float(np.std(dices)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std": float(np.std(ssims)),
            "pointing_accuracy": float(np.mean(pointings)),
        }
    
    # Mejor percentil global
    best_p_global = max(percentiles, key=lambda p: summary[f"p{p}"]["iou_mean"])
    summary["best_global_percentile"] = best_p_global
    summary["best_global_iou"] = summary[f"p{best_p_global}"]["iou_mean"]
    
    results["summary"] = summary
    
    # Guardar métricas
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Métricas guardadas en: {metrics_path}")
    logging.info(f"Mejor percentil global: p{best_p_global} (IoU={summary[f'p{best_p_global}']['iou_mean']:.4f})")
    
    # Generar visualizaciones para subset representativo
    # Seleccionar estratificado por clase y por calidad (buenas y malas IoU)
    glaucoma_samples = [(i, m) for i, m in enumerate(sample_metrics) if m[1] == 1]
    normal_samples = [(i, m) for i, m in enumerate(sample_metrics) if m[1] == 0]
    
    def select_diverse(samples, n):
        """Selecciona muestras diversas (alta y baja IoU)."""
        if len(samples) <= n:
            return samples
        # Ordenar por IoU
        sorted_samples = sorted(samples, key=lambda x: x[1][2])
        # Seleccionar uniformemente
        indices = np.linspace(0, len(sorted_samples) - 1, n, dtype=int)
        return [sorted_samples[i] for i in indices]
    
    selected_indices = []
    selected_indices.extend([idx for idx, _ in select_diverse(glaucoma_samples, n_viz_per_class)])
    selected_indices.extend([idx for idx, _ in select_diverse(normal_samples, n_viz_per_class)])
    
    logging.info(f"Generando {len(selected_indices)} visualizaciones...")
    
    for idx in selected_indices:
        image_id, label, best_iou, gradcam, gt_mask = sample_metrics[idx]
        
        # Reconstruir imagen original para visualización
        # Nota: necesitamos cargar la imagen cruda del dataset
        sample_info = data_module.annotations.get(image_id, {})
        if not sample_info:
            continue
            
        image_path = data_module.data_dir / sample_info.get("split", "val") / "Images" / sample_info["image_filename"]
        
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img = img.resize(data_module.image_size, Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        
        for p in percentiles:
            save_path = vis_dir / f"{image_id}_p{p}_gradcam_vs_gt.png"
            plot_gradcam_vs_gt(
                image=img_np,
                gradcam=gradcam,
                gt_mask=gt_mask,
                image_id=image_id,
                save_path=str(save_path),
                figure_size=config.get("visualization", {}).get("figure_size", [12, 10]),
                show=False,
                percentile=p,
            )
    
    logging.info(f"Visualizaciones guardadas en: {vis_dir}")
    
    return summary


def main() -> None:
    """Evalúa los 3 modelos few-shot secuencialmente."""
    parser = argparse.ArgumentParser(
        description="Evalúa Grad-CAM para modelos few-shot"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Ruta al archivo config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directorio base con modelos entrenados",
    )
    parser.add_argument(
        "--n-viz",
        type=int,
        default=10,
        help="Número de visualizaciones por clase",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Ruta al dataset REFUGE (sobreescribe config.yaml). Ej: /content/drive/MyDrive/.../REFUGE",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    # Cargar configuración
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Sobreescribir data_dir si se pasa por CLI
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
        logging.info(f"Usando data_dir desde CLI: {args.data_dir}")
    
    output_base = Path(args.output_dir)
    
    # Evaluar cada modelo few-shot
    fewshot_configs = [25, 30, 35]
    all_summaries = {}
    
    for n_g in fewshot_configs:
        model_dir = output_base / f"fewshot_{n_g}"
        model_path = model_dir / f"model_fewshot_{n_g}.pth"
        
        if not model_path.exists():
            logging.warning(f"Modelo no encontrado: {model_path}. Saltando...")
            continue
        
        logging.info("=" * 60)
        logging.info(f"Evaluando fewshot_{n_g}")
        logging.info("=" * 60)
        
        summary = evaluate_model(
            config=config,
            model_path=model_path,
            output_dir=model_dir,
            n_viz_per_class=args.n_viz,
        )
        
        all_summaries[f"fewshot_{n_g}"] = summary
    
    # Guardar resumen comparativo
    summary_path = output_base / "fewshot_summary_eval.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    logging.info("=" * 60)
    logging.info("Evaluación completada!")
    logging.info(f"Resumen guardado en: {summary_path}")
    
    # Imprimir tabla comparativa
    logging.info("\nTabla comparativa (IoU medio por percentil):")
    percentiles = [50, 60, 70, 80, 90, 100]
    header = "Modelo    | " + " | ".join([f"p{p:3d}" for p in percentiles])
    logging.info(header)
    logging.info("-" * len(header))
    
    for name, summary in all_summaries.items():
        row = f"{name:10s}|"
        for p in percentiles:
            iou = summary.get(f"p{p}", {}).get("iou_mean", 0.0)
            row += f" {iou:.3f} |"
        logging.info(row)
    
    logging.info("\nMejor percentil global por modelo:")
    for name, summary in all_summaries.items():
        best_p = summary.get("best_global_percentile", 0)
        best_iou = summary.get("best_global_iou", 0.0)
        logging.info(f"  {name}: p{best_p} (IoU={best_iou:.4f})")


if __name__ == "__main__":
    main()
