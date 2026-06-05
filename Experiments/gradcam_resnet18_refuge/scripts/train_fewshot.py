#!/usr/bin/env python3
# =============================================================================
# scripts/train_fewshot.py
# =============================================================================
# Entrena 3 modelos ResNet18 en configuración few-shot:
#   - fewshot_25: 25 glaucoma + 50 normal
#   - fewshot_30: 30 glaucoma + 60 normal
#   - fewshot_35: 35 glaucoma + 70 normal
#
# USO:
#   python scripts/train_fewshot.py --config config.yaml
#
# OUTPUT:
#   output/fewshot_25/model.pth
#   output/fewshot_30/model.pth
#   output/fewshot_35/model.pth
# =============================================================================

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Añadir modules/ al path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from gradcam_extractor import GradCAMExtractor, RefugeDataModule, set_global_seed


def setup_logging() -> None:
    """Configura logging básico."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def train_fewshot_model(
    config: dict,
    n_glaucoma: int,
    n_normal: int,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    """
    Entrena un modelo con configuración few-shot específica.
    
    Args:
        config: configuración completa del experimento
        n_glaucoma: número de ejemplos de glaucoma
        n_normal: número de ejemplos de normal
        output_dir: directorio donde guardar resultados
        device: dispositivo para entrenamiento
        
    Returns:
        dict con métricas de entrenamiento
    """
    set_global_seed(config.get("seed", 42))
    
    # Preparar data module con transforms sin augmentación
    data_module = RefugeDataModule(config)
    
    # Crear loaders few-shot
    train_loader = data_module.get_fewshot_train_loader(
        n_glaucoma=n_glaucoma,
        n_normal=n_normal,
        seed=config.get("seed", 42),
    )
    val_loader = data_module.get_val_loader()
    
    logging.info(
        f"Entrenando few-shot: {n_glaucoma} glaucoma + {n_normal} normal "
        f"({n_glaucoma + n_normal} total)"
    )
    
    # Crear y entrenar modelo
    extractor = GradCAMExtractor(config)
    metrics = extractor.train(train_loader, val_loader)
    
    # Guardar modelo
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"model_fewshot_{n_glaucoma}.pth"
    extractor.save(str(model_path))
    
    # Guardar métricas
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "n_glaucoma": n_glaucoma,
                "n_normal": n_normal,
                "total_train": n_glaucoma + n_normal,
                **metrics,
            },
            f,
            indent=2,
        )
    
    logging.info(
        f"Modelo fewshot_{n_glaucoma} guardado en {model_path}. "
        f"Best val acc: {metrics['best_val_acc']:.4f}"
    )
    
    return metrics


def main() -> None:
    """Entrena los 3 modelos few-shot secuencialmente."""
    parser = argparse.ArgumentParser(
        description="Entrena modelos ResNet18 en configuración few-shot"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Ruta al archivo config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directorio base para guardar modelos",
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
    
    # Configuración few-shot: pocos ejemplos = overfitting rápido
    if "few_shot" in config:
        config["classifier"]["epochs"] = config["few_shot"].get("epochs", 15)
        config["classifier"]["patience"] = config["few_shot"].get("patience", 3)
    else:
        config["classifier"]["epochs"] = 15
        config["classifier"]["patience"] = 3
    
    logging.info(f"Configuración few-shot: {config['classifier']['epochs']} épocas")
    
    output_base = Path(args.output_dir)
    
    # Configuraciones few-shot
    fewshot_configs = [
        {"n_glaucoma": 25, "n_normal": 50},
        {"n_glaucoma": 30, "n_normal": 60},
        {"n_glaucoma": 35, "n_normal": 70},
    ]
    
    all_metrics = {}
    
    for fs_cfg in fewshot_configs:
        n_g = fs_cfg["n_glaucoma"]
        n_n = fs_cfg["n_normal"]
        
        logging.info("=" * 60)
        logging.info(f"Entrenando modelo fewshot_{n_g} ({n_g}G + {n_n}N)")
        logging.info("=" * 60)
        
        out_dir = output_base / f"fewshot_{n_g}"
        metrics = train_fewshot_model(
            config=config,
            n_glaucoma=n_g,
            n_normal=n_n,
            output_dir=out_dir,
        )
        
        all_metrics[f"fewshot_{n_g}"] = {
            "n_glaucoma": n_g,
            "n_normal": n_n,
            **metrics,
        }
    
    # Guardar resumen comparativo
    summary_path = output_base / "fewshot_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    logging.info("=" * 60)
    logging.info("Entrenamiento few-shot completado!")
    logging.info(f"Resumen guardado en: {summary_path}")
    for name, m in all_metrics.items():
        logging.info(
            f"  {name}: {m['n_glaucoma']}G + {m['n_normal']}N | "
            f"Best val acc: {m['best_val_acc']:.4f} | "
            f"Épocas: {m['epochs_trained']}"
        )


if __name__ == "__main__":
    main()
