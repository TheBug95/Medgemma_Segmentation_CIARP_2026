# =============================================================================
# scripts/train.py
# =============================================================================
# Entrenamiento del clasificador ResNet18 con dataset REFUGE
#
# USO:
#   python scripts/train.py
# =============================================================================

import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.gradcam_extractor import GradCAMExtractor, RefugeDataModule, set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
_logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
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
    _logger.info(f"Semilla: {config['seed']}")

    _logger.info("Inicializando DataModule...")
    datamodule = RefugeDataModule(config)

    _logger.info("Cargando datos de entrenamiento...")
    train_loader = datamodule.get_train_loader()
    val_loader = datamodule.get_val_loader()
    _logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    _logger.info("Inicializando GradCAMExtractor...")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    _logger.info(f"Usando dispositivo: {device}")

    extractor = GradCAMExtractor(config)
    extractor.model = extractor.model.to(device)

    _logger.info("Entrenando modelo...")
    train_result = extractor.train(train_loader, val_loader)

    _logger.info(f"Mejor val accuracy: {train_result['best_val_acc']:.4f}")
    _logger.info(f"Epochs entrenados: {train_result['epochs_trained']}")

    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pth"
    extractor.save(str(model_path))

    _logger.info("Entrenamiento completado.")


if __name__ == "__main__":
    main()
