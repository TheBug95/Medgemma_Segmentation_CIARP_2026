# =============================================================================
# modules/data_module.py
# =============================================================================
# M1: DataModule - Carga y preprocesa datos REFUGE para el clasificador CNN
#
# RESPONSABILIDADES:
#   1. Cargar annotations.json y splits.json (NO regenerar splits)
#   2. Devolver 3 versiones de cada imagen según solicitud
#   3. Aplicar augmentations SOLO en train
#   4. Aplanar máscaras 3-clase (0=bg, 1=rim, 2=cup) a binaria (0=bg, 1=OD)
#
# CONFIG ESPERADA:
#   {
#       "data_dir": "Datasets/REFUGE",
#       "output_dir": "./data",
#       "image_size": [448, 448],
#       "batch_size": 16,
#       "num_workers": 2,
#       "seed": 42
#   }
#
# MÉTODOS PÚBLICOS:
#   - __init__(config: dict)
#   - get_train_loader() -> DataLoader  # con augmentations
#   - get_val_loader() -> DataLoader    # sin augmentations
#   - get_test_loader() -> DataLoader   # sin augmentations
#   - get_sample(idx: int) -> dict      # debug/inspección
#
# NOTAS:
#   - Normalización ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#   - Augmentations: flip, rotation ±15°, color jitter, random erasing, mixup
#   - Las máscaras GT se convierten de 3-clase a binarias (disc óptico vs fondo)
# =============================================================================