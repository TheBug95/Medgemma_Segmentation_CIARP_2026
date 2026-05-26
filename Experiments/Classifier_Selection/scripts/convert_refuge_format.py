# =============================================================================
# scripts/convert_refuge_format.py
# =============================================================================
# Convierte el dataset REFUGE al formato del proyecto (annotations.json + splits.json)
#
# INPUT:
#   - Datasets/REFUGE/train/index.json
#   - Datasets/REFUGE/val/index.json
#   - Datasets/REFUGE/test/index.json
#   - Datasets/REFUGE/*/Masks/*.png (para calcular vCDR)
#
# OUTPUT:
#   - data/annotations.json
#   - data/splits.json
#
# ANNOTATIONS.JSON SCHEMA:
#   {
#     "g0001": {
#       "image_filename": "g0001.jpg",
#       "label": "Pathological",
#       "disease_category": "glaucoma",
#       "disease_grading": {"cup_to_disc_ratio": 3, "vcdr": 0.75},
#       "mask_path": "Datasets/REFUGE/train/Masks/g0001.png",
#       "split": "train"
#     },
#     ...
#   }
#
# SPLITS.JSON SCHEMA:
#   {
#     "train": ["g0001", "g0002", ..., "n0040"],
#     "val": [...],
#     "test": [...]
#   }
#
# VCDR CALCULATION:
#   vCDR = cup_height / disc_height (desde máscaras 3-clase)
#   Escala 0-4: <0.3→0, 0.3-0.5→1, 0.5-0.65→2, 0.65-0.8→3, >0.8→4
#
# USO:
#   python scripts/convert_refuge_format.py
# =============================================================================