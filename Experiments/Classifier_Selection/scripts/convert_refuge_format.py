# =============================================================================
# scripts/convert_refuge_format.py
# =============================================================================
# Convierte el dataset REFUGE al formato del proyecto (annotations.json + splits.json)
#
# INPUT:
#   - Datasets/REFUGE/train/index.json
#   - Datasets/REFUGE/val/index.json
#   - Datasets/REFUGE/test/index.json
#
# OUTPUT:
#   - data/annotations.json
#   - data/splits.json
#
# ANNOTATIONS.JSON SCHEMA:
#   {
#     "g0001": {
#       "image_filename": "g0001.jpg",
#       "label": "glaucoma",
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
# USO:
#   python scripts/convert_refuge_format.py
# =============================================================================