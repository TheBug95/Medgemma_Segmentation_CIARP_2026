# =============================================================================
# scripts/few_shot.py
# =============================================================================
# Fine-tune con pocas muestras (N=50, N=100)
#
# RESPONSABILIDADES:
#   1. Fine-tunear modelo pre-entrenado con N muestras de train
#   2. Usar LAS MISMAS imágenes para todos los backbones (sampling fijo)
#   3. Evaluar en val set completo
#
# SAMPLING FIJO:
#   Para que la comparación sea justa, todos los backbones usan las mismas N imágenes.
#   Se usa seed=42 para muestreo aleatorio reproducible.
#
#   few_shot_indices = {
#     "N50": random.sample(train_ids, 50, seed=42),
#     "N100": random.sample(train_ids, 100, seed=42)
#   }
#
# FUNCIONES PRINCIPALES:
#   - get_few_shot_indices(train_ids, n_samples, seed) -> list[str]
#   - create_few_shot_loader(data_module, indices, batch_size) -> DataLoader
#   - train_few_shot(backbone_name, n_samples, data_module, config) -> dict
#
# RETORNO:
#   {
#     "N50": {
#       "accuracy": float,
#       "f1": float,
#       "epochs_trained": int
#     },
#     "N100": {
#       "accuracy": float,
#       "f1": float,
#       "epochs_trained": int
#     }
#   }
#
# CONFIGURACIÓN:
#   - Mismos hiperparámetros que training completo
#   - Mismo early stopping (patience=5)
#   - Máximo 30 epochs (puede converger antes con pocas muestras)
#
# USO:
#   from scripts.few_shot import train_few_shot
#   few_shot_results = train_few_shot("resnet18", data_module, config)
# =============================================================================