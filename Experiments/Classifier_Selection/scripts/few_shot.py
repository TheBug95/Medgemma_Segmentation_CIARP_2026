# =============================================================================
# scripts/few_shot.py
# =============================================================================
# Fine-tune con pocas muestras (support set de glaucoma) — Estrategia Multi-Seed
#
# RESPONSABILIDADES:
#   1. Fine-tunear modelo pre-entrenado con N muestras del support set
#   2. Usar LAS MISMAS imágenes para todos los backbones dentro de cada iteración
#   3. Evaluar en val set completo
#
# RESTRICCIÓN CRÍTICA — SUPPORT SET:
#   El support set está compuesto EXCLUSIVAMENTE por imágenes de glaucoma,
#   ya que es la clase clínica de interés y la clase minoritaria del dataset.
#   En REFUGE/train solo existen 40 imágenes con label="glaucoma".
#   Por eso N=50 y N=100 son IMPOSIBLES. Los tamaños válidos son:
#
#     N=25 → 25/40 muestras de glaucoma  (62.5% del disponible)
#     N=30 → 30/40 muestras de glaucoma  (75.0% del disponible)
#     N=35 → 35/40 muestras de glaucoma  (87.5% del disponible)
#
#   El val set se usa completo (40 glaucoma + 360 normal) para evaluar F1-macro.
#
# ESTRATEGIA DE SEEDS:
#   El experimento corre 5 iteraciones. En la iteración i, TODOS los backbones
#   reciben la misma seeds[i]. Esto garantiza:
#   - Mismo subconjunto de glaucoma para todos los backbones en esa iteración
#   - Mismas condiciones de inicialización (FC layer, dataloader shuffle)
#   - Comparabilidad directa entre backbones dentro de cada iteración
#
#   Llamada desde el orquestador:
#     for seed in config['few_shot']['seeds']:      # [42, 123, 456, 789, 1024]
#         for backbone in config['backbones']:      # mismo seed para TODOS
#             result = train_few_shot(backbone, data_module, config, seed=seed)
#
# FUNCIONES PRINCIPALES:
#   - get_glaucoma_indices(data_module, split='train') -> list[str]
#       Devuelve los IDs de las 40 imágenes de glaucoma en train.
#   - get_few_shot_indices(glaucoma_ids, n_samples, seed) -> list[str]
#       Muestrea N IDs de glaucoma. Determinista: misma seed → mismo subconjunto.
#   - create_few_shot_loader(data_module, indices, batch_size, seed) -> DataLoader
#       DataLoader con los N glaucoma seleccionados. seed controla el shuffle.
#   - train_few_shot(backbone_name, data_module, config, seed) -> dict
#       Fine-tune completo para N=25, N=30, N=35 con la seed dada.
#
# FIRMA PRINCIPAL:
#   train_few_shot(backbone_name: str, data_module: DataModule,
#                  config: dict, seed: int) -> dict
#
# RETORNO:
#   {
#     "seed": int,           # seed usada en esta iteración
#     "backbone": str,       # nombre del backbone
#     "N25": {
#       "f1_macro": float,   # F1-macro en val set completo (glaucoma + normal)
#       "accuracy": float,
#       "epochs_trained": int
#     },
#     "N30": {
#       "f1_macro": float,
#       "accuracy": float,
#       "epochs_trained": int
#     },
#     "N35": {
#       "f1_macro": float,
#       "accuracy": float,
#       "epochs_trained": int
#     }
#   }
#
# NOTAS:
#   - La seed inicializa: selección del subconjunto de glaucoma, pesos FC, shuffle
#   - Llamar set_global_seed(seed) al inicio de cada train_few_shot()
#   - El backbone (ImageNet) se mantiene fijo; solo varía la capa FC y el subset
#   - Early stopping patience=5, max_epochs=30 (desde config.yaml)
#   - Métricas evaluadas en el VAL set COMPLETO (no solo glaucoma)
#
# USO:
#   from scripts.few_shot import train_few_shot
#   result = train_few_shot("resnet18", data_module, config, seed=42)
# =============================================================================