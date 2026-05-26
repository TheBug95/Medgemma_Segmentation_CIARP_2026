# =============================================================================
# scripts/__init__.py
# =============================================================================
# Scripts utilitarios para el experimento de selección de clasificador.
#
# PROVEÉ:
#   - convert_refuge_format: Convierte REFUGE al schema del proyecto
#   - evaluate: Evaluación de clasificación (Accuracy, F1, Confusion Matrix)
#   - extract_gradcam: Grad-CAM + IoU vs GT
#   - few_shot: Fine-tune con N=50, N=100 (mismas imágenes para todos)
#   - benchmark_inference: VRAM y tiempo de inferencia
#
# NOTA:
#   - NO incluye training completo (solo few-shot)
#   - Labels son "glaucoma" o "normal" (minúsculas)
#
# USO DESDE NOTEBOOK:
#   import sys
#   sys.path.insert(0, '..')
#   from scripts import convert_refuge_format, evaluate, extract_gradcam, few_shot, benchmark
# =============================================================================