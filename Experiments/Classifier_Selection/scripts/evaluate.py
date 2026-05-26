# =============================================================================
# scripts/evaluate.py
# =============================================================================
# Evaluación de clasificación para el experimento few-shot
#
# RESPONSABILIDADES:
#   1. Calcular métricas de clasificación en un dataset
#   2. Generar matriz de confusión
#   3. Calcular precision, recall, f1 por clase
#
# FUNCIÓN PRINCIPAL:
#   evaluate_classification(model, data_loader, device) -> dict
#
# RETORNO:
#   {
#     "accuracy": float,
#     "f1_macro": float,
#     "f1_per_class": {"glaucoma": float, "normal": float},
#     "precision_per_class": {"glaucoma": float, "normal": float},
#     "recall_per_class": {"glaucoma": float, "normal": float},
#     "confusion_matrix": [[TN, FP], [FN, TP]],
#     "total_samples": int
#   }
#
# NOTAS:
#   - Labels: "glaucoma" o "normal" (minúsculas)
#   - F1-macro es la métrica CRÍTICA por el imbalance 10:1
#   - Accuracy no es buena métrica con imbalance (da ~90% solo prediciendo normal)
#
# USO:
#   from scripts.evaluate import evaluate_classification
#   metrics = evaluate_classification(model, val_loader, device="cuda")
# =============================================================================