# =============================================================================
# scripts/extract_gradcam.py
# =============================================================================
# Extracción de Grad-CAM y evaluación de calidad vs máscara GT
#
# RESPONSABILIDADES:
#   1. Extraer Grad-CAM de un modelo para un imagen
#   2. Calcular IoU entre Grad-CAM binarizado y máscara GT
#   3. Calcular pointing accuracy (¿punto máximo dentro del OD?)
#
# GRAD-CAM PROCEDURE:
#   1. Forward pass con imagen (3, 448, 448)
#   2. Backward pass en la clase predicha
#   3. Pooled gradients × activations de la capa objetivo
#   4. ReLU → heatmap sin normalizar
#   5. Normalizar a [0, 1]
#   6. Binarizar con percentil 95
#
# FUNCIONES PRINCIPALES:
#   - extract_gradcam(model, image, target_layer) -> ndarray (H, W)
#   - compute_iou(gradcam_mask, gt_mask) -> float
#   - compute_pointing_accuracy(gradcam, gt_mask) -> float
#   - evaluate_gradcam_dataset(model, data_loader, device) -> dict
#
# RETORNO:
#   {
#     "mean_iou": float,
#     "mean_pointing_accuracy": float,
#     "iou_per_sample": [float, ...],
#     "pointing_per_sample": [float, ...]
#   }
#
# CAPA OBJETIVO:
#   - Última capa conv del backbone
#   - ResNet18: layer4[-1].conv2
#   - EfficientNet: features[-1]
#   - DenseNet: features.denseblock4.denselayer16
#
# USO:
#   from scripts.extract_gradcam import evaluate_gradcam_dataset
#   gradcam_metrics = evaluate_gradcam_dataset(model, val_loader, device="cuda")
# =============================================================================