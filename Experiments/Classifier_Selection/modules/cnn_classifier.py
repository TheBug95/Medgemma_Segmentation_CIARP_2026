# =============================================================================
# modules/cnn_classifier.py
# =============================================================================
# M2: CNNClassifier - Implementación de 3 backbones + Grad-CAM + Trainer
#
# RESPONSABILIDADES:
#   1. Implementar 3 arquitecturas: ResNet-18, EfficientNet-B0, DenseNet-121
#   2. Generar Grad-CAM sobre la última capa conv
#   3. Proporcionar training loop con early stopping
#
# CLASES PROVEÍDAS:
#   - ResNet18Classifier(BackboneMixin)
#   - EfficientNetClassifier(BackboneMixin)
#   - DenseNetClassifier(BackboneMixin)
#   - ClassifierTrainer: Entrena cualquier backbone
#
# MÉTODOS PÚBLICOS:
#   - __init__(config: dict)
#   - train(train_loader, val_loader, epochs, lr, patience) -> dict
#   - predict(image: Tensor) -> {"prediction", "distribution"}
#   - get_gradcam(image: Tensor) -> ndarray (H, W) en [0, 1]
#   - save(path: str)
#   - load(path: str)
#
# GRAD-CAM:
#   - Capa objetivo: última conv del backbone
#   - Binarización: percentil 95
#   - Retorna heatmap normalizado en [0, 1]
#
# FREEZE STRATEGY:
#   - Freeze todo excepto último bloque conv + FC
#   - Solo descongelar parámetros a fine-tune
#
# CONFIG ESPERADA:
#   {
#       "backbone": "resnet18" | "efficientnet_b0" | "densenet121",
#       "num_classes": 2,
#       "pretrained": true,
#       "seed": 42
#   }
# =============================================================================