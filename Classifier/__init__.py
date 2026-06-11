# =============================================================================
# Classifier/__init__.py
# =============================================================================
# Clasificador CNN de produccion del pipeline (modulo M2): EfficientNet-B0 para
# glaucoma vs normal, con Grad-CAM. Ganador del experimento Classifier_Selection.
#
# Submodulos:
#   - efficientNet_init           -> EfficientNetClassifier (setup + train + predict + gradcam)
#   - efficientNet_classification -> inferencia de alto nivel (veredicto + gradcam visual)
#   - training.data_interface     -> lectura de splits/labels + loaders
#   - training.few_shot           -> entrenamiento+testing de las 5 repeticiones
#
# NOTA: el import de torch/torchvision se hace dentro de cada submodulo (no aqui),
# para poder importar utilidades que no dependen de torch sin tener torch instalado.
# =============================================================================

__all__: list[str] = []
