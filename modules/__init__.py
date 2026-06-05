# =============================================================================
# modules/__init__.py
# =============================================================================
# Módulos compartidos del proyecto MedGemma Segmentation.
# Cada módulo es independiente y con interfaz definida para conectarse
# como pieza de lego al pipeline.
# =============================================================================

from modules.gradcam_module import (
    GradCAMConfig,
    GradCAMExtractorModule,
    LayerResolver,
)

__all__ = [
    "GradCAMConfig",
    "GradCAMExtractorModule",
    "LayerResolver",
]
