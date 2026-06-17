# =============================================================================
# modules/backbones.py
# =============================================================================
# Fabrica compartida de backbones (torchvision) para el experimento.
#
# Provee DOS niveles:
#   - build_raw_backbone(name) -> BackboneParts : el modelo torchvision CRUDO
#       (cabeza original intacta, sin congelar) + sus piezas (capa objetivo de
#       Grad-CAM, bloque a descongelar, dimension de features, atributo de la
#       cabeza). Es la fuente UNICA del conocimiento "que backbone construir".
#       Lo usan tanto el fine-tune (CNNClassifier, que pone su cabeza entrenable
#       + freeze parcial) como el few-shot por prototipos.
#   - build_backbone(name) -> (backbone, target_layer, feat_dim) : extractor de
#       features CONGELADO con cabeza Identity (devuelve (B, D)), para el few-shot
#       por prototipos (PrototypeClassifier).
#
# DIMENSIONES DE EMBEDDING (D): resnet18=512 | efficientnet_b0=1280 | densenet121=1024
#
# CAPA OBJETIVO DE GRAD-CAM (ultima conv) / BLOQUE A DESCONGELAR (fine-tune):
#   resnet18        -> target=model.layer4   | last_block=model.layer4
#   efficientnet_b0 -> target=model.features | last_block=model.features[-1]
#   densenet121     -> target=model.features | last_block=model.features.denseblock4
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch.nn as nn
import torchvision.models as tvm

logger = logging.getLogger(__name__)

# Backbones soportados (todos de torchvision: tiene los tres y las capas objetivo
# de Grad-CAM documentadas; ademas evita una dependencia extra como timm).
SUPPORTED_BACKBONES = ("resnet18", "efficientnet_b0", "densenet121")


@dataclass
class BackboneParts:
    """Piezas de un backbone torchvision crudo (cabeza original intacta, sin congelar)."""

    model: nn.Module  # modelo torchvision completo (con su cabeza original)
    target_layer: nn.Module  # ultima capa conv (para Grad-CAM)
    target_path: str  # ruta string de target_layer (para LayerResolver del modulo compartido)
    last_block: nn.Module  # bloque a descongelar en el fine-tune
    feat_dim: int  # dimension del embedding (= in_features de la cabeza)
    head_attr: str  # nombre del atributo de la cabeza: "fc" | "classifier"


def build_raw_backbone(name: str, pretrained: bool = True) -> BackboneParts:
    """
    Construye el modelo torchvision CRUDO y devuelve sus piezas, SIN congelar ni
    tocar la cabeza.

    Fuente unica del conocimiento por arquitectura; la usan el fine-tune
    (CNNClassifier) y el few-shot por prototipos (build_backbone).

    Args:
        name: "resnet18" | "efficientnet_b0" | "densenet121".
        pretrained: cargar pesos ImageNet.

    Returns:
        BackboneParts(model, target_layer, last_block, feat_dim, head_attr).

    Raises:
        ValueError: si `name` no esta en SUPPORTED_BACKBONES.
    """
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Backbone '{name}' no soportado. Opciones: {SUPPORTED_BACKBONES}"
        )

    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        model = tvm.resnet18(weights=weights)
        return BackboneParts(model, model.layer4, "layer4", model.layer4, model.fc.in_features, "fc")

    if name == "densenet121":
        model = tvm.densenet121(weights=weights)
        return BackboneParts(
            model,
            model.features,
            "features",
            model.features.denseblock4,
            model.classifier.in_features,
            "classifier",
        )

    # efficientnet_b0
    model = tvm.efficientnet_b0(weights=weights)
    return BackboneParts(
        model,
        model.features,
        "features",
        model.features[-1],
        model.classifier[1].in_features,
        "classifier",
    )


def build_backbone(
    name: str, pretrained: bool = True, freeze: bool = True
) -> tuple[nn.Module, nn.Module, str, int]:
    """
    Construye un extractor de features para few-shot por prototipos.

    Toma el backbone crudo (build_raw_backbone) y reemplaza su cabeza por
    nn.Identity() —asi el forward de torchvision entrega el embedding ya agrupado
    (GAP) y aplanado: (B,3,H,W) -> (B, D)—. Con freeze=True congela todos los
    parametros (prototipos sin entrenar); con freeze=False quedan entrenables
    (para meta-training episodico del backbone).

    Args:
        name: "resnet18" | "efficientnet_b0" | "densenet121".
        pretrained: cargar pesos ImageNet.
        freeze: congelar el backbone (True) o dejarlo entrenable (False).

    Returns:
        (backbone, target_layer, target_path, feat_dim):
          - backbone: nn.Module que mapea (B,3,H,W) -> (B, D).
          - target_layer: la ultima capa conv (modulo) para Grad-CAM.
          - target_path: la ruta string de esa capa (para el modulo compartido).
          - feat_dim: D, la dimension del embedding.
    """
    parts = build_raw_backbone(name, pretrained=pretrained)
    # Cabeza -> Identity: el modelo devuelve el embedding (B, D) directamente.
    setattr(parts.model, parts.head_attr, nn.Identity())
    if freeze:
        for param in parts.model.parameters():
            param.requires_grad = False
        parts.model.eval()

    logger.info(
        "build_backbone(%s) | freeze=%s (ImageNet=%s) | feat_dim=%d",
        name,
        freeze,
        pretrained,
        parts.feat_dim,
    )
    return parts.model, parts.target_layer, parts.target_path, parts.feat_dim
