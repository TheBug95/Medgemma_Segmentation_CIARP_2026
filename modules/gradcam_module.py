# =============================================================================
# modules/gradcam_module.py
# =============================================================================
# Módulo independiente de extracción de Grad-CAM para cualquier CNN de PyTorch.
#
# RESPONSABILIDAD ÚNICA:
#   Recibir un modelo CNN ya entrenado + una imagen → devolver Grad-CAM.
#   NO entrena modelos, NO carga datasets, NO calcula métricas de segmentación.
#
# CLASES PROVEÍDAS:
#   - GradCAMConfig:           Configuración inmutable del extractor
#   - LayerResolver:           Resuelve nombres de capas a nn.Module
#   - BaseCAMExtractor:        Clase base abstracta (Strategy pattern)
#   - GradCAMExtractorModule:  Implementación de Grad-CAM estándar
#
# INTERFAZ PÚBLICA:
#   extractor = GradCAMExtractorModule(model, config)
#   heatmap   = extractor.extract(image_tensor)        → ndarray (H, W) [0, 1]
#   binary    = extractor.extract_binary(image_tensor)  → ndarray (H, W) {0, 1}
#   batch     = extractor.extract_batch(batch_tensor)   → list[ndarray]
#
# DISEÑO (Strategy Pattern):
#   BaseCAMExtractor define la interfaz. GradCAMExtractorModule es la
#   implementación de Grad-CAM canónico. Variantes futuras (Grad-CAM++,
#   Score-CAM, Eigen-CAM) heredan de BaseCAMExtractor sin modificar
#   el código existente.
#
# GRAD-CAM PROCEDIMIENTO (Selvaraju et al., 2017):
#   1. Forward pass: imagen → activaciones en capa objetivo
#   2. Backward pass: gradientes respecto a la clase objetivo
#   3. Global Average Pooling de gradientes (pesos por canal)
#   4. Combinación lineal ponderada de activaciones
#   5. ReLU → heatmap sin normalizar
#   6. Interpolación bilineal al tamaño original
#   7. Normalización min-max a [0, 1]
#   8. (Opcional) Suavizado gaussiano post-proceso
#   9. (Opcional) Binarización por percentil
#
# BACKBONES SOPORTADOS Y CAPAS OBJETIVO SUGERIDAS:
#   | Backbone       | target_layer              |
#   |----------------|---------------------------|
#   | ResNet18/34    | "layer4[-1]"              |
#   | ResNet50/101   | "layer4[-1]"              |
#   | EfficientNet   | "features[-1]"            |
#   | DenseNet121    | "features.denseblock4"    |
#   | VGG16          | "features[-1]"            |
#   | MobileNetV2    | "features[-1]"            |
#
# CONFIG ESPERADA:
#   GradCAMConfig(
#       target_layer="layer4[-1]",
#       percentile=95,
#       use_external=True,
#       sigma_smooth=0.0,
#       image_size=None,
#   )
# =============================================================================

from __future__ import annotations

import abc
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuración
# =============================================================================


@dataclass(frozen=True)
class GradCAMConfig:
    """
    Configuración inmutable para extracción de Grad-CAM.

    Attributes:
        target_layer: Nombre de la capa objetivo del modelo para capturar
            activaciones y gradientes. Soporta notación con punto e índices.
            Ejemplos: "layer4[-1]", "features.denseblock4", "features[-1]".
        percentile: Percentil para binarizar el heatmap en extract_binary().
            Valores típicos: 90-99. Default: 95.
        use_external: Si True, intenta usar pytorch-grad-cam como backend
            principal. Si falla o no está instalado, cae al backend manual
            con hooks. Default: True.
        sigma_smooth: Desviación estándar del suavizado gaussiano post-proceso.
            0.0 = desactivado (Grad-CAM canónico sin post-blur). Default: 0.0.
        image_size: Tamaño (H, W) al que interpolar el heatmap de salida.
            Si es None, se infiere del tensor de entrada. Default: None.
    """

    target_layer: str
    percentile: int = 95
    use_external: bool = True
    sigma_smooth: float = 0.0
    image_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Valida los parámetros de configuración."""
        if not self.target_layer:
            raise ValueError("target_layer no puede estar vacío")
        if not 0 <= self.percentile <= 100:
            raise ValueError(
                f"percentile debe estar en [0, 100], recibido: {self.percentile}"
            )
        if self.sigma_smooth < 0:
            raise ValueError(
                f"sigma_smooth debe ser >= 0, recibido: {self.sigma_smooth}"
            )
        if self.image_size is not None:
            if len(self.image_size) != 2 or any(s <= 0 for s in self.image_size):
                raise ValueError(
                    f"image_size debe ser (H, W) con valores positivos, "
                    f"recibido: {self.image_size}"
                )

    @classmethod
    def from_dict(cls, config: dict) -> GradCAMConfig:
        """
        Crea GradCAMConfig desde un diccionario (compatible con config.yaml).

        Acepta tanto la estructura plana como la anidada bajo clave 'gradcam'.
        Ejemplo:
            config = {"target_layer": "layer4[-1]", "percentile": 95}
            # o
            config = {"gradcam": {"target_layer": "layer4[-1]", "percentile": 95}}

        Args:
            config: Diccionario con los parámetros.

        Returns:
            Instancia de GradCAMConfig.
        """
        # Si viene anidado bajo clave 'gradcam', extraer
        if "gradcam" in config and isinstance(config["gradcam"], dict):
            gc_dict = config["gradcam"].copy()
        else:
            gc_dict = config.copy()

        # Mapeo de image_size desde lista a tupla (YAML deserializa como lista)
        if "image_size" in gc_dict and isinstance(gc_dict["image_size"], list):
            gc_dict["image_size"] = tuple(gc_dict["image_size"])
        # También buscar en la sección data del config original si no está en gradcam
        elif "image_size" not in gc_dict and "data" in config:
            data_size = config["data"].get("image_size")
            if data_size is not None:
                gc_dict["image_size"] = tuple(data_size)

        # Filtrar solo campos válidos del dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in gc_dict.items() if k in valid_fields}

        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GradCAMConfig:
        """
        Crea GradCAMConfig desde un archivo YAML.

        Args:
            path: Ruta al archivo YAML.

        Returns:
            Instancia de GradCAMConfig.
        """
        import yaml

        with open(path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)


# =============================================================================
# Layer Resolver
# =============================================================================


class LayerResolver:
    """
    Resuelve nombres de capas (strings) a módulos nn.Module de PyTorch.

    Soporta notación compleja incluyendo:
        - Acceso por atributo: "features.denseblock4"
        - Índices positivos y negativos: "layer4[-1]", "features[0]"
        - Combinaciones: "features.layer4[-1].conv2"

    Ejemplo:
        layer = LayerResolver.resolve(resnet18, "layer4[-1]")
        # Equivale a: resnet18.layer4[-1]
    """

    # Patrón regex para parsear tokens con índices opcionales.
    # Captura: nombre_atributo[índice] o solo nombre_atributo
    _TOKEN_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)(?:\[(-?\d+)\])?")

    @staticmethod
    def resolve(model: nn.Module, layer_name: str) -> nn.Module:
        """
        Resuelve un nombre de capa a su nn.Module correspondiente.

        Args:
            model: Modelo PyTorch.
            layer_name: String con la ruta a la capa, ej: "layer4[-1]".

        Returns:
            El nn.Module correspondiente a la capa.

        Raises:
            ValueError: Si la capa no se puede resolver.
        """
        current = model
        tokens = layer_name.split(".")

        for token in tokens:
            match = LayerResolver._TOKEN_PATTERN.fullmatch(token)
            if not match:
                raise ValueError(
                    f"Token inválido '{token}' en layer_name='{layer_name}'. "
                    f"Formato esperado: nombre o nombre[índice]"
                )

            attr_name = match.group(1)
            index_str = match.group(2)

            # Acceder al atributo
            if not hasattr(current, attr_name):
                available = [
                    name
                    for name, _ in current.named_children()
                ]
                raise ValueError(
                    f"El módulo {type(current).__name__} no tiene atributo "
                    f"'{attr_name}'. Atributos disponibles: {available}"
                )
            current = getattr(current, attr_name)

            # Aplicar índice si existe
            if index_str is not None:
                idx = int(index_str)
                try:
                    current = current[idx]
                except (IndexError, TypeError) as e:
                    raise ValueError(
                        f"No se pudo indexar {type(current).__name__} "
                        f"con índice [{idx}] en '{layer_name}': {e}"
                    ) from e

        return current

    @staticmethod
    def list_layers(
        model: nn.Module,
        filter_type: type | None = None,
        max_depth: int = 5,
    ) -> list[str]:
        """
        Lista todas las capas del modelo con sus nombres jerárquicos.

        Útil para debug: descubrir qué capas tiene un backbone desconocido.

        Args:
            model: Modelo PyTorch.
            filter_type: Si se especifica, solo lista capas de ese tipo.
                Ejemplo: nn.Conv2d para listar solo convolucionales.
            max_depth: Profundidad máxima de búsqueda recursiva.

        Returns:
            Lista de strings con nombres de capas (notación punto).
        """
        layers = []

        def _recurse(module: nn.Module, prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if filter_type is None or isinstance(child, filter_type):
                    layers.append(full_name)
                _recurse(child, full_name, depth + 1)

        _recurse(model, "", 0)
        return layers

    @staticmethod
    def suggest_target_layer(model: nn.Module) -> str:
        """
        Heurística para sugerir la capa objetivo automáticamente.

        Estrategia: buscar la última capa convolucional (o el bloque que la
        contiene) en el modelo. Para la mayoría de CNNs, la última capa
        convolucional produce las activaciones más semánticamente ricas.

        Heurísticas específicas por arquitectura:
            - ResNet: "layer4[-1]" (último BasicBlock/Bottleneck)
            - DenseNet: "features.denseblock4" (último DenseBlock)
            - EfficientNet: "features[-1]" (último bloque)
            - VGG: "features[-1]" o última Conv2d en features
            - Fallback: última Conv2d encontrada en el modelo

        Args:
            model: Modelo PyTorch.

        Returns:
            String con el nombre sugerido de la capa objetivo.

        Raises:
            ValueError: Si no se puede determinar una capa objetivo.
        """
        model_name = type(model).__name__.lower()

        # Heurística por nombre de arquitectura
        if "resnet" in model_name:
            # ResNet: el último bloque residual completo
            if hasattr(model, "layer4"):
                return "layer4[-1]"

        if "densenet" in model_name:
            if hasattr(model, "features"):
                # Buscar el último denseblock
                for name, _ in reversed(list(model.features.named_children())):
                    if "denseblock" in name:
                        return f"features.{name}"

        if "efficientnet" in model_name or "mobilenet" in model_name:
            if hasattr(model, "features"):
                return "features[-1]"

        if "vgg" in model_name:
            if hasattr(model, "features"):
                return "features[-1]"

        # Fallback genérico: buscar la última Conv2d en el modelo
        last_conv_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name = name

        if last_conv_name:
            logger.info(
                f"suggest_target_layer: usando fallback genérico, "
                f"última Conv2d encontrada: '{last_conv_name}'"
            )
            return last_conv_name

        raise ValueError(
            f"No se pudo determinar automáticamente la capa objetivo para "
            f"{type(model).__name__}. Especifique target_layer manualmente."
        )


# =============================================================================
# Base abstracta (Strategy Pattern para variantes futuras de CAM)
# =============================================================================


class BaseCAMExtractor(abc.ABC):
    """
    Clase base abstracta para extractores de Class Activation Maps.

    Define la interfaz que deben implementar todas las variantes de CAM
    (Grad-CAM, Grad-CAM++, Score-CAM, Eigen-CAM, etc.).

    Patrón Strategy: cada variante hereda de esta clase e implementa
    _compute_cam_weights(), que define cómo se calculan los pesos por
    canal de activación. El resto del pipeline (forward, backward,
    combinación, normalización) es compartido.

    Para agregar una nueva variante:
        1. Crear clase que herede de BaseCAMExtractor
        2. Implementar _compute_cam_weights()
        3. (Opcional) Override de _compute_heatmap() si la variante
           tiene un pipeline completamente diferente (ej. Score-CAM)
    """

    def __init__(
        self,
        model: nn.Module,
        config: GradCAMConfig,
        device: str | torch.device | None = None,
    ) -> None:
        """
        Inicializa el extractor registrando hooks en la capa objetivo.

        Args:
            model: CNN ya entrenada. Se pone en eval mode automáticamente.
                El modelo NO se modifica internamente (no se cambian pesos).
            config: Configuración del extractor.
            device: Dispositivo para cómputo. Si None, se auto-detecta
                del modelo.

        Raises:
            ValueError: Si la capa objetivo no existe en el modelo.
        """
        self._model = model
        self._config = config

        # Auto-detectar device del modelo
        if device is None:
            try:
                self._device = next(model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        # Resolver la capa objetivo usando LayerResolver
        self._target_layer = LayerResolver.resolve(model, config.target_layer)

        # Estado capturado por los hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Registrar hooks y guardar handles para cleanup
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

        # Cache del estado de disponibilidad de la librería externa
        self._external_available: Optional[bool] = None

        logger.info(
            f"Extractor {type(self).__name__} inicializado: "
            f"target_layer='{config.target_layer}' -> "
            f"{type(self._target_layer).__name__}, "
            f"device={self._device}"
        )

    @property
    def config(self) -> GradCAMConfig:
        """Retorna la configuración del extractor (solo lectura)."""
        return self._config

    @property
    def model(self) -> nn.Module:
        """Retorna el modelo asociado (solo lectura)."""
        return self._model

    def _register_hooks(self) -> None:
        """Registra forward y backward hooks en la capa objetivo."""

        def _forward_hook(
            module: nn.Module, input: tuple, output: torch.Tensor
        ) -> None:
            # Detach para no interferir con el grafo computacional principal
            self._activations = output.detach()

        def _backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            self._gradients = grad_output[0].detach()

        fwd_handle = self._target_layer.register_forward_hook(_forward_hook)
        bwd_handle = self._target_layer.register_full_backward_hook(
            _backward_hook
        )
        self._hook_handles.extend([fwd_handle, bwd_handle])

    @abc.abstractmethod
    def _compute_cam_weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula los pesos por canal de activación.

        Este es el punto de extensión del Strategy Pattern.
        Cada variante de CAM implementa su propia estrategia de pesos:
            - Grad-CAM: Global Average Pooling de gradientes
            - Grad-CAM++: Pesos ponderados por gradientes positivos
            - Score-CAM: No usa gradientes (override _compute_heatmap)

        Args:
            activations: Tensor (B, C, h, w) activaciones de la capa objetivo.
            gradients: Tensor (B, C, h, w) gradientes respecto a la clase.

        Returns:
            weights: Tensor (B, C, 1, 1) pesos por canal.
        """

    def _compute_heatmap(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Pipeline completo: imagen → heatmap normalizado [0, 1].

        Este método implementa los pasos 1-8 del procedimiento Grad-CAM.
        Las subclases normalmente solo necesitan override de
        _compute_cam_weights(). Override de este método solo si la
        variante tiene un pipeline fundamentalmente diferente.

        Args:
            image: Tensor (3, H, W) normalizado.
            target_class: Clase contra la cual computar gradientes.
                None = clase predicha por el modelo.

        Returns:
            heatmap: ndarray (H, W) en [0, 1].
            pred_info: dict con "predicted_class" y "probabilities".
        """
        self._model.eval()

        # Asegurar dimensión batch
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self._device)
        image.requires_grad_(True)

        # 1. Forward pass
        output = self._model(image)
        probabilities = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        pred_class = int(output.argmax(dim=1).item())

        # Determinar clase objetivo
        cls_target = target_class if target_class is not None else pred_class

        # 2. Backward pass respecto a la clase objetivo
        self._model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, cls_target] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Verificar que los hooks capturaron datos
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                f"Los hooks no capturaron activaciones/gradientes. "
                f"Verifique que target_layer='{self._config.target_layer}' "
                f"es una capa válida del modelo que participa en el forward pass."
            )

        # 3-4. Calcular pesos y combinación ponderada (delegado a subclase)
        weights = self._compute_cam_weights(self._activations, self._gradients)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)

        # 5. ReLU: solo activaciones positivas contribuyen
        cam = F.relu(cam)

        # 6. Interpolación al tamaño original de la imagen
        h, w = self._get_output_size(image)
        cam = F.interpolate(
            cam, size=(h, w), mode="bilinear", align_corners=False
        )

        # Convertir a numpy
        heatmap = cam.squeeze().cpu().detach().numpy()

        # 7. Normalización min-max a [0, 1]
        heatmap = self._normalize(heatmap)

        # 8. Suavizado gaussiano opcional
        if self._config.sigma_smooth > 0:
            heatmap = self._apply_smoothing(heatmap)

        pred_info = {
            "predicted_class": pred_class,
            "target_class": cls_target,
            "probabilities": probabilities,
        }

        return heatmap, pred_info

    def _get_output_size(self, image: torch.Tensor) -> tuple[int, int]:
        """
        Determina el tamaño de salida del heatmap.

        Prioridad:
            1. config.image_size si está definido
            2. Dimensiones espaciales del tensor de entrada

        Args:
            image: Tensor de entrada (B, 3, H, W).

        Returns:
            (H, W) tamaño de salida.
        """
        if self._config.image_size is not None:
            return self._config.image_size
        return image.shape[2], image.shape[3]

    @staticmethod
    def _normalize(heatmap: np.ndarray) -> np.ndarray:
        """Normaliza heatmap a [0, 1] con min-max."""
        h_min = heatmap.min()
        h_max = heatmap.max()
        if h_max > h_min:
            return (heatmap - h_min) / (h_max - h_min + 1e-8)
        # Si el heatmap es constante, retornar ceros
        return np.zeros_like(heatmap)

    def _apply_smoothing(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Aplica suavizado gaussiano y re-normaliza.

        Args:
            heatmap: ndarray (H, W) en [0, 1].

        Returns:
            heatmap suavizado y re-normalizado a [0, 1].
        """
        from scipy.ndimage import gaussian_filter

        smoothed = gaussian_filter(heatmap, sigma=self._config.sigma_smooth)
        return self._normalize(smoothed)

    # -----------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------

    def extract(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
        return_prediction: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """
        Extrae heatmap CAM continuo para una imagen.

        Intenta usar el backend externo (pytorch-grad-cam) si está
        configurado y disponible. Si falla, usa el backend manual.

        Args:
            image: Tensor normalizado (3, H, W).
                Debe estar preprocesado con la normalización del modelo
                (ej. ImageNet mean/std para ResNet).
            target_class: Clase contra la cual computar gradientes.
                None = clase predicha por el modelo. Solo se usa cuando
                el clasificador predice la clase de interés (ej. glaucoma).
            return_prediction: Si True, retorna también la predicción
                del modelo como segundo elemento del tuple.

        Returns:
            Si return_prediction=False:
                heatmap: ndarray (H, W) con valores en [0, 1]
            Si return_prediction=True:
                (heatmap, pred_info) donde pred_info = {
                    "predicted_class": int,
                    "target_class": int,
                    "probabilities": ndarray
                }
        """
        heatmap, pred_info = self._extract_with_backend(image, target_class)

        if return_prediction:
            return heatmap, pred_info
        return heatmap

    def extract_binary(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
        percentile: int | None = None,
    ) -> np.ndarray:
        """
        Extrae CAM binarizado usando un umbral por percentil.

        Pipeline:
            1. extract() → heatmap continuo [0, 1]
            2. Calcular threshold = np.percentile(heatmap, percentile)
            3. Binarizar: pixel >= threshold → 1, sino → 0

        Args:
            image: Tensor normalizado (3, H, W).
            target_class: Clase objetivo (None = predicha).
            percentile: Percentil de binarización. Si None, usa
                config.percentile. Valores típicos: 90-99.

        Returns:
            mask: ndarray (H, W) con valores 0.0 o 1.0 (float32).
        """
        p = percentile if percentile is not None else self._config.percentile
        heatmap = self.extract(image, target_class=target_class)
        threshold = np.percentile(heatmap, p)
        return (heatmap >= threshold).astype(np.float32)

    def extract_batch(
        self,
        images: torch.Tensor,
        target_class: int | None = None,
    ) -> list[np.ndarray]:
        """
        Procesa un batch completo imagen por imagen.

        Grad-CAM requiere backward pass por imagen, por lo que el
        procesamiento en batch se serializa internamente.

        Args:
            images: Tensor (B, 3, H, W) normalizado.
            target_class: Clase objetivo para todas las imágenes.
                None = clase predicha individualmente.

        Returns:
            Lista de B heatmaps, cada uno ndarray (H, W) en [0, 1].
        """
        if images.dim() != 4:
            raise ValueError(
                f"extract_batch espera tensor 4D (B, 3, H, W), "
                f"recibido: {images.dim()}D"
            )

        heatmaps = []
        for i in range(images.size(0)):
            hm = self.extract(images[i], target_class=target_class)
            heatmaps.append(hm)

        return heatmaps

    @torch.enable_grad()
    def _extract_with_backend(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Selecciona el backend apropiado para extraer el heatmap.

        Estrategia:
            1. Si use_external=True y la librería está disponible,
               usar pytorch-grad-cam
            2. Si falla o no está disponible, fallback al backend manual
            3. Una vez que se detecta que la librería no está disponible,
               no se vuelve a intentar (cache de estado)

        Args:
            image: Tensor (3, H, W).
            target_class: Clase objetivo.

        Returns:
            (heatmap, pred_info) tuple.
        """
        if self._config.use_external and self._external_available is not False:
            try:
                result = self._extract_external(image, target_class)
                self._external_available = True
                return result
            except Exception as e:
                self._external_available = False
                logger.warning(
                    f"pytorch-grad-cam no disponible ({e}), "
                    f"usando backend manual para todas las imágenes"
                )

        return self._compute_heatmap(image, target_class)

    @torch.enable_grad()
    def _extract_external(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Backend usando pytorch-grad-cam.

        Auto-instala la librería si no está disponible.

        Args:
            image: Tensor (3, H, W).
            target_class: Clase objetivo.

        Returns:
            (heatmap, pred_info) tuple.
        """
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import (
                ClassifierOutputTarget,
            )
        except ImportError:
            logger.warning(
                "pytorch-grad-cam no instalado. Instalando automáticamente..."
            )
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "grad-cam", "-q"]
            )
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import (
                ClassifierOutputTarget,
            )

        # Preparar imagen
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self._device).requires_grad_(True)

        # Obtener predicción (sin gradientes, solo inferencia)
        with torch.no_grad():
            output = self._model(image)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = int(output.argmax(dim=1).item())

        cls_target = target_class if target_class is not None else pred_class

        # Ejecutar pytorch-grad-cam
        target_layers = [self._target_layer]
        cam = GradCAM(model=self._model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(cls_target)]
        grayscale_cam = cam(input_tensor=image, targets=targets)[0]

        # Normalizar
        heatmap = self._normalize(grayscale_cam)

        # Suavizado opcional
        if self._config.sigma_smooth > 0:
            heatmap = self._apply_smoothing(heatmap)

        pred_info = {
            "predicted_class": pred_class,
            "target_class": cls_target,
            "probabilities": probabilities,
        }

        return heatmap, pred_info

    # -----------------------------------------------------------------
    # Cleanup de hooks
    # -----------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Remueve hooks registrados para liberar memoria.

        Llamar este método cuando el extractor ya no sea necesario,
        especialmente si se va a reutilizar el modelo con otro extractor.
        Después de llamar cleanup(), el extractor queda inutilizable.
        """
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._activations = None
        self._gradients = None
        logger.info(
            f"Hooks removidos de {type(self).__name__} "
            f"(target_layer='{self._config.target_layer}')"
        )

    def __del__(self) -> None:
        """Limpieza automática de hooks al destruir el objeto."""
        try:
            if self._hook_handles:
                self.cleanup()
        except Exception:
            # Silenciar errores durante garbage collection
            pass


# =============================================================================
# Implementación: Grad-CAM Estándar (Selvaraju et al., 2017)
# =============================================================================


class GradCAMExtractorModule(BaseCAMExtractor):
    """
    Implementación de Grad-CAM estándar (Selvaraju et al., 2017).

    Calcula los pesos por canal usando Global Average Pooling de los
    gradientes: α_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)

    Donde:
        - α_k: peso del canal k
        - y^c: score de la clase c (pre-softmax)
        - A^k_ij: activación del canal k en posición (i,j)
        - Z: número de pixeles en el feature map

    Uso:
        from modules.gradcam_module import GradCAMExtractorModule, GradCAMConfig

        config = GradCAMConfig(target_layer="layer4[-1]", percentile=95)
        extractor = GradCAMExtractorModule(model=my_cnn, config=config)

        # Heatmap continuo [0, 1]
        heatmap = extractor.extract(image_tensor)

        # Máscara binaria {0, 1}
        binary_mask = extractor.extract_binary(image_tensor)

        # Con predicción del modelo
        heatmap, pred_info = extractor.extract(image_tensor, return_prediction=True)

        # Batch
        heatmaps = extractor.extract_batch(batch_tensor)

        # Cleanup cuando ya no se necesite
        extractor.cleanup()
    """

    def _compute_cam_weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        """
        Grad-CAM: pesos = Global Average Pooling de gradientes.

        α_k = mean(gradients_k)  sobre dimensiones espaciales (H, W)

        Args:
            activations: Tensor (B, C, h, w).
            gradients: Tensor (B, C, h, w).

        Returns:
            weights: Tensor (B, C, 1, 1).
        """
        return gradients.mean(dim=(2, 3), keepdim=True)
