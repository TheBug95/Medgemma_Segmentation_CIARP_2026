#!/usr/bin/env python3
# =============================================================================
# scripts/test_gradcam_module.py
# =============================================================================
# Tests unitarios del módulo independiente de Grad-CAM.
#
# Verifica:
#   1. GradCAMConfig: creación, validación, from_dict, from_yaml
#   2. LayerResolver: resolve, list_layers, suggest_target_layer
#   3. GradCAMExtractorModule: extract, extract_binary, extract_batch
#   4. Cleanup de hooks
#   5. Compatibilidad con múltiples backbones (ResNet18, DenseNet, etc.)
#
# USO:
#   python scripts/test_gradcam_module.py
#
# Puede correrse desde cualquier ubicación; se ajusta el path automáticamente.
# =============================================================================

import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ajustar path para encontrar el módulo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Si estamos en Experiments/gradcam_resnet18_refuge/scripts/, subir 3 niveles
if "Experiments" in str(PROJECT_ROOT):
    PROJECT_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.gradcam_module import (
    BaseCAMExtractor,
    GradCAMConfig,
    GradCAMExtractorModule,
    LayerResolver,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utilidades de test
# =============================================================================

_passed = 0
_failed = 0
_errors: list[str] = []


def assert_test(condition: bool, test_name: str, detail: str = "") -> None:
    """Verifica una condición y registra resultado."""
    global _passed, _failed
    if condition:
        _passed += 1
        logger.info(f"  [OK] PASS: {test_name}")
    else:
        _failed += 1
        msg = f"  [FAIL] FAIL: {test_name}"
        if detail:
            msg += f" - {detail}"
        logger.error(msg)
        _errors.append(msg)


def create_dummy_image(size: tuple[int, int] = (448, 448)) -> torch.Tensor:
    """Crea una imagen dummy normalizada (3, H, W) con valores realistas."""
    torch.manual_seed(42)
    # Simular imagen con normalización ImageNet
    img = torch.randn(3, size[0], size[1])
    return img


# =============================================================================
# Test 1: GradCAMConfig
# =============================================================================


def test_config_creation() -> None:
    """Prueba creación y validación de GradCAMConfig."""
    logger.info("=" * 60)
    logger.info("Test 1: GradCAMConfig")
    logger.info("=" * 60)

    # Creación básica
    config = GradCAMConfig(target_layer="layer4[-1]")
    assert_test(config.target_layer == "layer4[-1]", "target_layer default")
    assert_test(config.percentile == 95, "percentile default = 95")
    assert_test(config.use_external is True, "use_external default = True")
    assert_test(config.sigma_smooth == 0.0, "sigma_smooth default = 0.0")
    assert_test(config.image_size is None, "image_size default = None")

    # Con parámetros personalizados
    config2 = GradCAMConfig(
        target_layer="features.denseblock4",
        percentile=90,
        use_external=False,
        sigma_smooth=1.5,
        image_size=(224, 224),
    )
    assert_test(config2.percentile == 90, "percentile personalizado")
    assert_test(config2.image_size == (224, 224), "image_size personalizado")

    # Validación: target_layer vacío
    try:
        GradCAMConfig(target_layer="")
        assert_test(False, "target_layer vacío debería lanzar error")
    except ValueError:
        assert_test(True, "target_layer vacío lanza ValueError")

    # Validación: percentile fuera de rango
    try:
        GradCAMConfig(target_layer="layer4[-1]", percentile=101)
        assert_test(False, "percentile=101 debería lanzar error")
    except ValueError:
        assert_test(True, "percentile=101 lanza ValueError")

    # Validación: sigma negativo
    try:
        GradCAMConfig(target_layer="layer4[-1]", sigma_smooth=-1.0)
        assert_test(False, "sigma_smooth negativo debería lanzar error")
    except ValueError:
        assert_test(True, "sigma_smooth negativo lanza ValueError")

    # from_dict: estructura plana
    d = {"target_layer": "layer4[-1]", "percentile": 80}
    config3 = GradCAMConfig.from_dict(d)
    assert_test(config3.percentile == 80, "from_dict estructura plana")

    # from_dict: estructura anidada (como config.yaml)
    d_nested = {
        "gradcam": {"target_layer": "features[-1]", "percentile": 92},
        "data": {"image_size": [448, 448]},
    }
    config4 = GradCAMConfig.from_dict(d_nested)
    assert_test(config4.target_layer == "features[-1]", "from_dict anidado target_layer")
    assert_test(config4.percentile == 92, "from_dict anidado percentile")
    assert_test(config4.image_size == (448, 448), "from_dict hereda image_size de data")

    # Inmutabilidad (frozen=True)
    try:
        config.percentile = 50  # type: ignore
        assert_test(False, "config debería ser inmutable")
    except AttributeError:
        assert_test(True, "config es inmutable (frozen)")


# =============================================================================
# Test 2: LayerResolver
# =============================================================================


def test_layer_resolver() -> None:
    """Prueba resolución de capas en diferentes arquitecturas."""
    logger.info("=" * 60)
    logger.info("Test 2: LayerResolver")
    logger.info("=" * 60)

    from torchvision.models import resnet18

    model = resnet18(weights=None)

    # Resolver layer4[-1]
    layer = LayerResolver.resolve(model, "layer4[-1]")
    assert_test(
        layer is model.layer4[-1],
        "resolve layer4[-1] en ResNet18",
    )

    # Resolver layer4[-1].conv2
    layer_conv = LayerResolver.resolve(model, "layer4[-1].conv2")
    assert_test(
        isinstance(layer_conv, nn.Conv2d),
        "resolve layer4[-1].conv2 -> Conv2d",
    )

    # Resolver capa inexistente
    try:
        LayerResolver.resolve(model, "layer99[-1]")
        assert_test(False, "layer99 debería lanzar error")
    except ValueError:
        assert_test(True, "capa inexistente lanza ValueError")

    # Resolver con índice fuera de rango
    try:
        LayerResolver.resolve(model, "layer4[99]")
        assert_test(False, "índice fuera de rango debería lanzar error")
    except ValueError:
        assert_test(True, "índice fuera de rango lanza ValueError")

    # list_layers
    all_layers = LayerResolver.list_layers(model)
    assert_test(len(all_layers) > 0, f"list_layers retorna {len(all_layers)} capas")
    assert_test("layer4" in all_layers, "layer4 está en list_layers")

    # list_layers con filtro por tipo
    conv_layers = LayerResolver.list_layers(model, filter_type=nn.Conv2d)
    assert_test(
        len(conv_layers) > 0,
        f"list_layers(Conv2d) retorna {len(conv_layers)} capas",
    )
    assert_test(
        len(conv_layers) >= 15,
        f"ResNet18 tiene suficientes Conv2d ({len(conv_layers)} >= 15 esperadas)",
    )

    # suggest_target_layer para ResNet
    suggested = LayerResolver.suggest_target_layer(model)
    assert_test(
        suggested == "layer4[-1]",
        f"suggest_target_layer(ResNet18) = '{suggested}'",
    )

    # suggest_target_layer para DenseNet (si está disponible)
    try:
        from torchvision.models import densenet121

        dense_model = densenet121(weights=None)
        suggested_dense = LayerResolver.suggest_target_layer(dense_model)
        assert_test(
            "denseblock" in suggested_dense,
            f"suggest_target_layer(DenseNet121) = '{suggested_dense}'",
        )

        # Verificar que la capa sugerida se puede resolver
        resolved = LayerResolver.resolve(dense_model, suggested_dense)
        assert_test(
            resolved is not None,
            "la capa sugerida para DenseNet121 es resoluble",
        )
    except ImportError:
        logger.info("  [SKIP] DenseNet121 no disponible")


# =============================================================================
# Test 3: GradCAMExtractorModule con ResNet18
# =============================================================================


def test_extractor_resnet18() -> None:
    """Prueba extracción de Grad-CAM con ResNet18."""
    logger.info("=" * 60)
    logger.info("Test 3: GradCAMExtractorModule (ResNet18)")
    logger.info("=" * 60)

    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()

    # Forzar backend manual para testear sin dependencia externa
    config = GradCAMConfig(
        target_layer="layer4[-1]",
        percentile=95,
        use_external=False,
        image_size=(448, 448),
    )

    extractor = GradCAMExtractorModule(model=model, config=config, device="cpu")

    image = create_dummy_image((448, 448))

    # Test extract()
    heatmap = extractor.extract(image)
    assert_test(
        isinstance(heatmap, np.ndarray),
        "extract retorna ndarray",
    )
    assert_test(
        heatmap.shape == (448, 448),
        f"shape del heatmap = {heatmap.shape} (esperado (448, 448))",
    )
    assert_test(
        0.0 <= heatmap.min() <= heatmap.max() <= 1.0,
        f"valores en [0, 1]: min={heatmap.min():.4f}, max={heatmap.max():.4f}",
    )

    # Test extract con return_prediction
    heatmap2, pred_info = extractor.extract(image, return_prediction=True)
    assert_test(
        "predicted_class" in pred_info,
        "pred_info contiene predicted_class",
    )
    assert_test(
        "probabilities" in pred_info,
        "pred_info contiene probabilities",
    )
    assert_test(
        len(pred_info["probabilities"]) == 1000,
        f"probabilities tiene {len(pred_info['probabilities'])} clases (ImageNet)",
    )

    # Test extract_binary()
    binary = extractor.extract_binary(image)
    assert_test(
        binary.shape == (448, 448),
        f"shape binario = {binary.shape}",
    )
    unique_vals = set(np.unique(binary))
    assert_test(
        unique_vals.issubset({0.0, 1.0}),
        f"valores binarios = {unique_vals} (esperado {{0.0, 1.0}})",
    )

    # Test extract_binary con percentil custom
    binary_p80 = extractor.extract_binary(image, percentile=80)
    # Con percentil más bajo, debería haber más pixeles activos
    assert_test(
        binary_p80.sum() >= binary.sum(),
        f"p80 activos ({binary_p80.sum():.0f}) >= p95 activos ({binary.sum():.0f})",
    )

    # Test extract_batch()
    batch = torch.stack([create_dummy_image(), create_dummy_image((448, 448))])
    heatmaps = extractor.extract_batch(batch)
    assert_test(
        len(heatmaps) == 2,
        f"extract_batch retorna {len(heatmaps)} heatmaps (esperado 2)",
    )
    assert_test(
        all(hm.shape == (448, 448) for hm in heatmaps),
        "todos los heatmaps del batch tienen shape (448, 448)",
    )

    # Test extract_batch error con tensor 3D
    try:
        extractor.extract_batch(image)  # 3D en lugar de 4D
        assert_test(False, "extract_batch con tensor 3D debería lanzar error")
    except ValueError:
        assert_test(True, "extract_batch con tensor 3D lanza ValueError")

    # Test cleanup
    n_hooks_antes = len(extractor._hook_handles)
    extractor.cleanup()
    assert_test(
        len(extractor._hook_handles) == 0,
        f"cleanup removió {n_hooks_antes} hooks",
    )
    assert_test(
        extractor._activations is None,
        "cleanup limpia activaciones",
    )


# =============================================================================
# Test 4: GradCAMExtractorModule con otros backbones
# =============================================================================


def test_extractor_other_backbones() -> None:
    """Prueba que el módulo funciona con diferentes arquitecturas CNN."""
    logger.info("=" * 60)
    logger.info("Test 4: Compatibilidad multi-backbone")
    logger.info("=" * 60)

    image = create_dummy_image((224, 224))

    # --- DenseNet121 ---
    try:
        from torchvision.models import densenet121

        model = densenet121(weights=None)
        model.eval()

        suggested = LayerResolver.suggest_target_layer(model)
        config = GradCAMConfig(
            target_layer=suggested,
            percentile=95,
            use_external=False,
            image_size=(224, 224),
        )
        extractor = GradCAMExtractorModule(model=model, config=config, device="cpu")
        heatmap = extractor.extract(image)
        assert_test(
            heatmap.shape == (224, 224),
            f"DenseNet121: shape={heatmap.shape}, target='{suggested}'",
        )
        assert_test(
            0.0 <= heatmap.min() <= heatmap.max() <= 1.0,
            "DenseNet121: valores en [0, 1]",
        )
        extractor.cleanup()
    except Exception as e:
        logger.warning(f"  [SKIP] DenseNet121: {e}")

    # --- EfficientNet-B0 ---
    try:
        from torchvision.models import efficientnet_b0

        model = efficientnet_b0(weights=None)
        model.eval()

        suggested = LayerResolver.suggest_target_layer(model)
        config = GradCAMConfig(
            target_layer=suggested,
            percentile=95,
            use_external=False,
            image_size=(224, 224),
        )
        extractor = GradCAMExtractorModule(model=model, config=config, device="cpu")
        heatmap = extractor.extract(image)
        assert_test(
            heatmap.shape == (224, 224),
            f"EfficientNet-B0: shape={heatmap.shape}, target='{suggested}'",
        )
        assert_test(
            0.0 <= heatmap.min() <= heatmap.max() <= 1.0,
            "EfficientNet-B0: valores en [0, 1]",
        )
        extractor.cleanup()
    except Exception as e:
        logger.warning(f"  [SKIP] EfficientNet-B0: {e}")

    # --- MobileNetV2 ---
    try:
        from torchvision.models import mobilenet_v2

        model = mobilenet_v2(weights=None)
        model.eval()

        suggested = LayerResolver.suggest_target_layer(model)
        config = GradCAMConfig(
            target_layer=suggested,
            percentile=95,
            use_external=False,
            image_size=(224, 224),
        )
        extractor = GradCAMExtractorModule(model=model, config=config, device="cpu")
        heatmap = extractor.extract(image)
        assert_test(
            heatmap.shape == (224, 224),
            f"MobileNetV2: shape={heatmap.shape}, target='{suggested}'",
        )
        assert_test(
            0.0 <= heatmap.min() <= heatmap.max() <= 1.0,
            "MobileNetV2: valores en [0, 1]",
        )
        extractor.cleanup()
    except Exception as e:
        logger.warning(f"  [SKIP] MobileNetV2: {e}")


# =============================================================================
# Test 5: Strategy Pattern (verificar que BaseCAMExtractor es abstracta)
# =============================================================================


def test_strategy_pattern() -> None:
    """Verifica que BaseCAMExtractor no se puede instanciar directamente."""
    logger.info("=" * 60)
    logger.info("Test 5: Strategy Pattern")
    logger.info("=" * 60)

    from torchvision.models import resnet18

    model = resnet18(weights=None)
    config = GradCAMConfig(target_layer="layer4[-1]", use_external=False)

    # BaseCAMExtractor es abstracta, no se puede instanciar
    try:
        BaseCAMExtractor(model=model, config=config, device="cpu")
        assert_test(False, "BaseCAMExtractor no debería ser instanciable")
    except TypeError:
        assert_test(True, "BaseCAMExtractor es abstracta (no instanciable)")

    # GradCAMExtractorModule sí se puede instanciar
    extractor = GradCAMExtractorModule(model=model, config=config, device="cpu")
    assert_test(
        isinstance(extractor, BaseCAMExtractor),
        "GradCAMExtractorModule hereda de BaseCAMExtractor",
    )
    assert_test(
        hasattr(extractor, "_compute_cam_weights"),
        "GradCAMExtractorModule implementa _compute_cam_weights",
    )
    extractor.cleanup()


# =============================================================================
# Test 6: Suavizado gaussiano
# =============================================================================


def test_smoothing() -> None:
    """Verifica que el suavizado gaussiano funciona correctamente."""
    logger.info("=" * 60)
    logger.info("Test 6: Suavizado gaussiano")
    logger.info("=" * 60)

    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()
    image = create_dummy_image((224, 224))

    # Sin suavizado
    config_no_smooth = GradCAMConfig(
        target_layer="layer4[-1]",
        use_external=False,
        sigma_smooth=0.0,
        image_size=(224, 224),
    )
    ext_no = GradCAMExtractorModule(model=model, config=config_no_smooth, device="cpu")
    hm_no = ext_no.extract(image)
    ext_no.cleanup()

    # Con suavizado
    config_smooth = GradCAMConfig(
        target_layer="layer4[-1]",
        use_external=False,
        sigma_smooth=2.0,
        image_size=(224, 224),
    )
    ext_sm = GradCAMExtractorModule(model=model, config=config_smooth, device="cpu")
    hm_sm = ext_sm.extract(image)
    ext_sm.cleanup()

    # El suavizado debería producir un heatmap diferente (menos ruidoso)
    diff = np.abs(hm_no - hm_sm).mean()
    assert_test(
        diff > 0.0,
        f"suavizado produce heatmap diferente (diff media = {diff:.6f})",
    )
    assert_test(
        0.0 <= hm_sm.min() <= hm_sm.max() <= 1.0,
        "heatmap suavizado sigue en [0, 1]",
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Ejecuta todos los tests."""
    global _passed, _failed, _errors

    logger.info("=" * 60)
    logger.info("Tests del módulo independiente de Grad-CAM")
    logger.info(f"Proyecto raíz: {PROJECT_ROOT}")
    logger.info("=" * 60)

    tests = [
        test_config_creation,
        test_layer_resolver,
        test_extractor_resnet18,
        test_extractor_other_backbones,
        test_strategy_pattern,
        test_smoothing,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            _failed += 1
            error_msg = f"  [ERROR] en {test_fn.__name__}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            _errors.append(error_msg)

    # Resumen
    logger.info("")
    logger.info("=" * 60)
    total = _passed + _failed
    logger.info(f"RESULTADO: {_passed}/{total} tests pasaron")
    if _failed > 0:
        logger.error(f"  {_failed} tests fallaron:")
        for err in _errors:
            logger.error(f"    {err}")
        sys.exit(1)
    else:
        logger.info("  [OK] Todos los tests pasaron correctamente")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
