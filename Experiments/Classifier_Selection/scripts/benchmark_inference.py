# =============================================================================
# scripts/benchmark_inference.py
# =============================================================================
# Benchmark de inferencia: VRAM y tiempo
#
# RESPONSABILIDADES:
#   1. Medir VRAM consumida (batch=1 y batch=16)
#   2. Medir tiempo de inferencia (ms/imagen)
#   3. Contar parámetros totales del modelo
#
# FUNCIONES PRINCIPALES:
#   - count_parameters(model) -> dict
#   - measure_vram(model, batch_size, device) -> float (MB)
#   - measure_inference_time(model, data_loader, device, num_runs) -> float (ms/img)
#   - run_benchmark(backbone_name, model, data_loader, device) -> dict
#
# RETORNO de run_benchmark:
#   {
#     "total_parameters":     int,
#     "trainable_parameters": int,
#     "vram_batch_1_mb":      float,  # MB
#     "vram_batch_16_mb":     float,  # MB
#     "inference_time_ms":    float   # ms por imagen, batch=1
#   }
#
# CÓMO MEDIR VRAM:
#   - torch.cuda.empty_cache()
#   - torch.cuda.reset_peak_memory_stats()
#   - forward pass con dummy tensor
#   - peak_memory = torch.cuda.max_memory_allocated() / 1024**2
#
# CÓMO MEDIR TIEMPO:
#   - Warmup de _WARMUP_RUNS pases (evita sesgo de inicialización)
#   - Promedio de num_runs forward passes (default 100, configurable en config.yaml)
#   - CUDA Events para GPU (perf_counter mide despacho CPU, no ejecución GPU)
#   - model.eval() + torch.no_grad()
#
# NOTAS:
#   - VRAM varía según batch size; se mide con batch=1 y batch=16
#   - La fórmula de selección usa vram_batch_1_mb para la normalización
#   - Retorna 0.0 para VRAM en CPU (métrica exclusivamente GPU)
#   - El benchmark es independiente de seed: mide costo de hardware, no de datos
#
# USO:
#   from scripts.benchmark_inference import run_benchmark
#   benchmark_results = run_benchmark("resnet18", model, val_loader, device)
# =============================================================================

from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger(__name__)

# Pases de warm-up antes de iniciar la medición de tiempo.
# Necesarios para llenar caches de kernels GPU y disparar compilaciones lazy
# (e.g. cuDNN autotuner) que de otro modo penalizarían el primer run medido.
_WARMUP_RUNS = 10


def count_parameters(model) -> dict:
    """Cuenta parámetros totales y entrenables de un CNNClassifier.

    Args:
        model: Instancia de CNNClassifier.

    Returns:
        {"total_parameters": int, "trainable_parameters": int}
    """
    net = model.model
    total     = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def measure_vram(model, batch_size: int, device: torch.device) -> float:
    """Mide el pico de VRAM (MB) durante un forward pass con el batch dado.

    Resetea el contador de pico *después* de vaciar el cache para que el
    valor retornado refleje únicamente pesos del modelo + activaciones del
    batch, sin residuos de operaciones previas.

    En CPU retorna 0.0 — VRAM es una métrica exclusiva de GPU.

    Args:
        model:      Instancia de CNNClassifier ya movida al device.
        batch_size: Tamaño del batch a simular.
        device:     Device objetivo.

    Returns:
        Pico de VRAM asignada en megabytes.
    """
    if device.type != "cuda":
        return 0.0

    net = model.model
    net.eval()

    # Tensor dummy en device — mismas dimensiones que las entradas reales
    # (definidas en SPEC: 448×448, 3 canales, normalizadas ImageNet).
    dummy = torch.randn(batch_size, 3, 448, 448, device=device)

    # empty_cache libera el pool cacheado (tensores no referenciados pero aún
    # reservados por PyTorch) para que el pico refleje solo lo que este forward
    # necesita, no memoria ociosa de pases anteriores.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        net(dummy)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return float(peak_mb)


def measure_inference_time(
    model,
    data_loader,
    device: torch.device,
    num_runs: int = 100,
) -> float:
    """Mide el tiempo medio de inferencia (ms/imagen) con batch_size=1.

    Usa CUDA Events en lugar de time.perf_counter() para GPU: los kernels
    GPU se encolan de forma asíncrona respecto a la CPU, por lo que
    perf_counter mide el tiempo de despacho del host, no la ejecución real
    del dispositivo. CUDA Events viven en la línea de tiempo GPU y son
    precisos para esta medición.

    El data_loader se usa únicamente para inferir las dimensiones espaciales
    de entrada; la medición en sí usa un tensor dummy en device para aislar
    el tiempo de inferencia del modelo del overhead de carga de datos y
    transferencia host→device.

    Args:
        model:       Instancia de CNNClassifier.
        data_loader: DataLoader de validación (solo para inferir forma de imagen).
        device:      Device objetivo.
        num_runs:    Número de pases cronometrados a promediar.

    Returns:
        Tiempo medio por imagen en milisegundos.
    """
    net = model.model
    net.eval()

    # Derivar dimensiones desde el loader en lugar de hardcodear, por si
    # la resolución cambia en el futuro sin tocar este script.
    sample_batch = next(iter(data_loader))
    c, h, w = sample_batch["image"].shape[1:]
    dummy = torch.randn(1, c, h, w, device=device)

    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(_WARMUP_RUNS):
            net(dummy)

        if use_cuda:
            # Asegurar que los pases de warm-up terminaron antes de arrancar
            # el cronómetro — la sincronización bloquea hasta que la GPU vacía
            # la cola de trabajo pendiente.
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_runs):
                net(dummy)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start.elapsed_time(end)  # ms totales sobre num_runs pases
        else:
            t0 = time.perf_counter()
            for _ in range(num_runs):
                net(dummy)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return float(elapsed_ms / num_runs)


def run_benchmark(
    backbone_name: str,
    model,
    data_loader,
    device: torch.device,
    num_runs: int = 100,
) -> dict:
    """Ejecuta el benchmark computacional completo para un backbone.

    Diseñado para llamarse una sola vez por backbone (independiente de seed),
    ya que el costo computacional depende de la arquitectura y el hardware,
    no de los datos ni de la inicialización aleatoria.

    Args:
        backbone_name: Nombre identificador del backbone (e.g. "resnet18").
                       Solo se usa para logging.
        model:         Instancia de CNNClassifier con pesos entrenados cargados.
        data_loader:   DataLoader de validación. Se pasa a measure_inference_time
                       para inferencia de forma; no se itera para VRAM.
        device:        Device objetivo.
        num_runs:      Pases a promediar para la medición de tiempo.

    Returns:
        {
            "total_parameters":     int,    # total de parámetros del modelo
            "trainable_parameters": int,    # parámetros con requires_grad=True
            "vram_batch_1_mb":      float,  # VRAM pico con batch=1  (MB)
            "vram_batch_16_mb":     float,  # VRAM pico con batch=16 (MB)
            "inference_time_ms":    float,  # ms por imagen, batch=1
        }
    """
    logger.info("Benchmarking '%s' en %s ...", backbone_name, device)

    params  = count_parameters(model)
    vram_1  = measure_vram(model, batch_size=1,  device=device)
    vram_16 = measure_vram(model, batch_size=16, device=device)
    time_ms = measure_inference_time(model, data_loader, device, num_runs=num_runs)

    result = {
        "total_parameters":     params["total_parameters"],
        "trainable_parameters": params["trainable_parameters"],
        "vram_batch_1_mb":      vram_1,
        "vram_batch_16_mb":     vram_16,
        "inference_time_ms":    time_ms,
    }

    logger.info(
        "%s | params=%.1fM (trainable=%.1fM) | VRAM b1=%.0f MB  b16=%.0f MB | %.2f ms/img",
        backbone_name,
        params["total_parameters"]  / 1e6,
        params["trainable_parameters"] / 1e6,
        vram_1,
        vram_16,
        time_ms,
    )

    return result
