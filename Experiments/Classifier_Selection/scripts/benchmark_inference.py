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
#   - count_parameters(model) -> int (total de parámetros)
#   - measure_vram(model, batch_size, device) -> float (MB)
#   - measure_inference_time(model, data_loader, device, num_runs) -> float (ms/img)
#   - run_benchmark(backbone_name, model, data_loader, device) -> dict
#
# RETORNO:
#   {
#     "total_parameters": int,
#     "trainable_parameters": int,
#     "vrám_batch_1": float,    # MB
#     "vram_batch_16": float,   # MB
#     "inference_time_ms": float  # ms por imagen
#   }
#
# CÓMO MEDIR VRAM:
#   - torch.cuda.empty_cache()
#   - torch.cuda.reset_peak_memory_stats()
#   - forward pass
#   - peak_memory = torch.cuda.max_memory_allocated() / 1024**2
#
# CÓMO MEDIR TIEMPO:
#   - Promedio de 100 forward passes (warmup 10)
#   - model.eval() + torch.no_grad()
#
# NOTAS:
#   - VRAM varía según batch size
#   - Tiempo de inferencia depende de GPU y carga del sistema
#   - Reportar ambas métricas para comparación completa
#
# USO:
#   from scripts.benchmark_inference import run_benchmark
#   benchmark_results = run_benchmark("resnet18", model, val_loader, device="cuda")
# =============================================================================