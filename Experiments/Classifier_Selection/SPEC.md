# SPEC.md - Especificación del Experimento de Selección de Clasificador CNN

## 1. Objetivo

Seleccionar el mejor backbone CNN (ResNet-18, EfficientNet-B0, DenseNet-121) para el clasificador del pipeline, evaluando **3 dimensiones** (sin training completo, solo few-shot con support set de glaucoma):
- Calidad del Grad-CAM (IoU con máscara GT)
- Few-shot F1-macro con N=25, 30, 35 (support set exclusivo de glaucoma, 5 iteraciones con seeds compartidas)
- Costo computacional (parámetros, VRAM, tiempo de inferencia)

> **Nota:** No se realiza training completo. El support set es exclusivamente de glaucoma (clase minoritaria de interés). En REFUGE/train hay 40 glaucomas → N=25, 30, 35. Se corren 5 iteraciones; en cada una todos los backbones usan la misma seed.

---

## 2. Backbones a Evaluar

| Nombre en código | Arquitectura | Feature Dim | Params | Import |
|-----------------|--------------|-------------|--------|--------|
| `resnet18` | ResNet-18 | 512 | 11.7M | `torchvision.models` |
| `efficientnet_b0` | EfficientNet-B0 | 1280 | 5.3M | `timm` |
| `densenet121` | DenseNet-121 | 1024 | 8.0M | `torchvision.models` |

---

## 3. Dataset

- **Origen:** `Datasets/REFUGE/` (train/, val/, test/)
- **Imágenes:** 400 por split, 40 glaucoma + 360 normal por split
- **Naming:** `g####.jpg`=glaucoma, `n####.jpg`=normal
- **Máscaras:** `Masks/g####.png` (0=background, 1=rim, 2=cup)
- **Labels:** `index.json` campo `Label` (1=glaucoma, 0=normal)

> **Restricción del support set:** Solo hay **40 imágenes de glaucoma** en train.
> El support set es exclusivamente de glaucoma (clase de interés y clase minoritaria).
> Esto limita los tamaños factibles a N < 40, de ahí N=25, 30, 35.

### Mapeo al Schema del Proyecto

| REFUGE | Proyecto |
|--------|----------|
| `Label: 1` | `label: "glaucoma"`, `disease_category: "glaucoma"` |
| `Label: 0` | `label: "normal"` |

### Conversión de vCDR Clínico a Escala 0-4

| vCDR | Escala 0-4 |
|------|------------|
| < 0.3 | 0 (Normal) |
| 0.3 - 0.5 | 1 (Normal-bajo) |
| 0.5 - 0.65 | 2 (Sospechoso) |
| 0.65 - 0.8 | 3 (Glaucoma leve) |
| > 0.8 | 4 (Glaucoma avanzado) |

---

## 4. Arquitectura del Clasificador

```
Input (3, 448, 448) - ImageNet normalized
    ↓
Backbone (pretrained ImageNet)
    ↓
Global Average Pooling → feature vector
    ↓
Dropout (p=0.5)
    ↓
Linear(num_classes=2)
    ↓
Softmax → distribution
```

### Freeze Strategy
- Freeze todo excepto último bloque conv + FC
- Fine-tune con Adam (lr=1e-4, weight_decay=1e-5)

### Training Config (Few-shot)
```yaml
optimizer: Adam
lr: 1e-4
weight_decay: 1e-5
scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
early_stopping: patience=5
max_epochs: 30
loss: CrossEntropyLoss
batch_size: 16
```

---

## 5. Augmentations (Solo Train)

- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness=0.1, contrast=0.1)
- Random erasing (p=0.2)
- Mixup (alpha=0.2)

---

## 6. Few-Shot: Mismo Sampling Fijo para Todos — 5 Iteraciones

El experimento corre **5 iteraciones** para cada backbone usando **las mismas 5 semillas**.
La regla fundamental es: **en la iteración i, TODOS los backbones usan la misma seeds[i]**.

Esto garantiza dos propiedades esenciales:
1. **Comparabilidad por iteración:** todos los backbones ven exactamente las mismas imágenes en cada iteración.
2. **Reproducibilidad:** cualquier investigador que ejecute iteración 0 obtendrá siempre el mismo subset.

```python
SEEDS = [42, 123, 456, 789, 1024]  # Definidas en config.yaml: few_shot.seeds

# Restricción: el support set es solo glaucoma (40 muestras disponibles en train)
# N=25 → 62.5% de los glaucoma disponibles
# N=30 → 75.0% de los glaucoma disponibles
# N=35 → 87.5% de los glaucoma disponibles

# Iteración 0: resnet18(seed=42), efficientnet_b0(seed=42), densenet121(seed=42)
# Iteración 1: resnet18(seed=123), efficientnet_b0(seed=123), densenet121(seed=123)
# ... y así hasta iteración 4

for i, seed in enumerate(SEEDS):            # i = 0..4
    for backbone in backbones:               # mismo seed para TODOS en esta iteración
        glaucoma_ids = get_glaucoma_indices(data_module, split='train')  # 40 IDs
        few_shot_indices = {
            "N25": random.sample(glaucoma_ids, 25, random_state=seed),
            "N30": random.sample(glaucoma_ids, 30, random_state=seed),
            "N35": random.sample(glaucoma_ids, 35, random_state=seed),
        }
        # Entrenar backbone con cada subset y esta seed
```

**Reporte final:** los resultados se agregan sobre las 5 iteraciones como **mean ± std**.
El score de selección se calcula sobre las medias.

---

## 7. Dimensiones de Evaluación

### Dim 1: Grad-CAM Quality
- IoU(Grad-CAM binarizado, GT mask binaria) usando percentil 95
- Pointing accuracy (¿punto máximo dentro del disco óptico?)

**Cómo extraer Grad-CAM:**
```python
# Grad-CAM sobre última capa conv del backbone
# 1. Forward pass
# 2. Backward en la clase predicha
# 3. Pooled gradients × activations
# 4. ReLU → heatmap
# 5. Binarizar con percentil 95
```

### Dim 2: Few-shot (support set de glaucoma)
- F1-macro@N25 — 25 de las 40 imágenes de glaucoma en train (62.5%)
- F1-macro@N30 — 30 de las 40 imágenes de glaucoma en train (75.0%)
- F1-macro@N35 — 35 de las 40 imágenes de glaucoma en train (87.5%)
- Evaluación en val set **completo** (40 glaucoma + 360 normal)
- Mismos subconjuntos para todos los backbones dentro de cada iteración

### Dim 3: Computacional
- Total parameters (M)
- VRAM batch=1 (MB)
- VRAM batch=16 (MB)
- Tiempo de inferencia (ms/imagen)

---

## 8. Fórmula de Selección

```
VRAM_norm = (VRAM - min_VRAM) / (max_VRAM - min_VRAM + 1e-8)
F1_mean   = mean(F1@N25_mean, F1@N30_mean, F1@N35_mean)   # medias sobre las 5 iteraciones
Score     = 0.40 × F1_mean + 0.40 × IoU_GradCAM_mean + 0.20 × (1 - VRAM_norm)
```

Todas las métricas son la **media sobre las 5 iteraciones** (seeds 42, 123, 456, 789, 1024).  
Winner = backbone con mayor Score

---

## 9. Estructura de Resultados

Cada iteración genera sus propios archivos. Al final se agrega sobre las 5 iteraciones.

```
results/
├── resnet18/
│   ├── seed_42/
│   │   ├── model_N25.pth
│   │   ├── model_N30.pth
│   │   ├── model_N35.pth
│   │   ├── gradcam_metrics.json    # evaluado con model_N35
│   │   └── few_shot_metrics.json   # {N25, N30, N35} → {f1_macro, accuracy, epochs}
│   ├── seed_123/
│   └── ...  (seed_456, seed_789, seed_1024)
├── efficientnet_b0/
│   └── ...
└── densenet121/
    └── ...

selection_summary.json    # Tabla comparativa (mean ± std) + winner
selection_report.md        # Análisis detallado
```

---

## 10. Archivos del Experimento

| Archivo | Descripción |
|---------|-------------|
| `config.yaml` | Configuración centralizada |
| `modules/data_module.py` | M1 - Data loading |
| `modules/cnn_classifier.py` | M2 - Backbones + Grad-CAM + Trainer |
| `scripts/convert_refuge_format.py` | Convierte REFUGE → annotations.json + splits.json |
| `scripts/evaluate.py` | Evaluación de clasificación (Accuracy, F1, Confusion Matrix) |
| `scripts/extract_gradcam.py` | Grad-CAM + IoU vs GT |
| `scripts/few_shot.py` | Fine-tune con N=25, 30, 35 (support set de glaucoma; mismas imágenes por iteración) |
| `scripts/benchmark_inference.py` | VRAM y tiempo |
| `notebook/experiment_orchestrator.ipynb` | Orquestador Colab |

---

## 11. Prerrequisitos

```bash
torch>=2.0
torchvision
timm
scikit-learn
scipy
pillow
numpy
pyyaml
matplotlib
seaborn
```