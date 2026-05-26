# SPEC.md - Especificación del Experimento de Selección de Clasificador CNN

## 1. Objetivo

Seleccionar el mejor backbone CNN (ResNet-18, EfficientNet-B0, DenseNet-121) para el clasificador del pipeline, evaluando **3 dimensiones** (sin training completo, solo few-shot):
- Calidad del Grad-CAM (IoU con máscara GT)
- Few-shot (Accuracy con N=50, N=100, usando mismas imágenes para todos)
- Costo computacional (parámetros, VRAM, tiempo de inferencia)

> **Nota:** No se realiza training completo. El experimento usa few-shot (N=50, N=100) para comparar backbones.

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

## 6. Few-Shot: Mismo Sampling Fijo para Todos

Para que la comparación sea justa, todos los backbones usan las **mismas N imágenes**:

```python
few_shot_indices = {
    "N50": random.sample(train_ids, 50, seed=42),   # Mismos 50 para todos
    "N100": random.sample(train_ids, 100, seed=42), # Mismos 100 para todos
}
```

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

### Dim 2: Few-shot
- Accuracy@N50 (50 imágenes)
- Accuracy@N100 (100 imágenes)
- F1-macro@N50, F1-macro@N100
- Mismas imágenes para todos los backbones

### Dim 3: Computacional
- Total parameters (M)
- VRAM batch=1 (MB)
- VRAM batch=16 (MB)
- Tiempo de inferencia (ms/imagen)

---

## 8. Fórmula de Selección

```
VRAM_norm = (VRAM - min_VRAM) / (max_VRAM - min_VRAM)
Score = 0.40 × Mean(F1@N50, F1@N100) + 0.40 × IoU_GradCAM + 0.20 × (1 - VRAM_norm)
```

Winner = backbone con mayor Score

---

## 9. Estructura de Resultados

```
results/
├── resnet18/
│   ├── model_N50.pth
│   ├── model_N100.pth
│   ├── gradcam_metrics.json
│   └── few_shot_metrics.json
├── efficientnet_b0/
│   └── ...
└── densenet121/
    └── ...

selection_summary.json    # Tabla comparativa + winner
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
| `scripts/few_shot.py` | Fine-tune con N=50, N=100 (mismas imágenes para todos) |
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