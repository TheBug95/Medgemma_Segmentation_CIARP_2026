# Grad-CAM with ResNet18 on REFUGE

## Descripción

Experimento para extraer Grad-CAM de una ResNet18 pre-entrenada usando el dataset REFUGE. Genera visualizaciones comparativas (Grad-CAM vs máscara GT ground truth) y calcula métricas de IoU, SSIM y pointing accuracy.

## Estructura

```
gradcam_resnet18_refuge/
├── config.yaml                      # Configuración centralizada
├── modules/
│   └── gradcam_extractor.py         # CNNClassifier + Grad-CAM
├── scripts/
│   ├── train.py                     # Entrenamiento del modelo
│   ├── extract_and_evaluate.py       # Extracción + métricas + visualización
│   └── visualize.py                 # Funciones de plotting
├── output/
│   ├── model.pth                    # Modelo entrenado
│   └── metrics.json                 # Métricas aggregadas
├── visualizations/                  # PNGs 2x2 por imagen
└── README.md
```

## Parámetros configurables (config.yaml)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `seed` | 42 | Semilla global |
| `data.image_size` | [448, 448] | Tamaño de entrada CNN |
| `data.batch_size` | 16 | Batch size |
| `classifier.epochs` | 30 | Epochs de entrenamiento |
| `classifier.lr` | 0.0001 | Learning rate |
| `gradcam.percentile` | 95 | Percentil para binarizar |
| `visualization.num_samples` | 20 | Muestras a visualizar |
| `visualization.save_images` | true | Guardar PNGs |
| `visualization.show_images` | false | plt.show() |

## Métricas

| Métrica | Descripción |
|---------|-------------|
| **IoU** | `TP / (TP + FP + FN)` entre Grad-CAM binarizado y máscara GT |
| **SSIM** | Similitud estructural (manual, ventana 11x11) entre Grad-CAM normalizado [0,1] y máscara GT |
| **Pointing Accuracy** | 1.0 si el máximo de Grad-CAM cae dentro del optic disc, 0.0 si no |

## Uso

```bash
# Entrenar y evaluar (default)
python scripts/extract_and_evaluate.py

# Usar modelo pre-entrenado (solo evaluación)
python scripts/extract_and_evaluate.py --no-train --model-path ./output/model.pth

# Solo entrenar
python scripts/train.py
```

## Visualización 2x2

Cada imagen genera un panel con:

```
(a) Imagen original              (b) Grad-CAM heatmap superpuesto
(c) Máscara GT binarizada       (d) Overlay: GT (blue) vs Grad-CAM contour (red)
```

## Formato de máscaras GT

| Valor | Estructura |
|-------|------------|
| 0 | Background |
| 1 | Rim (neuroretinal border) |
| 2 | Cup (optic cup) |

Binaria: conversión a `mask > 0 → 1` (optic disc completo = rim + cup).

## Pipeline

```
Dataset REFUGE (test)
    ↓
ResNet18 training (train split)
    ↓
Grad-CAM extraction (layer4[-1].conv2)
    ↓
Comparación vs GT mask
    ↓
Visualización 2x2 + métricas JSON
```
