# Classifier — EfficientNet-B0 (glaucoma vs normal)

Clasificador CNN del pipeline (módulo **M2**). Dada una foto de fondo de ojo:

- decide **glaucoma** o **normal**, y
- si es **glaucoma**, además devuelve un **Grad-CAM**: un mapa de calor que muestra *dónde miró*
  la red para decidirlo (como mapa numérico y como imagen PNG con el calor superpuesto sobre la foto).
  Si es **normal**, solo devuelve el veredicto.

EfficientNet-B0 fue el **ganador** del experimento `Experiments/Classifier_Selection` (el más ligero
y con mejor F1 few-shot). Esta carpeta es su versión "de producción", limpia y enfocada solo en él.

---

## Estructura

```
Classifier/
├── config.yaml                     # toda la configuración (NADA se hardcodea en el código)
├── efficientNet_init.py            # la clase del modelo: setup + entrenamiento + predict + Grad-CAM
├── efficientNet_classification.py  # INFERENCIA: clasificar una imagen (+ Grad-CAM si glaucoma)
├── splits/                         # los split_repetition_*.json (listas train/val/test por repetición)
├── training/
│   ├── data_interface.py           # leer splits, leer etiquetas, armar y submuestrear los datos
│   ├── few_shot.py                 # entrenar+testear las 5 repeticiones
│   └── train_orchestrator.ipynb    # notebook para correr todo en Colab
├── checkpoints/                    # (se crea al entrenar) fs_weights_split_<i>.pth  — NO se versiona
└── results/                        # (se crea al testear) predicciones + Grad-CAM    — NO se versiona
```

---

## Los datos

El entrenamiento se basa en **5 archivos** `split_repetition_{1..5}.json` (ya en `splits/`). Cada uno
trae sus propias listas `train` / `validation` / `test`. **El split no trae la etiqueta**
glaucoma/normal: se lee del `.json` de anotación de cada imagen (campo `label`:
`Pathological → glaucoma`, `Normal → normal`).

- Las **imágenes/máscaras/anotaciones** NO se versionan (son privadas): viven en **Google Drive** y
  se montan en Colab. La ruta raíz se configura en `config.yaml → data.data_root`.
- Los **splits** (pequeños, solo rutas) sí se versionan en `splits/`.

Las rutas de un split son relativas a `data_root` (ej. `data_root/"1209/1209_left.jpg"`).

---

## Entrenar (protocolo de 5 repeticiones)

Para la repetición `i` (1→5): se usa `split_repetition_i.json` con la seed `i`
(`[42, 123, 456, 789, 1024]`). Se entrena con su sección **`train_fewshot`** (30 imágenes ya
balanceadas: 15 glaucoma + 15 normal), se usa `validation` para *early stopping*, se guardan los pesos
`fs_weights_split_i.pth`, y se testea con su `test` guardando **predicciones + Grad-CAM** de las
predichas glaucoma.

**Opción A — Notebook (recomendado, en Colab con GPU):** abre
[`training/train_orchestrator.ipynb`](training/train_orchestrator.ipynb) y corre las celdas.

**Opción B — Terminal:**
```bash
cd Classifier
python -m training.few_shot --config config.yaml
```

Salidas:
- `checkpoints/fs_weights_split_1.pth` … `fs_weights_split_5.pth` (5 modelos).
- `results/split_<i>/predictions.json` — por imagen: id, etiqueta real, clase predicha, probabilidades.
- `results/split_<i>/gradcam/<id>_gradcam.png` y `.npy` — solo de las predichas glaucoma.
- `results/training_summary.json` — resumen (rutas + métricas de validación por repetición).

> Las métricas finales (accuracy / F1 / matriz de confusión) **no** se calculan aquí: se derivan
> después a partir de los `predictions.json` (es lo que se acordó).

---

## Clasificar una imagen (inferencia)

**Terminal:**
```bash
cd Classifier
python efficientNet_classification.py \
    --image /ruta/a/foto.jpg \
    --checkpoint checkpoints/fs_weights_split_1.pth \
    --out-dir results/infer
```

**Python:**
```python
from efficientNet_classification import load_config, load_classifier, classify_image

config = load_config("config.yaml")
clf = load_classifier("checkpoints/fs_weights_split_1.pth", config)
res = classify_image("foto.jpg", clf, config, out_dir="results/infer")

print(res["prediction"], res["distribution"])
# Si es glaucoma: res["gradcam"]["overlay_image"] (PIL), ["heatmap"] (array), ["overlay_path"] (PNG)
# Si es normal:  res["gradcam"] is None
```

---

## Configuración clave (`config.yaml`)

| Sección | Qué controla |
|---|---|
| `data.data_root` | Carpeta raíz de las imágenes (en Colab: el Drive). |
| `data.split_files` | Los 5 archivos de split (orden = orden de repeticiones). |
| `data.label_map` | Mapeo del campo `label` → clase (`Pathological`/`Normal`). |
| `training.seeds` | Una seed por repetición. |
| `training.train_section` | Sección del split para entrenar (`train_fewshot`). |
| `gradcam` | Capa objetivo, colormap y transparencia del overlay. |

---

## Notas sobre los datos

- La etiqueta sale del campo `label` del `.json` de anotación (confirmado: es un array `[{...}]` con
  `Pathological`/`Normal`). `read_label` toma el primer registro; el `label_map` es configurable.
- El entrenamiento usa la sección `train_fewshot` (ya balanceada, 15 glaucoma + 15 normal). Si quisieras
  submuestrearla aún más, pon `training.train_subsample.size` a un entero en el `config.yaml`.

---

## Notas técnicas

- El **Grad-CAM** lo extrae el módulo compartido `modules/gradcam_module.py` de la raíz del repo
  (una sola implementación de Grad-CAM en todo el proyecto), con capa objetivo `features[-1]`.
- Preprocesamiento fijo del modelo: imágenes a **448×448** y normalización **ImageNet**.
- Reproducibilidad: cada repetición fija su seed global y `cudnn.deterministic=True`.
- Entorno: requiere `torch`, `torchvision`, `pyyaml`, `pillow`, `numpy`, `matplotlib` (todo presente
  en Colab). Localmente, sin `torchvision`, solo corre la lógica de datos (`data_interface`).
