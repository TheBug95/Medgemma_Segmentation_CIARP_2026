# M9: Orchestrator — Conexión y Ejecución del Pipeline Completo

## 1. Propósito

El Orchestrator es el **director de orquesta**. No tiene lógica propia de machine learning — su trabajo es conectar los 8 módulos anteriores en el orden correcto, ejecutar las 18 configuraciones experimentales (3 pipelines × 6 condiciones), y guardar los resultados de forma estructurada.

Es la pieza que convierte módulos independientes en un pipeline funcional.

---

## 2. Inicialización

Al arrancar, el Orchestrator:

1. Lee el archivo `config.yaml` completo
2. Inicializa cada módulo pasándole su sección de configuración correspondiente:
   - `DataModule` con la sección `data`
   - `CNNClassifier` con la sección `classifier`
   - `SAMSegmenter` con la sección `sam`
   - `PipelineA_LoRA` con la sección `pipeline_a`
   - `PipelineB_WSSS` con la sección `pipeline_b`
   - `PipelineC_FSLFD` con la sección `pipeline_c`
   - `MedGemmaConditioner` con la sección `medgemma`
   - `Evaluator` con la sección `evaluation`
3. Carga los modelos preentrenados desde sus checkpoints (CNN, SAM, SAM-LoRA, calibración FSL/FD)
4. Verifica que todo está listo (modelos cargados, datos accesibles, GPU disponible)

---

## 3. Flujo de Ejecución Completo

Para cada imagen del set de test, el Orchestrator ejecuta los siguientes pasos:

### Paso 1A: Clasificación

Pasa la imagen por el **CNNClassifier** y obtiene:
- La predicción de enfermedad (ej: "glaucoma")
- La distribución de probabilidades (ej: {glaucoma: 0.92, normal: 0.08})
- El Grad-CAM de la imagen (mapa de calor de 448×448)

Estos resultados se guardan porque se reutilizan múltiples veces.

### Paso 1B: Segmentación (3 pipelines)

Se ejecutan los **3 pipelines en paralelo** (o secuencialmente si la VRAM no alcanza):

**Pipeline A (LoRA):**
- Pasa la imagen por SAM-LoRA → obtiene una máscara directa

**Pipeline B (WSSS):**
- Pasa la imagen por SAM → obtiene N candidatas
- Pasa el Grad-CAM y las candidatas al módulo WSSS → selecciona la mejor por IoU

**Pipeline C (FSL/FD):**
- Reutiliza las mismas N candidatas de SAM (no hay que regenerarlas)
- Pasa la predicción del clasificador, las candidatas y la imagen al módulo FSL/FD → filtra y selecciona

Al final se tienen **3 máscaras**: una por pipeline.

### Paso 2: MedGemma (6 condiciones por pipeline)

Para cada una de las 3 máscaras, se ejecutan las **6 condiciones de ablation**:

| Condición | Se envía a MedGemma |
|-----------|--------------------|
| A | Imagen cruda + prompt genérico |
| B | Imagen con máscara overlay + prompt que menciona la región |
| C1 | Imagen cruda + prompt con la predicción |
| C2 | Imagen cruda + prompt con la distribución completa |
| D1 | Imagen con overlay + prompt con predicción + región |
| D2 | Imagen con overlay + prompt con distribución + región |

Cada ejecución produce un texto descriptivo. En total: 3 máscaras × 6 condiciones = **18 textos** por imagen.

### Paso 3: Evaluación

Para cada uno de los 18 textos:
- Métricas de segmentación: IoU, Dice, SSIM de la máscara usada vs la GT
- Métricas de texto: BERTScore y precisión de hallazgo del texto vs la descripción del experto

---

## 4. Estructura de Resultados

Los resultados se guardan en archivos JSON organizados así:

```
results/
├── pipeline_a_lora/
│   ├── condition_A.json
│   ├── condition_B.json
│   ├── condition_C1.json
│   ├── condition_C2.json
│   ├── condition_D1.json
│   └── condition_D2.json
├── pipeline_b_wsss/
│   ├── condition_A.json
│   └── ...
└── pipeline_c_fslfd/
    ├── condition_A.json
    └── ...
```

Cada archivo JSON contiene una lista de resultados, uno por imagen:
- `image_id`: identificador de la imagen
- `segmentation_metrics`: {iou, dice, ssim}
- `text_metrics`: {bertscore_f1, finding_mentioned}
- `generated_text`: el texto producido por MedGemma
- `prompt_used`: el prompt exacto enviado
- `classification`: {prediction, distribution}
- `config`: la configuración usada
- `seed`: la semilla
- `timestamp`: cuándo se ejecutó

---

## 5. Análisis Comparativo

Después de ejecutar todas las configuraciones, el Orchestrator genera un resumen comparativo:

### 5.1 Tabla de métricas promedio

Se calcula el promedio ± desviación estándar de cada métrica por configuración:

| | A | B | C1 | C2 | D1 | D2 |
|---|---|---|---|---|---|---|
| **LoRA** | BERTScore ± σ | ... | ... | ... | ... | ... |
| **WSSS** | ... | ... | ... | ... | ... | ... |
| **FSL/FD** | ... | ... | ... | ... | ... | ... |

### 5.2 Tests estadísticos

Se aplica el test de Wilcoxon pareado para cada comparación relevante:
- A vs B, A vs C1, A vs C2, A vs D2 (dentro de cada pipeline)
- LoRA-D2 vs WSSS-D2 vs FSL-D2 (entre pipelines)

Se reporta p-valor y effect size.

### 5.3 Tabla de segmentación

| Pipeline | IoU | Dice | SSIM |
|----------|-----|------|------|
| LoRA | ... | ... | ... |
| WSSS | ... | ... | ... |
| FSL/FD | ... | ... | ... |

---

## 6. Optimización de VRAM

El Orchestrator debe gestionar la VRAM porque no todos los modelos caben simultáneamente. La estrategia es:

1. **Cargar CNN y SAM** para los pasos 1A y 1B
2. Al terminar paso 1B para todas las imágenes: **descargar SAM** de la GPU
3. **Cargar MedGemma** para el paso 2
4. Al terminar paso 2: **descargar MedGemma**
5. **Cargar BERTScore** para el paso 3

Alternativamente, procesar imagen por imagen cargando y descargando modelos según se necesitan (más lento pero usa menos VRAM).

---

## 7. Reproducibilidad

El Orchestrator es responsable de:
- Llamar a `set_global_seed(42)` al inicio
- Pasar la semilla a cada módulo
- Guardar la configuración completa junto con los resultados
- Verificar que `splits.json` existe y es el correcto

---

## 8. Verificaciones

1. Los resultados de las 18 configuraciones se guardaron correctamente
2. Cada JSON tiene la estructura esperada (todos los campos presentes)
3. Las métricas de segmentación son idénticas dentro de cada pipeline (la máscara no cambia entre condiciones)
4. Los resultados son reproducibles: dos ejecuciones con la misma semilla dan los mismos números
5. No hay imágenes faltantes (cada imagen del test set aparece en cada JSON)
