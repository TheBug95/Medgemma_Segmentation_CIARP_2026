# AGENTS.md — Instrucciones para Asistentes de IA

## Contexto del Proyecto

Este es un proyecto de investigación en **segmentación oftalmológica** que combina:
- Un clasificador CNN para identificar enfermedades oculares
- SAM (Segment Anything Model) para generar máscaras de segmentación
- MedGemma (VLM médico de Google) para generar descripciones clínicas

El objetivo es medir si **condicionar MedGemma** con información previa (máscara de segmentación y/o clasificación) mejora la calidad de sus descripciones comparado con un baseline sin condicionamiento.

## Arquitectura: 3 Pipelines Comparables

Existen **3 pipelines** que comparten clasificador y MedGemma pero difieren en cómo obtienen la máscara:

- **Pipeline A (LoRA):** SAM fine-tuned con LoRA → devuelve máscara directa (NO selecciona entre candidatas)
- **Pipeline B (WSSS):** SAM genera candidatas → se selecciona la que tenga mayor IoU con el Grad-CAM del clasificador CNN
- **Pipeline C (FSL/FD):** SAM genera candidatas → se filtran con KDE usando el threshold de la enfermedad predicha por el clasificador

Cada pipeline se evalúa con **6 condiciones de ablation** para MedGemma (ver sección Condiciones).

## Datos

Los datos son **privados y cerrados**. No deben publicarse, subirse a repositorios públicos, ni incluirse en logs.

Cada imagen tiene 4 anotaciones:
- `disease_category`: categoría de enfermedad (cataract, glaucoma, AMD, DR, normal)
- `disease_grading`: escala clínica (ej: LOCSIII para cataratas)
- `segmentation_mask`: máscara ground truth binaria
- `expert_description`: descripción textual del oftalmólogo

## Reglas de Código

### Estructura modular obligatoria
- Cada módulo está en `modules/` con una clase principal y una interfaz definida en `experimental_design.md`
- **NO mezclar** la lógica de un módulo con otro. Si necesitas funcionalidad de otro módulo, impórtalo y llama a su interfaz pública
- Cada módulo debe funcionar de forma independiente con un test unitario propio

### Replicabilidad
- **Semilla global:** `SEED = 42` definida en `config.yaml`
- Todo módulo debe recibir `seed` en su config y llamar a `set_global_seed(seed)` al inicializarse
- Los splits del dataset se cargan desde `splits.json` (nunca regenerar splits)
- Usar `torch.backends.cudnn.deterministic = True`

### Convenciones de código
- **Idioma del código:** Inglés (nombres de variables, funciones, clases)
- **Idioma de comentarios:** Español (explicar la lógica en español)
- **Type hints** obligatorios en todas las funciones públicas
- **Docstrings** obligatorios en todas las clases y métodos públicos
- Formato: `black` con línea máxima de 100 caracteres

### Configuración
- Toda configuración va en `config.yaml`, NO hardcodeada en el código
- Los módulos reciben su sección del config como `dict` en el constructor
- Rutas de archivos: usar `pathlib.Path`, nunca strings crudos

### Logging y resultados
- Usar `logging` estándar de Python, NO `print()`
- Los resultados se guardan en `results/` como JSON con estructura definida
- Cada resultado debe incluir: timestamp, config usada, semilla, métricas

## Las 6 Condiciones de Ablation (Paso 2)

MedGemma recibe 2 entradas: imagen y prompt. Las condiciones varían qué se envía:

| Cond | Imagen | Prompt | Parámetros del método `generate()` |
|------|--------|--------|-------------------------------------|
| A | Cruda | Genérico | `mask=None, prediction=None, distribution=None` |
| B | +Máscara overlay rojo | Menciona región | `mask=mask, prediction=None, distribution=None` |
| C1 | Cruda | +Solo clase predicha | `mask=None, prediction="cataract", distribution=None` |
| C2 | Cruda | +Distribución completa | `mask=None, prediction=None, distribution={"cataract": 0.80, ...}` |
| D1 | +Máscara overlay rojo | +Clase + región | `mask=mask, prediction="cataract", distribution=None` |
| D2 | +Máscara overlay rojo | +Distribución + región | `mask=mask, prediction=None, distribution={"cataract": 0.80, ...}` |

## Distinciones Críticas entre Pipelines

### Pipeline A (LoRA) — Segmentador DIRECTO
- SAM fue **entrenado** con LoRA usando máscaras GT
- En inferencia: imagen → SAM-LoRA → máscara directa
- **NO genera candidatas**, **NO selecciona**
- No depende del clasificador para la máscara

### Pipeline B (WSSS) — Selector por Grad-CAM
- SAM genera N candidatas (modo AMG)
- El clasificador CNN genera Grad-CAM de la imagen
- Se binariza el Grad-CAM (percentil 95)
- Se calcula IoU entre cada candidata y el Grad-CAM
- Se selecciona la candidata con mayor IoU
- **DEPENDE** del clasificador (Grad-CAM viene del clasificador)

### Pipeline C (FSL/FD) — Filtro por KDE
- SAM genera N candidatas (modo AMG)
- Se extrae embedding MedSigLIP de cada candidata
- El clasificador dice "cataract" → se usan los thresholds KDE de cataract
- Se evalúa log-densidad de cada candidata contra ese KDE
- Solo pasan las que están dentro de [Θ_min, Θ_max]
- **DEPENDE** del clasificador (la clase selecciona qué threshold usar)

## Métricas

### Segmentación (constantes por pipeline, no cambian entre condiciones A-D)
- **IoU:** TP / (TP + FP + FN)
- **Dice:** 2·TP / (2·TP + FP + FN)
- **SSIM:** similitud estructural entre máscara predicha y GT

### Texto (varían por condición)
- **BERTScore (F1):** usando BiomedBERT
- **Precisión de hallazgo:** ¿el texto menciona la patología correcta? (bool)
- **Likert 1-5:** evaluación manual por oftalmólogo (cuando disponible)

### Estadísticas
- Test de Wilcoxon signed-rank pareado para comparar condiciones
- Significancia: p < 0.05
- Reportar effect size: r = Z / sqrt(N)

## Qué NO Hacer

1. **NO modificar MedGemma internamente.** Se usa como caja negra. No agregar LoRA a MedGemma hasta que se indique explícitamente.
2. **NO inventar columnas o campos de datos.** Si un campo no existe en la especificación, preguntar.
3. **NO mezclar lógica entre módulos.** Cada módulo tiene su interfaz definida.
4. **NO generar splits nuevos.** Siempre cargar desde `splits.json`.
5. **NO hardcodear rutas, hiperparámetros o semillas.** Todo va en `config.yaml`.
6. **NO subir datos a repositorios públicos.** Los datos son privados.
7. **NO usar `print()`.** Usar `logging`.
8. **NO asumir** que Pipeline A "selecciona" entre candidatas. LoRA devuelve la máscara directamente.

## Entorno de Ejecución

- **GPU:** NVIDIA L4 o T4 (≥16GB VRAM)
- **Plataforma:** Google Colab Pro o GCP
- **Python:** 3.10+
- **PyTorch:** 2.x
- **Dependencias clave:** transformers, segment-anything, torchvision, scikit-learn, scipy, bert-score

## Archivos de Referencia

- `config.yaml` → Configuración centralizada
- `splits.json` → Particiones del dataset (inmutables)
- `experimental_design.md` → Diseño experimental con interfaces de cada módulo
- `pipeline_mejorado_informe.md` → Diagrama del pipeline y condiciones de ablation
