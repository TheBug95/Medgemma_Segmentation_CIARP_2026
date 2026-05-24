# M1: DataModule — Carga, Preprocesamiento y Gestión de Datos

## 1. Propósito

El DataModule es la **primera pieza** del pipeline. Es el único módulo que toca disco. Todos los demás módulos reciben sus datos a través de él. Su trabajo es:

- Cargar las imágenes y sus anotaciones clínicas
- Mantener las particiones (train/val/test) fijas para reproducibilidad
- Preparar cada imagen en los formatos que necesitan los demás módulos
- Proveer lotes (batches) listos para consumir

---

## 2. Datos de Entrada

### 2.1 Estructura esperada en disco

El dataset se organiza en una carpeta con:
- Una subcarpeta `images/` con todas las imágenes (PNG o JPG, RGB)
- Una subcarpeta `masks/` con las máscaras ground truth (escala de grises, donde 0=fondo y 255=patología).
  El nombre del archivo de máscara se deriva del `image_filename` (ej: `1209_right.jpg` → `masks/1209_right_mask.png`)
- Un archivo `annotations.json` que contiene las anotaciones clínicas de cada imagen

### 2.2 Estructura del archivo `annotations.json`

`annotations.json` es un **array de objetos JSON**, donde cada objeto representa una imagen con sus anotaciones clínicas:

```json
{
  "id": 151,
  "image_filename": "1209_right.jpg",
  "label": "Pathological",
  "transcription": "Clinical photograph of the right eye showing...",
  "doctor_name": "Dr. Gabriel Alejandro Osorio Navarro",
  "session_id": "a7a8e254-de9b-47e8-a6f8-42f158e445fc",
  "locs_data": {
    "conditions": ["glaucoma"],
    "glaucoma": {
      "cup_to_disc_ratio": 3,
      "neuroretinal_rim": 3,
      "disc_hemorrhage": 0,
      "peripapillary_atrophy": 2,
      "rnfl_defect": 2,
      "disc_pallor": 1,
      "vessel_changes": 3
    }
  },
  "source": "gcp",
  "creator_username": "gabriel_alejandro",
  "created_at": "2026-04-21T15:29:41.837597+00:00",
  "updated_at": "2026-04-21T15:31:08.958086+00:00",
  "filename_hash": "e027c0d144a8defe046a901a9adf02c4533bf78cf390e87b6081ef62107b551a",
  "session_hash": "6faeb11472b5507fd4598a95eb34776ba08dd86095841958e2dafa012dda1d57",
  "patient_metadata": null,
  "is_complete": true
}
```

Campos relevantes para el pipeline:

| Campo | Tipo | Ejemplo | Descripción |
|-------|------|---------|-------------|
| `image_filename` | str | `"1209_right.jpg"` | Nombre del archivo de imagen (ruta en `images/`) |
| `label` | str | `"Pathological"` | Clasificación binaria: `"Pathological"` o `"Normal"` |
| `transcription` | str | `"Clinical photograph..."` | Descripción textual del oftalmólogo |
| `locs_data.conditions` | list[str] | `["glaucoma"]` | Lista de condiciones detectadas (para casos normales: `["normal"]`) |
| `locs_data.<enfermedad>` | dict | Grading estructurado | Campos de grading específicos de la enfermedad (ver sección 2.3) |
| `doctor_name` | str | `"Dr. Gabriel..."` | Nombre del oftalmólogo que realizó la anotación |

**Mapeo a los campos usados por el pipeline:**

- `image_filename` → ruta de imagen: `images/{image_filename}`
- `image_filename` → ruta de máscara GT: `masks/{image_filename_sin_ext}_mask.png`
- `locs_data.conditions[0]` → `disease_category`
- `locs_data.<enfermedad>` → `disease_grading`
- `transcription` → `expert_description`

### 2.3 Escala de Grading para Glaucoma

Cuando `locs_data.conditions` contiene `"glaucoma"`, el objeto `locs_data.glaucoma` contiene los siguientes 7 campos de grading, cada uno correspondiente a un dropdown de clasificación clínica:

| Campo (Field ID) | Rango | Significado de cada valor |
|------------------|-------|---------------------------|
| `cup_to_disc_ratio` | 0–4 | 0: ≤0.3 (normal) / 1: 0.4–0.5 (borderline) / 2: 0.6–0.7 (suspicious) / 3: 0.8–0.9 (advanced cupping) / 4: 1.0 (total cupping) |
| `neuroretinal_rim` | 0–4 | 0: Normal (ISNT rule preserved) / 1: ISNT rule violation / 2: Focal notching / 3: Diffuse thinning / 4: Near-total or total rim loss |
| `disc_hemorrhage` | 0–1 | 0: None / 1: Present (splinter hemorrhage at or near disc margin) |
| `peripapillary_atrophy` | 0–2 | 0: None / 1: Beta-zone PPA only / 2: Large or progressive beta-zone PPA |
| `rnfl_defect` | 0–3 | 0: No visible RNFL defect / 1: Wedge-shaped defect (localized) / 2: Diffuse RNFL thinning / 3: Both wedge defect and diffuse thinning |
| `disc_pallor` | 0–2 | 0: Normal color / 1: Mild pallor / 2: Significant pallor |
| `vessel_changes` | 0–3 | 0: Normal vessel pattern / 1: Bayoneting / 2: Nasalization of vessels / 3: Both bayoneting and nasalization |

**Nota:** Esta escala es específica para glaucoma. Si en el futuro se agregan otras enfermedades (ej: catarata), se documentarán sus respectivos campos de grading en `locs_data.<enfermedad>`.

---

## 3. Generación de Splits (Particiones)

### 3.1 ¿Por qué un archivo `splits.json`?

Para que **todos los módulos y todos los investigadores** usen exactamente las mismas imágenes en train/val/test. Si cada uno genera sus propios splits, los resultados no son comparables. El archivo `splits.json` se genera UNA sola vez y nunca se vuelve a generar.

### 3.2 Cómo se generan las particiones

1. Tomar todas las imágenes del dataset
2. Dividir en 70% entrenamiento, 15% validación, 15% test
3. La división debe ser **estratificada por `label`** (Pathological / Normal). Esto asegura que cada split mantenga la misma proporción de casos patológicos y normales que el dataset original. Si el dataset tiene 60% de casos patológicos, cada split (train, val, test) también tendrá ~60%
4. Usar `random_state=42` (la semilla global) para que la división sea siempre la misma
5. Guardar las listas de `image_id` de cada split en `splits.json`

### 3.3 Contenido de `splits.json`

El archivo contiene tres listas de IDs de imágenes:
- `"train"`: lista de IDs para entrenamiento (70%)
- `"val"`: lista de IDs para validación (15%)
- `"test"`: lista de IDs para evaluación final (15%)
- `"seed"`: la semilla usada (42)

---

## 4. Preprocesamiento de Imágenes

Cada modelo downstream requiere un formato diferente. El DataModule prepara **3 versiones** de cada imagen:

### 4.1 Versión para la CNN (clasificador)

- **Tamaño**: Redimensionar a 448×448 píxeles. Se usa este tamaño porque coincide con MedSigLIP, lo que facilita comparaciones espaciales directas
- **Normalización**: Aplicar la normalización estándar de ImageNet (restar la media y dividir por la desviación estándar de cada canal RGB). Esto es necesario porque los backbones CNN (ResNet, EfficientNet, DenseNet) fueron preentrenados con esta normalización
- **Data augmentation** (solo en entrenamiento):
  - Flip horizontal aleatorio (50% de probabilidad): simula que la imagen podría estar espejada
  - Rotación aleatoria de ±15°: las patologías pueden aparecer en cualquier orientación
  - Ajuste aleatorio de brillo y contraste (±20%): simula variaciones de iluminación entre equipos
- **En validación y test**: NO se aplica augmentation. Solo redimensionar + normalizar

### 4.2 Versión para SAM (segmentador)

- **Tamaño**: Redimensionar a 1024×1024 píxeles. Este es el tamaño nativo de SAM y redimensionar a otro tamaño degrada las máscaras
- **Formato**: Array NumPy de tipo uint8 (valores 0-255), RGB
- **NO normalizar**: SAM tiene su propio preprocesamiento interno

### 4.3 Versión Raw (para MedGemma y visualización)

- **Tamaño**: Resolución original de la imagen, sin redimensionar
- **Formato**: Array NumPy uint8 RGB
- **Sin preprocesamiento**: MedGemma tiene su propio `AutoProcessor` que se encarga de todo

### 4.4 Preprocesamiento de la máscara GT

La máscara ground truth se prepara en **2 tamaños**:
- **448×448**: Para evaluar contra Grad-CAM y para el overlay de MedGemma
- **1024×1024**: Para evaluar contra las máscaras generadas por SAM

En ambos casos:
- Se carga como imagen en escala de grises
- Se redimensiona usando interpolación **NEAREST** (no bilinear). Esto es crítico: la máscara es binaria (0 o 1). Si se usara interpolación bilinear, los bordes tendrían valores intermedios (0.3, 0.7) que no son ni fondo ni patología. NEAREST preserva los bordes nítidos
- Se binariza: cualquier valor > 127 se convierte en 1, el resto en 0
- Se convierte a tensor de PyTorch con shape (1, H, W)

---

## 5. Dataset de PyTorch

### 5.1 Qué retorna cada muestra

Cuando se pide una imagen al Dataset, retorna un diccionario con:

| Campo | Shape/Tipo | Descripción |
|-------|-----------|-------------|
| `image_id` | str | Identificador único (`image_filename` sin extensión) |
| `image_cnn` | Tensor (3, 448, 448) | Imagen normalizada para la CNN |
| `image_sam` | ndarray (1024, 1024, 3) | Imagen uint8 para SAM |
| `image_raw` | ndarray (H, W, 3) | Imagen original sin procesar |
| `mask_gt` | Tensor (1, 448, 448) | Máscara GT a resolución CNN |
| `mask_gt_sam` | Tensor (1, 1024, 1024) | Máscara GT a resolución SAM |
| `label` | str | "Pathological" o "Normal" |
| `disease_category` | str | Condición detectada, derivada de `locs_data.conditions[0]` |
| `disease_grading` | dict | Grading estructurado (ej: `{"cup_to_disc_ratio": 3, ...}`). `None` si es normal |
| `expert_description` | str | Descripción del oftalmólogo (campo `transcription`) |
| `doctor_name` | str | Nombre del oftalmólogo que realizó la anotación |

### 5.2 Collate function personalizada

PyTorch normalmente apila todos los elementos de un batch en un solo tensor. Pero nuestro dataset contiene mezcla de tensores, arrays NumPy y strings, que no se pueden apilar igual. Se necesita una función de collate personalizada que:
- **Apila como tensor** los campos que son tensores (`image_cnn`, `mask_gt`)
- **Agrupa en listas** los campos que son arrays NumPy o strings (`image_sam`, `image_raw`, `disease_category`, etc.)

---

## 6. Clase Principal: DataModule

La clase DataModule encapsula todo lo anterior. Al inicializarse:

1. Recibe la sección `data` del `config.yaml`
2. Carga el archivo `annotations.json`
3. Carga `splits.json` (nunca lo regenera)
4. Prepara las transformaciones (train con augmentation, eval sin augmentation)

Expone estos métodos:
- `get_train_loader()` → DataLoader con shuffle=True, augmentation, batch_size=16
- `get_val_loader()` → DataLoader con shuffle=False, sin augmentation
- `get_test_loader()` → DataLoader con shuffle=False, sin augmentation
- `get_sample(idx)` → Una muestra individual para debugging
- `get_class_distribution()` → Distribución de clases por split (para verificar estratificación)

---

## 7. Verificaciones que debe pasar el módulo

Antes de considerar M1 como "terminado", debe pasar estas verificaciones:

1. **Los splits no se solapan**: Ninguna imagen puede estar en train Y val al mismo tiempo
2. **Los splits cubren todo el dataset**: La suma de train + val + test debe ser igual al total de imágenes
3. **Formato correcto de cada sample**: `image_cnn` debe tener shape (3, 448, 448), `image_sam` debe ser (1024, 1024, 3), `mask_gt` debe ser (1, 448, 448) y binaria (solo 0s y 1s)
4. **DataLoader funcional**: Se puede crear un batch y recorrerlo sin errores
5. **Estratificación correcta**: La proporción de cada enfermedad debe ser similar en los tres splits (verificar con `get_class_distribution()`)

---

## 8. Consideraciones Especiales

### 8.1 Datos en Google Drive (Colab)

En Colab, los datos se montan desde Google Drive. La ruta sería algo como `/content/drive/MyDrive/dataset/`. Los workers del DataLoader (`num_workers > 0`) pueden causar problemas con Drive montado — si hay errores, usar `num_workers=0`.

### 8.2 Memoria

NO cargar todo el dataset en memoria. El Dataset de PyTorch carga cada imagen bajo demanda (cada vez que se pide una muestra, lee de disco). Esto es intencional y necesario para datasets grandes.

### 8.3 Máscara GT a múltiples resoluciones

La máscara GT se necesita en 2 tamaños porque SAM opera a 1024×1024 y la CNN a 448×448. Ambas versiones usan interpolación NEAREST para preservar los bordes binarios.
