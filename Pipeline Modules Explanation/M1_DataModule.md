# M1: DataModule — Carga, Preprocesamiento y Gestión de Datos

## 1. Propósito

El DataModule es la **primera pieza** del pipeline. Es el único módulo que toca disco. Todos los demás módulos reciben sus datos a través de él. Su trabajo es:

- Cargar las imágenes y sus 4 anotaciones
- Mantener las particiones (train/val/test) fijas para reproducibilidad
- Preparar cada imagen en los formatos que necesitan los demás módulos
- Proveer lotes (batches) listos para consumir

---

## 2. Datos de Entrada

### 2.1 Estructura esperada en disco

El dataset se organiza en una carpeta con:
- Una subcarpeta `images/` con todas las imágenes (PNG o JPG, RGB)
- Una subcarpeta `masks/` con las máscaras ground truth (escala de grises, donde 0=fondo y 255=patología)
- Un archivo `annotations.csv` que conecta cada imagen con sus anotaciones

### 2.2 Campos del archivo de anotaciones

Cada fila de `annotations.csv` corresponde a una imagen y tiene estos campos:

| Campo | Tipo | Ejemplo | Descripción |
|-------|------|---------|-------------|
| `image_id` | Texto | "img_0001" | Identificador único |
| `image_path` | Texto | "images/img_0001.png" | Ruta relativa a la imagen |
| `mask_path` | Texto | "masks/img_0001_mask.png" | Ruta relativa a la máscara GT |
| `disease_category` | Texto | "cataract" | Categoría: cataract, glaucoma, AMD, DR, normal |
| `disease_grading` | Texto | "C3N2P1" | Escala clínica. Para cataratas: LOCSIII |
| `expert_description` | Texto | "Se observa opacidad..." | Descripción del oftalmólogo |

---

## 3. Generación de Splits (Particiones)

### 3.1 ¿Por qué un archivo `splits.json`?

Para que **todos los módulos y todos los investigadores** usen exactamente las mismas imágenes en train/val/test. Si cada uno genera sus propios splits, los resultados no son comparables. El archivo `splits.json` se genera UNA sola vez y nunca se vuelve a generar.

### 3.2 Cómo se generan las particiones

1. Tomar todas las imágenes del dataset
2. Dividir en 70% entrenamiento, 15% validación, 15% test
3. La división debe ser **estratificada por `disease_category`**: esto significa que si el dataset tiene 30% de cataratas, cada split (train, val, test) también tendrá ~30% de cataratas. Esto evita que por azar una clase quede sub-representada en algún split
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
| `image_id` | str | Identificador único |
| `image_cnn` | Tensor (3, 448, 448) | Imagen normalizada para la CNN |
| `image_sam` | ndarray (1024, 1024, 3) | Imagen uint8 para SAM |
| `image_raw` | ndarray (H, W, 3) | Imagen original sin procesar |
| `mask_gt` | Tensor (1, 448, 448) | Máscara GT a resolución CNN |
| `mask_gt_sam` | Tensor (1, 1024, 1024) | Máscara GT a resolución SAM |
| `disease_category` | str | Categoría de enfermedad |
| `disease_grading` | str | Escala clínica (ej: LOCSIII) |
| `expert_description` | str | Texto del oftalmólogo |

### 5.2 Collate function personalizada

PyTorch normalmente apila todos los elementos de un batch en un solo tensor. Pero nuestro dataset contiene mezcla de tensores, arrays NumPy y strings, que no se pueden apilar igual. Se necesita una función de collate personalizada que:
- **Apila como tensor** los campos que son tensores (`image_cnn`, `mask_gt`)
- **Agrupa en listas** los campos que son arrays NumPy o strings (`image_sam`, `image_raw`, `disease_category`, etc.)

---

## 6. Clase Principal: DataModule

La clase DataModule encapsula todo lo anterior. Al inicializarse:

1. Recibe la sección `data` del `config.yaml`
2. Carga el archivo `annotations.csv`
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
