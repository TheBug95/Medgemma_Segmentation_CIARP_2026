# M3: SAMSegmenter — Segmentación con SAM 2

## 1. Propósito

El módulo SAMSegmenter proporciona dos variantes del segmentador basado en **SAM 2.1 Tiny**:

1. **SAMSegmenter** (sin entrenar): Usa SAM 2 en modo AMG para generar N máscaras candidatas automáticamente
2. **LoRASAMSegmenter** (fine-tuned): Hereda de SAMSegmenter, añade LoRA al Image Encoder para producir máscaras directas más precisas

Ambas variantes alimentan los tres pipelines:
- **Pipeline B (WSSS):** genera candidatas → selecciona por Grad-CAM
- **Pipeline C (FSL/FD):** genera candidatas → filtra por KDE
- **Predicción directa:** devuelve la máscara sin selección adicional

---

## 2. SAM 2 — Arquitectura

### 2.1 ¿Qué cambió de SAM 1 a SAM 2?

SAM 2 reemplaza el Image Encoder basado en ViT por una arquitectura **Hiera** (Hierarchical Vision Transformer). Esto le da:

- Mejor eficiencia computacional (menos FLOPs para la misma calidad)
- Mejor rendimiento en benchmarks de segmentación
- Soporte nativo para segmentación de vídeo (aunque aquí solo usamos imágenes)

### 2.2 Componentes internos

1. **Image Encoder (Hiera):** Recibe la imagen y produce embeddings jerárquicos en múltiples resoluciones. Este es el componente más costoso (~100ms) pero se ejecuta UNA sola vez por imagen
2. **Prompt Encoder:** Codifica los prompts (puntos, cajas). En modo AMG se usa una grilla automática de puntos. En LoRA se usa un punto de la máscara GT
3. **Mask Decoder:** Toma los embeddings + prompts y produce las máscaras. Rápido (~10ms por prompt)

### 2.3 Tamaños disponibles

| Modelo | Params | FPS (A100) | Arquitectura | HuggingFace ID |
|--------|--------|------------|-------------|----------------|
| **SAM 2.1 Tiny** | **38.9M** | 91.2 | Hiera-T | `facebook/sam2.1-hiera-tiny` |
| SAM 2.1 Small | 46M | 84.8 | Hiera-S | `facebook/sam2.1-hiera-small` |
| SAM 2.1 Base+ | 80.8M | 64.1 | Hiera-B+ | `facebook/sam2.1-hiera-base-plus` |
| SAM 2.1 Large | 224.4M | 39.5 | Hiera-L | `facebook/sam2.1-hiera-large` |

**Elección: SAM 2.1 Tiny** — 38.9M parámetros, ~0.5GB VRAM. Cabe en T4/L4 junto con MedGemma 4B (~8GB) y la CNN (~0.1GB), con margen de ~7GB para operaciones intermedias.

---

## 3. Variante 1: SAMSegmenter (Sin Entrenar)

### 3.1 Qué hace

Carga SAM 2.1 Tiny preentrenado y lo usa en **modo AMG (Automatic Mask Generation)** para generar múltiples máscaras candidatas sin intervención del usuario.

### 3.2 Modo AMG paso a paso

1. SAM genera una **grilla regular de puntos** sobre la imagen (ej: 32×32 = 1024 puntos)
2. Para cada punto, el Mask Decoder genera 3 máscaras posibles (objeto pequeño, mediano, grande)
3. Se filtran por calidad: solo se conservan las de IoU predicho alto y estabilidad alta
4. Se aplica NMS (Non-Maximum Suppression) para eliminar duplicados
5. Se retornan las máscaras finales ordenadas por `predicted_iou` descendente

### 3.3 Interfaz

```python
class SAMSegmenter:
    def __init__(self, config: dict):
        """
        config = {
            "model_type": "hiera_t",
            "checkpoint": "facebook/sam2.1-hiera-tiny",
            "config_file": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "device": "cuda",
            "points_per_side": 32,
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.8,
            "seed": 42
        }
        """

    def generate_candidates(self, image_raw: ndarray) -> list[dict]:
        """
        Input: image_raw (H, W, 3) uint8 — imagen RGB sin normalizar
        Returns: lista de candidatas ordenadas por predicted_iou descendente:
            {
                "mask": ndarray (H, W) bool,
                "predicted_iou": float,
                "stability_score": float,
                "area": int,
                "area_ratio": float,
                "bbox": (x, y, w, h)
            }
        """
```

### 3.4 Filtrado de candidatas

Después de que AMG genera las máscaras crudas, se aplican filtros adicionales:
- **Área mínima**: descartar máscaras que cubren <0.5% de la imagen (ruido)
- **Área máxima**: descartar máscaras que cubren >80% de la imagen (fondo completo)
- **Ordenamiento**: por `predicted_iou` descendente (la más confiable primero)

### 3.5 Uso en los pipelines

- **WSSS (Pipeline B):** recibe las candidatas → selecciona la que mejor coincide con el Grad-CAM
- **FSL/FD (Pipeline C):** recibe las candidatas → filtra por KDE con MedSigLIP embeddings
- **Predicción directa:** se toma la candidata con mayor `predicted_iou` como resultado final

---

## 4. Variante 2: LoRASAMSegmenter (Fine-tuned con LoRA)

### 4.1 Qué hace

Hereda de SAMSegmenter (mantiene `generate_candidates()`). Añade LoRA al Image Encoder para adaptar SAM 2 a imágenes oftalmológicas, y proporciona `predict()` para generar una máscara directa sin necesidad de candidatas.

### 4.2 ¿Qué es LoRA?

LoRA (Low-Rank Adaptation) no modifica los pesos originales del modelo. Agrega matrices pequeñas (adapters) en paralelo a las capas de atención:

- El peso original W queda **congelado**
- Se crean dos matrices A (input×rank) y B (rank×output)
- A se inicializa con distribución normal, B en **ceros** (el adapter empieza sin efecto)
- Forward pass: `output = W·x + α·(B·A·x)`
- Solo se entrenan A, B y el Mask Decoder

**Ventajas:**
- ~1-5% de parámetros entrenables
- No destruye conocimiento previo (pesos base congelados)
- Funciona con pocos datos (paper BIP: IoU=0.97 desde 42 imágenes)
- Adapters pequeños: ~10MB por enfermedad vs ~300MB del modelo completo

### 4.3 Qué partes de SAM 2 se adaptan

| Componente | Estado | Detalle |
|------------|--------|---------|
| Image Encoder (Hiera) | **LoRA en Q y V** | Se insertan adapters en `q_proj` y `v_proj` de cada bloque de atención |
| Prompt Encoder | **Congelado** | No se modifica |
| Mask Decoder | **Desbloqueado** | Se entrena completo (~4M params), necesita adaptarse a máscaras médicas |

### 4.4 Interfaz

```python
class LoRASAMSegmenter(SAMSegmenter):
    def __init__(self, config: dict):
        """
        Hereda config de SAMSegmenter y añade:
        config = {
            ... (mismos campos que SAMSegmenter) ...
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_epochs": 50,
            "lora_lr": 0.0001
        }
        """

    def train(self, train_images: list, train_masks: list,
              val_images: list, val_masks: list) -> dict:
        """
        Entrena SAM 2 con LoRA usando las imágenes y máscaras GT.
        Returns: {
            "history": list[dict],      # IoU por época
            "best_val_iou": float,
            "best_epoch": int
        }
        """

    def predict(self, image_raw: ndarray) -> dict:
        """
        Genera una máscara directa (sin candidatas).
        Input: image_raw (H, W, 3) uint8
        Returns: {
            "mask": ndarray (H, W) bool,
            "confidence": float
        }
        """

    def save(self, path: str) -> None:
        """Guarda solo adapters LoRA + Mask Decoder (~10MB)."""

    def load(self, path: str) -> None:
        """Carga adapters sobre el modelo base reconstruido."""
```

### 4.5 Preparación de datos para entrenamiento

SAM necesita prompts para generar máscaras. Durante el entrenamiento se usan **puntos muestreados de la máscara GT**:

1. Tomar la máscara GT de la imagen
2. Encontrar todos los píxeles que son `1` (patología)
3. Muestrear aleatoriamente 1 punto de esos píxeles
4. Usar ese punto como prompt de tipo "foreground"

Si la máscara GT está vacía (imagen normal), se usa el punto central de la imagen.

### 4.6 Entrenamiento

**Configuración:**
- Loss: `0.5 × Dice_loss + 0.5 × BCE_loss`
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Épocas: 50
- Sin batches: una imagen a la vez (SAM 2 no soporta batches nativamente en modo entrenamiento)

**Cada época:**
1. Mezclar aleatoriamente el orden de las imágenes
2. Para cada imagen:
   - Muestrear un punto de la máscara GT como prompt
   - Codificar imagen con Image Encoder (con LoRA)
   - Codificar punto con Prompt Encoder
   - Decodificar máscara con Mask Decoder
   - Calcular loss contra máscara GT
   - Backpropagation y actualizar pesos (solo adapters + mask decoder)
3. Evaluar en validación (IoU promedio)
4. Si el IoU de validación es el mejor, guardar pesos

### 4.7 Inferencia (Predicción directa)

1. Codificar imagen con Image Encoder (con adapters LoRA entrenados)
2. Usar **punto central** de la imagen como prompt (no tenemos GT en inferencia)
3. Mask Decoder produce la máscara
4. Aplicar sigmoid + binarizar con umbral 0.5
5. Retornar máscara binaria + confianza (IoU predicho)

### 4.8 Guardar y cargar

**Se guarda:** adapters LoRA (matrices A y B) + pesos del Mask Decoder (~10MB)
**Se carga:** se reconstruye SAM 2 base desde checkpoint original, se cargan los adapters encima

### 4.9 Uso en los pipelines

- **WSSS (Pipeline B):** `generate_candidates()` genera candidatas (con LoRA activo) → selección por Grad-CAM
- **FSL/FD (Pipeline C):** `generate_candidates()` genera candidatas (con LoRA activo) → filtrado por KDE
- **Predicción directa:** `predict()` devuelve la máscara sin selección adicional

### 4.10 Resultados esperados (del paper BIP con SAM 1)

| N (imágenes) | IoU medio | Desviación |
|--------------|-----------|------------|
| 42 | 0.969 | ±0.040 |
| 84 | 0.970 | ±0.022 |
| 126 | 0.972 | ±0.019 |
| 210 | 0.972 | ±0.019 |

Con apenas 42 imágenes se alcanza casi el techo. Más datos mejoran la estabilidad (menor σ) pero no la precisión.

---

## 5. Parámetros del AMG

| Parámetro | Default | Qué controla |
|-----------|---------|-------------|
| `points_per_side` | 32 | Densidad de la grilla: 32×32 = 1024 puntos. Más puntos = más candidatas pero más lento |
| `pred_iou_thresh` | 0.7 | Umbral mínimo del IoU predicho. Sube = menos candidatas pero mejor calidad estimada |
| `stability_score_thresh` | 0.8 | Estabilidad de la máscara ante perturbaciones del prompt. Sube = máscaras más robustas |
| `min_mask_region_area` | 100 | Área mínima en píxeles para aceptar una máscara. Filtra ruido microscópico |

---

## 6. Parámetros de LoRA

| Parámetro | Default | Qué controla |
|-----------|---------|-------------|
| `lora_rank` | 8 | Rango de las matrices A y B. Más alto = más capacidad pero más parámetros |
| `lora_alpha` | 16 | Factor de escala del adapter. Controla cuánto influye LoRA en la salida |
| `lora_epochs` | 50 | Número de épocas de entrenamiento |
| `lora_lr` | 0.0001 | Learning rate para AdamW |

---

## 7. Consideraciones para Imágenes Médicas

### 7.1 SAM no entiende anatomía

SAM segmenta por **bordes visuales**, no por significado clínico. En fondo de ojo:
- Segmenta bien: disco óptico (borde claro), vasos (alto contraste), lesiones grandes
- Segmenta mal: opacidades sutiles del cristalino (bajo contraste), drusas pequeñas

### 7.2 LoRA compensa parcialmente

Con fine-tuning, SAM aprende que en imágenes oftalmológicas "algo interesante" significa "patología". Pero sigue sin entender semántica clínica — simplemente aprende patrones visuales asociados a las máscaras GT.

### 7.3 Resolución

Las máscaras se generan a 1024×1024. Para evaluar contra GT (448×448) o superponer en MedGemma, redimensionar con interpolación NEAREST (preserva bordes binarios).

---

## 8. Gestión de VRAM

| Componente | VRAM aproximada |
|------------|-----------------|
| SAM 2 Tiny | ~0.5 GB |
| MedGemma 4B | ~8 GB |
| CNN | ~0.1 GB |
| **Total** | **~8.6 GB** |

Cabe en T4 (16GB) con margen de ~7GB. Cuando SAM ya no se necesita, mover a CPU y liberar caché de CUDA.

---

## 9. Verificaciones

1. `generate_candidates` retorna una lista no vacía con los campos correctos
2. Las máscaras son booleanas de shape (H, W) con H, W correctos
3. Los IoU predichos están entre 0 y 1
4. Las áreas cumplen los filtros (entre 0.5% y 80%)
5. Las candidatas están ordenadas por IoU descendente
6. LoRA insertado correctamente: parámetros entrenables < 10% del total
7. `predict` retorna `mask` (booleana, shape correcta) y `confidence` (float)
8. Save/Load produce resultados idénticos
9. Con la misma semilla, se generan las mismas máscaras (reproducibilidad)
