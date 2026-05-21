# M3: SAMSegmenter — Generación de Máscaras Candidatas

## 1. Propósito

El SAMSegmenter tiene una única responsabilidad: recibir una imagen y generar N máscaras candidatas de segmentación usando SAM en modo automático (AMG).

Este módulo es usado por:
- **Pipeline B (WSSS):** toma las candidatas y selecciona la que mejor se alinea con el Grad-CAM
- **Pipeline C (FSL/FD):** toma las candidatas y filtra con los thresholds KDE

Pipeline A (LoRA) **NO usa** este módulo porque tiene su propia versión de SAM fine-tuned que devuelve la máscara directamente.

---

## 2. ¿Qué es SAM y cómo funciona?

### 2.1 SAM en una frase

SAM (Segment Anything Model) es un modelo de segmentación "generalista" entrenado con 1,000 millones de máscaras. Dado cualquier tipo de imagen, SAM puede segmentar automáticamente todos los "objetos" visibles sin saber qué son.

### 2.2 Componentes de SAM

SAM tiene 3 partes internas:

1. **Image Encoder (ViT):** Recibe la imagen completa y produce una representación comprimida (embeddings). Este paso es costoso (~100ms) pero se hace UNA sola vez por imagen
2. **Prompt Encoder:** Codifica los prompts que le dicen a SAM dónde buscar. Los prompts pueden ser puntos, cajas, o texto. En modo AMG no se dan prompts específicos — se genera una grilla automática de puntos
3. **Mask Decoder:** Toma los embeddings de la imagen + los prompts y produce las máscaras. Este paso es rápido (~10ms por prompt)

### 2.3 Modo AMG (Automatic Mask Generation)

En modo AMG, SAM no recibe prompts específicos del usuario. En su lugar:

1. Genera una **grilla regular de puntos** sobre toda la imagen (ej: 32×32 = 1024 puntos)
2. Para cada punto, el mask decoder genera 3 máscaras posibles (a diferentes escalas: objeto pequeño, mediano, grande)
3. Se filtran por calidad: solo se conservan las que tienen un IoU predicho alto y una estabilidad alta
4. Se aplica NMS (Non-Maximum Suppression) para eliminar máscaras duplicadas o casi idénticas
5. Se retornan las máscaras finales ordenadas por calidad

### 2.4 ¿Por qué SAM no es suficiente por sí solo?

SAM genera máscaras de **cualquier cosa visible**: vasos sanguíneos, disco óptico, reflejos de luz, artefactos del equipo, etc. No tiene concepto de "patología" vs "anatomía normal". Por eso necesitamos los selectores de Pipeline B o C para elegir cuál de las candidatas corresponde a la patología que nos interesa.

Además, el paper BIP demuestra que los **scores de confianza de SAM no son confiables** en imágenes médicas. Una máscara con score 0.95 puede ser un artefacto y una con 0.70 puede ser la patología real.

---

## 3. Variantes de SAM disponibles

| Variante | Tamaño | Velocidad | Notas |
|----------|--------|-----------|-------|
| **SAM ViT-Tiny** | 38.9M params | Rápido | Buena relación calidad/velocidad |
| **SAM ViT-Base** | 93.7M params | Medio | Mayor precisión |
| **SAM ViT-Large** | 312M params | Lento | Máxima precisión pero alto costo VRAM |
| **SAM 2 Tiny** | 38.9M params | Rápido | Versión mejorada de SAM |
| **MedSAM** | 93.7M params | Medio | Fine-tuned en 1.5M imágenes médicas |

**Recomendación:** SAM 2 Tiny o MedSAM según VRAM disponible. SAM 2 Tiny cabe en Colab T4 junto con MedGemma.

---

## 4. Parámetros del Generador

| Parámetro | Default | Qué controla |
|-----------|---------|-------------|
| `points_per_side` | 32 | Densidad de la grilla: 32×32 = 1024 puntos. Más puntos = más candidatas pero más lento |
| `pred_iou_thresh` | 0.7 | Umbral mínimo del IoU predicho por SAM. Sube = menos candidatas pero mejor calidad estimada |
| `stability_score_thresh` | 0.8 | Qué tan estable es la máscara ante perturbaciones del prompt. Sube = máscaras más robustas |
| `min_mask_region_area` | 100 | Área mínima en píxeles para aceptar una máscara. Filtra ruido microscópico |

---

## 5. Generación y Filtrado de Candidatas

### 5.1 Qué hace el método `generate_candidates`

Recibe una imagen (array NumPy uint8 RGB de 1024×1024) y:

1. La pasa por SAM en modo AMG
2. SAM devuelve todas las máscaras que detectó (pueden ser 50-200)
3. Se aplican filtros adicionales:
   - **Área mínima**: descartar máscaras que cubren menos del 0.5% de la imagen (probablemente ruido)
   - **Área máxima**: descartar máscaras que cubren más del 80% de la imagen (probablemente el fondo completo)
4. Se ordenan por `predicted_iou` descendente (la más confiable según SAM primero)

### 5.2 Qué retorna para cada candidata

Cada candidata es un diccionario con:
- `mask`: array booleano de (H, W) — la máscara binaria
- `predicted_iou`: float entre 0 y 1 — confianza de SAM (no confiable en médicas)
- `stability_score`: float entre 0 y 1 — estabilidad de la máscara
- `area`: entero — cantidad de píxeles que cubre
- `area_ratio`: float — proporción de la imagen que cubre (area / total)
- `bbox`: tupla (x, y, w, h) — bounding box de la máscara

---

## 6. Consideraciones para Imágenes Médicas

### 6.1 SAM no entiende anatomía

SAM segmenta por **bordes visuales**, no por significado clínico. En fondo de ojo:
- ✅ Segmenta bien: disco óptico (borde claro), vasos (alto contraste), lesiones grandes
- ❌ Segmenta mal: opacidades sutiles del cristalino (bajo contraste), drusas pequeñas

### 6.2 MedSAM como alternativa

MedSAM fue fine-tuned en ~1.5M de imágenes médicas. Puede ser mejor para regiones con bajo contraste. La interfaz del módulo es la misma — solo cambia el checkpoint que se carga.

### 6.3 Resolución

Las máscaras se generan a 1024×1024. Para evaluar contra el GT (448×448) o superponer en MedGemma, hay que redimensionarlas usando interpolación NEAREST (para preservar bordes binarios, igual que las máscaras GT del DataModule).

---

## 7. Gestión de VRAM

SAM comparte GPU con MedGemma y la CNN:
- MedGemma 4B: ~8 GB
- SAM Tiny: ~0.5 GB
- CNN: ~0.1 GB
- **Total: ~8.6 GB** → cabe en T4 (16GB) con margen

Cuando SAM ya no se necesita, se debe mover a CPU y liberar caché de CUDA.

---

## 8. Verificaciones

1. `generate_candidates` retorna una lista no vacía
2. Cada candidata tiene los campos requeridos (mask, predicted_iou, etc.)
3. Las máscaras son booleanas de shape (1024, 1024)
4. Los IoU predichos están entre 0 y 1
5. Las áreas cumplen los filtros (entre 0.5% y 80%)
6. Las candidatas están ordenadas por IoU descendente
7. Con la misma semilla, se generan las mismas máscaras (reproducibilidad)
