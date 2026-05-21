# M4: PipelineA_LoRA — SAM Fine-tuned (Segmentador Directo)

## 1. Propósito

Pipeline A es el **segmentador más preciso**. En lugar de generar candidatas y seleccionar, entrena SAM con LoRA para que devuelva la máscara correcta en un solo paso.

La diferencia fundamental con Pipeline B y C: aquí **NO hay candidatas ni selección**. SAM recibe la imagen y produce la máscara directamente.

---

## 2. ¿Qué es LoRA?

### 2.1 El problema

SAM tiene millones de parámetros. Fine-tunear todos requiere mucha VRAM y datos, y se arriesga el "catastrophic forgetting" (olvidar lo aprendido en el preentrenamiento).

### 2.2 La solución

LoRA (Low-Rank Adaptation) **no modifica** los pesos originales. Agrega matrices pequeñas (adapters) en paralelo a las capas de atención del Transformer:

- El peso original W queda **congelado**
- Se agregan dos matrices pequeñas A y B (de rango bajo, ej: rank=8)
- La salida se calcula como: `output = W·x + α·(B·A·x)`
- Al inicio, B está inicializada en ceros, así que el adapter no hace nada y el modelo funciona exactamente igual al original
- Durante el entrenamiento, A y B aprenden las adaptaciones necesarias

### 2.3 Ventajas

- **Muy pocos parámetros entrenables**: con rank=8, solo el 1-5% de los parámetros se entrenan
- **No destruye el conocimiento previo**: pesos originales congelados
- **Funciona con pocos datos**: el paper BIP muestra IoU=0.97 desde N=42 imágenes
- **Adapters pequeños**: ~10MB por enfermedad (vs ~300MB del modelo completo)

---

## 3. ¿Qué partes de SAM se adaptan?

SAM tiene 3 componentes. LoRA se aplica **solo al Image Encoder**:

- **Image Encoder (ViT)**: Se insertan adapters LoRA en las capas de atención Q (query) y V (value) de cada bloque del Transformer. Se eligen Q y V porque la literatura muestra que es el mejor balance rendimiento/eficiencia
- **Prompt Encoder**: Se mantiene congelado, no se modifica
- **Mask Decoder**: Se desbloquea completamente para entrenar (es pequeño, ~4M params, y necesita adaptarse a las máscaras médicas)

---

## 4. Inserción de LoRA

### 4.1 La capa LoRA

Cada capa LoRA envuelve una capa Linear existente del modelo. Funciona así:

1. Se guarda la capa original (congelada, no se modifica)
2. Se crean dos matrices nuevas: A (de tamaño input×rank) y B (de tamaño rank×output)
3. A se inicializa con distribución normal (valores aleatorios pequeños)
4. B se inicializa en **ceros** — esto garantiza que al inicio el adapter no afecta la salida
5. Durante el forward pass: se calcula la salida original + la salida del adapter multiplicada por un factor α

### 4.2 Proceso de inserción

Se recorren todos los bloques de atención del Image Encoder de SAM. En cada bloque:
- Se reemplaza `q_proj` (query projection) con una versión LoRA
- Se reemplaza `v_proj` (value projection) con una versión LoRA
- `k_proj` y `out_proj` se dejan sin modificar

Al final, se imprime cuántos parámetros son entrenables vs totales. Debería ser alrededor del 1-5%.

---

## 5. Preparación de datos para entrenamiento

SAM necesita **prompts** (puntos o cajas) para generar máscaras. Durante el entrenamiento, se usan **puntos muestreados de la máscara GT** como prompt:

1. Tomar la máscara GT de la imagen actual
2. Encontrar todos los píxeles que son "1" (patología)
3. Muestrear aleatoriamente 1 punto de esos píxeles
4. Usar ese punto como prompt de tipo "foreground" (punto positivo)

Esto le dice a SAM: "hay algo interesante en esta coordenada, segmentalo". Con LoRA, SAM aprende que en imágenes oftalmológicas, "algo interesante" significa "patología", y produce la máscara correcta.

Si la máscara GT está vacía (imagen normal), se usa el punto central de la imagen.

---

## 6. Entrenamiento

### 6.1 Configuración

- **Optimizer**: AdamW con lr=1e-4, weight_decay=1e-4
- **Épocas**: 50
- **Sin batches**: se procesa una imagen a la vez (SAM no soporta batches nativamente)

### 6.2 Loss function: Dice + BCE

Se usa una combinación de dos losses estándar en segmentación médica:

- **Dice Loss**: Mide la superposición global entre la máscara predicha y la GT. Penaliza especialmente los falsos negativos (no detectar patología). Se calcula como 1 menos el coeficiente Dice
- **BCE Loss** (Binary Cross-Entropy): Mide el error pixel a pixel. Da gradientes más suaves y estables

La combinación es: `total_loss = 0.5 × Dice_loss + 0.5 × BCE_loss`. Esto balancea la visión global (Dice) con la precisión pixel a pixel (BCE).

### 6.3 Cada época hace:

1. Mezclar aleatoriamente el orden de las imágenes de entrenamiento
2. Para cada imagen:
   - Muestrear un punto de la máscara GT como prompt
   - Codificar la imagen con el Image Encoder (con LoRA)
   - Codificar el punto con el Prompt Encoder
   - Decodificar la máscara con el Mask Decoder
   - Calcular la loss combinada contra la máscara GT
   - Backpropagation y actualizar pesos (solo adapters LoRA + mask decoder)
3. Al final de la época, evaluar en validación (calcular IoU promedio)
4. Si el IoU de validación es el mejor hasta ahora, guardar los pesos

---

## 7. Inferencia (Predicción)

A diferencia de Pipeline B y C, aquí no se generan candidatas. El proceso es:

1. Codificar la imagen con el Image Encoder (con los adapters LoRA entrenados)
2. Usar un **punto central** de la imagen como prompt (en inferencia no tenemos la GT)
3. El Mask Decoder produce una máscara
4. Aplicar sigmoid (convertir logits a probabilidades) y binarizar con umbral 0.5
5. Retornar la máscara binaria + la confianza (IoU predicho por SAM)

El resultado es un diccionario con:
- `mask`: array booleano de (H, W) — la máscara de segmentación
- `confidence`: float — confianza del modelo

---

## 8. Guardar y cargar (solo adapters)

Al guardar Pipeline A, **NO se guarda todo SAM**. Solo se guardan:
- Los pesos de los adapters LoRA (matrices A y B de cada capa)
- Los pesos del Mask Decoder (que también se entrenó)

Esto resulta en archivos de ~10MB por enfermedad (vs ~300MB si se guardara todo SAM).

Al cargar, se reconstruye SAM base desde su checkpoint original y se cargan los adapters encima. Los pesos base no se modifican.

---

## 9. Resultados esperados (del paper BIP)

| N (imágenes de entrenamiento) | IoU medio | Desviación |
|-------------------------------|-----------|------------|
| 42 | 0.969 | ±0.040 |
| 84 | 0.970 | ±0.022 |
| 126 | 0.972 | ±0.019 |
| 210 | 0.972 | ±0.019 |

Con apenas **42 imágenes** se alcanza casi el techo. Más datos mejoran la estabilidad (menor σ) pero no la precisión.

---

## 10. Verificaciones

1. LoRA fue insertado correctamente: los parámetros entrenables deben ser menos del 10% del total
2. `predict` retorna un diccionario con `mask` (booleano, shape correcta) y `confidence` (float)
3. Save/Load produce resultados idénticos
4. Con la misma semilla y datos, el entrenamiento es reproducible
