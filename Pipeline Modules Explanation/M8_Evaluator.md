# M8: Evaluator — Métricas y Análisis Estadístico

## 1. Propósito

El Evaluator es el módulo de **medición**. Calcula todas las métricas necesarias para comparar los 3 pipelines y las 6 condiciones, y aplica los tests estadísticos para determinar si las diferencias son significativas.

No entrena nada ni genera nada. Solo mide y compara.

---

## 2. Dos Tipos de Métricas

### 2.1 Métricas de segmentación (comparan máscara predicha vs máscara GT)

Estas métricas son **constantes** dentro de cada pipeline. La máscara no cambia entre condiciones A-D (lo que cambia es el texto de MedGemma). Se calculan una vez por pipeline.

**IoU (Intersection over Union)**

Mide qué tanto se solapan la máscara predicha y la GT.
- Se calcula como: píxeles correctamente detectados / (correctos + falsos positivos + falsos negativos)
- Rango: 0 (sin solapamiento) a 1 (solapamiento perfecto)
- Es la métrica estándar en segmentación semántica y la que usa el paper BIP

**Dice Coefficient**

Equivalente al F1-Score pero para segmentación. Muy similar a IoU pero ligeramente más generoso.
- Se calcula como: 2 × píxeles correctos / (2 × correctos + falsos positivos + falsos negativos)
- Rango: 0 a 1
- Es la métrica estándar en segmentación médica (papers de segmentación de tumores, órganos, etc.)

**SSIM (Structural Similarity Index)**

Mide similitud estructural entre la máscara predicha y la GT, considerando luminancia, contraste y estructura.
- A diferencia de IoU/Dice que solo miran superposición, SSIM captura si la "forma" de la máscara es correcta incluso si está ligeramente desplazada
- Rango: -1 a 1 (en la práctica para máscaras binarias: 0 a 1)
- Captura errores que IoU/Dice ignoran: por ejemplo, una máscara con el borde correcto pero interior incorrecto

### 2.2 Métricas de texto (comparan texto generado vs descripción del experto)

Estas métricas **varían** entre condiciones A-D dentro del mismo pipeline. Son las que miden el efecto del condicionamiento.

**BERTScore (F1)**

Mide similitud semántica entre el texto generado por MedGemma y la descripción del oftalmólogo.
- Usa un modelo de lenguaje (BiomedBERT) para obtener embeddings de cada palabra/token
- Calcula la similitud coseno entre los embeddings de ambos textos
- Retorna Precisión, Recall y F1. Usamos F1 como métrica principal
- Rango: 0 a 1 (en la práctica, textos médicos suelen estar entre 0.3-0.8)
- Ventaja sobre métricas como BLEU: captura sinónimos y paráfrasis (ej: "opacidad del cristalino" y "lens opacity" tienen alto BERTScore aunque las palabras son diferentes)

Se usa BiomedBERT (en lugar de BERT genérico) porque está preentrenado en textos biomédicos y entiende mejor la terminología clínica.

**Precisión de hallazgo**

Métrica binaria simple: ¿el texto generado por MedGemma menciona la patología correcta?
- Se busca si alguna de las keywords de la enfermedad aparece en el texto (ej: para glaucoma buscar "glaucoma", "cupping", "optic nerve", "disc", "rim", etc.)
- Retorna True o False
- Útil como métrica de "sanidad": si MedGemma no menciona la enfermedad correcta, la descripción es fundamentalmente incorrecta

**Likert 1-5 (evaluación manual)**

Un oftalmólogo lee el texto generado y le asigna un puntaje:
- 1: Completamente incorrecto o irrelevante
- 2: Menciona algo pero con errores significativos
- 3: Parcialmente correcto pero incompleto
- 4: Correcto con detalles menores faltantes
- 5: Descripción clínica completa y precisa

Esta métrica es la más valiosa pero requiere evaluación manual. Se aplica cuando sea posible, no en todos los runs.

---

## 3. Tests Estadísticos

### 3.1 ¿Por qué tests estadísticos?

No basta con decir "Condición D2 tiene BERTScore 0.72 y Condición A tiene 0.65". Necesitamos saber si esa diferencia es **estadísticamente significativa** o si podría ser producto del azar.

### 3.2 Test de Wilcoxon Signed-Rank (pareado)

Es el test que usamos para todas las comparaciones. Se elige porque:
- Es **pareado**: compara los scores de las mismas imágenes bajo dos condiciones diferentes (cada imagen es su propio control)
- Es **no paramétrico**: no asume que los datos tienen distribución normal (los BERTScores raramente son normales)
- Es robusto con muestras pequeñas

**Cómo funciona:**
1. Para cada imagen, calcular la diferencia entre el score de Condición X y Condición Y
2. Ordenar las diferencias por magnitud (ignorando el signo)
3. Asignar rangos
4. Sumar los rangos de las diferencias positivas y negativas por separado
5. El estadístico de prueba es el menor de esas dos sumas
6. Comparar contra la distribución teórica para obtener el p-valor

**Interpretación:**
- p < 0.05 → la diferencia ES significativa (rechazamos la hipótesis nula de que son iguales)
- p ≥ 0.05 → la diferencia NO es significativa (no podemos afirmar que son diferentes)

### 3.3 Effect Size (tamaño del efecto)

Además del p-valor, se reporta el **effect size** para indicar qué tan grande es la diferencia:

- Se calcula como: r = Z / √N, donde Z es el estadístico estandarizado y N es el número de pares
- Interpretación:
  - r < 0.3: efecto pequeño
  - 0.3 ≤ r < 0.5: efecto mediano
  - r ≥ 0.5: efecto grande

Un resultado puede ser estadísticamente significativo pero con efecto pequeño (ej: p=0.03 pero r=0.1). Esto significa que la diferencia es real pero clínicamente irrelevante.

---

## 4. Resultado del Método `evaluate_segmentation`

Recibe la máscara predicha y la GT. Retorna un diccionario con:
- `iou`: float
- `dice`: float
- `ssim`: float

---

## 5. Resultado del Método `evaluate_text`

Recibe el texto generado y el texto de referencia. Retorna un diccionario con:
- `bertscore_f1`: float
- `finding_mentioned`: bool

---

## 6. Resultado del Método `statistical_test`

Recibe dos listas de scores (uno por condición) y retorna:
- `statistic`: valor del estadístico de prueba
- `p_value`: probabilidad de observar esa diferencia por azar
- `significant`: booleano (p < 0.05)
- `effect_size`: magnitud del efecto (r)

---

## 7. Verificaciones

1. IoU y Dice retornan 1.0 cuando la predicción es idéntica a la GT
2. IoU y Dice retornan 0.0 cuando no hay solapamiento
3. BERTScore retorna un valor alto cuando se compara un texto consigo mismo
4. El test de Wilcoxon retorna p ≈ 1.0 cuando ambas listas son idénticas
5. El test de Wilcoxon retorna p < 0.05 cuando las listas son claramente diferentes
