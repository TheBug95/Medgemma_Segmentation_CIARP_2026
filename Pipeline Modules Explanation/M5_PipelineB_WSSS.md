# M5: PipelineB_WSSS — Selección de Máscara por Grad-CAM

## 1. Propósito

Pipeline B usa Weakly Supervised Semantic Segmentation (WSSS). Su trabajo es: dadas N máscaras candidatas de SAM y un mapa Grad-CAM del clasificador, **seleccionar la candidata que mejor coincide con la región señalada por el Grad-CAM**.

No entrena nada. Es puramente un algoritmo de selección.

---

## 2. Dependencias

Este módulo recibe datos de otros dos módulos:
- **M2 (CNNClassifier):** le provee el Grad-CAM de la imagen
- **M3 (SAMSegmenter):** le provee la lista de máscaras candidatas

---

## 3. Flujo Paso a Paso

### Paso 1: Recibir el Grad-CAM

El clasificador CNN genera un Grad-CAM para la imagen: un mapa de calor de 448×448 con valores entre 0 y 1. Los valores altos indican dónde la CNN "miró" para hacer su predicción.

### Paso 2: Binarizar el Grad-CAM

Convertir el mapa de calor continuo en una máscara binaria (0 o 1):

1. Calcular el percentil 95 de los valores del Grad-CAM. Esto encuentra el valor por debajo del cual está el 95% de los píxeles
2. Cualquier píxel con valor ≥ a ese umbral se marca como 1 (zona activa), el resto como 0
3. El resultado es una máscara que cubre aproximadamente el 5% de la imagen — la zona donde la CNN concentró más atención

¿Por qué percentil 95? Porque queremos solo las regiones de **máxima activación**, no todo lo que tenga algún nivel de activación. Un percentil más bajo (ej: 80) daría una zona demasiado amplia y menos discriminativa.

### Paso 3: Redimensionar las candidatas

Las candidatas de SAM están a 1024×1024 y el Grad-CAM está a 448×448. Hay que llevarlas al mismo tamaño. Se redimensiona el Grad-CAM binarizado a 1024×1024 (o las candidatas a 448×448) usando interpolación NEAREST para preservar bordes binarios.

### Paso 4: Calcular IoU entre cada candidata y el Grad-CAM

Para cada máscara candidata de SAM:

1. Calcular la **intersección**: píxeles que son "1" tanto en la candidata como en el Grad-CAM binarizado
2. Calcular la **unión**: píxeles que son "1" en al menos una de las dos
3. IoU = intersección / unión

El IoU mide qué tanto se solapan las dos máscaras. Un IoU de 1.0 significa superposición perfecta, 0.0 significa que no se solapan en nada.

### Paso 5: Seleccionar la mejor

Se elige la candidata con el **mayor IoU** contra el Grad-CAM. Esta es la máscara que más coincide con la región donde la CNN detectó la patología.

---

## 4. Resultado

El módulo retorna un diccionario con:
- `mask`: la máscara binaria seleccionada (la mejor candidata)
- `iou_with_gradcam`: el IoU entre esa candidata y el Grad-CAM (indica qué tan buena fue la coincidencia)
- `candidate_index`: el índice de la candidata seleccionada dentro de la lista original

---

## 5. ¿Por qué funciona?

La intuición es: si la CNN aprendió correctamente que "esta imagen tiene catarata", entonces su Grad-CAM apuntará a la zona del cristalino. SAM, por otro lado, segmenta múltiples regiones de la imagen (disco óptico, vasos, cristalino, etc.). Al calcular IoU entre el Grad-CAM y cada candidata, estamos preguntando: "¿cuál de estas regiones segmentadas coincide con donde la CNN vio la patología?"

---

## 6. Limitaciones

- **Dependencia del clasificador**: Si la CNN clasifica incorrectamente (dice "glaucoma" cuando es "catarata"), el Grad-CAM apuntará a la zona equivocada y se seleccionará una máscara incorrecta
- **Grad-CAM difuso**: A veces el Grad-CAM es muy amplio y cubre múltiples regiones, lo que reduce su poder discriminativo
- **Candidatas insuficientes**: Si SAM no generó una candidata que cubra la patología real, no hay nada bueno que seleccionar

---

## 7. Parámetro configurable

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `gradcam_percentile` | 95 | Percentil para binarizar el Grad-CAM. Más alto = zona más pequeña y focal |

---

## 8. Verificaciones

1. Dadas candidatas y un Grad-CAM, retorna un resultado con los campos correctos
2. La máscara seleccionada es efectivamente la de mayor IoU (verificar recalculando manualmente)
3. Si no hay candidatas, retorna un resultado vacío o un error controlado
4. El `iou_with_gradcam` retornado es correcto (verificar contra cálculo manual)
