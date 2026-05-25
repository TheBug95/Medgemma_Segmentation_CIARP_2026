# M2: CNNClassifier — Clasificación Oftalmológica + Grad-CAM

## 1. Propósito

El CNNClassifier tiene **3 responsabilidades**:

1. **Clasificar** la imagen según la enfermedad detectada (ej: glaucoma, normal)
2. **Producir una distribución de probabilidades** sobre las categorías (para condicionar MedGemma)
3. **Generar Grad-CAM** de la última capa convolucional (requerido únicamente por Pipeline B — WSSS)

Es **compartido** por los 3 pipelines.

---

## 2. Arquitectura

El clasificador usa transfer learning: un backbone CNN preentrenado en ImageNet, al cual se le reemplaza la última capa FC por una nueva que clasifica en N clases configurables (actualmente 2: glaucoma, normal).

**Diseño extensible:** La capa FC se define con `num_classes` como parámetro de configuración. Agregar una nueva enfermedad (ej: catarata) solo requiere cambiar `num_classes` en la configuración y reentrenar; no se modifica la arquitectura del clasificador.

El flujo es: Imagen (448×448) → Backbone CNN → Global Average Pooling → Dropout (50%) → FC → Softmax → Distribución de probabilidades.

De la última capa convolucional del backbone se extraen los feature maps para Grad-CAM.

### Backbones candidatos

| Backbone | Vector de features | Parámetros |
|----------|-------------------|------------|
| ResNet-18 | 512 dims | 11.7M |
| ResNet-34 | 512 dims | 21.8M |
| ResNet-50 | 2048 dims | 25.6M |
| EfficientNet-B0 | 1280 dims | 5.3M |
| DenseNet-121 | 1024 dims | 8.0M |

---

## 3. Entrenamiento

### 3.1 Estrategia de fine-tuning

Se **congelan todas las capas** excepto el último bloque convolucional y la capa FC. Las capas bajas (bordes, texturas) ya funcionan bien desde ImageNet. Solo se adaptan las capas altas a imágenes oftalmológicas.

### 3.2 Configuración

- **Optimizer**: Adam, lr=1e-4, weight_decay=1e-5
- **Loss**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau — si val_loss no mejora en 3 épocas, reduce lr a la mitad
- **Early stopping**: patience=5 épocas sin mejora → detener y restaurar mejores pesos
- **Épocas máximas**: 30

### 3.3 Cada época hace:

1. **Train**: recorrer batches, forward pass, calcular loss, backward, actualizar pesos
2. **Validación**: forward pass sin actualizar pesos, medir loss/accuracy/F1
3. **Decisiones**: ajustar lr, guardar si es el mejor, verificar early stopping

---

## 4. Predicción

Dada una imagen normalizada, retorna:
- `prediction`: clase ganadora (ej: "glaucoma")
- `distribution`: probabilidades por clase (ej: {"glaucoma": 0.92, "normal": 0.08})
- `predicted_index`: índice numérico

---

## 5. Grad-CAM

**Nota:** La generación de Grad-CAM solo se requiere para el Pipeline B (WSSS), donde se usa para seleccionar la mejor candidata de SAM vía IoU. En los pipelines A y C, `get_gradcam()` no se invoca.

### 5.1 ¿Qué es?

Un mapa de calor (448×448) que muestra qué regiones de la imagen contribuyeron más a la predicción. Valores cercanos a 1 = zonas "calientes" (importantes). Valores cercanos a 0 = irrelevantes.

### 5.2 Cómo funciona paso a paso

1. **Registrar hooks** en la última capa conv del backbone: uno captura los feature maps (forward), otro los gradientes (backward)
2. **Forward pass**: obtener predicción y capturar feature maps (ej: 512 canales de 14×14)
3. **Backward pass**: calcular gradientes del score de la clase predicha respecto a los feature maps
4. **Calcular pesos por canal**: promediar cada gradiente espacialmente (14×14 → 1 valor por canal)
5. **Combinar**: multiplicar cada feature map por su peso y sumar → mapa de 14×14
6. **ReLU**: eliminar valores negativos (solo contribuciones positivas)
7. **Redimensionar**: agrandar de 14×14 a 448×448 con interpolación bilinear
8. **Normalizar**: escalar a [0, 1]
9. **Limpiar hooks** para no afectar futuras inferencias

Para Pipeline B, el Grad-CAM se **binariza** con percentil 95 (solo el 5% más activo = "1") y se calcula IoU contra las candidatas de SAM.

---

## 6. Experimento de Selección de Backbone

### 6.1 Protocolo

Para cada backbone × 5 semillas: entrenar, medir 4 dimensiones:

**D1 — Clasificación**: Accuracy, F1-macro, matriz de confusión

**D2 — Calidad Grad-CAM**: IoU(Grad-CAM binarizado, GT mask), pointing accuracy (¿el punto máximo cae dentro de la GT?), cobertura (% de GT cubierta)

**D3 — Costo**: Parámetros, VRAM, tiempo de inferencia

**D4 — Few-shot**: Accuracy con N=50, N=100, N=full

### 6.2 Fórmula de selección

```
Score = 0.30×F1 + 0.30×IoU_GradCAM + 0.20×Acc_fewshot + 0.20×(1-VRAM)
```

Si empatan, preferir el más ligero.

---

## 7. Verificaciones

1. Forward pass produce logits del shape correcto
2. Predict retorna distribución que suma 1.0
3. Grad-CAM retorna array 448×448 en [0, 1]
4. Save/Load produce resultados idénticos
