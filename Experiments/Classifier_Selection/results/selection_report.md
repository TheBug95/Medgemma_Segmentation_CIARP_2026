# Selection Report - Experimento de Selección de Clasificador CNN

*Este documento se genera automáticamente tras ejecutar el experimento.*

---

## 1. Resumen Ejecutivo

**Backbone ganador:** `<<WINNER>>`

**Score:** `<<SCORE>>`

### Tabla Comparativa

| Métrica | ResNet-18 | EfficientNet-B0 | DenseNet-121 | Winner |
|---------|-----------|----------------|--------------|--------|
| Accuracy | | | | |
| F1-macro | | | | |
| IoU Grad-CAM | | | | |
| Acc@N50 | | | | |
| Acc@N100 | | | | |
| VRAM (batch=1) | | | | |
| Params (M) | | | | |
| **Score** | | | | |

---

## 2. Análisis por Dimensión

### 2.1 Clasificación
[Análisis de F1-macro, confusión de clases, performance en glaucoma vs normal]

### 2.2 Grad-CAM Quality
[Análisis de IoU y pointing accuracy - qué tan bien el Grad-CAM alinea con la máscara GT]

### 2.3 Computacional
[Análisis de trade-off accuracy vs costo computacional]

### 2.4 Few-shot
[Análisis de generalización con pocas muestras]

---

## 3. Recomendación

[Explicación de por qué se eligió el backbone ganador]

---

## 4. Configuración Usada

```yaml
<<CONFIG_YAML_USED>>
```

---

## 5. Detalles Técnicos

- **Fecha del experimento:** `<<TIMESTAMP>>`
- **Dataset:** REFUGE (train/val/test)
- **Total imágenes:** 1200 (400 train, 400 val, 400 test)
- **Clases:** glaucoma (40 por split) vs normal (360 por split)