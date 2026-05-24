# M7: MedGemmaConditioner — Generación de Descripciones Clínicas

## 1. Propósito

MedGemmaConditioner encapsula la interacción con MedGemma 4B (Vision-Language Model médico de Google). Su trabajo es: dada una imagen y opcionalmente una máscara y/o información de clasificación, **generar una descripción clínica** de los hallazgos oftalmológicos.

Este módulo implementa las **6 condiciones de ablation** que son el corazón del experimento comparativo. Cada condición varía qué información recibe MedGemma para medir cómo el condicionamiento afecta la calidad del texto generado.

---

## 2. MedGemma como Caja Negra

MedGemma se usa **sin modificación interna**. No se le aplica LoRA, no se le fine-tunea, no se cambian sus pesos. Se usa exactamente como viene de HuggingFace. Lo único que cambia entre condiciones es **qué se le envía como entrada**.

MedGemma recibe 2 entradas:
- **Imagen**: puede ser la imagen cruda o la imagen con la máscara superpuesta en rojo
- **Texto (prompt)**: puede ser genérico o incluir información del clasificador

---

## 3. Las 6 Condiciones de Ablation

### Condición A — Baseline puro

- **Imagen**: Cruda, sin modificar
- **Prompt**: "Describe the ophthalmological findings in this fundus image."
- **Qué prueba**: Capacidad base de MedGemma sin ninguna ayuda. Es la línea de referencia

### Condición B — Solo máscara

- **Imagen**: La imagen original con la máscara superpuesta como overlay semi-transparente en rojo
- **Prompt**: "The region highlighted in red was identified by an automatic segmentation system. Describe the ophthalmological findings, focusing on the highlighted region."
- **Qué prueba**: ¿Mostrarle DÓNDE está la patología mejora la descripción?

### Condición C1 — Solo predicción (clase)

- **Imagen**: Cruda, sin modificar
- **Prompt**: "An ophthalmological classifier identifies the primary finding in this fundus image as: glaucoma. Describe the ophthalmological findings."
- **Qué prueba**: ¿Decirle QUÉ enfermedad es (sin decirle dónde) mejora la descripción?

### Condición C2 — Solo distribución completa

- **Imagen**: Cruda, sin modificar
- **Prompt**: "An ophthalmological classifier analyzed this fundus image and estimates: glaucoma (92%), normal (8%). Describe the ophthalmological findings."
- **Qué prueba**: ¿Dar la distribución de probabilidades completa (incluyendo la incertidumbre del clasificador) mejora la descripción? ¿MedGemma genera un texto más cauteloso cuando sabe que el clasificador tiene dudas?

### Condición D1 — Máscara + predicción

- **Imagen**: Con overlay rojo de la máscara
- **Prompt**: "An ophthalmological classifier identifies the primary finding as: glaucoma. The region highlighted in red indicates the area where this finding is located. Describe the findings focusing on the highlighted region."
- **Qué prueba**: ¿Combinar DÓNDE y QUÉ (sin distribución) da mejora adicional?

### Condición D2 — Máscara + distribución completa

- **Imagen**: Con overlay rojo de la máscara
- **Prompt**: "An ophthalmological classifier estimates: glaucoma (92%), normal (8%). The region highlighted in red indicates the area where the main finding is located. Describe the findings in detail, focusing on the highlighted region and its relationship with the suggested diagnosis."
- **Qué prueba**: ¿El condicionamiento máximo (DÓNDE + QUÉ + incertidumbre) produce la mejor descripción?

---

## 4. Cómo se Genera el Overlay de Máscara

Cuando la condición requiere overlay (B, D1, D2):

1. Tomar una copia de la imagen original
2. Donde la máscara es "1" (patología), mezclar el color del píxel original con rojo puro:
   - Nuevo píxel = 60% del original + 40% de rojo (255, 0, 0)
   - Esto produce una zona roja semi-transparente que permite ver la anatomía debajo
3. Donde la máscara es "0" (fondo), dejar la imagen sin modificar

El resultado es la imagen original con la zona de patología teñida de rojo.

---

## 5. Cómo se Construyen los Prompts

El módulo tiene un **template por condición**. Los templates de C1, C2, D1 y D2 tienen placeholders que se llenan dinámicamente:

- `{prediction}`: se reemplaza con la clase predicha (ej: "glaucoma")
- `{distribution}`: se reemplaza con la distribución formateada (ej: "glaucoma (92%), normal (8%)")

La distribución se formatea ordenando las clases de mayor a menor probabilidad.

---

## 6. Parámetros de Generación

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `max_new_tokens` | 512 | Longitud máxima del texto generado |
| `torch_dtype` | bfloat16 | Precisión reducida para ahorrar VRAM |
| `device_map` | "auto" | Distribución automática en GPU |
| `do_sample` | False | Generación determinística (greedy) para reproducibilidad |

Se usa generación greedy (no sampling) para que el mismo input siempre produzca el mismo output. Esto es necesario para reproducibilidad.

---

## 7. Flujo de Ejecución del Método `generate`

1. **Validar condición**: Verificar que los parámetros proporcionados son consistentes con la condición. Por ejemplo, condición B requiere `mask`, condición C1 requiere `prediction`, etc.
2. **Preparar imagen**: Si la condición requiere overlay, generar la imagen con máscara superpuesta. Si no, usar la imagen cruda.
3. **Construir prompt**: Seleccionar el template de la condición y llenar los placeholders.
4. **Preprocesar**: Usar el `AutoProcessor` de MedGemma para convertir la imagen y el texto al formato que espera el modelo.
5. **Generar**: Llamar a `model.generate()` con los inputs preprocesados.
6. **Decodificar**: Convertir los tokens generados de vuelta a texto legible.
7. **Retornar**: El texto generado, la condición usada, el prompt exacto, y si se usó overlay.

---

## 8. Resultado

El módulo retorna un diccionario con:
- `text`: el texto generado por MedGemma (ej: "The fundus image shows...")
- `condition`: la condición usada (ej: "D2")
- `prompt_used`: el prompt exacto que se envió (para auditoría y reproducibilidad)
- `image_was_overlaid`: booleano — si se superpuso la máscara

---

## 9. Consideraciones de VRAM

MedGemma 4B en bfloat16 consume ~8GB de VRAM. Es el modelo más grande del pipeline. Se recomienda:
- Cargar con `device_map="auto"` para distribuir entre GPU y CPU si es necesario
- Liberar la caché de CUDA antes de cargar MedGemma si SAM ya terminó
- En Colab T4 (16GB), MedGemma cabe si no hay otros modelos grandes cargados simultáneamente

---

## 10. Verificaciones

1. Cada una de las 6 condiciones genera un texto no vacío
2. El prompt usado corresponde al template correcto de la condición
3. Los placeholders se llenaron correctamente (no quedan `{prediction}` literales en el texto)
4. La imagen con overlay tiene las dimensiones correctas y los píxeles rojos donde corresponde
5. Con `do_sample=False` y la misma semilla, dos ejecuciones producen el mismo texto
6. El módulo rechaza combinaciones inválidas (ej: condición B sin mask)
