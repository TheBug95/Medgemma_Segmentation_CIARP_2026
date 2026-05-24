# M6: PipelineC_FSLFD — Filtrado de Máscaras por KDE

## 1. Propósito

Pipeline C usa Few-Shot Learning con Feature Distribution (FSL/FD). Su trabajo es: dadas N máscaras candidatas de SAM, **filtrar y seleccionar** la que tenga una distribución de características más consistente con las máscaras conocidas de esa enfermedad.

A diferencia de Pipeline B (que usa localización espacial con Grad-CAM), Pipeline C opera en el **espacio de embeddings**: compara las características visuales de cada candidata contra un modelo estadístico construido a partir de ejemplos previos.

---

## 2. Dependencias

- **M2 (CNNClassifier):** le provee la predicción de enfermedad (ej: "catarata") para seleccionar qué modelo KDE usar
- **M3 (SAMSegmenter):** le provee la lista de máscaras candidatas
- **MedSigLIP:** extractor de embeddings (viene cargado con MedGemma, 0 costo extra)

---

## 3. Conceptos Clave

### 3.1 ¿Qué es un embedding?

Un embedding es un vector numérico (ej: 768 números) que **resume** las características visuales de una imagen o región. Imágenes visualmente similares tendrán embeddings cercanos en el espacio vectorial. MedSigLIP genera embeddings de 768 dimensiones específicamente entrenados para imágenes médicas.

### 3.2 ¿Qué es KDE?

KDE (Kernel Density Estimation) es un método estadístico que, dado un conjunto de puntos, **estima la función de densidad** subyacente. Es como construir un "mapa de probabilidad" en el espacio de embeddings.

En nuestro caso: dados k=15-20 embeddings de máscaras GT de una enfermedad (ej: cataratas), KDE construye un modelo de "cómo se ven normalmente las máscaras de catarata" en el espacio de 768 dimensiones.

### 3.3 ¿Qué son los thresholds OOD?

OOD = Out-Of-Distribution. Los thresholds `[Θ_min, Θ_max]` definen una "ventana de normalidad". Si el embedding de una candidata tiene una log-densidad dentro de esta ventana, se considera **in-distribution** (se parece a las máscaras conocidas). Si cae fuera, se considera **out-of-distribution** (no se parece) y se descarta.

---

## 4. Fase de Calibración (se hace UNA vez antes de la inferencia)

### Para cada enfermedad del dataset, se realiza:

**Paso 1: Reunir el support set**

Tomar k=15-20 pares (imagen, máscara GT) de esa enfermedad del set de entrenamiento. Estas son las "muestras de referencia" que definen cómo se ve esa patología.

**Paso 2: Extraer embeddings**

Para cada par (imagen, máscara GT):
1. Aplicar la máscara sobre la imagen (hacer cero todo lo que no es patología)
2. Pasar la imagen enmascarada por MedSigLIP → obtener un embedding de 768 dimensiones
3. Acumular estos embeddings

Al final se tiene una matriz de (k, 768): k embeddings de 768 dimensiones cada uno.

**Paso 3: Ajustar KDE**

Usar los k embeddings para ajustar un estimador de densidad kernel (KDE) por cada dimensión del embedding (768 KDEs unidimensionales). Esto modelo la distribución "normal" de cómo se ven las máscaras de esa enfermedad.

**Paso 4: Calcular thresholds OOD**

Evaluar la log-densidad de cada uno de los k embeddings del support set contra el KDE recién ajustado. Esto da k valores de log-densidad. A partir de estos:

- Calcular Q1 (percentil 25) y Q3 (percentil 75)
- IQR = Q3 - Q1
- Θ_min = Q1 - 1.5 × IQR (límite inferior)
- Θ_max = Q3 + 1.5 × IQR (límite superior)

Esta es la regla del IQR, la misma que se usa para detectar outliers en un boxplot. Cualquier candidata cuya log-densidad caiga fuera de [Θ_min, Θ_max] se considera atípica.

**Resultado de la calibración:** Para cada enfermedad se almacena:
- El modelo KDE ajustado
- Los thresholds [Θ_min, Θ_max]

---

## 5. Fase de Inferencia (Selección de Máscara)

### Paso 1: Obtener la predicción del clasificador

El clasificador CNN del Paso 1A dice: "esta imagen tiene **catarata**".

### Paso 2: Seleccionar el KDE y thresholds correspondientes

Con la predicción "glaucoma", se buscan el KDE y los thresholds `[Θ_min_glaucoma, Θ_max_glaucoma]` que se calibraron previamente para glaucoma.

Esta es la dependencia crítica con el clasificador: **sin la predicción, no sabemos contra qué distribución comparar**.

### Paso 3: Evaluar cada candidata

Para cada máscara candidata de SAM:
1. Aplicar la máscara sobre la imagen original (hacer cero todo fuera de la candidata)
2. Pasar la imagen enmascarada por MedSigLIP → embedding de 768 dims
3. Evaluar la log-densidad `ℓ*` de ese embedding contra el KDE de catarata:
   - `ℓ* = Σⱼ log p̂ⱼ(z*ⱼ)` donde j recorre las 768 dimensiones
   - Esto es la suma de las log-densidades marginales por dimensión

### Paso 4: Filtrar

Solo pasan las candidatas cuya log-densidad cae dentro de la ventana: `Θ_min ≤ ℓ* ≤ Θ_max`.

Las que están fuera se descartan: no se parecen lo suficiente a las máscaras conocidas de catarata.

### Paso 5: Seleccionar la mejor

De las que pasaron el filtro, se selecciona la de **mayor ℓ*** (la más "típica" según la distribución).

---

## 6. Resultado

El módulo retorna un diccionario con:
- `mask`: la máscara binaria seleccionada
- `log_density`: el ℓ* de la máscara seleccionada
- `in_ood_window`: booleano — si pasó el filtro OOD
- `candidate_index`: índice dentro de la lista original

Si ninguna candidata pasa el filtro, se retorna la de mayor ℓ* (la menos atípica) con `in_ood_window=False` como señal de baja confianza.

---

## 7. ¿Por qué funciona?

La intuición es: las patologías reales de una enfermedad tienen una "firma visual" característica en el espacio de embeddings. Las máscaras de SAM que cubren la patología real tendrán un embedding similar a las máscaras GT de esa enfermedad. Las máscaras que cubren vasos, disco óptico, o artefactos tendrán embeddings muy diferentes y caerán fuera de la ventana OOD.

---

## 8. Limitaciones

- **Dependencia doble**: Depende del clasificador (para seleccionar el KDE correcto) Y de MedSigLIP (para los embeddings). Si el clasificador falla, se usa el KDE equivocado
- **Support set pequeño**: Con solo k=15-20 ejemplos, el KDE puede no ser representativo
- **Dimensionalidad alta**: 768 dimensiones con pocos ejemplos puede dar estimaciones de densidad inestables. Se mitiga usando KDE marginales (una por dimensión) en lugar de KDE multivariado
- **Dominio**: El paper BIP validó esto en segmento anterior (pupila). Para fondo de ojo es experimental

---

## 9. Verificaciones

1. La calibración produce un KDE y thresholds válidos para cada enfermedad
2. `select_mask` retorna un resultado con los campos correctos
3. Las candidatas filtradas efectivamente caen dentro de [Θ_min, Θ_max]
4. La candidata seleccionada es la de mayor ℓ* entre las que pasan el filtro
5. El módulo maneja el caso de 0 candidatas que pasan el filtro (retornar la menos atípica)
