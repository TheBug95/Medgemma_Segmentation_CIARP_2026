# REFUGE Dataset — Guia de Segmentacion

## Dataset: Retinal Fundus Glaucoma Challenge

Particion de **train** con 400 imagenes de fondo de ojo:
- **40 glaucoma** (prefijo `g`)
- **360 normales** (prefijo `n`)

---

## Estructura de Carpetas

| Carpeta | Formato | Contenido |
|---|---|---|
| `Images/` | `.jpg` | Fotografias de retina originales (2124x2056) |
| `Images_Cropped/` | `.jpg` | Recortadas alrededor del disco optico |
| `Masks/` | `.png` | Mascaras de segmentacion (0=fondo, 1=rim, 2=copa) |
| `Masks_Cropped/` | `.png` | Mascaras recortadas (misma codificacion) |
| `gts/` | `.bmp` | Misma segmentacion que Masks/, codificada como 0=copa, 128=rim, 255=fondo |
| `illustrations/` | `.jpg` | Visualizaciones con anotaciones superpuestas |
| `index.json` | JSON | Metadatos: coordenadas fovea, label (1=glaucoma, 0=normal) |

---

## Codificacion de las Mascaras (`Masks/`)

| Valor | Estructura | Color en overlay |
|---|---|---|
| `0` | Fondo (todo fuera del disco optico) | Sin cambios |
| `1` | **Rim** (borde neuroretinal del disco) | Verde |
| `2` | **Copa optica** (optic cup) | Rojo |

El **disco optico completo** = Rim (1) + Copa (2).

---

## Metricas Clinicas

### 1. vCDR — Cup-to-Disc Ratio Vertical (Estandar Clinico)

Es la metrica clinica estandar para diagnostico de glaucoma. Se calcula como la
**altura vertical de la copa** dividido la **altura vertical del disco**:

```
vCDR = altura_vertical_copa / altura_vertical_disco
```

Se mide en el eje Y (vertical) usando los puntos extremos de los contornos.

Umbrales clinicos:
- **Normal**: vCDR < 0.5
- **Glaucoma Sospechoso**: 0.5 <= vCDR < 0.65
- **Glaucoma**: vCDR >= 0.65

### 2. CDR por Area (NO es clinico)

Simplemente la proporcion de pixeles de copa vs disco total. No se usa en practica
clinica pero es util como referencia.

```
CDR_area = area_copa / area_disco
```

### 3. RDR Real / DDLS (Disco-to-Limit Summary)

Mide el **ancho minimo del rim** en cualquier direccion radial desde el centro
del disco, dividido por el **radio maximo del disco**:

```
RDR = ancho_minimo_rim / radio_maximo_disco
```

Se calcula lanzando 360 rayos desde el centro de la copa hacia el borde del disco,
midiendo el ancho del rim en cada direccion y tomando el minimo.

- **RDR bajo** = rim muy delgado en algun punto = senal de glaucoma
- **RDR alto** = rim uniforme y grueso = normal

### 4. RIM Ratio (1 - CDR_area)

Proporcion del area del rim vs area del disco. No es RDR real, solo referencia.

---

## Diferencia entre Glaucoma y Normal

Ambos grupos tienen copa y disco. La diferencia esta en:

| | Glaucoma | Normal |
|---|---|---|
| **vCDR medio** | **~0.6-0.7** | **~0.4-0.5** |
| **RDR medio** | **Bajo (~0.1-0.3)** | **Alto (~0.4-0.7)** |
| Copa | Agrandada verticalmente | Pequena verticalmente |
| Rim | **Muy delgado** en al menos un punto | Grueso en todas direcciones |

En glaucoma la copa se expande y el borde (rim) se vuelve asimetrico y delgado
en la region donde progresion la enfermedad.

---

## Visualizacion de Overlays

Leyenda de colores en los overlays:
- **Rojo** = Copa optica
- **Verde** = Borde del disco (rim)
- **Amarillo** = Linea vertical del disco (vCDR)
- **Cyan** = Linea vertical de la copa (vCDR)
- **Magenta** = Rayo con rim minimo (RDR real)

---

## Ejemplos Visuales — Imagenes Recortadas

### Glaucoma — `g0001`

```
============================================================
vCDR (vertical)      = 0.6232   <- Estandar clinico
CDR_area             = 0.4159   <- Proporcion de areas
------------------------------------------------------------
RDR REAL (DDLS)      = 0.2734   <- Ancho minimo del rim / radio disco
RIM_ratio (1-CDR)    = 0.5841   <- NO es RDR real, solo area
------------------------------------------------------------
Altura disco (px)    = 353
Altura copa (px)     = 220
Radio disco max (px) = 179.2
Ancho rim min (px)   = 49.0
============================================================
```

Copa moderada, rim con ancho minimo de 49px.

---

### Normal — `n0001`

```
============================================================
vCDR (vertical)      = 0.4580   <- Estandar clinico
CDR_area             = 0.2457   <- Proporcion de areas
------------------------------------------------------------
RDR REAL (DDLS)      = 0.2419   <- Ancho minimo del rim / radio disco
RIM_ratio (1-CDR)    = 0.7543   <- NO es RDR real, solo area
------------------------------------------------------------
Altura disco (px)    = 393
Altura copa (px)     = 180
Radio disco max (px) = 219.4
Ancho rim min (px)   = 53.1
============================================================
```

Copa pequena verticalmente (vCDR bajo), rim ancho y uniforme.

---

### Glaucoma Severo — `g0028`

```
============================================================
vCDR (vertical)      = 0.7924   <- Estandar clinico
CDR_area             = 0.6250   <- Proporcion de areas
------------------------------------------------------------
RDR REAL (DDLS)      = 0.1076   <- Ancho minimo del rim / radio disco
RIM_ratio (1-CDR)    = 0.3750   <- NO es RDR real, solo area
------------------------------------------------------------
Altura disco (px)    = 289
Altura copa (px)     = 229
Radio disco max (px) = 158.8
Ancho rim min (px)   = 17.1
============================================================
```

Copa muy grande verticalmente (vCDR altisimo), rim minimo de solo 17px
(extremadamente delgado).

---

## Comparacion lado a lado

| Imagen | Label | vCDR | CDR_area | RDR_real | Rim_min_px | Diagnostico |
|---|---|---|---|---|---|---|
| `g0001` | Glaucoma | 0.6232 | 0.4159 | 0.2734 | 49.0 | Glaucoma sospechoso |
| `n0001` | Normal | 0.4580 | 0.2457 | 0.2419 | 53.1 | Normal |
| `g0028` | Glaucoma | **0.7924** | 0.6250 | **0.1076** | **17.1** | Glaucoma severo |

Observaciones clave:
- `g0028` tiene vCDR de 0.79 (muy por encima del umbral de 0.65) Y el rim
  minimo es solo 17px (RDR=0.11, extremadamente bajo).
- `n0001` tiene vCDR bajo (0.46) Y el rim minimo de 53px es casi 3x mas
  grueso que en `g0028`.

---

## Scripts para Generar Overlays

### Con imagenes recortadas
```bash
python3 overlay.py g0001
python3 overlay.py n0001
```

### Con imagenes completas
```bash
python3 overlay_full.py g0001
python3 overlay_full.py n0001
```

Los scripts generan un overlay con:
- Copa en rojo, rim en verde
- Lineas de referencia vertical (amarillo=disco, cyan=copa)
- Linea magenta mostrando la direccion del rim mas delgado

---

## Dependencias

```bash
pip install opencv-python Pillow numpy
```

---

## Codigos Originales del Dataset

El dataset REFUGE original codifica las mascaras como:
- `0` = copa optica
- `128` = borde neuroretinal (rim)
- `255` = fondo

Los scripts hacen un **remapeo automatico** a la codificacion interna:
- `2` = copa
- `1` = rim
- `0` = fondo

Esto permite trabajar con una codificacion consistente sin importar el origen
de la mascara.

---

## Como se Clasifica Glaucoma?

**1. Por vCDR (clinico):**
Si vCDR > 0.65 -> glaucoma. Este es el criterio de referencia en oftalmologia.

**2. Por RDR/DDLS:**
Si el ancho minimo del rim es muy pequeno respecto al radio del disco ->
glaucoma. Detecta perdida focal de fibras que el vCDR puede no capturar.

**3. Combinacion:**
vCDR alto + RDR bajo = glaucoma probable con alta especificidad.
