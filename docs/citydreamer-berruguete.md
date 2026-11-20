# CityDreamer — qué es, cómo funciona y qué hemos hecho con Berruguete

> Documento de trabajo de David Antizar (Ntizar). Repo original:
> [hzxie/CityDreamer](https://github.com/hzxie/CityDreamer) · CVPR 2024 ·
> *CityDreamer: Compositional Generative Model of Unbounded 3D Cities* (S-Lab, NTU).

---

## 1. Qué hace, en una frase

Genera **ciudades 3D infinitas** (no un edificio suelto ni una escena cerrada): un
modelo de disposición urbana inventa el plano, otro pinta el suelo y otro pinta cada
edificio, y se componen por máscaras. La cámara orbita y se renderiza un vídeo aéreo.

Lo importante para nosotros: **acepta el plano de una ciudad real** (`hf.png` +
`seg.png` derivados de OpenStreetMap), así que se le puede pedir *este* barrio y no uno
inventado. Eso es exactamente lo que hemos preparado para **Berruguete (Tetuán, Madrid)**.

## 2. Arquitectura: tres generadores, un compositor

| Módulo | Peso | Qué produce | Red |
|---|---|---|---|
| **Unbounded Layout Generator** | `sampler.pth` | height field (m) + mapa semántico de la ciudad | VQ-VAE + **difusión absorbente** (masked, tipo MaskGIT) sobre tokens del VQ-VAE |
| **Background Stuff Generator** | `CityDreamer-Bgnd.pth` | suelo: calzada, acera, hierba, agua | **GANCraft** con *hash-grid* 3D (NeRF-like) + renderer volumétrico |
| **Building Instance Generator** | `CityDreamer-Fgnd.pth` | **cada** edificio, con su estilo | GANCraft en modo *local* (una instancia por pasada) |

La composición es la clave del paper: el fondo se renderiza **una vez** y luego cada
edificio se renderiza por separado y se pega con su máscara de instancia. Eso es lo que
permite ciudades "unbounded": nunca generas toda la ciudad de golpe, generas una ventana
y la vas desplazando.

Flujo de inferencia (`scripts/inference.py`):

```
hf.png + seg.png  ──►  instancias (etiquetas pares = fachada, impares = cubierta)
                      │
                      ├─► volumen 3D (1536×1536×640, extrusión CUDA)
                      │      └─► ray-voxel intersection (voxlib, CUDA) por cámara
                      ├─► background: parches de 108×192 → GANCraft global
                      └─► por edificio visible: parche → GANCraft local → máscara
                             └─► composición → frame → 24 frames → mp4
```

## 3. Las escalas (esto es lo que hay que tener claro)

- **1 px = 1 m**. El raster del layout va en metros: una huella de 20 m ocupa 20 px.
  (Ojo: `zoom_level` del pipeline OSM solo afecta al **grosor de las calles**, no a la
  escala; el original usa zoom 18.)
- **`LAYOUT_VOL_SIZE = 1536`**: la ventana que ve la red = 1,5 km × 1,5 km.
- **`EXTENDED_VOL_SIZE = 2880`**: el PNG de entrada debe ser ≥ 2880×2880; se recorta el
  centro. El margen sobrante (672 px por lado) se usa para poder recortar el volumen de
  cada edificio (`BUILDING_VOL_SIZE = 672`) sin quedarse sin datos.
- **`LAYOUT_MAX_HEIGHT = 640`**: la altura se normaliza a 640 m. Un edificio de 26 m
  ocupa el 4 % del rango vertical → en Madrid todo sale bajo. En Nueva York, una torre
  de 200 m ocupa el 31 %. **El modelo está entrenado con rascacielos**: es la limitación
  estética real, no un bug nuestro.
- **`--patch_height/--patch_width`**: trocea la imagen de salida (540×960) en parches para
  no reventar la VRAM. El paper usa 5×5 con una RTX 3090 (24 GB).

## 4. Requisitos reales (medidos, no supuestos)

- Python 3.8 + `torch 1.13.1+cu117` (o torch 2.x, ver aviso abajo).
- **Tres extensiones CUDA propias** que hay que compilar (`nvcc` + MSVC en Windows):
  - `extensions/extrude_tensor` — extruye el mapa 2D a volumen `int32` 1536×1536×640.
  - `extensions/voxlib` — ray-voxel intersection en perspectiva.
  - `extensions/grid_encoder` — hash-grid del GANCraft.
- **VRAM**: el volumen de instancias es **1536·1536·640·4 B = 6,0 GB** él solo, más
  `hf_seg` (8×2880² float32 = 265 MB) y los pesos (363 + 100 MB). En una **RTX 3060
  Laptop de 6 GB** no entra: no es cuestión de ajustar parches, es el tensor del volumen.
- Pesos: están en **HuggingFace** (`hzxie/city-dreamer`, sin captcha y gratis):
  `CityDreamer-Bgnd.pth` (363 MB) y `CityDreamer-Fgnd.pth` (100 MB). El `sampler.pth`
  **no** está en HF, pero **no hace falta si el plano viene de OSM** (no se genera el
  layout). El gateway oficial (`gateway.infinitescript.com`) está detrás de Cloudflare +
  reCAPTCHA, de ahí que tiremos de HF.

## 5. Qué hemos hecho (y qué hubo que tocar)

### 5.1 Un plano real de Berruguete

`scripts/berruguete_osm_fetch.py` — descarga 3,2 × 3,2 km de Madrid por **Overpass**
(la API `/map` limita a 50.000 nodos y no llega) e inyecta el `<bounds>` que
`utils/osm_helper.py` necesita para calcular la resolución.

`scripts/berruguete_layout.py` — rasteriza el barrio al formato de CityDreamer:

- calles con su ancho real (clase 1), huellas de edificio (clase 2), zonas verdes (3),
  obra (4), agua (5), suelo (6);
- altura = `height` de OSM; si no hay, `building:levels × 4,26`; si tampoco, la mediana
  del barrio;
- salida `hf.png` (uint16, metros) + `seg.png` (modo P con la paleta de
  `utils/helpers.get_seg_map_palette`) + `metadata.json`.

Resultado medido del extracto:

| Métrica | Valor |
|---|---|
| Área descargada | 3,2 × 3,2 km (18,5 MB de XML, 12.570 huellas) |
| Raster generado | 4.171 × 3.773 px (1 px = 1 m) |
| Ventana central 2880² | 1.586 edificios con instancia |
| Cobertura edificada | 22,6 % |
| Altura p50 / p95 / máx | 15 / 40 / 110 m |

### 5.2 Tres defectos del repo original que hubo que arreglar

1. **`inference.py` ignoraba el plano de OSM**: la llamada estaba comentada
   (`# hf, seg, ... = get_city_layout(city_osm_dir)`) y siempre generaba el layout con el
   sampler. Parcheado: si `--city_osm_dir` tiene `hf.png`, se usa el plano real y **no se
   carga el sampler** (que además no está en HF).
2. **`get_footprint_height_stat` revienta en Europa**: calcula percentiles de alturas y
   los datos de Madrid solo traen `building:levels` → lista vacía → `IndexError`. Se
   sustituye por un relleno tolerante (`levels × 4,26`, luego mediana del barrio).
3. **Las zonas verdes dependían de teselas de Carto**: hoy devuelven **PNG de 0 bytes**
   (200 OK sin cuerpo) y el script original se queda en un `while` infinito reintentando.
   Se sustituye por el dato real de OSM (`leisure=park|garden|…`, `landuse=grass|forest|…`,
   `natural=wood|water`).
4. **`scripts/osm_image_fetcher.py` no acepta `--zoom` por línea de órdenes**:
   `add_argument("--zoom", default=15)` va **sin `type=int`**, así que con `--zoom 18`
   llega un `str` y revienta en `2.0**zoom_level` (`TypeError: unsupported operand
   type(s) for ** or pow(): 'float' and 'str'`). Con el valor por defecto no falla, de ahí
   que no se hubiera visto. Ya no lo usamos (ver punto 3), pero si alguien lo rescata:
   `--zoom`, `type=int`.

### 5.2b El endpoint de Overpass cae, y hay que tener plan B

`overpass-api.de` devolvió **502 Bad Gateway** en el primer intento con un recuadro de
3,2 × 3,2 km. `scripts/berruguete_osm_fetch.py` recorre tres espejos en orden
(`overpass-api.de` → `overpass.kumi.systems` → `overpass.private.coffee`) y el segundo
sirvió el extracto completo: **18,06 MB, 124.759 nodos, 23.432 ways, 602 relations**.
Con un solo endpoint, la tarea habría muerto por infraestructura ajena.


### 5.3 Pesos y aspecto

`output/CityDreamer-Bgnd.pth` y `output/CityDreamer-Fgnd.pth` descargados de HF.

## 6. Cómo se ejecuta (gratis, sin créditos de NaN)

En una GPU de ≥16 GB (Kaggle T4 gratis, Colab T4):

```bash
git clone https://github.com/hzxie/CityDreamer.git && cd CityDreamer
mkdir -p output && cd output
curl -L -o CityDreamer-Bgnd.pth "https://huggingface.co/hzxie/city-dreamer/resolve/main/CityDreamer-Bgnd.pth?download=true"
curl -L -o CityDreamer-Fgnd.pth "https://huggingface.co/hzxie/city-dreamer/resolve/main/CityDreamer-Fgnd.pth?download=true"
cd ..
pip install -r requirements.txt
for e in extensions/*/; do (cd "$e" && pip install .); done

# el plano del barrio
mkdir -p data/osm && git clone --depth 1 https://github.com/Ntizar/berruguete-citydreamer /tmp/b
cp -r /tmp/b/layout data/osm/ES-Madrid-Berruguete

python3 scripts/inference.py \
  --city_osm_dir data/osm/ES-Madrid-Berruguete \
  --patch_height 3 --patch_width 3 \
  --n_frames 8
```

Salida: `output/rendering.mp4` (o sube `--n_frames 24` para el vídeo completo, más lento:
el coste es por edificio visible y por frame).

## 7. Límites honestos

- **6 GB de VRAM no dan** para este modelo (ver §4). El camino gratis es GPU en la nube.
- **Madrid no es Nueva York**: con alturas de 20-30 m el modelo tenderá a pintar fachadas
  bajas o a inventar torres donde no toca. Si queremos que "quede Berruguete", hay que
  decidir una **exageración vertical** (p. ej. ×2 en `hf.png`) y asumirlo como arte.
- Los edificios son inventados: el GAN **no** reproduce la arquitectura real de cada
  portal, solo respeta huella y altura. El ladrillo cara vista, los miradores y las
  cornisas de Tetuán no están en el dataset.
- La licencia es **NTU S-Lab 1.0**: uso e investigación con atribución; revisar antes de
  nada comercial.

## 8. Un pueblo sin datos buenos: Pontejos (Cantabria)

El mismo código, apuntado a `43.4073, -3.8001`, produce esto. No falla nada: lo que cambia
es la **escala del fenómeno**.

| Ventana central 2880 px | Berruguete (Madrid) | Pontejos (Cantabria) |
|---|---|---|
| Superficie edificada | **34,8 %** | **1,2 %** |
| Viario | 18,8 % | 2,3 % |
| Verde | 15,4 % | 9,7 % |
| Altura mediana / p95 | 25 / 42 m | **8 / 12 m** |
| Huellas en el extracto | 12.570 (3,2 km) | 1.572 (1,7 km) |
| Con `building:levels` | 2.286 de 3.688 | 932 de 1.205 |
| Con `height` | 0 | 7 |
| Sin nada | 1.402 | 266 |

**El problema no es la falta de datos: es que CityDreamer necesita densidad.** Normaliza la
altura a 640 m, así que una casa de 8 m ocupa el **1,2 %** del rango vertical: literalmente
invisible. Y como el 98,8 % del lienzo está vacío, el GAN rellena el hueco con lo que
aprendió (fachadas urbanas), no con prados. Para un pueblo hay que cambiar de herramienta,
no de parámetros.

### 8.1 Qué datos SÍ hay en España (probado hoy)

- **Catastro (WFS INSPIRE, gratis y sin clave)**: devuelve `BuildingPart` con
  `numberOfFloorsAboveGround` — es decir, **plantas reales por edificio**, incluidos pajares y
  naves. Probado con un bbox de Pontejos: `HTTP 200`, 25.830 B de GML y plantas 3/1/1/1/2.
  Endpoint: `https://ovc.catastro.meh.es/INSPIRE/wfsBU.aspx` (`typeNames=bu:BuildingPart`).
  En Madrid OSM ya trae `building:levels`; en un pueblo Catastro es **mejor que OSM**.
- **PNOA / ortofoto del IGN (WMS, gratis)**: `https://www.ign.es/wms-inspire/pnoa-ma`,
  capa `OI.OrthoimageCoverage`. Probado sobre Pontejos: **JPEG real de 512×512, 57.745 B**
  — cubiertas, calles, prados y arbolado a ~25 cm/px en la capa nativa. Es la **textura** que
  hoy no tenemos en el render.
- **MDT / LiDAR (IGN, 2 m)**: para que el terreno no sea plano (Cantabria tiene relieve).
- **Mapillary (CC-BY-SA, API gratuita)**: fotos a pie de calle → únicas imágenes de fachada
  con licencia libre.

### 8.2 La receta para un pueblo

1. **Geometría**: huellas y plantas de **Catastro** (WGS84, WFS) + calles y usos de OSM.
2. **Relieve**: MDT del IGN como base del height field (el modelo lo ignora, el visor no).
3. **Textura de suelo y cubiertas**: ortofoto PNOA por teselas WMS → UV sobre las huellas.
4. **Fachadas**: Mapillary o fotos propias; si no hay, color plano por tipología
   (pizarra/teja, muro encalado) en vez de inventar con el GAN.
5. **Si se quiere "generativo"**: CityDreamer solo como *relleno de estilo* en áreas sin
   dato, y avisando de que inventará arquitectura urbana.
6. **La vía que gana en un pueblo**: **fotogrametría / Gaussian Splatting** con un móvil
   (2-3 min de vídeo en círculo). Da el pueblo **real y texturado**, y entrena en una
   3060 de 6 GB sin problema porque la escena es pequeña — justo lo que CityDreamer no puede.

## 9. El suelo: sobre el terreno real, sí — dentro del modelo, no

Pregunta justa: ¿y apoyar todo esto en el **LiDAR del suelo español**? Hay que separar dos
cosas que se confunden.

### 9.1 Lo que NO se puede: meter el MDT dentro de CityDreamer

El volumen se construye **extruyendo desde z = 0**: `extensions/extrude_tensor` rellena el
voxel `[i, j, z]` con la clase del píxel mientras `z < hf[i, j]`. No hay canal de base. Y
la red normaliza la altura con `NORMALIZE_DELIMETER = [1536, 1536, 640]`, así que un
desplazamiento de base no se lee como «este edificio está a 720 m», se lee como «este
edificio es altísimo». Además el hash-grid aprendió suelo plano: darle terreno produce
fachadas desplazadas y artefactos, no realismo. Se podría hackear (base variable por
columna en el kernel + reentrenar), pero es I+D, no un flag.

**Consecuencia práctica**: CityDreamer va a renderizar Berruguete como si la ciudad fuera
una mesa de billar. Y no lo es.

### 9.2 Lo que SÍ, y está medido

Cotas reales bajo los edificios (Terrarium/AWS, ~3,5 m/px):

| Sitio | Cotas | Desnivel bajo los edificios | Desviación típica |
|---|---|---|---|
| **Berruguete** (Madrid) | 644 – 744 m | **100 m** | 16,2 m |
| **Pontejos** (Cantabria) | 1 – 44 m | 43 m | 7,8 m |

Cien metros de desnivel no son un detalle: el barrio no está en una plataforma y el modelo
lo ignora. En Pontejos el caserío está en una zona baja y las naves suben por la ladera —
plano, eso desaparece.

### 9.3 Fuentes de relieve, por exactitud

| Fuente | Resolución | Estado | Cuándo |
|---|---|---|---|
| **MDT02 / PNOA-LiDAR (IGN)** | **2 m** | WCS (`servicios.ign.es`) **no responde desde esta red** (reset); se baja por hojas del Centro de Descargas del CNIG | La buena para España; exige descargar hoja |
| **Terrarium (AWS S3)** | ~3,5 m/px a z15 | Probado: 9/9 teselas, sin clave | Lo usado aquí: inmediato y global |
| **Copernicus DEM 30 m** | 30 m | Probado: `HTTP 200`, `image/tiff` | Fuera de España o cuando el detalle no importa |
| **PNOA ortofoto (IGN)** | hasta 25 cm | Probado: JPEG 512×512 real | Textura de suelo y cubiertas |

### 9.4 Reglas finas que hacen que se vea «exacto»

- **La cota del edificio no es la de su centroide**: se toma la **mínima** de su huella,
  para que no flote en la esquina baja, y se levanta un zócalo hasta la máxima (muro de
  contención). Sin eso, en pendiente, medio barrio flota.
- **El terreno no se dibuja como losa** en matplotlib: ordena colecciones por profundidad
  media y la pinta encima de los edificios. O `computed_zorder = False` + `zorder`
  manual, o nada.
- **Exageración vertical aparte**: en un visor con relieve, la altura del edificio y el
  relieve compiten; con 100 m de desnivel y 25 m de casas, el terreno se come la escena.

### 9.5 Reparto final

- **Visor / malla** → MDT02 (si se descarga) o Terrarium + PNOA: sitio **fiel**, con
  relieve y textura real. Corre sin GPU.
- **CityDreamer** → solo el aspecto inventado, en ventana **llana**, y declarando que el
  suelo va plano.
- **Splatting** → si el pueblo tiene que ser *él mismo*: el relieve y la textura vienen
  del vídeo, no de una capa.

## 10. Y el modelo de lenguaje, ¿qué pinta aquí?

Un LLM (qwen3.6, deepseek, el que sea) **no genera geometría 3D**. No hay prompt que
convierta "hazme Berruguete" en una malla: quien genera la ciudad es el GAN de
CityDreamer con **sus** pesos. Lo que sí hace el LLM —y es donde ha aportado en esta
sesión— es **construir el andamio**: leer el repo, detectar los tres defectos, escribir
el pipeline OSM→layout, generar el visor y montar el cuaderno de GPU. Eso es trabajo de
ingeniería real, y sale gratis.
