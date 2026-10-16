# Segmentación de agua de inundación con U-Net en Sentinel-1: Sen1Floods11 y Lima 2017

Examen parcial de Redes Neuronales (UNI FIIS, 2026-II), Pregunta 2.

Entrenamos una **U-Net propia** para segmentar agua de inundación en imágenes de radar Sentinel-1 (bandas VV y VH, en dB) con los chips etiquetados a mano de **Sen1Floods11**. Luego medimos cómo generaliza a un evento que la red nunca vio (**Bolivia**) y a una región de prueba que armamos nosotros: **Lima, marzo de 2017** (Niño Costero), con las etiquetas de Copernicus EMS **EMSN038**.

## Ejecutar todo con un solo comando

En **Google Colab con GPU T4** (*Entorno de ejecución → Cambiar tipo → T4*):

```bash
!git clone https://github.com/<usuario>/<repo>.git
%cd <repo>
!bash run_all.sh
```

`run_all.sh` hace, en orden:

1. Instala las dependencias (`requirements.txt`).
2. Corre una prueba rápida de las losses con máscara, las métricas y la U-Net (`tests/test_basico.py`).
3. Descarga Sen1Floods11 desde el bucket público `gs://sen1floods11/v1.1` (`scripts/download_data.sh`).
4. Entrena y evalúa todo (`python -m src.pipeline`). Las tablas quedan en `results/` y las figuras en `results/figuras/`.

| Variante | Comando | Descarga | Tiempo aprox. (T4) |
|---|---|---|---|
| Completo (con E4/E5, etiquetas débiles) | `bash run_all.sh` | ≈ 8 GB | ≈ 2–3 h |
| Sin etiquetas débiles | `SIN_DEBILES=1 bash run_all.sh` | ≈ 1 GB | ≈ 1 h |

Los experimentos que ya tienen `outputs/<nombre>/history.csv` **no se vuelven a entrenar**. Si la sesión se corta, al volver a ejecutar el comando se **retoma** desde la última época guardada.

### Reutilizar los modelos ya entrenados (Google Drive)

Las rutas se cambian con variables de entorno (`src/config.py`). Para usar los datos y los checkpoints de nuestro Drive sin reentrenar:

```bash
!FLOODS_DATA=/content/drive/MyDrive/tarea2_floods/data/sen1floods11 \
 FLOODS_OUT=/content/drive/MyDrive/tarea2_floods/outputs \
 python -m src.pipeline --pasos final evaluacion lima
```

El paso `final` no reentrena lo que ya existe: solo vuelve a tomar las decisiones con valid y escribe `results/eleccion.json`, que usan los pasos siguientes.

### Datos de Lima (paso manual)

Estos dos archivos no se pueden descargar automáticamente. Van en `data/lima/`. Si faltan, el pipeline **omite Lima** y avisa.

| Archivo | Cómo obtenerlo |
|---|---|
| `S1_Lima_20170317_VV_VH_dB.tif` | `python scripts/gee_export_lima.py --proyecto <proyecto-earth-engine>`. Exporta a Drive el Sentinel-1 del 17-mar-2017 (S1B, órbita descendente, track 171, VV+VH, 10 m, EPSG:4326) |
| `EMSN038_vectores.zip` | Bajar `EMSN038_Geodatabase.zip` (2.3 GB) desde https://mapping.emergency.copernicus.eu/activations/EMSN038/ y quitarle los rásters con `python scripts/make_emsn038_light_zip.py EMSN038_Geodatabase.zip data/lima/EMSN038_vectores.zip` (≈ 98 MB) |

## Estructura

```
configs/experimentos.yaml   experimentos E1–E5, receta base, regla de mejora y semillas
src/
  config.py                 rutas (cambiables con variables de entorno) y semilla
  data.py                   lectura, exclusiones, normalización (solo con train), chips débiles, Dataset
  model.py                  U-Net con attention gates opcionales
  losses.py                 BCE, Dice y BCE+Dice que ignoran los píxeles sin dato (−1)
  metrics.py                IoU, F1, precisión, recall, omisión, comisión (clase agua)
  train.py                  entrenamiento con AMP, retome desde last.pt, carga de modelos
  evaluate.py               tablas de valid, test, Bolivia, por evento, por chip, semillas
  lima.py                   máscaras EMSN038, predicción por chips de 512 y métricas en Lima
  pipeline.py               orquestador (pasos: ablacion, final, evaluacion, lima)
scripts/                    descarga de Sen1Floods11, export de Earth Engine, zip liviano de EMSN038
tests/test_basico.py        pruebas rápidas en CPU
notebooks/                  exploración y desarrollo (Colab + Drive), ver abajo
results/                    tablas CSV y figuras (las genera el pipeline)
```

### Notebooks

Son el registro del desarrollo, pensados para Colab con Google Drive. El código reproducible es `src/`.

| Notebook | Contenido |
|---|---|
| `01_exploracion.ipynb` | Splits, metadata, un chip por dentro, inventario de los 446 chips |
| `02b_ablaciones.ipynb` | E1–E5: losses y etiquetas débiles |
| `02_modelado.ipynb` | Camino final: E3, E6 (Attention U-Net), E7 (100 épocas), modelo final, test, Bolivia y semillas |
| `03_lima.ipynb` | Región de prueba de Lima: export de Earth Engine, etiquetas EMSN038, evaluación |

## Metodología (resumen)

- **Datos:** 446 chips dorados de 512×512 a 10 m. Se excluyen 7 inservibles (6 de Ghana con ≥ 99 % sin etiqueta y `Paraguay_34417` sin imagen), así que quedan 250 / 86 / 88 / 15 (train / valid / test / Bolivia). Los píxeles sin radar (NaN) se marcan como −1 y no cuentan ni en la loss ni en las métricas.
- **Normalización:** recorte a los percentiles 0.1–99.9 y estandarización por banda, **solo con train**. La misma normalización se aplica a Bolivia y a Lima.
- **Modelo:** U-Net de 4 niveles (32–256 filtros, ≈ 7.8 M de parámetros), 2 canales de entrada, salida en logits.
- **Entrenamiento:** AdamW (lr 1e-3), coseno, 40 épocas, recortes aleatorios de 256 con espejos y rotaciones, AMP.
- **Selección:** todo se decide en **valid**. Un cambio se queda solo si mejora el IoU en más de 0.02, porque la variación entre corridas es ≈ 0.01. Test, Bolivia y Lima se miran una sola vez, al final.
- **Modelo final:** la receta ganadora reentrenada con train + valid (336 chips), con 3 semillas.
- **Línea base:** umbral de Otsu (`S1OtsuLabelHand`; en Lima, Otsu sobre VH de la escena).

## Resultados principales (IoU de la clase agua, por píxel)

**Ablación (valid):**

| Otsu | E1 BCE | E2 Dice | **E3 BCE+Dice** | E4 + débiles | E5 pre débiles → dorados | E6 Attention | E7 100 épocas |
|---|---|---|---|---|---|---|---|
| 0.507 | 0.623 | 0.640 | **0.643** | 0.577 | 0.620 | 0.643 | 0.639 |

Las etiquetas débiles no ayudan, y ni las *attention gates* ni entrenar más superan la regla de +0.02. El modelo elegido es **E3**.

**Test y Bolivia (media ± desviación de 3 semillas):**

| | Test | Bolivia (evento no visto) |
|---|---|---|
| Otsu | 0.555 | **0.718** (umbral ajustado por evento) |
| E3, solo train | 0.665 ± 0.002 | 0.669 ± 0.029 |
| E3, train + valid (final) | **0.671 ± 0.003** | 0.627 ± 0.016 |

**Lima 2017:** el IoU es ≈ 0 contra EMSN038, tanto para la red como para Otsu. La huella de Copernicus marca **lodo y depósitos del huaico**, no agua visible. El agua de los ríos Rímac y Chillón es turbulenta, va cargada de sedimento y corre por un cauce angosto, así que no se ve oscura en el radar. La mayor parte de la comisión está en **sombras de radar** de los cerros. El mar sí se detecta completo. Copernicus tampoco detectó agua estancada con Sentinel-1 en este evento (reporte técnico EMSN038).

**Tipos de fallo:**
1. Agua que no se ve oscura (vegetación inundada, agua turbulenta) → omisión.
2. Superficies lisas que no son agua (campos, suelo húmedo) → comisión.
3. Ríos finos (1–3 píxeles) → omisión.
4. Sombra de radar en laderas (Lima) → comisión.

## Reproducibilidad

- Semilla 42 en pesos, orden de los chips y aumentaciones. Las semillas 1 y 2 miden la variación entre corridas.
- En GPU, cuDNN y la precisión mixta no son 100 % deterministas: los números pueden variar ≈ 0.01 entre corridas (por eso se reportan 3 semillas).

## Referencias

- Ronneberger, Fischer y Brox (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
- Bonafilia, Tellman, Anderson e Issenberg (2020). *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1*. CVPR Workshops.
- Oktay et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL.
- Copernicus Emergency Management Service (2017). *EMSN038: Floods in Lima, Peru*. Reporte técnico y geodatabase.
- Otsu (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Trans. SMC.
