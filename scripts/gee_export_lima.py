"""Exporta a Google Drive el Sentinel-1 (VV, VH en dB) de Lima del 17-mar-2017 con Google Earth Engine.

Se corre una sola vez (necesita una cuenta registrada en Earth Engine y un proyecto de Google Cloud):

    python scripts/gee_export_lima.py --proyecto ee-tarea2-floods

El archivo aparece en Drive en tarea2_floods_lima/S1_Lima_20170317_VV_VH_dB.tif (5–15 min). Hay que copiarlo a
data/lima/. Es la misma colección (COPERNICUS/S1_GRD: calibrada, corregida por terreno, en dB) con la que se
construyó Sen1Floods11, así que la imagen queda en el mismo formato que los chips de entrenamiento.
"""
import argparse

import ee

AOI = [-77.20, -12.10, -76.80, -11.75]        # [oeste, sur, este, norte]: ríos Rímac y Chillón


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proyecto', required=True, help='ID del proyecto de Earth Engine')
    args = parser.parse_args()

    ee.Authenticate()
    ee.Initialize(project=args.proyecto)
    aoi = ee.Geometry.Rectangle(AOI)

    # 17-mar-2017, Sentinel-1B, órbita descendente, track 171, VV+VH (las pasadas del 18 y 24 son solo VV)
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(aoi)
          .filterDate('2017-03-17', '2017-03-18')
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('relativeOrbitNumber_start', 171)))
    print('Escenas:', [f['id'] for f in s1.getInfo()['features']])
    lima = s1.select(['VV', 'VH']).mosaic().clip(aoi).toFloat()

    tarea = ee.batch.Export.image.toDrive(
        image=lima.unmask(-9999),               # GeoTIFF no guarda NaN de forma fiable: −9999 = sin dato
        description='S1_Lima_20170317',
        folder='tarea2_floods_lima',
        fileNamePrefix='S1_Lima_20170317_VV_VH_dB',
        region=aoi,
        crs='EPSG:4326',
        scale=10,
        maxPixels=1e9,
    )
    tarea.start()
    print('Exportación iniciada: https://code.earthengine.google.com/tasks')


if __name__ == '__main__':
    main()
