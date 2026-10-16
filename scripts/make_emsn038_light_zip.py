"""Crea un zip liviano del geodatabase de EMSN038 con solo las capas vectoriales.

El zip original (EMSN038_Geodatabase.zip, 2.3 GB) se descarga desde el navegador en
https://mapping.emergency.copernicus.eu/activations/EMSN038/ . Casi todo su peso son 6 rásters
(cambio en Sentinel-1 y modelo de elevación) que este proyecto no usa. Sin ellos quedan ≈ 98 MB.

    python scripts/make_emsn038_light_zip.py EMSN038_Geodatabase.zip data/lima/EMSN038_vectores.zip
"""
import sys
import zipfile


def main(origen, destino):
    with zipfile.ZipFile(origen) as zi:
        grandes = {i.filename.split('/')[-1].split('.')[0] for i in zi.infolist() if i.file_size > 100e6}
        print('Tablas ráster excluidas:', sorted(grandes))
        with zipfile.ZipFile(destino, 'w', zipfile.ZIP_DEFLATED) as zo:
            for i in zi.infolist():
                if i.filename.split('/')[-1].split('.')[0] not in grandes:
                    zo.writestr(i, zi.read(i.filename))
    print('Listo:', destino)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
