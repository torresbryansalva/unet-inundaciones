"""Región de prueba de Lima (17-mar-2017): máscaras EMSN038, predicción por chips de 512 y métricas.

Necesita dos archivos en data/lima/ que no se pueden descargar sin intervención (ver README):
- S1_Lima_20170317_VV_VH_dB.tif   → scripts/gee_export_lima.py (Google Earth Engine)
- EMSN038_vectores.zip            → geodatabase de Copernicus EMS (descarga desde el navegador)
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import rowcol

from .config import LIMA, RESULTS
from .data import load_norm_stats, normalize
from .evaluate import show_errores
from .metrics import WaterMetrics
from .train import DEVICE, con_semilla, load_model

S1_TIF = LIMA / 'S1_Lima_20170317_VV_VH_dB.tif'
GDB_ZIP = LIMA / 'EMSN038_vectores.zip'
MASCARAS_TIF = LIMA / 'Lima_mascaras_EMSN038.tif'
CHIP = 512
ZONAS = {                                   # [oeste, sur, este, norte]
    'Rimac (Huachipa-Carapongo)': [-76.97, -12.04, -76.82, -11.93],
    'Chillon (Comas-Carabayllo)': [-77.12, -11.97, -76.98, -11.80],
}
FIGS = RESULTS / 'figuras'


def leer_s1():
    with rasterio.open(S1_TIF) as src:
        img = src.read().astype('float32')
        perfil = src.profile
    img[img == -9999] = np.nan          # Earth Engine exporta "sin dato" como -9999
    return img, perfil


def crear_mascaras(perfil, sin_imagen):
    """Rasteriza las capas de EMSN038 sobre la grilla del S1 (−1 fuera del AOI de Copernicus o sin radar).

    La huella (Potential_Maximum_flood) tiene dos tipos de polígono en la columna `source`:
    'river' (el río) y 'maximun height' (la extensión del huaico). Se evalúan juntos y por separado.
    """
    import geopandas as gpd
    from rasterio import features

    gdb = f'/vsizip/{GDB_ZIP.resolve().as_posix()}/EMSN038_UTM18S_v0.gdb'
    pmf = gpd.read_file(gdb, layer='Potential_Maximum_flood')
    cauce = gpd.read_file(gdb, layer='WatrcrsA')
    aoi = gpd.read_file(gdb, layer='EMSN038_AOI')
    H, W = perfil['height'], perfil['width']

    def rasterizar(g):
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        lineas = g.geom_type.str.contains('Line')
        g = g.set_geometry(g.geometry.where(~lineas, g.geometry.buffer(5))).to_crs(perfil['crs'])
        return features.rasterize(((geom, 1) for geom in g.geometry), out_shape=(H, W),
                                  transform=perfil['transform'], fill=0, dtype='uint8')

    dentro_aoi = rasterizar(aoi).astype(bool)

    def a_mascara(capa):
        lab = rasterizar(capa).astype('int8')
        lab[~dentro_aoi | sin_imagen] = -1
        return lab

    mascaras = {
        'Huella máxima': a_mascara(pmf),
        'Río (EMSN038)': a_mascara(pmf[pmf['source'] == 'river']),
        'Huaico (EMSN038)': a_mascara(pmf[pmf['source'] == 'maximun height']),
        'Cauce': a_mascara(cauce),
    }
    salida = {'driver': 'GTiff', 'height': H, 'width': W, 'count': len(mascaras), 'dtype': 'int8',
              'crs': perfil['crs'], 'transform': perfil['transform'], 'compress': 'deflate'}
    with rasterio.open(MASCARAS_TIF, 'w', **salida) as dst:
        for i, (nombre, m) in enumerate(mascaras.items(), start=1):
            dst.write(m, i)
            dst.set_band_description(i, nombre)
    print('Máscaras guardadas en', MASCARAS_TIF)


def leer_mascaras():
    with rasterio.open(MASCARAS_TIF) as src:
        return {src.descriptions[i]: src.read(i + 1) for i in range(src.count)}


@torch.no_grad()
def predecir(model, img, sin_imagen, stats):
    """Corta la escena en chips de 512 (bordes rellenos con NaN), predice cada uno y pega el resultado."""
    H, W = img.shape[1:]
    NF, NC = -(-H // CHIP), -(-W // CHIP)
    pad = np.full((2, NF * CHIP, NC * CHIP), np.nan, dtype='float32')
    pad[:, :H, :W] = img
    pred = np.zeros(pad.shape[1:], dtype=bool)
    for i in range(NF):
        for j in range(NC):
            x = pad[:, i * CHIP:(i + 1) * CHIP, j * CHIP:(j + 1) * CHIP]
            if np.isnan(x).all():
                continue
            t = torch.from_numpy(normalize(x, stats))[None].to(DEVICE)
            pred[i * CHIP:(i + 1) * CHIP, j * CHIP:(j + 1) * CHIP] = (model(t)[0, 0] > 0).cpu().numpy()
    return pred[:H, :W] & ~sin_imagen


def iou_por_chip(pred, lab):
    ious = []
    for i in range(0, lab.shape[0], CHIP):
        for j in range(0, lab.shape[1], CHIP):
            m = WaterMetrics(); m.update(pred[i:i + CHIP, j:j + CHIP], lab[i:i + CHIP, j:j + CHIP])
            if m.tp + m.fp + m.fn > 0:
                ious.append(m.compute()['iou'])
    return float(np.mean(ious)) if ious else float('nan')


def evaluar(elegido, final, semillas):
    from skimage.filters import threshold_otsu

    if not S1_TIF.exists():
        print(f'Lima: falta {S1_TIF} → se omite (ver README, sección "Datos de Lima")')
        return
    img, perfil = leer_s1()
    sin_imagen = np.isnan(img).any(axis=0)
    if not MASCARAS_TIF.exists():
        if not GDB_ZIP.exists():
            print(f'Lima: falta {GDB_ZIP} → se omite (ver README, sección "Datos de Lima")')
            return
        crear_mascaras(perfil, sin_imagen)
    mascaras = leer_mascaras()
    stats = load_norm_stats()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    recetas = {'Solo train': elegido, 'Train + valid': final}
    preds = {(receta, s): predecir(load_model(con_semilla(nombre, s)), img, sin_imagen, stats)
             for receta, nombre in recetas.items() for s in semillas}

    umbral = threshold_otsu(img[1][~sin_imagen])          # línea base: Otsu sobre VH de la escena
    otsu = (img[1] < umbral) & ~sin_imagen
    print(f'Lima · umbral de Otsu en VH: {umbral:.2f} dB')

    filas = []
    for ref, lab in mascaras.items():
        evaluable = lab != -1
        for (modelo, s), pred in [(('Otsu', None), otsu)] + list(preds.items()):
            m = WaterMetrics(); m.update(pred, lab)
            filas.append({'referencia': ref, 'modelo': modelo, 'semilla': s, **m.compute(),
                          'iou_por_chip': iou_por_chip(pred, lab), 'pct_pred_agua': 100 * pred[evaluable].mean()})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTS / 'resultados_lima.csv', index=False)
    resumen = tabla.groupby(['referencia', 'modelo'], sort=False)[
        ['iou', 'precision', 'recall', 'comision', 'iou_por_chip', 'pct_pred_agua']].agg(['mean', 'std'])
    resumen.to_csv(RESULTS / 'resumen_lima.csv')
    print('\nresumen_lima\n', resumen.round(3))

    votos = sum(p.astype('uint8') for p in preds.values())
    print(f'Lima · % de píxeles "agua" donde coinciden los {len(preds)} modelos:',
          round(100 * (votos == len(preds)).sum() / max((votos > 0).sum(), 1), 1))

    # Figuras: escena completa y mapas de error por zona (modelo final, primera semilla)
    pred_final = preds[('Train + valid', semillas[0])]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img[0], cmap='gray', vmin=-25, vmax=0)
    ax.imshow(np.ma.masked_where(~pred_final, pred_final), cmap='autumn', alpha=0.8)
    ax.contour(mascaras['Huella máxima'] == 1, levels=[0.5], colors='cyan', linewidths=0.6)
    ax.set_title('Rojo = agua según la red · cian = huella EMSN038'); ax.axis('off')
    plt.tight_layout(); plt.savefig(FIGS / 'lima_escena.png', dpi=120); plt.close()

    for k, (zona, (oeste, sur, este, norte)) in enumerate(ZONAS.items()):
        f0, c0 = rowcol(perfil['transform'], oeste, norte)
        f1, c1 = rowcol(perfil['transform'], este, sur)
        f, c = slice(max(f0, 0), f1), slice(max(c0, 0), c1)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img[0][f, c], cmap='gray', vmin=-25, vmax=0); axes[0].set_title(f'{zona} · VV'); axes[0].axis('off')
        show_errores(axes[1], pred_final[f, c], mascaras['Huella máxima'][f, c], 'Red vs huella máxima')
        show_errores(axes[2], pred_final[f, c], mascaras['Cauce'][f, c], 'Red vs cauce')
        show_errores(axes[3], otsu[f, c], mascaras['Huella máxima'][f, c], 'Otsu vs huella máxima')
        plt.tight_layout(); plt.savefig(FIGS / f'lima_errores_{k + 1}.png', dpi=110); plt.close()
