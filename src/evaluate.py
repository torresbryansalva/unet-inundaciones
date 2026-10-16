"""Tablas y figuras del informe: valid (ablación), test/Bolivia, por evento, por chip y semillas."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from .config import DATA, OUT, RESULTS
from .metrics import WaterMetrics
from .train import con_semilla, evaluate, load_model, predict, split_loader

FIGS = RESULTS / 'figuras'


def otsu_mask(chip):
    with rasterio.open(DATA / 'S1OtsuLabelHand' / f'{chip}_S1OtsuLabelHand.tif') as src:
        return src.read(1) == 1


def eval_otsu(datos, split):
    m = WaterMetrics()
    for chip in datos.ids(split):
        m.update(otsu_mask(chip), datos.get(chip)[1])
    return m


def eval_split(model, datos, split):
    return evaluate(model, split_loader(datos, split))[1]


def tabla_valid(datos, nombres, archivo):
    """Otsu + cada experimento en valid, con la época del mejor checkpoint y el IoU medio de las últimas 5 épocas."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    filas = {'Otsu': eval_otsu(datos, 'valid').compute()}
    hist = {}
    for n in nombres:
        filas[n] = eval_split(load_model(n), datos, 'valid').compute()
        h = pd.read_csv(OUT / n / 'history.csv')
        hist[n] = h
        filas[n].update({'mejor_epoca': int(h.loc[h['iou'].idxmax(), 'epoch']), 'iou_ult5': h['iou'].tail(5).mean()})
    tabla = pd.DataFrame(filas).T
    tabla.to_csv(RESULTS / f'{archivo}.csv')
    print(f'\n{archivo}\n', tabla.round(4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for n, h in hist.items():
        ax1.plot(h['epoch'], h['iou'], label=n)
        ax2.plot(h['epoch'], h['recall'], label=n)
    for ax, t in [(ax1, 'iou'), (ax2, 'recall')]:
        ax.axhline(filas['Otsu'][t], color='gray', ls='--', label='Otsu')
        ax.set_title(f'{t} en valid'); ax.set_xlabel('época'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIGS / f'curvas_{archivo}.png', dpi=120); plt.close()
    return tabla


def show_errores(ax, pred, lab, title):
    rgb = np.zeros((*lab.shape, 3))
    rgb[:] = [0.89, 0.84, 0.72]                      # no agua bien detectada (beige)
    rgb[pred & (lab == 1)] = [0.12, 0.44, 0.70]      # acierto de agua (azul)
    rgb[pred & (lab == 0)] = [0.84, 0.15, 0.16]      # agua inventada (rojo)
    rgb[~pred & (lab == 1)] = [1.00, 0.55, 0.00]     # agua no vista (naranja)
    rgb[lab == -1] = [0.60, 0.60, 0.60]              # sin dato (gris)
    ax.imshow(rgb); ax.set_title(title); ax.axis('off')


def evaluacion_final(datos, elegido, final):
    """Otsu / solo train (elegido) / red final en test y Bolivia, por evento y por chip (con los 6 peores)."""
    FIGS.mkdir(parents=True, exist_ok=True)
    m_elegido, m_final = load_model(elegido), load_model(final)

    met = {}
    for split in ['test', 'bolivia']:
        met[f'Otsu · {split}'] = eval_otsu(datos, split)
        met[f'Solo train · {split}'] = eval_split(m_elegido, datos, split)
        met[f'Red · {split}'] = eval_split(m_final, datos, split)
    tabla = pd.DataFrame({k: m.compute() for k, m in met.items()}).T
    tabla.to_csv(RESULTS / 'resultados_final.csv')
    print('\nresultados_final\n', tabla.round(4))
    for split in ['test', 'bolivia']:
        met[f'Red · {split}'].confusion().to_csv(RESULTS / f'confusion_{split}.csv')

    evaluacion = datos.chips[datos.chips['split'].isin(['test', 'bolivia'])]
    filas_evento, filas_chip = {}, []
    for evento, grupo in evaluacion.groupby('evento'):
        m_otsu, m_red = WaterMetrics(), WaterMetrics()
        for r in grupo.itertuples():
            lab = datos.get(r.chip)[1]
            pred = predict(m_final, datos, r.chip)
            m_otsu.update(otsu_mask(r.chip), lab)
            m_red.update(pred, lab)
            m = WaterMetrics(); m.update(pred, lab)
            filas_chip.append({'chip': r.chip, 'split': r.split, 'pct_agua': r.pct_agua, **m.compute(),
                               'tp': m.tp, 'fp': m.fp, 'fn': m.fn})
        red = m_red.compute()
        filas_evento[evento] = {'chips': len(grupo), 'pct_agua': grupo['pct_agua'].mean(),
                                'pct_sin_dato': grupo['pct_sin_dato'].mean(), 'iou_otsu': m_otsu.compute()['iou'],
                                'iou_red': red['iou'], 'recall_red': red['recall'], 'comision_red': red['comision']}
    por_evento = pd.DataFrame(filas_evento).T.sort_values('iou_red')
    por_evento.to_csv(RESULTS / 'por_evento.csv')
    print('\npor_evento\n', por_evento.round(3))

    por_chip = pd.DataFrame(filas_chip)
    por_chip.to_csv(RESULTS / 'por_chip_test_bolivia.csv', index=False)
    con_algo = por_chip[por_chip[['tp', 'fp', 'fn']].sum(axis=1) > 0]       # métrica de los autores de Sen1Floods11
    print('IoU promedio por chip:', con_algo.groupby('split')['iou'].mean().round(4).to_dict())

    peores = por_chip[por_chip['pct_agua'] >= 1].nsmallest(6, 'iou')
    fig, axes = plt.subplots(len(peores), 3, figsize=(12, 4 * len(peores)))
    for row, r in zip(axes, peores.itertuples()):
        img, lab = datos.get(r.chip)
        row[0].imshow(img[0], cmap='gray', vmin=-25, vmax=0); row[0].set_title(f'{r.chip} · VV'); row[0].axis('off')
        row[1].imshow(img[1], cmap='gray', vmin=-30, vmax=-5); row[1].set_title('VH'); row[1].axis('off')
        show_errores(row[2], predict(m_final, datos, r.chip), lab, f'Errores · IoU {r.iou:.2f}')
    plt.tight_layout(); plt.savefig(FIGS / 'peores_chips.png', dpi=110); plt.close()


def tabla_semillas(datos, elegido, final, semillas):
    """Las 2 recetas × 3 semillas en test y Bolivia (y valid para 'solo train'): media ± desviación."""
    filas = []
    for s in semillas:
        for receta, nombre in [('Solo train', elegido), ('Train + valid', final)]:
            m = load_model(con_semilla(nombre, s))
            fila = {'receta': receta, 'semilla': s}
            if receta == 'Solo train':
                fila['iou_valid'] = eval_split(m, datos, 'valid').compute()['iou']
            for split in ['test', 'bolivia']:
                met = eval_split(m, datos, split).compute()
                fila[f'iou_{split}'], fila[f'recall_{split}'] = met['iou'], met['recall']
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTS / 'resultados_semillas.csv', index=False)
    resumen = tabla.groupby('receta')[['iou_valid', 'iou_test', 'iou_bolivia', 'recall_test', 'recall_bolivia']].agg(['mean', 'std'])
    resumen.to_csv(RESULTS / 'resumen_semillas.csv')
    print('\nresumen_semillas\n', resumen.round(4))
