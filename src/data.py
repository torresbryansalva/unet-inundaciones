"""Lectura de Sen1Floods11, exclusiones, normalización (solo con train), aumentaciones y Dataset."""
import json

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .config import DATA, OUT, SEED, SPLITS, WEAK


def read_chip(img_path, label_path):
    """Imagen cruda (2, 512, 512) en dB con NaN, y máscara int8 con -1 / 0 / 1."""
    with rasterio.open(img_path) as src:
        img = src.read().astype('float32')
    with rasterio.open(label_path) as src:
        lab = src.read(1).astype('int8')
    return img, lab


def load_chips():
    """Tabla de chips dorados y CACHE {chip: (img, máscara)}.

    - Se excluyen los chips inservibles: ≥ 99 % de la máscara en -1 (6 de Ghana) o imagen ≥ 99 % NaN
      (Paraguay_34417). Quedan 250 / 86 / 88 / 15.
    - Los píxeles sin medición de radar (NaN) pasan a -1 en la máscara: no se evalúan.
    """
    parts = []
    for s in SPLITS:
        df = pd.read_csv(DATA / 'splits' / f'flood_{s}_data.csv', header=None, names=['s1_file', 'label_file'])
        df['split'] = s
        parts.append(df)
    chips = pd.concat(parts, ignore_index=True)
    chips['chip'] = chips['s1_file'].str.replace('_S1Hand.tif', '', regex=False)
    chips['evento'] = chips['chip'].str.split('_').str[0]

    cache, filas = {}, []
    for r in tqdm(chips.itertuples(), total=len(chips), desc='Leyendo chips'):
        img, lab = read_chip(DATA / 'S1Hand' / r.s1_file, DATA / 'LabelHand' / r.label_file)
        filas.append({'pct_agua': 100 * (lab == 1).sum() / max((lab != -1).sum(), 1),
                      'pct_sin_dato': 100 * (lab == -1).mean(),
                      'pct_nan': 100 * np.isnan(img[0]).mean()})
        lab[np.isnan(img).any(axis=0)] = -1
        cache[r.chip] = (img, lab)
    chips = pd.concat([chips, pd.DataFrame(filas)], axis=1)

    chips['excluir'] = (chips['pct_sin_dato'] >= 99) | (chips['pct_nan'] >= 99)
    for chip in chips.loc[chips['excluir'], 'chip']:
        del cache[chip]
    print('Excluidos:', chips.loc[chips['excluir'], 'chip'].tolist())
    chips = chips[~chips['excluir']].reset_index(drop=True)
    print('Chips por split:', chips['split'].value_counts().to_dict())
    return chips, cache


def compute_norm_stats(chips, cache):
    """Percentiles 0.1–99.9 y media/desviación por banda, con 20,000 píxeles por chip de train."""
    rng = np.random.default_rng(SEED)
    muestras = []
    for chip in chips.loc[chips['split'] == 'train', 'chip']:
        px = cache[chip][0].reshape(2, -1)
        px = px[:, ~np.isnan(px).any(axis=0)]
        idx = rng.choice(px.shape[1], size=min(20000, px.shape[1]), replace=False)
        muestras.append(px[:, idx])
    muestras = np.concatenate(muestras, axis=1)
    lo = np.percentile(muestras, 0.1, axis=1)
    hi = np.percentile(muestras, 99.9, axis=1)
    recortado = np.clip(muestras, lo[:, None], hi[:, None])
    stats = {'lo': lo.tolist(), 'hi': hi.tolist(),
             'mean': recortado.mean(axis=1).tolist(), 'std': recortado.std(axis=1).tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(OUT / 'norm_stats.json', 'w'), indent=2)     # se reutiliza para Bolivia y Lima
    return stats


def load_norm_stats():
    return json.load(open(OUT / 'norm_stats.json'))


def normalize(img, stats):
    """dB crudo → recorta extremos, estandariza por banda y pone 0 donde no hay dato."""
    lo, hi = np.array(stats['lo'])[:, None, None], np.array(stats['hi'])[:, None, None]
    mean, std = np.array(stats['mean'])[:, None, None], np.array(stats['std'])[:, None, None]
    x = (np.clip(img, lo, hi) - mean) / std
    return np.nan_to_num(x, nan=0.0).astype('float32')


def load_weak(chips):
    """Chips débiles sin Bolivia y sin solape geográfico > 1 % con valid / test / bolivia (evita fuga de datos)."""
    archivos = sorted((WEAK / 'S1Weak').glob('*_S1Weak.tif'))
    weak = pd.DataFrame({'chip': [a.name.replace('_S1Weak.tif', '') for a in archivos]})
    weak = weak[~weak['chip'].str.startswith('Bolivia_')].reset_index(drop=True)
    weak['evento'] = weak['chip'].str.split('_').str[0]

    def bounds(path):
        with rasterio.open(path) as src:
            return tuple(src.bounds)

    evaluacion = chips[chips['split'].isin(['valid', 'test', 'bolivia'])]
    gb = np.array([bounds(DATA / 'S1Hand' / f) for f in tqdm(evaluacion['s1_file'], desc='Límites dorados')])
    wb = np.array([bounds(WEAK / 'S1Weak' / f'{c}_S1Weak.tif') for c in tqdm(weak['chip'], desc='Límites débiles')])
    ancho = np.minimum(wb[:, None, 2], gb[None, :, 2]) - np.maximum(wb[:, None, 0], gb[None, :, 0])
    alto = np.minimum(wb[:, None, 3], gb[None, :, 3]) - np.maximum(wb[:, None, 1], gb[None, :, 1])
    comun = np.clip(ancho, 0, None) * np.clip(alto, 0, None)
    weak['solape_eval'] = comun.max(axis=1) / ((wb[:, 2] - wb[:, 0]) * (wb[:, 3] - wb[:, 1]))

    print(f'Chips débiles: {len(weak)} · con solape > 1 % (excluidos): {(weak["solape_eval"] > 0.01).sum()}')
    return weak[weak['solape_eval'] <= 0.01].reset_index(drop=True)


class Datos:
    """Todo lo que necesita el entrenamiento: tabla de chips, CACHE en RAM, normalización y (opcional) débiles."""

    def __init__(self, con_debiles=False):
        self.chips, self.cache = load_chips()
        self.stats = compute_norm_stats(self.chips, self.cache)
        self.weak = load_weak(self.chips) if con_debiles else None

    def ids(self, *splits):
        return self.chips.loc[self.chips['split'].isin(splits), 'chip'].tolist()

    def get(self, chip):
        """Dorados: desde la RAM. Débiles: desde el disco en cada lectura (son 4,000+, no caben en RAM)."""
        if chip in self.cache:
            return self.cache[chip]
        img, lab = read_chip(WEAK / 'S1Weak' / f'{chip}_S1Weak.tif',
                             WEAK / 'S1OtsuLabelWeak' / f'{chip}_S1OtsuLabelWeak.tif')
        lab[np.isnan(img).any(axis=0)] = -1
        return img, lab


def augment(img, lab, rng):
    if rng.random() < 0.5:
        img, lab = img[:, :, ::-1], lab[:, ::-1]                     # espejo horizontal
    if rng.random() < 0.5:
        img, lab = img[:, ::-1, :], lab[::-1, :]                     # espejo vertical
    k = int(rng.integers(4))
    return np.rot90(img, k, axes=(1, 2)), np.rot90(lab, k)          # rotación 0/90/180/270°


def random_crop(img, lab, size, rng):
    H, W = lab.shape
    y, x = int(rng.integers(0, H - size + 1)), int(rng.integers(0, W - size + 1))
    return img[:, y:y + size, x:x + size], lab[y:y + size, x:x + size]


class FloodDataset(Dataset):
    """Train: espejos, rotaciones y recorte aleatorio. Valid/test: chip completo de 512 sin cambios."""

    def __init__(self, ids, datos, train=False, crop=None):
        self.ids = list(ids)
        self.datos = datos
        self.train = train
        self.crop = crop

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img, lab = self.datos.get(self.ids[i])
        img = normalize(img, self.datos.stats)
        if self.train:
            rng = np.random.default_rng(int(torch.randint(2**31, (1,))))    # semilla desde torch → reproducible
            img, lab = augment(img, lab, rng)
            if self.crop:
                img, lab = random_crop(img, lab, self.crop, rng)
        img = torch.from_numpy(np.ascontiguousarray(img))
        lab = torch.from_numpy(np.ascontiguousarray(lab).astype('int64'))
        return img, lab
