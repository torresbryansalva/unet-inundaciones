"""Pruebas rápidas en CPU (segundos). Se corren con: python -m tests.test_basico"""
import numpy as np
import torch

from src.data import normalize
from src.losses import LOSSES
from src.metrics import WaterMetrics
from src.model import UNet


def test_losses_ignoran_sin_dato():
    """Cambiar la predicción solo en un píxel -1 no debe cambiar la loss."""
    target = torch.tensor([[[1, 0, -1]]])
    logits_a = torch.tensor([[[[2.0, -2.0, 5.0]]]])
    logits_b = torch.tensor([[[[2.0, -2.0, -5.0]]]])
    for nombre, fn in LOSSES.items():
        assert torch.isclose(fn(logits_a, target), fn(logits_b, target)), nombre


def test_metricas():
    pred = np.array([True, True, False, False, True])
    target = np.array([1, 0, 1, 0, -1])            # el último no cuenta
    m = WaterMetrics()
    m.update(pred, target)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)
    assert abs(m.compute()['iou'] - 1 / 3) < 1e-9


def test_unet_forma():
    for attention in [False, True]:
        salida = UNet(attention=attention)(torch.randn(1, 2, 64, 64))
        assert salida.shape == (1, 1, 64, 64)


def test_normalize_sin_nan():
    stats = {'lo': [-25, -30], 'hi': [0, -5], 'mean': [-10, -17], 'std': [3, 3]}
    img = np.full((2, 4, 4), np.nan, dtype='float32')
    img[:, 0, 0] = [-10, -17]
    x = normalize(img, stats)
    assert not np.isnan(x).any() and x[0, 0, 0] == 0


if __name__ == '__main__':
    for nombre, fn in list(globals().items()):
        if nombre.startswith('test_'):
            fn()
            print('OK', nombre)
