"""Entrenamiento: una época, evaluación, run_experiment (con retome) y carga de modelos."""
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import OUT, SEED
from .data import FloodDataset, normalize
from .losses import LOSSES
from .metrics import WaterMetrics
from .model import UNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def con_semilla(nombre, s):
    """La semilla 42 usa el nombre base; las demás llevan sufijo: E3_bce_dice_s1, E3_bce_dice_s2…"""
    return nombre if s == SEED else f'{nombre}_s{s}'


def train_one_epoch(model, loader, optimizer, loss_fn, scaler):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, loss_fn=LOSSES['bce']):
    model.eval()
    m = WaterMetrics()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
            logits = model(x)
        total += loss_fn(logits.float(), y).item() * x.size(0)
        n += x.size(0)
        m.update((logits[:, 0] > 0).cpu().numpy(), y.cpu().numpy())
    return total / n, m


def split_loader(datos, split):
    return DataLoader(FloodDataset(datos.ids(split), datos), batch_size=4, shuffle=False, num_workers=2)


def run_experiment(name, datos, loss='bce_dice', epochs=40, lr=1e-3, batch_size=8, crop=256, attention=False,
                   train_splits=('train',), weak_per_epoch=0, init_from=None, seed=SEED, force=False):
    """Entrena y guarda en OUT/<name>/ (best.pt, history.csv).

    - Si ya terminó (existe history.csv), no reentrena (force=True para reentrenar).
    - Si se cortó a mitad, retoma desde la última época guardada (last.pt).
    - Si 'valid' está en train_splits, no hay con qué elegir: se guarda la última época.
    - weak_per_epoch: chips débiles elegidos al azar que se suman a los dorados en cada época (E4, E5a).
    - init_from: experimento cuyo best.pt es el punto de partida (E5).
    """
    exp_dir = OUT / name
    if not force and (exp_dir / 'history.csv').exists():
        print(f'{name}: ya entrenado → se lee {exp_dir / "history.csv"}')
        return pd.read_csv(exp_dir / 'history.csv')
    if weak_per_epoch and datos.weak is None:
        raise ValueError(f'{name} necesita chips débiles: crea Datos(con_debiles=True)')
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n=== {name} ===')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    elige_con_valid = 'valid' not in train_splits
    gold = datos.ids(*train_splits)
    valid_loader = split_loader(datos, 'valid')

    model = UNet(attention=attention).to(DEVICE)
    if init_from:
        model.load_state_dict(torch.load(OUT / init_from / 'best.pt', map_location=DEVICE)['model'])
        print(f'Pesos iniciales: {init_from}/best.pt')
    loss_fn = LOSSES[loss]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    config = {'loss': loss, 'lr': lr, 'epochs': epochs, 'crop': crop, 'attention': attention,
              'train_splits': list(train_splits), 'weak_per_epoch': weak_per_epoch, 'init_from': init_from, 'seed': seed}

    history, best_iou, inicio = [], -1.0, 1
    if not force and (exp_dir / 'last.pt').exists():
        ck = torch.load(exp_dir / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['rng'])
        rng.bit_generator.state = ck['np_rng']
        history, best_iou, inicio = ck['history'], ck['best_iou'], ck['epoch'] + 1
        print(f'{name}: se retoma desde la época {inicio}')

    train_loader = None
    for epoch in range(inicio, epochs + 1):
        t0 = time.time()
        if train_loader is None or weak_per_epoch:         # con débiles: otra muestra al azar en cada época
            ids = gold + (rng.choice(datos.weak['chip'], weak_per_epoch, replace=False).tolist() if weak_per_epoch else [])
            train_loader = DataLoader(FloodDataset(ids, datos, train=True, crop=crop),
                                      batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler)
        fila = {'epoch': epoch, 'train_loss': train_loss}
        if elige_con_valid:
            val_loss, val_m = evaluate(model, valid_loader, loss_fn)
            fila.update({'val_loss': val_loss, **val_m.compute()})
            mejora = fila['iou'] > best_iou
        else:
            mejora = epoch == epochs
        scheduler.step()
        history.append(fila)
        if mejora:
            best_iou = fila.get('iou', best_iou)
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'config': config}, exp_dir / 'best.pt')
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': torch.get_rng_state(),
                    'np_rng': rng.bit_generator.state, 'history': history, 'best_iou': best_iou},
                   exp_dir / 'last.pt')
        info = f"val {fila['val_loss']:.4f} | IoU {fila['iou']:.4f}" if elige_con_valid else 'sin valid'
        print(f"{epoch:3d} | train {train_loss:.4f} | {info} | {time.time() - t0:.0f}s {'✓' if mejora else ''}")

    history = pd.DataFrame(history)
    history.to_csv(exp_dir / 'history.csv', index=False)
    os.remove(exp_dir / 'last.pt')
    return history


def load_model(name):
    ckpt = torch.load(OUT / name / 'best.pt', map_location=DEVICE)
    model = UNet(attention=ckpt['config'].get('attention', False)).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    return model.eval()


@torch.no_grad()
def predict(model, datos, chip):
    img, _ = datos.get(chip)
    x = torch.from_numpy(normalize(img, datos.stats))[None].to(DEVICE)
    return (model(x)[0, 0] > 0).cpu().numpy()
