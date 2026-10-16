"""Rutas y constantes del proyecto.

Todas las rutas se pueden cambiar con variables de entorno. Por ejemplo, en Colab, para reutilizar
los datos y los modelos ya entrenados que están en Google Drive:

    FLOODS_DATA=/content/drive/MyDrive/tarea2_floods/data/sen1floods11
    FLOODS_OUT=/content/drive/MyDrive/tarea2_floods/outputs
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = Path(os.environ.get('FLOODS_DATA', ROOT / 'data' / 'sen1floods11'))    # chips dorados + splits
WEAK = Path(os.environ.get('FLOODS_WEAK', ROOT / 'data' / 'weak'))            # chips débiles (S1Weak, S1OtsuLabelWeak)
LIMA = Path(os.environ.get('FLOODS_LIMA', ROOT / 'data' / 'lima'))            # S1 de Lima + geodatabase EMSN038
OUT = Path(os.environ.get('FLOODS_OUT', ROOT / 'outputs'))                    # checkpoints e historiales
RESULTS = Path(os.environ.get('FLOODS_RESULTS', ROOT / 'results'))            # tablas CSV y figuras del informe

SEED = 42
SPLITS = ['train', 'valid', 'test', 'bolivia']
