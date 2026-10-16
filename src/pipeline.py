"""Orquestador: reproduce todo el estudio en el mismo orden que los notebooks.

    python -m src.pipeline                      # todo
    python -m src.pipeline --sin-debiles        # sin E4/E5 (evita descargar 7 GB de chips débiles)
    python -m src.pipeline --pasos evaluacion lima

Pasos:
1. ablacion   → E1–E5 en valid (losses y etiquetas débiles)          → results/resultados_valid.csv
2. final      → E6 (attention) y E7 (100 épocas), elección con valid,
                modelo final con train+valid y 3 semillas              → results/eleccion.json
3. evaluacion → test, Bolivia, por evento, por chip, semillas         → results/resultados_final.csv …
4. lima       → región de prueba de Lima (si están sus datos)         → results/resultados_lima.csv

Los experimentos ya entrenados (con history.csv en outputs/) no se vuelven a entrenar.
"""
import argparse
import json

import yaml

from .config import RESULTS, ROOT
from .data import Datos

CFG = yaml.safe_load(open(ROOT / 'configs' / 'experimentos.yaml', encoding='utf-8'))
PASOS = ['ablacion', 'final', 'evaluacion', 'lima']


def paso_ablacion(datos, con_debiles):
    from .evaluate import tabla_valid
    from .train import run_experiment

    hechos = []
    for nombre, args in CFG['ablacion'].items():
        if args.get('weak_per_epoch') and not con_debiles:
            print(f'{nombre}: se omite (usa chips débiles y se pidió --sin-debiles)')
            continue
        run_experiment(nombre, datos, **args)
        hechos.append(nombre)
    tabla_valid(datos, hechos, 'resultados_valid')


def paso_final(datos):
    """Misma regla en cada decisión: lo nuevo se queda solo si supera por más de `regla_mejora` de IoU en valid."""
    from .evaluate import tabla_valid
    from .train import con_semilla, run_experiment

    regla, base_cfg = CFG['regla_mejora'], CFG['ablacion'][CFG['base']]
    run_experiment(CFG['base'], datos, **base_cfg)
    run_experiment('E6_attention', datos, **base_cfg, attention=True)
    t = tabla_valid(datos, [CFG['base'], 'E6_attention'], 'resultados_valid_E3_E6')
    att = bool(t.loc['E6_attention', 'iou'] - t.loc[CFG['base'], 'iou'] > regla)
    base = 'E6_attention' if att else CFG['base']

    e7 = f"E7_{'attention' if att else 'unet'}_{CFG['epocas_largas']}ep"
    run_experiment(e7, datos, **{**base_cfg, 'epochs': CFG['epocas_largas']}, attention=att)
    t = tabla_valid(datos, [base, e7], 'resultados_valid_epocas')
    largo = bool(t.loc[e7, 'iou'] - t.loc[base, 'iou'] > regla)
    elegido = e7 if largo else base
    epocas = CFG['epocas_largas'] if largo else base_cfg.get('epochs', 40)

    final = f'FINAL_{elegido}_train_valid'
    receta = {**base_cfg, 'epochs': epocas, 'attention': att}
    for s in CFG['semillas']:
        run_experiment(con_semilla(elegido, s), datos, **receta, seed=s)
        run_experiment(con_semilla(final, s), datos, **receta, train_splits=('train', 'valid'), seed=s)

    eleccion = {'attention': att, 'epocas': epocas, 'elegido': elegido, 'final': final, 'semillas': CFG['semillas']}
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(eleccion, open(RESULTS / 'eleccion.json', 'w'), indent=2)
    print('\nElección (con valid):', eleccion)


def leer_eleccion():
    if not (RESULTS / 'eleccion.json').exists():
        raise SystemExit('Falta results/eleccion.json: corre antes el paso "final" (--pasos final …)')
    return json.load(open(RESULTS / 'eleccion.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pasos', nargs='+', choices=PASOS, default=PASOS)
    parser.add_argument('--sin-debiles', action='store_true', help='omite E4/E5 (chips débiles)')
    args = parser.parse_args()

    datos = None
    if set(args.pasos) & {'ablacion', 'final', 'evaluacion'}:
        datos = Datos(con_debiles='ablacion' in args.pasos and not args.sin_debiles)

    if 'ablacion' in args.pasos:
        paso_ablacion(datos, con_debiles=not args.sin_debiles)
    if 'final' in args.pasos:
        paso_final(datos)
    if 'evaluacion' in args.pasos:
        from .evaluate import evaluacion_final, tabla_semillas
        e = leer_eleccion()
        evaluacion_final(datos, e['elegido'], e['final'])
        tabla_semillas(datos, e['elegido'], e['final'], e['semillas'])
    if 'lima' in args.pasos:
        from .lima import evaluar
        e = leer_eleccion()
        evaluar(e['elegido'], e['final'], e['semillas'])


if __name__ == '__main__':
    main()
