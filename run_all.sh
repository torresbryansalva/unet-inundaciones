#!/usr/bin/env bash
# Reproduce todo el estudio con un solo comando (pensado para Colab con GPU T4):
#   bash run_all.sh                  → todo (descarga ≈ 8 GB con los chips débiles)
#   SIN_DEBILES=1 bash run_all.sh    → sin E4/E5 (descarga ≈ 1 GB)
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/4 Dependencias"
pip install -q -r requirements.txt

echo "== 2/4 Prueba rápida (losses con máscara, métricas, forma de la U-Net)"
python -m tests.test_basico

echo "== 3/4 Datos de Sen1Floods11"
bash scripts/download_data.sh

echo "== 4/4 Experimentos, evaluación y Lima"
if [ "${SIN_DEBILES:-0}" = "1" ]; then
  python -m src.pipeline --sin-debiles
else
  python -m src.pipeline
fi

echo "Listo: tablas en results/ y figuras en results/figuras/"
