#!/usr/bin/env bash
# Descarga Sen1Floods11 v1.1 desde el bucket público de Google Cloud (necesita gsutil; Colab ya lo trae).
# Si algo ya está descargado, no se vuelve a bajar.
#   SIN_DEBILES=1 bash scripts/download_data.sh   → omite los 7 GB de chips débiles (E4/E5)
set -euo pipefail

DATA=${FLOODS_DATA:-data/sen1floods11}
WEAK=${FLOODS_WEAK:-data/weak}
GCS=gs://sen1floods11/v1.1

mkdir -p "$DATA/splits"
[ -f "$DATA/splits/flood_train_data.csv" ] || gsutil -m -q cp "$GCS/splits/flood_handlabeled/*.csv" "$DATA/splits/"
[ -f "$DATA/Sen1Floods11_Metadata.geojson" ] || gsutil -q cp "$GCS/Sen1Floods11_Metadata.geojson" "$DATA/"

# Chips dorados (446): radar, máscara manual y máscara de Otsu (línea base)
for carpeta in S1Hand LabelHand S1OtsuLabelHand; do
  [ -d "$DATA/$carpeta" ] || gsutil -m -q cp -r "$GCS/data/flood_events/HandLabeled/$carpeta" "$DATA/"
  echo "$carpeta: $(ls "$DATA/$carpeta" | wc -l) archivos (esperado 446)"
done

# Chips débiles sin Bolivia (4,160): Bolivia es la región reservada para medir generalización
if [ "${SIN_DEBILES:-0}" != "1" ]; then
  for carpeta in S1Weak S1OtsuLabelWeak; do
    mkdir -p "$WEAK/$carpeta"
    gsutil -m -q rsync -x "Bolivia_.*" "$GCS/data/flood_events/WeaklyLabeled/$carpeta" "$WEAK/$carpeta"
    echo "$carpeta: $(ls "$WEAK/$carpeta" | wc -l) archivos (esperado 4160)"
  done
fi
