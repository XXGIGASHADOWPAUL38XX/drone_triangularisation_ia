#!/usr/bin/env bash
# Run at the repository root on the remote machine; never starts an SSH connection.
set -euo pipefail
PY=${PY:-.venv/bin/python}
DATA=${DATA:-artifacts/pilot-v2}
RUN=${RUN:-runs/night-v2}
DEVICE=${DEVICE:-cuda}
TRIALS=${TRIALS:-24}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
if [[ "$DEVICE" == cuda ]]; then
  "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA inaccessible: vérifier nvidia-smi et la wheel torch"; print(torch.cuda.get_device_name())'
fi
[[ -f "$DATA/manifest.json" && -f "$DATA/windows.parquet" ]] || { echo 'Préparer le corpus avant cet appel.' >&2; exit 2; }
mkdir -p "$RUN"
for task in classification distance; do
  if [[ ! -f "$RUN/$task/manifest.json" ]]; then
    "$PY" -m edth.train --data "$DATA/windows.parquet" --task "$task" --out "$RUN/$task" --device "$DEVICE" --trials "$TRIALS" 2>&1 | tee -a "$RUN/$task.log"
  else
    echo "$task déjà finalisé ; pas de nouvelle sélection sur le test."
  fi
  MPLCONFIGDIR="$RUN/.matplotlib" "$PY" -m edth.plot --run "$RUN/$task"
done
"$PY" -m edth.release pack --data "$DATA" --classification "$RUN/classification" --distance "$RUN/distance" --out "$RUN/bundle"
"$PY" -m edth.evaluate --bundle "$RUN/bundle" --data "$DATA/windows.parquet" | tee "$RUN/evaluation.json"
