#!/usr/bin/env bash
# Entraîne les deux modèles gelés puis évalue le holdout, sur la machine RTX.
# Emploi, à la racine du dépôt :  bash run_server.sh
# Personnalisation : CACHE=... OUT=... PY=... bash run_server.sh
set -euo pipefail
PY=${PY:-.venv/bin/python}
CACHE=${CACHE:-artifacts/windows-24000.parquet}
OUT=${OUT:-runs/final-24k}

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA inaccessible : vérifier nvidia-smi et la wheel torch"'

"$PY" 01_train.py --task classification --model cnn --device cuda --cache "$CACHE" --out "$OUT/cls-cnn"
"$PY" 02_evaluate.py --run "$OUT/cls-cnn" --cache "$CACHE"
"$PY" 01_train.py --task distance --model mlp --device cuda --cache "$CACHE" --out "$OUT/dist-mlp"
"$PY" 02_evaluate.py --run "$OUT/dist-mlp" --cache "$CACHE"
"$PY" 03_compare.py --runs-dir "$OUT" --out "$OUT/comparatif.csv"
echo "Terminé : $OUT (les deux modèles gelés + évaluation du holdout)."
