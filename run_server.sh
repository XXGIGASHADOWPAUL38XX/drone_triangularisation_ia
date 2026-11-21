#!/usr/bin/env bash
# Entraîne les deux modèles gelés et leurs baselines sur la RTX, puis compare.
# Emploi, à la racine du dépôt :  bash run_server.sh
set -euo pipefail
PY=${PY:-.venv/bin/python}

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA inaccessible : vérifier nvidia-smi et la wheel torch"'

"$PY" 02_ppc_classification/main.py --model cnn      --device cuda
"$PY" 02_ppc_classification/main.py --model logistic --relative --name logit
"$PY" 03_ppc_distance/main.py      --model mlp      --device cuda
"$PY" 03_ppc_distance/main.py      --model ridge    --name ridge
"$PY" 03_ppc_distance/main.py      --model constant --name constant
"$PY" 04_compare/main.py
echo "Terminé : 02/output et 03/output contiennent les runs ; tableau dans 04_compare/output/comparatif.csv."
