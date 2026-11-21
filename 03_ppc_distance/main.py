"""Étape 3 : entraîner et évaluer le régresseur de distance (si le drone est présent).

  python main.py                                        # MLP gelé (valeur HPO)
  python main.py --model ridge --name ridge
  python main.py --model constant --name constant
  python main.py --model mlp --features handcrafted --name mlp-handcrafted

Entrée  input/windows-<rate>-bands200-avg4-descriptors.parquet
Sortie  output/<nom>/ : model.joblib, metrics.json (validation + holdout, MAE
        bornée et non bornée), config.json, curves.json, learning-curves.png
N'apprend que sur les fenêtres des prises positives à 1-5 m, en niveaux
ABSOLUS ; la sortie est bornée à [1, 5] m. Modèles : mlp (gelé), ridge,
constant.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                       # ml.py
sys.path.insert(0, str(ROOT / "01_ppc_features"))   # dsp.py : registre des features
from dsp import DEFAULT_FEATURES, FEATURE_SETS, default_window_cache, resolve_features

import ml

STAGE = Path(__file__).resolve().parent
INPUT, OUTPUT = STAGE / "input", STAGE / "output"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["mlp", "ridge", "constant"], default="mlp")
    p.add_argument("--input", type=Path)
    p.add_argument("--features", default=DEFAULT_FEATURES,
                   help=f"groupes : {', '.join(FEATURE_SETS)} (constant ignore)")
    p.add_argument("--relative", action="store_true",
                   help="retirer le niveau global (contre-indiqué pour la distance)")
    p.add_argument("--name", help="nom du run, dérivé du modèle et du cache par défaut")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="défaut : valeur gelée de l'HPO")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--hidden-units", type=int, choices=[8, 16, 32, 64], default=16)
    p.add_argument("--depth", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.input is None:
        a.input = default_window_cache(INPUT)
    if not a.input.is_file():
        raise SystemExit(f"cache introuvable : {a.input} (relancer l'étape 01)")
    m = re.search(r"(\d+)(.*)$", a.input.stem)
    a.rate = int(m.group(1)) if m else None
    if a.name is None:
        a.name = f"{a.model}-{a.rate // 1000}k{m.group(2)}" if m else a.model
        if a.model == "mlp" and (a.depth, a.hidden_units) != (1, 16):
            a.name += f"-d{a.depth}-h{a.hidden_units}"
    a.out = OUTPUT
    a.resolve_features = resolve_features  # le registre des groupes vit dans l'étape 01
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA demandé mais indisponible : vérifier nvidia-smi et la wheel torch.")
    ml.experiment("distance", a)


if __name__ == "__main__":
    main()
