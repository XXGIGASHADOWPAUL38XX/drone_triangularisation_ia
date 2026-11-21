"""Étape 3 : entraîner et évaluer le régresseur de distance (si le drone est présent).

  python main.py                                        # MLP gelé (valeur HPO)
  python main.py --model ridge --name ridge
  python main.py --model constant --name constant
  python main.py --model mlp --features b48 --name mlp-b48

Entrée  input/windows-<rate>.parquet  (posé par l'étape 01)
Sortie  output/<nom>/ : model.joblib, metrics.json (dev + holdout, MAE bornée
        et non bornée), config.json, cv.csv, curves.json, learning-curves.png
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
from dsp import resolve_features

import ml

STAGE = Path(__file__).resolve().parent
INPUT, OUTPUT = STAGE / "input", STAGE / "output"


def main():
    default = INPUT / "windows-24000.parquet"
    if not default.is_file():
        candidates = sorted(INPUT.glob("windows-*.parquet"))
        default = candidates[0] if candidates else default
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["mlp", "ridge", "constant"], default="mlp")
    p.add_argument("--input", type=Path, default=default)
    p.add_argument("--features", default="mel,b48,rms", help="groupes plates : mel, b48, rms (constant ignore)")
    p.add_argument("--relative", action="store_true",
                   help="retirer le niveau global (contre-indiqué pour la distance)")
    p.add_argument("--name", help="nom du run (défaut : <modèle>-<taux>k)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="défaut : valeur gelée de l'HPO")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not a.input.is_file():
        raise SystemExit(f"cache introuvable : {a.input} (relancer l'étape 01)")
    a.rate = int(m.group(1)) if (m := re.search(r"(\d+)", a.input.stem)) else None
    if a.name is None:
        a.name = f"{a.model}-{a.rate // 1000}k" if a.rate else a.model
    a.out = OUTPUT
    a.resolve_features = resolve_features  # le registre des groupes vit dans l'étape 01
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA demandé mais indisponible : vérifier nvidia-smi et la wheel torch.")
    ml.experiment("distance", a)


if __name__ == "__main__":
    main()
