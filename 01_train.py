"""Entraîne un modèle par validation croisée groupée, SANS toucher au holdout.

Emploi (une commande = une expérience) :
  python 01_train.py --task classification --model cnn   --out runs/cls-24k
  python 01_train.py --task distance      --model mlp   --out runs/dist-24k
  python 01_train.py --task classification --model logistic --relative --out runs/cls-logit
  python 01_train.py --task distance --model ridge --features b48 --out runs/dist-b48

Architectures gelées d'après l'HPO préliminaire (les hyperparamètres par
défaut sont ceux du run RTX) : CNN log-mel relative pour la détection, MLP
16 cachées sur niveaux absolus pour la distance. Le holdout `test` n'est lu
ni pour entraîner ni pour choisir : il n'est évalué que par 02_evaluate.py.

Sélection : trois plis de développement groupés par événement temporel,
poids égal par prise, scaler ajusté sur le train du pli uniquement. Score =
exactitude équilibrée pondérée (détection) ou MAE en mètres (distance),
calculé sur les prédictions croisées de tous les plis.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import data
from features import resolve_features
from models import build

# Défauts figés par l'HPO du run RTX ; les surcharger est un choix délibéré.
FROZEN = {('classification', 'cnn'): dict(lr=7.7e-3, dropout=0.15),
          ('distance', 'mlp'): dict(lr=3e-3, dropout=0.15)}


def code_digest():
    sha = hashlib.sha256()
    for name in ('audio.py', 'features.py', 'models.py', 'data.py'):
        sha.update(Path(name).read_bytes())
    return sha.hexdigest()


def build_model(a, cols):
    frozen = FROZEN.get((a.task, a.model), dict(lr=3e-3, dropout=0.15))
    lr = a.lr if a.lr is not None else frozen['lr']
    dropout = a.dropout if a.dropout is not None else frozen['dropout']
    relative = (not a.no_relative) if a.model == 'cnn' else a.relative
    params = {}
    if a.model in ('cnn', 'mlp'):
        params = dict(lr=lr, dropout=dropout, relative=relative)
    if a.model == 'mlp':
        params.update(width=a.width, depth=a.depth)
    if a.model == 'logistic':
        params = dict(strength=a.strength, relative=relative)
    if a.model == 'ridge':
        params = dict(strength=a.strength, relative=relative)
    model = build(a.task, a.model, **params)
    return model, params


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=['classification', 'distance'], required=True)
    p.add_argument('--model', required=True,
                   help='cnn | mlp | logistic (classification) ; mlp | ridge | constant (distance)')
    p.add_argument('--cache', type=Path, default=Path('artifacts/windows-24000.parquet'))
    p.add_argument('--features', default='mel,b48,rms',
                   help="groupes plates : mel, b48, rms — vide = tout. Ignoré par cnn et constant")
    p.add_argument('--relative', action='store_true',
                   help="retire le niveau global de la fenêtre (logistique/mlp/ridge)")
    p.add_argument('--no-relative', action='store_true',
                   help='désactive la soustraction de niveau du cnn (gelée ON par défaut)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--depth', type=int, default=1)
    p.add_argument('--strength', type=float, default=1.0, help='C (logistique) ou alpha (ridge)')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.relative and a.no_relative:
        p.error('--relative et --no-relative sont incompatibles')
    out = Path(a.out)
    if (out / 'config.json').is_file():
        raise SystemExit(f'{out} contient déjà un run : choisir un autre --out ou le supprimer.')
    out.mkdir(parents=True, exist_ok=True)
    if a.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise SystemExit('CUDA demandé mais indisponible : vérifier nvidia-smi et la wheel torch.')
    cols = None if a.model in ('cnn', 'constant') else resolve_features(a.features)
    frame = data.task_frame(data.load(a.cache), a.task)
    if a.model == 'cnn' and 'logmel' not in frame.columns:
        raise SystemExit("le modèle cnn exige la colonne 'logmel' du cache.")
    dev, test = data.dev(frame), data.test(frame)
    folds = sorted(int(f) for f in dev.fold.unique())
    if len(folds) < 2 or test.empty:
        raise SystemExit('il faut au moins deux plis de développement et un holdout non vide.')
    name = data.selection_metric(a.task)
    rows, pooled, curves, best_epochs = [], [], {}, []
    for f in folds:
        tr, va = dev[dev.fold != f], dev[dev.fold == f]
        if a.task == 'classification' and tr.drone_present.nunique() != 2:
            raise SystemExit(f'le train du pli {f} ne contient pas les deux classes.')
        model, params = build_model(a, cols)
        model.fit(tr, va, device=a.device, epochs=a.epochs, patience=a.patience, seed=a.seed + f, cols=cols)
        pred = model.predict(va)
        pooled.append(va.assign(pred=pred))
        m = data.report(va, pred, a.task)['overall']
        rows.append({'fold': f, name: m[name]})
        curves[f'fold_{f}'] = model.curve
        if a.model in ('cnn', 'mlp'):
            best_epochs.append(model.best_epoch)
        print(f'pli {f} : {name} = {m[name]:.4f}')
    pooled = pd.concat(pooled)
    cv = data.report(pooled, pooled.pred.to_numpy(), a.task)
    # Réajustement final sur tout le développement ; pour un réseau, le nombre
    # d'époques est la médiane des meilleurs points de contrôle des plis.
    final_epochs = a.epochs if not best_epochs else max(1, int(np.median(best_epochs)))
    final, params = build_model(a, cols)
    final.fit(dev, None, device=a.device, epochs=final_epochs, seed=a.seed, cols=cols)
    joblib.dump(final, out / 'model.joblib')
    (out / 'cv.csv').write_text(pd.DataFrame(rows).to_csv(index=False))
    (out / 'curves.json').write_text(json.dumps(
        {'folds': curves, 'final': final.curve, 'fold_best_epochs': best_epochs,
         'final_epochs': final_epochs}, indent=2) + '\n')
    with open(a.cache, 'rb') as fp:
        cache_sha = hashlib.file_digest(fp, 'sha256').hexdigest()
    config = dict(task=a.task, model=a.model, cache=a.cache.name, cache_sha256=cache_sha,
                  rate=int(a.cache.stem.rsplit('-', 1)[-1]) if a.cache.stem.rsplit('-', 1)[-1].isdigit() else None,
                  features=a.features, columns=cols, params=params, device=a.device,
                  epochs=a.epochs, patience=a.patience, final_epochs=final_epochs, seed=a.seed,
                  code_sha256=code_digest(), dev_cv=cv)
    (out / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    print(f"{a.task}/{a.model} : dev {name} = {cv['overall'][name]:.4f} -> {out} (holdout non consulté)")


if __name__ == '__main__':
    main()
