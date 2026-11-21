"""Évalue un run sur les prises réservées (holdout). Ne sert jamais à choisir.

Emploi :
  python 02_evaluate.py --run runs/cls-24k
  python 02_evaluate.py --run runs/dist-24k

Le holdout n'est lu qu'ici. Chaque question (un taux, un jeu de features)
ne doit donner lieu qu'à une seule consultation de ces prises ; comparer des
variantes se fait sur les scores de développement croisé de 03_compare.py.
Écrit dans le répertoire du run : metrics.json et learning-curves.png.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import data


def plot_curves(run, task):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves = json.loads((run / 'curves.json').read_text())
    folds = curves.get('folds', {})
    if not folds or not folds[next(iter(folds))]:
        print('pas de courbes (modèle non neuronal) : figure ignorée.')
        return
    metric = 'val_accuracy' if task == 'classification' else 'val_mae'
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for fold, rows in folds.items():
        epochs = [r['epoch'] for r in rows]
        line, = axes[0].plot(epochs, [r['train_loss'] for r in rows], label=f'Train {fold}')
        if rows and 'val_loss' in rows[0]:
            axes[0].plot(epochs, [r['val_loss'] for r in rows], '--', color=line.get_color(), label=f'Val {fold}')
        if rows and metric in rows[0]:
            axes[1].plot(epochs, [r[metric] for r in rows], label=f'Val {fold}')
    axes[0].set(xlabel='Époque', title='Loss pondérée par prise'); axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Époque', title=metric); axes[1].legend(fontsize=8)
    fig.suptitle('Validation croisée (développement) — le holdout n\'y figure pas')
    fig.savefig(run / 'learning-curves.png'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=None,
                   help='cache à évaluer (défaut : artifacts/windows-<rate>.parquet du run)')
    a = p.parse_args()
    config = json.loads((a.run / 'config.json').read_text())
    cache = a.cache
    if cache is None:
        cache = Path('artifacts') / f"windows-{config.get('rate')}.parquet"
    if not cache.is_file():
        raise SystemExit(f'cache introuvable : {cache} (préciser --cache)')
    model = joblib.load(a.run / 'model.joblib')
    task = config['task']
    frame = data.task_frame(data.load(cache), task)
    test = data.test(frame)
    result = ({'cache': cache.name, 'task': task, 'model': config['model'], 'features': config['features'],
               'scope': 'Holdout groupé de la même campagne ; seuil 0.5, pas de lissage ni hystérésis.',
               'overall_dev_cv': config['dev_cv']['overall']}
              | data.report(test, model.predict(test), task))
    if task == 'distance':
        result['unbounded'] = data.report(test, model.predict(test, bounded=False), task)['overall']
    (a.run / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot_curves(a.run, task)
    overall = result['overall']
    show = ('balanced_accuracy', 'precision', 'recall') if task == 'classification' else ('mae_m', 'bias_m')
    print(json.dumps({k: overall[k] for k in ('takes', 'windows', *show)}, indent=2))


if __name__ == '__main__':
    main()
