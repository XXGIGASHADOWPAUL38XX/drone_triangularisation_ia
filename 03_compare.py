"""Tableau comparatif des runs : le "qu'est-ce qui marche, qu'est-ce qui est mort".

Emploi :
  python 03_compare.py                    # tous les sous-dossiers de runs/
  python 03_compare.py --runs-dir runs --out runs/comparatif.csv

Chaque ligne = un run (une expérience). Les scores de développement croisé
sont comparables entre eux ; les scores de holdout ne servent qu'à confirmer
les choix déjà faits sur le développement, pas à en tester de nouveaux.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import data


def collect(runs_dir):
    rows = []
    for run in sorted(Path(runs_dir).glob('*/config.json')):
        d = run.parent
        config = json.loads(run.read_text())
        task = config['task']
        name = data.selection_metric(task)
        row = dict(run=d.name, task=task, model=config['model'], features=config['features'],
                   rate=config.get('rate'), **config.get('params', {}),
                   dev_cv=config['dev_cv']['overall'][name])
        metrics_file = d / 'metrics.json'
        if metrics_file.is_file():
            overall = json.loads(metrics_file.read_text())['overall']
            row[f'test_{name}'] = overall[name]
            row['test_takes'] = overall['takes']
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs-dir', type=Path, default=Path('runs'))
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()
    table = collect(a.runs_dir)
    if table.empty:
        raise SystemExit(f'aucun run trouvé dans {a.runs_dir} (lancer 01_train.py d\'abord).')
    cols = [c for c in ('run', 'task', 'model', 'features', 'rate', 'dev_cv',
                        'test_balanced_accuracy', 'test_mae_m', 'test_takes') if c in table]
    table = table[cols].sort_values(['task', 'dev_cv'])
    print(table.to_string(index=False))
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(a.out, index=False)
        print(f'écrit : {a.out}')


if __name__ == '__main__':
    main()
