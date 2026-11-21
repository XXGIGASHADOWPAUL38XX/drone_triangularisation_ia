"""Chargement du cache de fenêtres, contrôles d'intégrité et métriques.

Toutes les métriques pondèrent les prises à masse égale : une prise longue
ne compte pas plus qu'une prise courte. Les 9 748 fenêtres se chevauchent ;
elles ne sont pas 9 748 observations indépendantes.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, mean_absolute_error,
                             precision_score, recall_score)

from models import weights

KEYS = ['sample_id', 'device_id', 'group_id', 'split', 'fold', 'drone_present', 'distance_m']


def load(cache):
    """Lit le parquet et applique les contrôles d'intégrité, sans regarder les scores."""
    d = pd.read_parquet(cache)
    missing = [c for c in KEYS if c not in d]
    if missing:
        raise ValueError(f'colonnes manquantes dans le cache : {missing}')
    id_cols = [c for c in KEYS if c != 'distance_m']  # distance_m est NaN pour les négatives
    if d[id_cols].isna().any().any() or not d.split.isin(['dev', 'test', 'excluded']).all():
        raise ValueError('identifiants, split ou étiquettes invalides')
    if not pd.api.types.is_bool_dtype(d.drone_present):
        raise ValueError('drone_present doit être booléen')
    per_take = d.groupby('sample_id')[['group_id', 'device_id', 'drone_present', 'distance_m']]
    if (per_take.nunique(dropna=False) > 1).any().any():
        raise ValueError('métadonnées incohérentes au sein d une prise')
    if (d.groupby('group_id').split.nunique() > 1).any():
        raise ValueError('un groupe temporel traverse dev et test : fuite de données')
    folds = pd.to_numeric(d.loc[d.split == 'dev', 'fold'], errors='coerce')
    if not np.isfinite(folds).all() or (folds % 1 != 0).any():
        raise ValueError('plis de développement invalides')
    if (d[d.split == 'dev'].groupby('group_id').fold.nunique() > 1).any():
        raise ValueError('un groupe est réparti sur plusieurs plis')
    present = d.drone_present
    if d.loc[present, 'distance_m'].isna().any() or d.loc[~present, 'distance_m'].notna().any():
        raise ValueError('présence et distance incohérentes')
    return d


def task_frame(d, task):
    """Lignes utiles à la tâche. La distance n'apprend que sur les présents 1-5 m.

    La prise confirmée à 0,1 m reste utilisable pour la détection ; elle est
    hors plage pour la régression."""
    if task == 'distance':
        return d[d.drone_present & d.distance_m.between(1, 5)].copy()
    if task == 'classification':
        return d.copy()
    raise ValueError(f'tâche inconnue : {task!r}')


def dev(d):
    return d[d.split == 'dev'].copy()


def test(d):
    return d[d.split == 'test'].copy()


def _metrics(d, task):
    pred = d.pred.to_numpy()
    w = weights(d)
    out = dict(takes=int(d.sample_id.nunique()), windows=len(d))
    if task == 'distance':
        out['mae_m'] = float(mean_absolute_error(d.distance_m, pred, sample_weight=w))
        out['bias_m'] = float(np.average(pred - d.distance_m, weights=w))
    else:
        y, p = d.drone_present.to_numpy(), pred >= .5
        out['accuracy'] = float(accuracy_score(y, p, sample_weight=w))
        out['balanced_accuracy'] = (float(balanced_accuracy_score(y, p, sample_weight=w))
                                    if len(np.unique(y)) == 2 else None)
        out['precision'] = float(precision_score(y, p, sample_weight=w, zero_division=0))
        out['recall'] = float(recall_score(y, p, sample_weight=w, zero_division=0))
    return out


def report(d, pred, task):
    """Métriques globales, par téléphone et, en distance, par distance cible."""
    d = d.assign(pred=np.asarray(pred))
    out = {'overall': _metrics(d, task)}
    out['by_device'] = {str(k): _metrics(v, task) for k, v in d.groupby('device_id')}
    if task == 'distance':
        out['by_distance'] = {str(k): _metrics(v, task) for k, v in d.groupby('distance_m')}
    return out


def selection_metric(task):
    return 'mae_m' if task == 'distance' else 'balanced_accuracy'
