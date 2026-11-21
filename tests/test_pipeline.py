"""Tests du pipeline : parité 24 kHz historique, intégrité des splits, smoke des modèles."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data as datamod
from audio import DROPOUT_SAMPLES, Decimator, antialias_fir, iter_windows
from features import FLAT_COLS, LOGMEL, n_frames, resolve_features, window_features
from models import Constant, Linear, Mlp, CnnPresence, weights

ROOT = Path(__file__).resolve().parent.parent
COLLECTOR = Path('/home/noe/Projects/EDTH_data-collector/data')
PILOT = ROOT / 'artifacts/pilot-v2/windows.parquet'


# ---------------------------------------------------------------- audio/features

def test_fir_24k_est_le_reference_historique():
    fir = antialias_fir(24000)
    ref = np.sinc(2 * 11000 / 48000 * (np.arange(161) - 80)) * np.kaiser(161, 8.6)
    ref /= ref.sum()
    assert fir.shape == (161,)
    assert np.array_equal(fir, ref)
    assert antialias_fir(48000) is None


def test_decimateur_donne_le_meme_resultat_en_un_ou_plusieurs_appels():
    rng = np.random.default_rng(0)
    pcm = rng.integers(-3000, 3000, 50000, dtype=np.int16)
    once = Decimator(24000).process(pcm)
    dec = Decimator(24000)
    parts = [dec.process(pcm[i:i + 733]) for i in range(0, len(pcm), 733)]
    assert np.array_equal(np.concatenate(parts), once)


@pytest.mark.parametrize('rate,frames', [(16000, 14), (24000, 22), (48000, 45)])
def test_nombre_de_trames_par_taux(rate, frames):
    assert n_frames(rate) == frames


def test_features_fenetres_synthetiques():
    rng = np.random.default_rng(1)
    for rate in (16000, 24000, 48000):
        t = np.arange(rate) / rate
        pcm = (12000 * np.sin(2 * np.pi * 5000 * t)).astype(np.int16) + rng.integers(-50, 50, rate, dtype=np.int16)
        f = window_features(pcm, rate)
        assert set(FLAT_COLS) <= set(f)
        assert f[LOGMEL].shape == (32 * n_frames(rate),)
        assert np.isfinite(f[LOGMEL]).all() and np.isfinite([f['b4_8'], f['rms_db']]).all()
        assert f[LOGMEL].max() <= f['rms_db'] + 5.  # un filtre mel ne dépasse pas trop le niveau global
        # détermination : même fenêtre, mêmes features
        assert np.array_equal(f[LOGMEL], window_features(pcm, rate)[LOGMEL])


def test_garde_apres_coupure_numerique():
    rate = 24000
    pcm = np.ones(10 * rate, dtype=np.int16) * 100
    pcm[2 * rate:2 * rate + DROPOUT_SAMPLES] = 0  # coupure de 40 ms à 48 kHz
    starts = [i for i, _ in iter_windows(pcm, rate)]
    # fin de coupure en échantillons cibles (facteur 2) + garde de 500 ms
    allowed = rate + DROPOUT_SAMPLES // 2 + rate // 2
    assert starts, 'aucune fenêtre après la coupure'
    assert all(i >= allowed for i in starts)      # rien ne chevauche ni ne suit de trop près
    assert min(starts) == 42000                  # première grille admise (pas 250 ms)


def test_resolve_features():
    assert resolve_features('mel,b48,rms') == FLAT_COLS
    assert resolve_features('b48') == ['b4_8']
    assert resolve_features('') == FLAT_COLS
    with pytest.raises(ValueError):
        resolve_features('mel,zzz')


# ---------------------------------------------------------------- splits

def _meta_rows():
    rows = []
    for g in range(8):
        for k, dist in enumerate([None, 1., 3., 5.]):
            # groupes espacés de 10 s : la tolérance de 1 s ne doit pas les fusionner
            rows.append(dict(sample_id=f'{g}-{k}', device_id=f'D{g % 2}', start_utc=10. * g + 0.1 * k,
                             drone_present=dist is not None, distance_m=dist, duration_s=0.5, valid_windows=20))
    return rows


def test_partitions_groupees_sans_fuite():
    from importlib import util
    spec = util.spec_from_file_location('prepare', ROOT / '00_prepare.py')
    prepare = util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    rows = _meta_rows()
    prepare.events(rows)
    prepare.assign_splits(rows)
    part = pd.DataFrame(rows)
    assert (part.groupby('group_id').split.nunique() == 1).all()  # aucun groupe ne traverse dev/test
    assert part[part.split == 'test'].drone_present.nunique() == 2  # classes présentes au holdout
    for f in part[part.split == 'dev'].fold.unique():
        tr = part[(part.split == 'dev') & (part.fold != f)]
        assert tr.drone_present.nunique() == 2


def test_controles_detectent_les_incoherences():
    df = pd.DataFrame(dict(
        sample_id=['a', 'a'], device_id=['D', 'D'], group_id=['g', 'g'], split=['dev', 'test'],
        fold=[0, -1], drone_present=[True, True], distance_m=[2., 2.], f0=[1., 1.]))
    with pytest.raises(ValueError):
        _load_frame(df)


def _load_frame(df):
    # data.load attend un parquet : passer par un fichier temporaire
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'w.parquet'
        df.to_parquet(p)
        return datamod.load(p)


# ---------------------------------------------------------------- modèles

def _corpus(logmel_len=32 * 22, n_takes=12):
    rng = np.random.default_rng(7)
    rows = []
    for take in range(n_takes):
        dist = None if take % 4 == 0 else float(1 + (take % 3))
        level = -60. if dist is None else -15. - (dist or 0)
        for w in range(3 + take % 2):
            rows.append(dict(sample_id=f't{take}', device_id=f'D{take % 2}', group_id=f'g{take // 4}',
                             split='dev' if take < 9 else 'test', fold=int(take % 3) if take < 9 else -1,
                             drone_present=dist is not None, distance_m=dist,
                             logmel=level + rng.normal(0, .1, logmel_len).astype('float32'),
                             **{c: level + rng.normal(0, .1) for c in FLAT_COLS}))
    return pd.DataFrame(rows)


def test_mlp_distance_smoke_et_bornage():
    df = _corpus()
    dev = datamod.dev(datamod.task_frame(df, 'distance'))  # exclut les prises négatives (NaN)
    model = Mlp('distance', width=4, lr=.01).fit(dev, dev.iloc[:3], epochs=2, patience=2, cols=FLAT_COLS)
    pred = model.predict(dev)
    assert np.isfinite(pred).all() and (pred >= 1.).all() and (pred <= 5.).all()
    assert np.isfinite(model.predict(dev, bounded=False)).all()  # la sortie non bornée reste calculable
    assert weights(dev).sum() == pytest.approx(dev.sample_id.nunique())  # masse 1 par prise


def test_logistique_et_constant():
    df = _corpus()
    dev = datamod.dev(df)
    logit = Linear('classification', kind='logistic').fit(dev, cols=FLAT_COLS)
    p = logit.predict(dev)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    c = Constant()
    assert np.allclose(c.fit(dev).predict(dev), 3.0)


def test_cnn_smoke():
    df = _corpus()
    dev = datamod.dev(df)
    model = CnnPresence(channels=4).fit(dev, dev.iloc[:4], epochs=1, patience=1, seed=0)
    p = model.predict(dev)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    assert 0 < model.parameter_count() < 2000
    assert model.best_epoch == 1


# ---------------------------------------------------------------- intégration réelle

@pytest.mark.skipif(not (COLLECTOR / 'samples').is_dir() or not PILOT.is_file(),
                    reason='corpus local ou cache historique absent')
def test_parite_cache_24k_contre_pilot_v2(tmp_path):
    out = tmp_path / 'artifacts'
    out.mkdir()
    subprocess.run([sys.executable, '00_prepare.py', '--local', str(COLLECTOR),
                    '--rate', '24000', '--out-dir', str(out)], check=True, cwd=ROOT)
    old = pd.read_parquet(PILOT).sort_values(['sample_id', 'start_sample_24k']).reset_index(drop=True)
    new = pd.read_parquet(out / 'windows-24000.parquet').sort_values(['sample_id', 'start_sample']).reset_index(drop=True)
    assert len(old) == len(new) == 9748
    assert (old.split.values == new.split.values).all()
    assert (old.fold.values == new.fold.values).all()
    assert (old.group_id.values == new.group_id.values).all()
    assert (old.start_sample_24k.values == new.start_sample.values).all()
    for c in FLAT_COLS:
        assert np.array_equal(old[c].to_numpy(float), new[c].to_numpy(float)), c
    lm_old = np.stack([np.asarray(v, dtype=np.float32) for v in old[LOGMEL]])
    lm_new = np.stack([np.asarray(v, dtype=np.float32) for v in new[LOGMEL]])
    assert np.array_equal(lm_old, lm_new)
