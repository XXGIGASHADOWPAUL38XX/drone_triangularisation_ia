"""Tests : parité 24 kHz historique, splits sans fuite, smoke des modèles, étape 01 complète."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                     # ml.py
sys.path.insert(0, str(ROOT / "01_ppc_features"))  # dsp.py
import dsp
import ml

COLLECTOR = Path("/home/noe/Projects/EDTH_data-collector/data")
PILOT = ROOT / "artifacts/pilot-v2/windows.parquet"


# ---------------------------------------------------------------- audio / features

def test_fir_24k_est_la_reference_historique():
    fir = dsp.antialias_fir(24000)
    ref = np.sinc(2 * 11000 / 48000 * (np.arange(161) - 80)) * np.kaiser(161, 8.6)
    ref /= ref.sum()
    assert np.array_equal(fir, ref)
    assert dsp.antialias_fir(48000) is None


def test_decimateur_donne_le_meme_resultat_en_un_ou_plusieurs_appels():
    pcm = np.random.default_rng(0).integers(-3000, 3000, 50000, dtype=np.int16)
    once = dsp.Decimator(24000).process(pcm)
    dec = dsp.Decimator(24000)
    parts = [dec.process(pcm[i:i + 733]) for i in range(0, len(pcm), 733)]
    assert np.array_equal(np.concatenate(parts), once)


@pytest.mark.parametrize("rate,frames", [(16000, 14), (24000, 22), (48000, 45)])
def test_nombre_de_trames_par_taux(rate, frames):
    assert dsp.n_frames(rate) == frames


def test_features_fenetres_synthetiques():
    rng = np.random.default_rng(1)
    for rate in (16000, 24000, 48000):
        t = np.arange(rate) / rate
        pcm = (12000 * np.sin(2 * np.pi * 5000 * t)).astype(np.int16) + rng.integers(-50, 50, rate, dtype=np.int16)
        f = dsp.window_features(pcm, rate)
        assert set(dsp.FLAT_COLS) <= set(f)
        assert f[dsp.LOGMEL].shape == (32 * dsp.n_frames(rate),)
        assert np.isfinite(f[dsp.LOGMEL]).all()
        assert np.array_equal(f[dsp.LOGMEL], dsp.window_features(pcm, rate)[dsp.LOGMEL])  # déterminisme


def test_garde_apres_coupure_numerique():
    rate = 24000
    pcm = np.ones(10 * rate, dtype=np.int16) * 100
    pcm[2 * rate:2 * rate + dsp.DROPOUT] = 0  # coupure de 40 ms à 48 kHz
    starts = [i for i, _ in dsp.iter_windows(pcm, rate)]
    allowed = rate + dsp.DROPOUT // 2 + rate // 2  # fin de coupure (taux cible) + garde 500 ms
    assert starts and all(i >= allowed for i in starts)
    assert min(starts) == 42000  # première grille admise (pas 250 ms)


def test_resolve_features():
    assert dsp.resolve_features("mel,b48,rms") == dsp.FLAT_COLS
    assert dsp.resolve_features("b48") == ["b4_8"]
    assert dsp.resolve_features("") == dsp.FLAT_COLS
    with pytest.raises(ValueError):
        dsp.resolve_features("mel,zzz")


# ---------------------------------------------------------------- splits

def _meta_rows():
    rows = []
    for g in range(8):
        for k, dist in enumerate([None, 1., 3., 5.]):
            # groupes espacés de 10 s : la tolérance de 1 s ne doit pas les fusionner
            rows.append(dict(sample_id=f"{g}-{k}", device_id=f"D{g % 2}", start_utc=10. * g + 0.1 * k,
                             drone_present=dist is not None, distance_m=dist, duration_s=0.5, valid_windows=20))
    return rows


def test_partitions_groupees_sans_fuite():
    from importlib import util
    spec = util.spec_from_file_location("features_main", ROOT / "01_ppc_features" / "main.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = _meta_rows()
    module.events(rows)
    module.assign_splits(rows)
    part = pd.DataFrame(rows)
    assert (part.groupby("group_id").split.nunique() == 1).all()  # aucun groupe ne traverse dev/test
    assert part[part.split == "test"].drone_present.nunique() == 2
    for f in part[part.split == "dev"].fold.unique():
        assert part[(part.split == "dev") & (part.fold != f)].drone_present.nunique() == 2


def test_controles_detectent_les_incoherences(tmp_path):
    df = pd.DataFrame(dict(sample_id=["a", "a"], device_id=["D", "D"], group_id=["g", "g"],
                           split=["dev", "test"], fold=[0, -1], drone_present=[True, True],
                           distance_m=[2., 2.], f0=[1., 1.]))
    p = tmp_path / "w.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError):
        ml.load(p)


# ---------------------------------------------------------------- modèles

def _corpus(logmel_len=32 * 22, n_takes=13):
    """Corpus synthétique : groupes de 4 prises alignés sur dev/test (jamais à cheval)."""
    rng = np.random.default_rng(7)
    rows = []
    for take in range(n_takes):
        dist = None if take % 4 == 0 else float(1 + (take % 3))
        level = -60. if dist is None else -15. - (dist or 0)
        for w in range(3 + take % 2):
            rows.append(dict(sample_id=f"t{take}", device_id=f"D{take % 2}", group_id=f"g{take // 4}",
                             split="dev" if take < 8 else "test", fold=int((take // 4) % 3) if take < 8 else -1,
                             drone_present=dist is not None, distance_m=dist,
                             logmel=level + rng.normal(0, .1, logmel_len).astype("float32"),
                             **{c: level + rng.normal(0, .1) for c in dsp.FLAT_COLS}))
    return pd.DataFrame(rows)


def test_mlp_distance_smoke_et_bornage():
    dev = ml.task_frame(_corpus(), "distance")
    dev = dev[dev.split == "dev"]
    model = ml.build("distance", "mlp", 0.01, 0.15).fit(dev, dev.iloc[:3], epochs=2, patience=2, cols=dsp.FLAT_COLS)
    pred = model.predict(dev)
    assert np.isfinite(pred).all() and (pred >= 1.).all() and (pred <= 5.).all()
    assert np.isfinite(model.predict(dev, bounded=False)).all()
    assert ml.weights(dev).sum() == pytest.approx(dev.sample_id.nunique())  # masse 1 par prise


def test_baselines():
    dev = _corpus()
    dev = dev[dev.split == "dev"]
    logit = ml.build("classification", "logistic").fit(dev, cols=dsp.FLAT_COLS)
    p = logit.predict(dev)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    ridge = ml.build("distance", "ridge").fit(ml.task_frame(dev, "distance"), cols=dsp.FLAT_COLS)
    assert np.isfinite(ridge.predict(ml.task_frame(dev, "distance"))).all()
    const = ml.build("distance", "constant")
    assert np.allclose(const.fit(dev).predict(dev), 3.0)


def test_cnn_smoke():
    dev = _corpus()
    dev = dev[dev.split == "dev"]
    model = ml.build("classification", "cnn", 0.01, 0.0).fit(dev, dev.iloc[:4], epochs=1, patience=1, seed=0)
    p = model.predict(dev)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    assert 0 < model.parameter_count() < 2000


def test_experience_complete(tmp_path):
    """L'étape 02 de bout en bout sur un corpus synthétique (holdout inclus)."""
    from types import SimpleNamespace
    corpus = _corpus()
    cache = tmp_path / "windows-24000.parquet"
    corpus.to_parquet(cache)
    a = SimpleNamespace(model="mlp", input=cache, features="mel,b48,rms", relative=False,
                        name="smoke", out=tmp_path, rate=24000, device="cpu",
                        epochs=1, patience=1, lr=None, dropout=None, seed=42,
                        resolve_features=dsp.resolve_features)
    ml.experiment("classification", a)
    out = tmp_path / "smoke"
    for name in ("model.joblib", "config.json", "metrics.json", "cv.csv", "curves.json"):
        assert (out / name).is_file(), name
    result = json.loads((out / "metrics.json").read_text())
    assert result["task"] == "classification" and np.isfinite(result["dev_cv"]["balanced_accuracy"])
    assert set(result["test"]) >= {"overall", "by_device"}


# ---------------------------------------------------------------- intégration réelle

@pytest.mark.skipif(not (COLLECTOR / "samples").is_dir() or not PILOT.is_file(),
                    reason="corpus local ou cache historique absent")
def test_parite_cache_24k_contre_pilot_v2(tmp_path):
    stage = tmp_path / "01_ppc_features"
    shutil.copytree(ROOT / "01_ppc_features", stage)
    subprocess.run([sys.executable, str(stage / "main.py"), "--data", str(COLLECTOR), "--rate", "24000"],
                   check=True, cwd=stage)
    old = pd.read_parquet(PILOT).sort_values(["sample_id", "start_sample_24k"]).reset_index(drop=True)
    new = pd.read_parquet(stage / "output" / "windows-24000.parquet")
    new = new.sort_values(["sample_id", "start_sample"]).reset_index(drop=True)
    assert len(old) == len(new) == 9748
    assert (old.split.values == new.split.values).all()
    assert (old.fold.values == new.fold.values).all()
    assert (old.group_id.values == new.group_id.values).all()
    assert (old.start_sample_24k.values == new.start_sample.values).all()
    for c in dsp.FLAT_COLS:
        assert np.array_equal(old[c].to_numpy(float), new[c].to_numpy(float)), c
    lm_old = np.stack([np.asarray(v, dtype=np.float32) for v in old[dsp.LOGMEL]])
    lm_new = np.stack([np.asarray(v, dtype=np.float32) for v in new[dsp.LOGMEL]])
    assert np.array_equal(lm_old, lm_new)
