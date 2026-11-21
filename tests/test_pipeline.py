"""Tests : fenêtres et features, parité 24 kHz historique, splits sans fuite, smoke des modèles."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # ml.py
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


@pytest.mark.parametrize("rate,size", [(24000, 733), (16000, 733), (16000, 1024)])
def test_decimateur_donne_le_meme_resultat_en_un_ou_plusieurs_appels(rate, size):
    # 733 et 1024 ne sont pas multiples de 3 : le signe de la phase compte à 16 kHz
    pcm = np.random.default_rng(0).integers(-3000, 3000, 50000, dtype=np.int16)
    once = dsp.Decimator(rate).process(pcm)
    dec = dsp.Decimator(rate)
    parts = [dec.process(pcm[i:i + size]) for i in range(0, len(pcm), size)]
    assert np.array_equal(np.concatenate(parts), once)


@pytest.mark.parametrize("rate,frames", [(16000, 14), (24000, 22), (48000, 45)])
def test_nombre_de_trames_par_taux(rate, frames):
    assert dsp.n_frames(rate, 1000) == frames


def test_features_fenetres_synthetiques():
    rng = np.random.default_rng(1)
    for rate in (16000, 24000, 48000):
        width = dsp.window_samples(rate, 250)
        t = np.arange(width) / rate
        pcm = (12000 * np.sin(2 * np.pi * 5000 * t)).astype(np.int16) + rng.integers(-50, 50, width, dtype=np.int16)
        f = dsp.window_features(pcm, rate, window_ms=250)
        assert set(dsp.FFT_COLS) <= set(f) and "rms_db" in f
        assert f[dsp.LOGFFT].shape == (dsp.N_BINS * dsp.n_frames(rate, 250),)
        assert np.isfinite(f[dsp.LOGFFT]).all()
        assert np.array_equal(f[dsp.LOGFFT], dsp.window_features(pcm, rate, window_ms=250)[dsp.LOGFFT])  # déterminisme
        bands = dsp.window_features(pcm, rate, window_ms=250, band_level_group=dsp.DEFAULT_BAND_GROUP)
        assert set(bands) == set(dsp.BAND_LEVEL_GROUPS[dsp.DEFAULT_BAND_GROUP])
        assert np.isfinite(list(bands.values())).all()


def test_garde_apres_coupure_numerique():
    rate = 24000
    pcm = np.ones(10 * rate, dtype=np.int16) * 100
    pcm[2 * rate:2 * rate + dsp.DROPOUT] = 0  # coupure de 40 ms à 48 kHz
    starts = [i for i, _ in dsp.iter_windows(pcm, rate, window_ms=1000, stride_ms=250)]
    allowed = rate + dsp.DROPOUT // 2 + rate // 2  # fin de coupure (taux cible) + garde 500 ms
    assert starts and all(i >= allowed for i in starts)
    assert min(starts) == 42000  # première grille admise (pas 250 ms)


def test_resolve_features():
    assert dsp.resolve_features("band_levels_avg4") == dsp.BAND_LEVEL_GROUPS["band_levels_avg4"]
    assert dsp.resolve_features("rms") == ["rms_db"]
    assert dsp.resolve_features("") == dsp.BAND_LEVEL_GROUPS[dsp.DEFAULT_BAND_GROUP]
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
    assert (part.groupby("group_id").split.nunique() == 1).all()  # aucun groupe ne traverse les partitions
    for split in ("train", "validation", "test"):
        assert part[part.split == split].drone_present.nunique() == 2
    assert part[part.split == "test"].distance_m.dropna().nunique() >= 2


def test_controles_detectent_les_incoherences(tmp_path):
    df = pd.DataFrame(dict(sample_id=["a", "a"], device_id=["D", "D"], group_id=["g", "g"],
                           split=["train", "test"], drone_present=[True, True],
                           distance_m=[2., 2.], f0=[1., 1.]))
    p = tmp_path / "w.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError):
        ml.load(p)


# ---------------------------------------------------------------- modèles

def _corpus(n_takes=13, bins=10, frames=3):
    """Corpus synthétique : bandes avg4 + spectre logfft, un groupe par prise,
    aligné sur train/validation/test (jamais à cheval)."""
    rng = np.random.default_rng(7)
    band = dsp.BAND_LEVEL_GROUPS["band_levels_avg4"]
    rows = []
    for take in range(n_takes):
        dist = None if take % 4 == 0 else float(1 + (take % 3))
        level = -60. if dist is None else -15. - (dist or 0)
        split = "train" if take < 8 else ("validation" if take < 10 else "test")
        for w in range(3 + take % 2):
            rows.append(dict(sample_id=f"t{take}", device_id=f"D{take % 2}", group_id=f"g{take}",
                             split=split, drone_present=dist is not None, distance_m=dist,
                             rms_db=level + rng.normal(0, .1),
                             logfft=(level + rng.normal(0, .1, bins * frames)).astype("float32"),
                             **{c: level + rng.normal(0, .1) for c in band},
                             **{f"fft_{i}": level + rng.normal(0, .1) for i in range(bins)}))
    return pd.DataFrame(rows)


def test_mlp_distance_smoke_et_bornage():
    frame = ml.task_frame(_corpus(), "distance")
    train, valid = frame[frame.split == "train"], frame[frame.split == "validation"]
    model = ml.build("distance", "mlp", 0.01, 0.15).fit(
        train, valid, epochs=2, patience=2, cols=dsp.BAND_LEVEL_GROUPS["band_levels_avg4"])
    pred = model.predict(frame)
    assert np.isfinite(pred).all() and (pred >= 1.).all() and (pred <= 5.).all()
    assert np.isfinite(model.predict(frame, bounded=False)).all()
    assert ml.weights(train).sum() == pytest.approx(train.sample_id.nunique())  # masse 1 par prise


def test_baselines():
    frame = _corpus()
    train = frame[frame.split == "train"]
    dist = ml.task_frame(frame, "distance")
    cols = dsp.BAND_LEVEL_GROUPS["band_levels_avg4"]
    logit = ml.build("classification", "logistic").fit(train, cols=cols)
    p = logit.predict(train)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    ridge = ml.build("distance", "ridge").fit(dist[dist.split == "train"], cols=cols)
    assert np.isfinite(ridge.predict(dist)).all()
    const = ml.build("distance", "constant")
    assert np.allclose(const.fit(train).predict(train), 3.0)


def test_cnn_smoke():
    frame = _corpus()
    train, valid = frame[frame.split == "train"], frame[frame.split == "validation"]
    model = ml.build("classification", "cnn", 0.01, 0.0).fit(train, valid, epochs=1, patience=1, seed=0)
    p = model.predict(frame)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    assert 0 < model.parameter_count() < 2000


def test_experience_complete(tmp_path):
    """L'étape 02 de bout en bout sur un corpus synthétique (holdout inclus)."""
    from types import SimpleNamespace
    corpus = _corpus()
    cache = tmp_path / "windows-48000-bands200-avg4-w100-s20.parquet"
    corpus.to_parquet(cache)
    a = SimpleNamespace(model="mlp", input=cache, features="band_levels_avg4", relative=False,
                        name="smoke", out=tmp_path, rate=48000, device="cpu",
                        epochs=1, patience=1, lr=None, dropout=None, seed=42,
                        resolve_features=dsp.resolve_features)
    ml.experiment("classification", a)
    out = tmp_path / "smoke"
    for name in ("model.joblib", "config.json", "metrics.json", "curves.json"):
        assert (out / name).is_file(), name
    result = json.loads((out / "metrics.json").read_text())
    assert result["task"] == "classification"
    assert np.isfinite(result["validation"]["overall"]["balanced_accuracy"])
    assert set(result["test"]) >= {"overall", "by_device"}
    config = json.loads((out / "config.json").read_text())
    assert config["columns"] == dsp.BAND_LEVEL_GROUPS["band_levels_avg4"]
    assert config["validation"]["balanced_accuracy"] == result["validation"]["overall"]["balanced_accuracy"]


# ---------------------------------------------------------------- intégration réelle

@pytest.mark.skipif(not (COLLECTOR / "samples").is_dir() or not PILOT.is_file(),
                    reason="corpus local ou cache historique absent")
def test_parite_cache_24k_contre_pilot_v2(tmp_path):
    """Le dsp actuel redonne la grille 1 s / 250 ms, les partitions migrées et le RMS du pilot."""
    stage = tmp_path / "01_ppc_features"
    shutil.copytree(ROOT / "01_ppc_features", stage, ignore=shutil.ignore_patterns("data", "output"))
    subprocess.run([sys.executable, str(stage / "main.py"), "--data", str(COLLECTOR), "--rate", "24000",
                    "--window-ms", "1000", "--stride-ms", "250", "--with-descriptors"],
                   check=True, cwd=stage)
    old = pd.read_parquet(PILOT).sort_values(["sample_id", "start_sample_24k"]).reset_index(drop=True)
    new = pd.read_parquet(stage / "output" / "windows-24000-bands200-avg4-descriptors.parquet")
    new = new.sort_values(["sample_id", "start_sample"]).reset_index(drop=True)
    assert len(old) == len(new) == 9748
    expect = np.where(old.split == "test", "test", np.where(old.fold == 0, "validation", "train"))
    assert (expect == new.split.values).all()  # dev fold 0 -> validation, folds 1-2 -> train
    assert (old.group_id.values == new.group_id.values).all()
    assert (old.start_sample_24k.values == new.start_sample.values).all()
    assert np.array_equal(old.rms_db.to_numpy(float), new.rms_db.to_numpy(float))
