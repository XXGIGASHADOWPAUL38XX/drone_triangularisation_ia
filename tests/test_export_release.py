"""Export de release : manifeste conforme au contrat, parité rejouable depuis le bundle.

Rejoue le lot de parité indépendamment de l'export : décode les extraits
PCM48, redécime par blocs, recalcule les features et les prédictions depuis
les pickles du bundle — tout doit être identique aux valeurs attendues.
"""
import base64
import hashlib
import json
import sys
from importlib import util
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # ml.py
sys.path.insert(0, str(ROOT / "01_ppc_features"))  # dsp.py
import dsp

DET = ROOT / "02_ppc_classification" / "output" / "cnn-24k"
DIS = ROOT / "03_ppc_distance" / "output" / "mlp-24k"
CACHE = ROOT / "02_ppc_classification" / "input" / "windows-24000.parquet"
DATA = ROOT / "01_ppc_features" / "input" / "data"


def _export(out):
    spec = util.spec_from_file_location("export_main", ROOT / "05_export_release" / "main.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = SimpleNamespace(detection=DET, distance=DIS, cache=None, data=DATA,
                        out=out, release="test-r0", force=False)
    return mod.build(a)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    if not (DET / "model.joblib").is_file() or not CACHE.is_file() or not (DATA / "samples").is_dir():
        pytest.skip("runs gelés, cache ou corpus absents")
    return _export(tmp_path_factory.mktemp("release"))


def test_bundle_complet_et_manifeste_conforme(bundle):
    m = json.loads((bundle / "manifest.json").read_text())
    fir = dsp.antialias_fir(24000)
    assert m["contract"] == "edth.model-release.v1"
    assert m["input"]["transport"] == {"rate_hz": 48000, "format": "pcm_s16le", "channels": 1}
    assert m["input"]["model_rate_hz"] == 24000
    assert m["input"]["fir"]["coefficients_sha256"] == hashlib.sha256(fir.tobytes()).hexdigest()
    assert m["input"]["window"] == {"ms": 1000, "step_ms": 250, "samples": 24000, "step_samples": 6000}
    assert m["input"]["qc"] == {"zero_run_ms": 40, "guard_ms": 500, "start_guard_ms": 500}
    assert m["features"]["flat_columns"] == list(dsp.FLAT_COLS)
    assert m["features"]["logmel"]["frames"] == dsp.n_frames(24000)
    assert m["detection"]["threshold"] == 0.5 and m["detection"]["rule"] == "single_window"
    assert m["distance"]["bounds_m"] == [1.0, 5.0]
    for sub in ("detection", "distance"):
        for name in ("model.joblib", "config.json", "metrics.json"):
            assert (bundle / sub / name).is_file(), f"{sub}/{name}"
    for name in ("dsp.py", "ml.py"):
        digest = hashlib.sha256((bundle / "frozen" / name).read_bytes()).hexdigest()
        assert digest == m["frozen"]["files"][f"frozen/{name}"]


def test_parite_rejouable_depuis_le_bundle(bundle):
    fix = json.loads((bundle / "parity" / "fixture.json").read_text())
    rate = fix["model_rate_hz"]
    det = joblib.load(bundle / "detection" / "model.joblib")
    dis = joblib.load(bundle / "distance" / "model.joblib")
    assert len(fix["cases"]) >= 3
    for case in fix["cases"]:
        pcm = np.frombuffer(base64.b64decode(case["excerpt"]["pcm48_b64"]), "<i2")
        off = case["excerpt"]["window_start_offset_24k"]
        dec = dsp.Decimator(rate)
        stream = np.concatenate([dec.process(pcm[i:i + fix["chunk_hint_samples_48k"]])
                                 for i in range(0, len(pcm), fix["chunk_hint_samples_48k"])])
        win = stream[off:off + rate]
        assert np.array_equal(win, np.frombuffer(base64.b64decode(case["expected"]["window_pcm24_b64"]), "<i2"))
        feats = dsp.window_features(win, rate)
        for c, v in case["expected"]["flat"].items():
            assert float(feats[c]) == v, c
        assert np.array_equal(np.asarray(feats[dsp.LOGMEL], np.float32),
                              np.asarray(case["expected"]["logmel"], np.float32))
        df = pd.DataFrame([feats])
        assert float(np.ravel(det.predict(df))[0]) == pytest.approx(case["expected"]["score"], abs=1e-6)
        assert float(np.ravel(dis.predict(df))[0]) == pytest.approx(case["expected"]["distance_m"], abs=1e-6)
        assert case["expected"]["present"] == (case["expected"]["score"] >= 0.5)
