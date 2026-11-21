"""Export de release v2 : manifeste conforme, parité rejouable depuis le bundle.

Chaîne complète sur un corpus synthétique : étape 01 (fenêtres + manifeste),
entraînements 02/03, export, puis rejeu indépendant du lot de parité avec le
code FIGÉ du bundle (frozen/) — tout doit être identique aux valeurs attendues.
"""
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from datetime import datetime, timezone
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
import ml

CACHE_NAME = "windows-48000-bands200-avg4-w100-s20.parquet"
FEATURES = "band_levels_avg4"


def _wav(path, pcm48):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(pcm48.tobytes())


def _corpus(root):
    """8 groupes de 4 prises de 2 s : bruit faible, ou sinus 3 kHz si le drone est présent."""
    rng = np.random.default_rng(3)
    t = np.arange(2 * 48000) / 48000
    for g in range(8):
        for k, dist in enumerate([None, 1.0, 3.0, 5.0]):
            sid = f"take-{g}-{k}"
            pcm = rng.integers(-200, 200, len(t)).astype(np.int16)
            if dist is not None:
                pcm = pcm + (int(9000 / dist) * np.sin(2 * np.pi * 3000 * t)).astype(np.int16)
            _wav(root / "samples" / sid / "audio.wav", pcm.astype(np.int16))
            when = datetime.fromtimestamp(1700000000 + 10 * g + 0.1 * k, tz=timezone.utc)
            (root / "samples" / sid / "metadata.json").write_text(json.dumps(dict(
                sample_id=sid, device_id=f"D{g % 2}", captured_at=when.isoformat(),
                drone_present=dist is not None, distance_m=dist, duration_seconds=2.0,
                mode="hand", capture_settings=dict(capture_contract="edth.web-pcm.v1"))))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    corpus = tmp_path_factory.mktemp("corpus")
    _corpus(corpus)
    stage = tmp_path_factory.mktemp("stage") / "01_ppc_features"
    shutil.copytree(ROOT / "01_ppc_features", stage, ignore=shutil.ignore_patterns("data", "output"))
    (stage / "input" / "splits.json").unlink(missing_ok=True)  # partitions fraîches pour ce corpus
    subprocess.run([sys.executable, str(stage / "main.py"), "--data", str(corpus)], check=True, cwd=stage)
    cache = stage / "output" / CACHE_NAME
    assert cache.is_file(), "l'étape 01 devait écrire le cache par défaut 48 kHz w100-s20"
    runs = tmp_path_factory.mktemp("runs")
    for task in ("classification", "distance"):
        # même convention que les étapes réelles : <étape>/input/<cache + manifeste>, <étape>/output/<run>
        (runs / task / "input").mkdir(parents=True)
        shutil.copyfile(cache, runs / task / "input" / cache.name)
        shutil.copyfile(cache.with_name("manifest-" + cache.stem[len("windows-"):] + ".json"),
                        runs / task / "input" / ("manifest-" + cache.stem[len("windows-"):] + ".json"))
        a = SimpleNamespace(model="mlp", input=cache, features=FEATURES, relative=False,
                            name="mlp-smoke", out=runs / task / "output", rate=48000, device="cpu",
                            epochs=12, patience=4, lr=None, dropout=None, seed=42,
                            resolve_features=dsp.resolve_features)
        ml.experiment(task, a)
    spec = util.spec_from_file_location("export_main", ROOT / "05_export_release" / "main.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(SimpleNamespace(detection=runs / "classification" / "output" / "mlp-smoke",
                                     distance=runs / "distance" / "output" / "mlp-smoke",
                                     cache=None, data=corpus,
                                     out=tmp_path_factory.mktemp("release"),
                                     release="test-v2", force=False))


def test_bundle_complet_et_manifeste_conforme(bundle):
    m = json.loads((bundle / "manifest.json").read_text())
    assert m["contract"] == "edth.model-release.v2"
    assert m["input"]["transport"] == {"rate_hz": 48000, "format": "pcm_s16le", "channels": 1}
    assert m["input"]["model_rate_hz"] == 48000
    assert m["input"]["fir"] is None  # 48 kHz natif : aucun filtre
    assert m["input"]["window"] == {"ms": 100, "step_ms": 20, "samples": 4800, "step_samples": 960}
    assert m["input"]["qc"] == {"zero_run_ms": 40, "guard_ms": 500, "start_guard_ms": 500}
    assert m["features"]["cache"] == CACHE_NAME
    assert m["features"]["band_levels"]["count"] == 50
    assert m["features"]["local_only"] is True
    assert m["features"]["recompute"] == {"band_level_group": "band_levels_avg4", "include_rms": False}
    assert m["detection"]["threshold"] == 0.5 and m["detection"]["rule"] == "single_window"
    assert m["detection"]["columns"] == dsp.BAND_LEVEL_GROUPS["band_levels_avg4"]
    assert m["distance"]["bounds_m"] == [1.0, 5.0]
    assert m["training"]["seed"] == 143
    for sub in ("detection", "distance"):
        for name in ("model.joblib", "config.json", "metrics.json"):
            assert (bundle / sub / name).is_file(), f"{sub}/{name}"
    for name in ("dsp.py", "features.py", "ml.py"):
        digest = hashlib.sha256((bundle / "frozen" / name).read_bytes()).hexdigest()
        assert digest == m["frozen"]["files"][f"frozen/{name}"]


def test_parite_rejouable_depuis_le_frozen(bundle):
    fix = json.loads((bundle / "parity" / "fixture.json").read_text())
    saved = {n: sys.modules.get(n) for n in ("features", "dsp", "ml")}
    try:
        frozen = bundle / "frozen"
        for name in ("features", "dsp", "ml"):  # dsp fait « from features import ... »
            spec = util.spec_from_file_location(name, frozen / f"{name}.py")
            module = util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        fdsp = sys.modules["dsp"]
        det = joblib.load(bundle / "detection" / "model.joblib")
        dis = joblib.load(bundle / "distance" / "model.joblib")
        assert fix["contract"] == "edth.model-release.v2" and len(fix["cases"]) >= 3
        for case in fix["cases"]:
            pcm = np.frombuffer(base64.b64decode(case["excerpt"]["pcm48_b64"]), "<i2")
            rate = fix["model_rate_hz"]
            width = fdsp.window_samples(rate, case["expected"]["window_ms"])
            dec = fdsp.Decimator(rate)
            stream = np.concatenate([dec.process(pcm[i:i + fix["chunk_hint_samples_48k"]])
                                     for i in range(0, len(pcm), fix["chunk_hint_samples_48k"])])
            off = case["excerpt"]["window_start_offset"]
            win = stream[off:off + width]
            assert np.array_equal(win, np.frombuffer(base64.b64decode(case["expected"]["window_pcm_b64"]), "<i2"))
            feats = fdsp.window_features(win, rate, window_ms=case["expected"]["window_ms"],
                                         **case["expected"]["recompute"])
            for c, v in case["expected"]["flat"].items():
                assert float(feats[c]) == v, c
            df = pd.DataFrame([feats])
            assert float(np.ravel(det.predict(df))[0]) == pytest.approx(case["expected"]["score"], abs=1e-6)
            assert float(np.ravel(dis.predict(df))[0]) == pytest.approx(case["expected"]["distance_m"], abs=1e-6)
            assert case["expected"]["present"] == (case["expected"]["score"] >= 0.5)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
