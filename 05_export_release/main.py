"""Étape 5 : figer deux runs en un bundle de release conforme au contrat d'intégration.

  python main.py                  # r0 : cnn-24k + mlp-24k -> output/<release>/
  python main.py --push           # puis push vers HF_MODELS_REPO (clé du .env)
  python main.py --push-only --release <id>   # pousser un bundle déjà exporté

Entrée  deux dossiers de run (02, 03), le cache parquet de 01, le corpus
Sortie  output/<release-id>/ : manifest.json, frozen/, detection/, distance/,
        parity/fixture.json — autovalidé : chaque cas de parité est rejoué
        (décimation fraîche par blocs de 50 ms, features, prédictions) et
        doit être identique aux valeurs attendues, sinon refus.
Contrat edth.model-release.v1 — doc Outline « Contrat d'intégration ML V1 »
(B2 — IA audio). Le contenu de output/ n'est pas versionné : le canal de
diffusion est le repo HF privé, l'application pinne une révision.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # ml.py
sys.path.insert(0, str(ROOT / "01_ppc_features"))  # dsp.py : registre des features
import dsp

STAGE = Path(__file__).resolve().parent
INPUT02, INPUT03 = ROOT / "02_ppc_classification", ROOT / "03_ppc_distance"
CONTRACT = "edth.model-release.v1"
THRESHOLD = 0.5
CHUNK_48K = 2400  # 50 ms à 48 kHz : le rejeu simule le flux live par blocs
RUN_FILES = ("model.joblib", "config.json", "metrics.json")
FROZEN_SOURCES = {"dsp.py": ROOT / "01_ppc_features" / "dsp.py", "ml.py": ROOT / "ml.py"}


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _rate_of(cache_name):
    m = re.search(r"windows-(\d+)\.parquet", cache_name)
    if not m:
        raise SystemExit(f"nom de cache illisible : {cache_name!r}")
    return int(m.group(1))


def _load_run(path, task):
    for name in RUN_FILES:
        if not (path / name).is_file():
            raise SystemExit(f"run {task} incomplet : {path / name} absent")
    return json.loads((path / "config.json").read_text())


def _input_desc(cfg):
    if cfg["model"] == "cnn":
        return "logmel relative (logmel − rms_db ; standardisation interne au pickle)"
    if cfg["model"] == "constant":
        return "aucune (référence constante 3 m)"
    suffix = " relatives" if cfg.get("relative") else " absolues"
    return f"flat_columns{suffix} ({', '.join(cfg['columns'])})"


def _select_cases(d):
    """Déterministe, split dev uniquement : 1 prise négative + distances extrêmes."""
    dev = d[d.split == "dev"]
    neg, pos = dev[~dev.drone_present], dev[dev.drone_present & dev.distance_m.between(1, 5)]
    if neg.empty or pos.empty:
        raise SystemExit("il faut au moins une prise négative et une positive (1-5 m) en dev")
    dists = sorted(pos.distance_m.unique())
    picks = [neg.sample_id.min(),
             pos[pos.distance_m == dists[0]].sample_id.min(),
             pos[pos.distance_m == dists[-1]].sample_id.min()]
    rows = []
    for sid in dict.fromkeys(picks):  # ordre stable, sans doublon
        take = dev[dev.sample_id == sid].sort_values("start_sample")
        rows.append(take.iloc[0])
    return rows


def _case(row, pcm48, rate, det, dis):
    """Un cas de parité autovalidé : extrait 48 kHz -> fenêtre -> features -> prédictions."""
    start24 = int(row.start_sample)
    start48 = 2 * start24 - dsp.GUARD  # 500 ms d'amorce : garde de début + état FIR
    if start48 < 0:
        raise SystemExit(f"prise {row.sample_id} : fenêtre trop tôt pour l'amorce de parité")
    excerpt = pcm48[start48:start48 + dsp.GUARD + 2 * rate]
    if len(excerpt) != dsp.GUARD + 2 * rate:
        raise SystemExit(f"prise {row.sample_id} : WAV trop court pour l'extrait de parité")
    # 1) référence offline : décimation continue de toute la prise, comme l'étape 01
    ref = dsp.Decimator(rate).process(pcm48)[start24:start24 + rate]
    feats = dsp.window_features(ref, rate)
    for c in dsp.FLAT_COLS:
        if float(feats[c]) != float(row[c]):
            raise SystemExit(f"feature {c} diverge du cache : dsp.py ou le cache a changé")
    if not np.array_equal(np.asarray(feats[dsp.LOGMEL], np.float32),
                          np.asarray(row[dsp.LOGMEL], np.float32)):
        raise SystemExit("logmel diverge du cache : dsp.py ou le cache a changé")
    # 2) rejeu de l'extrait comme un flux : décimateur neuf, blocs de 50 ms
    dec = dsp.Decimator(rate)
    stream = np.concatenate([dec.process(excerpt[i:i + CHUNK_48K])
                             for i in range(0, len(excerpt), CHUNK_48K)])
    live = stream[dsp.GUARD // 2:dsp.GUARD // 2 + rate]
    if not np.array_equal(live, ref):
        raise SystemExit(f"prise {row.sample_id} : le rejeu par blocs ne redonne pas la fenêtre")
    # 3) prédictions depuis les pickles du bundle (fenêtre unique)
    df = pd.DataFrame([feats])
    score = float(np.ravel(det.predict(df))[0])
    dist = float(np.ravel(dis.predict(df))[0])
    return dict(
        sample_id=str(row.sample_id), device_id=str(row.device_id), split=str(row.split),
        drone_present=bool(row.drone_present),
        distance_m=None if pd.isna(row.distance_m) else float(row.distance_m),
        excerpt=dict(pcm48_b64=_b64(excerpt), samples_48k=int(len(excerpt)),
                     window_start_offset_24k=dsp.GUARD // 2, guard_samples_48k=int(dsp.GUARD)),
        expected=dict(window_pcm24_b64=_b64(ref),
                      flat={c: float(feats[c]) for c in dsp.FLAT_COLS},
                      logmel=np.asarray(feats[dsp.LOGMEL], np.float32).tolist(),
                      score=score, distance_m=dist, present=bool(score >= THRESHOLD)),
    )


def build(a):
    det_cfg = _load_run(a.detection, "détection")
    dis_cfg = _load_run(a.distance, "distance")
    rate = _rate_of(det_cfg["cache"])
    if _rate_of(dis_cfg["cache"]) != rate:
        raise SystemExit("les deux runs n'ont pas le même taux : bundle incohérent, refuser")
    if a.release is None:
        a.release = f"edth.audio.{rate // 1000}k-1s.r0"
    fir = dsp.antialias_fir(rate)
    fir_sha = None if fir is None else hashlib.sha256(fir.tobytes()).hexdigest()
    cutoff = None if fir is None else (11000 if rate == 24000 else 7333)
    bundle = Path(a.out) / a.release
    if bundle.exists() and not a.force:
        raise SystemExit(f"{bundle} existe déjà : --force pour écraser")
    tmp = Path(a.out) / f".tmp-{a.release}"
    if tmp.exists():
        shutil.rmtree(tmp)
    for sub in ("frozen", "detection", "distance", "parity"):
        (tmp / sub).mkdir(parents=True)
    for src, sub in ((a.detection, "detection"), (a.distance, "distance")):
        for name in RUN_FILES:
            shutil.copyfile(src / name, tmp / sub / name)
    for name, src in FROZEN_SOURCES.items():  # code d'inférence figé, validé par le rejeu ci-dessous
        shutil.copyfile(src, tmp / "frozen" / name)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = dict(
        contract=CONTRACT, release=a.release,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        frozen=dict(repo_commit=commit,
                    files={f"frozen/{n}": hashlib.sha256(src.read_bytes()).hexdigest()
                           for n, src in FROZEN_SOURCES.items()}),
        input=dict(
            transport=dict(rate_hz=dsp.MASTER, format="pcm_s16le", channels=1),
            model_rate_hz=rate,
            fir=None if fir is None else dict(taps=len(fir), cutoff_hz=cutoff, kaiser_beta=8.6,
                                             coefficients_sha256=fir_sha),
            window=dict(ms=1000, step_ms=250, samples=rate, step_samples=rate // 4),
            qc=dict(zero_run_ms=dsp.DROPOUT * 1000 // dsp.MASTER,
                    guard_ms=dsp.GUARD * 1000 // dsp.MASTER, start_guard_ms=500)),
        features=dict(
            fft=dsp.FFT, hop=dsp.HOP, analysis_window="periodic-hann",
            scale="dBFS absolus, plancher -120 dB",
            flat_columns=list(dsp.FLAT_COLS),
            logmel=dict(mels=dsp.N_BANDS, frames=dsp.n_frames(rate), dtype="float32",
                        layout=f"mel-major : indice = mel × {dsp.n_frames(rate)} + trame")),
        detection=dict(file="detection/model.joblib", model=det_cfg["model"],
                       input=_input_desc(det_cfg), threshold=THRESHOLD, rule="single_window"),
        distance=dict(file="distance/model.joblib", model=dis_cfg["model"],
                      input=_input_desc(dis_cfg), bounds_m=[1.0, 5.0],
                      gate="présence établie par detection"),
        training=dict(cache=det_cfg["cache"], seed=143,
                      metrics=["detection/metrics.json", "distance/metrics.json"],
                      dev_cv=dict(detection=det_cfg["dev_cv"], distance=dis_cfg["dev_cv"])),
    )
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    cache = Path(a.cache) if a.cache else a.detection.parent.parent / "input" / det_cfg["cache"]
    if not cache.is_file():
        raise SystemExit(f"cache introuvable : {cache} (relancer l'étape 01)")
    det = joblib.load(tmp / "detection" / "model.joblib")
    dis = joblib.load(tmp / "distance" / "model.joblib")
    wav_root = Path(a.data)
    cases = []
    for row in _select_cases(pd.read_parquet(cache)):
        wav = wav_root / "samples" / row.sample_id / "audio.wav"
        if not wav.is_file():
            raise SystemExit(f"WAV maître introuvable : {wav}")
        case = _case(row, dsp.read_wav48(wav), rate, det, dis)
        cases.append(case)
        print(f"  parité : {case['sample_id'][:8]} present={case['drone_present']} "
              f"d={case['distance_m']} score={case['expected']['score']:.3f} "
              f"present_pred={case['expected']['present']} "
              f"dist={case['expected']['distance_m']:.2f}")
    fixture = dict(contract=CONTRACT, release=a.release, model_rate_hz=rate,
                   transport_rate_hz=dsp.MASTER, chunk_hint_samples_48k=CHUNK_48K,
                   replay="une session = un extrait (amorce 500 ms + fenêtre 1 s) ; "
                          "décimation continue par blocs, fenêtre à window_start_offset_24k",
                   cases=cases)
    (tmp / "parity" / "fixture.json").write_text(
        json.dumps(fixture, separators=(",", ":"), allow_nan=False))
    if bundle.exists():
        shutil.rmtree(bundle)  # --force
    tmp.rename(bundle)
    print(f"release {a.release} : {len(cases)} cas de parité autovalidés -> {bundle}")
    return bundle


def push(a):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    repo = os.getenv("HF_MODELS_REPO")
    if not repo:
        raise SystemExit("HF_MODELS_REPO absent du .env : push impossible")
    bundle = Path(a.out) / a.release
    if not (bundle / "manifest.json").is_file():
        raise SystemExit(f"bundle introuvable ou incomplet : {bundle}")
    from huggingface_hub import HfApi
    api = HfApi(token=os.getenv("HF_TOKEN") or True)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    info = api.upload_folder(folder_path=str(bundle), repo_id=repo, repo_type="model",
                             path_in_repo=a.release,
                             commit_message=f"release {a.release} ({CONTRACT})")
    rev = getattr(info, "oid", None)
    if not rev:
        try:
            rev = api.list_repo_commits(repo, repo_type="model")[0].commit_id
        except Exception:
            rev = "voir l'historique du repo"
    print(f"poussé : {repo} -> {a.release} ; révision à pinner côté app : {rev}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detection", type=Path, default=INPUT02 / "output" / "cnn-24k")
    p.add_argument("--distance", type=Path, default=INPUT03 / "output" / "mlp-24k")
    p.add_argument("--cache", type=Path, default=None,
                   help="parquet de l'étape 01 (défaut : input du run détection)")
    p.add_argument("--data", type=Path, default=ROOT / "01_ppc_features" / "input" / "data",
                   help="corpus des WAV maîtres (extraits de parité)")
    p.add_argument("--out", type=Path, default=STAGE / "output")
    p.add_argument("--release", default=None, help="défaut : edth.audio.<taux>k-1s.r0")
    p.add_argument("--force", action="store_true", help="écraser un bundle existant")
    p.add_argument("--push", action="store_true", help="pousser le bundle vers HF_MODELS_REPO après l'export")
    p.add_argument("--push-only", action="store_true", help="pousser un bundle déjà exporté")
    a = p.parse_args()
    if a.push_only:
        if a.release is None:
            found = [q.parent for q in sorted(Path(a.out).glob("*/manifest.json"))]
            if len(found) != 1:
                raise SystemExit("--push-only : préciser --release (ou exactement un bundle dans --out)")
            a.release = found[0].name
        push(a)
        return
    build(a)
    if a.push:
        push(a)


if __name__ == "__main__":
    main()
