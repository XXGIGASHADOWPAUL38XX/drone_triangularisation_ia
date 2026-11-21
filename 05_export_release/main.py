"""Étape 5 : figer deux runs en un bundle de release conforme au contrat d'intégration.

  python main.py                # les deux runs par défaut -> output/<release>/
  python main.py --push         # puis push vers HF_MODELS_REPO (clé du .env)
  python main.py --push-only --release <id>   # pousser un bundle déjà exporté

Entrée  deux dossiers de run (02, 03), leurs caches parquet + manifestes de
        l'étape 01, le corpus des WAV maîtres
Sortie  output/<release-id>/ : manifest.json, frozen/, detection/, distance/,
        parity/fixture.json — autovalidé : chaque cas de parité est rejoué
        (flux par blocs de 50 ms, features locales, prédictions) et doit être
        identique aux valeurs attendues, sinon refus.
Contrat edth.model-release.v2 : la fenêtre, la géométrie des bandes et la
recette de recalcul des features sont lues dans les manifestes de l'étape 01 —
plus rien n'est codé en dur ici. Les deux tâches peuvent venir de caches de
PAS différents (50 ms détection, 20 ms distance) : même fenêtre, mêmes
features et même corpus exigés, seuls les pas peuvent différer ; les bornes
de distance viennent de ml.py. Features LOCALES de fenêtre uniquement : les
descripteurs normalisés par prise ne sont pas rejouables en temps réel et
sont refusés. Le contenu de output/ n'est pas versionné : le canal de
diffusion est le repo HF privé, l'application pinne une révision.
"""
import argparse
import base64
import csv
import hashlib
import json
import os
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
import ml

STAGE = Path(__file__).resolve().parent
INPUT02, INPUT03 = ROOT / "02_ppc_classification", ROOT / "03_ppc_distance"
CONTRACT = "edth.model-release.v2"
THRESHOLD = 0.5
CHUNK_48K = 2400  # 50 ms à 48 kHz : le rejeu simule le flux live par blocs
RUN_FILES = ("model.joblib", "config.json", "metrics.json")
DETECTION_RUN = "mlp-48k-bands200-avg4-w100-s50"   # cache classification de l'étape 01 (pas 50 ms)
DISTANCE_RUN = "mlp-48k-bands200-avg4-w100-s20"    # cache distance (pas 20 ms)
FROZEN_SOURCES = {"dsp.py": ROOT / "01_ppc_features" / "dsp.py",
                  "features.py": ROOT / "01_ppc_features" / "features.py",
                  "ml.py": ROOT / "ml.py"}


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _wav_index(root):
    """sample_id -> WAV maître, comme l'étape 01 : layout samples/ ou AudioFolder CSV."""
    if (root / "samples").is_dir():
        return {p.parent.name: p for p in sorted((root / "samples").glob("*/audio.wav"))}
    path = root / "data" / "metadata.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {row["sample_id"]: root / "data" / row["file_name"] for row in csv.DictReader(f)}


def _load_run(path, task):
    for name in RUN_FILES:
        if not (path / name).is_file():
            raise SystemExit(f"run {task} incomplet : {path / name} absent")
    return json.loads((path / "config.json").read_text())


def _stage_manifest(cache):
    """Manifeste de l'étape 01 : seule source de vérité sur la fenêtre et les features."""
    path = cache.with_name("manifest-" + cache.stem[len("windows-"):] + ".json")
    if not path.is_file():
        raise SystemExit(f"manifeste du cache absent : {path} (relancer l'étape 01)")
    m = json.loads(path.read_text())
    missing = [k for k in ("rate", "window_ms", "stride_ms", "stride_samples") if k not in m]
    if missing:
        raise SystemExit(f"manifeste {path.name} sans {missing} : format antérieur au "
                         "contrat v2 — régénérer le cache avec l'étape 01 actuelle")
    return m


def _band_group(mcache):
    """Nom du groupe de bandes du cache, d'après la géométrie du manifeste."""
    levels = mcache.get("band_levels")
    if levels is None:
        return None
    for name, (count, average) in dsp.BAND_LEVEL_LAYOUTS.items():
        if (levels["source_bands"], levels["average"]) == (count, average):
            return name
    raise SystemExit(f"géométrie de bandes inconnue : {levels}")


def _recompute(mcache, cache_columns):
    """Arguments de dsp.window_features qui régénèrent les colonnes locales du cache."""
    group = _band_group(mcache)
    if group:
        return dict(band_level_group=group, include_rms="rms_db" in cache_columns)
    return dict(include_b48=bool(mcache.get("include_b48")))


def _refuse_recording_features(cfg, task):
    """Les descripteurs normalisés par prise ne sont pas rejouables fenêtre par fenêtre."""
    cols = cfg.get("columns")
    if cols is None:  # cnn (entrée logfft locale) ou constant (aucune entrée)
        return
    refused = [c for c in cols
               if c in dsp.FEATURE_SETS["handcrafted"]
               or c in dsp.FEATURE_SETS[dsp.BAND_DESCRIPTOR_GROUP]]
    if refused:
        raise SystemExit(f"run {task} : features par prise non rejouables en temps réel : "
                         f"{refused} ; réentraîner sur des features locales de fenêtre")


def _input_desc(cfg):
    if cfg["model"] == "cnn":
        return "logfft relatif (logfft − rms_db ; standardisation interne au pickle)"
    if cfg["model"] == "constant":
        return "aucune (référence constante 3 m)"
    suffix = " relatives" if cfg.get("relative") else " absolues"
    return f"flat_columns{suffix} ({', '.join(cfg['columns'])})"


def _required_columns(det_cfg, dis_cfg):
    """Colonnes que les deux pickles consomment réellement (cnn : logfft + rms)."""
    cols = []
    for cfg in (det_cfg, dis_cfg):
        cols.extend(cfg.get("columns") or ([] if cfg["model"] == "constant" else [dsp.LOGFFT, "rms_db"]))
    return list(dict.fromkeys(cols))


def _select_cases(d):
    """Déterministe, split validation uniquement : 1 prise négative + distances extrêmes."""
    valid = d[d.split == "validation"]
    neg = valid[~valid.drone_present]
    pos = valid[valid.drone_present & valid.distance_m.between(ml.DISTANCE_MIN_M, ml.DISTANCE_MAX_M)]
    if neg.empty or pos.empty:
        raise SystemExit("il faut au moins une prise négative et une positive (1-7 m) en validation")
    dists = sorted(pos.distance_m.unique())
    picks = [neg.sample_id.min(),
             pos[pos.distance_m == dists[0]].sample_id.min(),
             pos[pos.distance_m == dists[-1]].sample_id.min()]
    rows = []
    for sid in dict.fromkeys(picks):  # ordre stable, sans doublon
        take = valid[valid.sample_id == sid].sort_values("start_sample")
        rows.append(take.iloc[0])
    return rows


def _case(row, pcm48, rate, window_ms, recompute, required, det, dis):
    """Un cas de parité autovalidé : extrait 48 kHz -> fenêtre -> features -> prédictions."""
    factor = dsp.MASTER // rate
    width = dsp.window_samples(rate, window_ms)
    start = int(row.start_sample)
    start48 = factor * start - dsp.GUARD  # 500 ms d'amorce : garde de début + état FIR
    if start48 < 0:
        raise SystemExit(f"prise {row.sample_id} : fenêtre trop tôt pour l'amorce de parité")
    excerpt = pcm48[start48:start48 + dsp.GUARD + factor * width]
    if len(excerpt) != dsp.GUARD + factor * width:
        raise SystemExit(f"prise {row.sample_id} : WAV trop court pour l'extrait de parité")
    # 1) référence offline : chaîne continue de toute la prise, comme l'étape 01
    y = pcm48 if factor == 1 else dsp.Decimator(rate).process(pcm48)
    ref = y[start:start + width]
    feats = dsp.window_features(ref, rate, window_ms=window_ms, **recompute)
    missing = [c for c in required if c not in feats]
    if missing:
        raise SystemExit(f"features non recalculables pour les modèles : {missing}")
    for c in feats:
        if c == dsp.LOGFFT:
            if not np.array_equal(np.asarray(feats[dsp.LOGFFT], np.float32),
                                  np.asarray(row[dsp.LOGFFT], np.float32)):
                raise SystemExit("logfft diverge du cache : dsp.py ou le cache a changé")
        elif float(feats[c]) != float(row[c]):
            raise SystemExit(f"feature {c} diverge du cache : dsp.py ou le cache a changé")
    # 2) rejeu de l'extrait comme un flux : chaîne neuve, blocs de 50 ms
    #    (Decimator est un passe-plat à 48 kHz : le rejeu vaut pour tous les taux)
    dec = dsp.Decimator(rate)
    stream = np.concatenate([dec.process(excerpt[i:i + CHUNK_48K])
                             for i in range(0, len(excerpt), CHUNK_48K)])
    offset = dsp.GUARD // factor
    if not np.array_equal(stream[offset:offset + width], ref):
        raise SystemExit(f"prise {row.sample_id} : le rejeu par blocs ne redonne pas la fenêtre")
    # 3) prédictions depuis les pickles du bundle (fenêtre unique)
    df = pd.DataFrame([feats])
    score = float(np.ravel(det.predict(df))[0])
    dist = float(np.ravel(dis.predict(df))[0])
    expected = dict(window_pcm_b64=_b64(ref), window_ms=window_ms, recompute=dict(recompute),
                   flat={c: float(feats[c]) for c in feats if c != dsp.LOGFFT},
                   score=score, distance_m=dist, present=bool(score >= THRESHOLD))
    if dsp.LOGFFT in feats:
        expected["logfft"] = np.asarray(feats[dsp.LOGFFT], np.float32).tolist()
    return dict(
        sample_id=str(row.sample_id), device_id=str(row.device_id), split=str(row.split),
        drone_present=bool(row.drone_present),
        distance_m=None if pd.isna(row.distance_m) else float(row.distance_m),
        excerpt=dict(pcm48_b64=_b64(excerpt), samples_48k=int(len(excerpt)),
                     window_start_offset=int(offset), guard_samples_48k=int(dsp.GUARD)),
        expected=expected,
    )


def build(a):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)  # provenance HF_DATASET/HF_REVISION si présentes
    det_cfg = _load_run(a.detection, "détection")
    dis_cfg = _load_run(a.distance, "distance")
    cache_det = Path(a.cache) if a.cache else a.detection.parent.parent / "input" / det_cfg["cache"]
    cache_dis = Path(a.cache) if a.cache else a.distance.parent.parent / "input" / dis_cfg["cache"]
    for label, cache in (("détection", cache_det), ("distance", cache_dis)):
        if not cache.is_file():
            raise SystemExit(f"cache {label} introuvable : {cache} (relancer l'étape 01)")
    m_det, m_dis = _stage_manifest(cache_det), _stage_manifest(cache_dis)
    # cohérence des deux caches : même fenêtre, mêmes features, même corpus —
    # seul le pas (stride) a le droit de différer entre les deux tâches
    for key in ("rate", "window_ms", "fir", "band_levels", "logfft"):
        if m_det.get(key) != m_dis.get(key):
            raise SystemExit(f"les caches des deux runs divergent sur {key!r} : bundle incohérent, refuser")
    if (m_det["source"], m_det["seed"]) != (m_dis["source"], m_dis["seed"]):
        raise SystemExit("les caches des deux runs ne viennent pas du même corpus : refuser")
    rate, window_ms = m_det["rate"], m_det["window_ms"]
    _refuse_recording_features(det_cfg, "détection")
    _refuse_recording_features(dis_cfg, "distance")
    if a.release is None:
        a.release = f"edth.audio.{rate // 1000}k-w{window_ms}.r2"
    fir = dsp.antialias_fir(rate)
    fir_sha = None if fir is None else hashlib.sha256(fir.tobytes()).hexdigest()
    cutoff = None if fir is None else (11000 if rate == 24000 else 7333)
    width = dsp.window_samples(rate, window_ms)
    frame = pd.read_parquet(cache_det)  # cas de parité depuis le cache détection
    recompute = _recompute(m_det, frame.columns)
    if recompute != _recompute(m_dis, pd.read_parquet(cache_dis).columns):
        raise SystemExit("les recettes de features des deux caches diffèrent : refuser")
    required = _required_columns(det_cfg, dis_cfg)
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
            window=dict(ms=window_ms, samples=width),
            qc=dict(zero_run_ms=dsp.DROPOUT * 1000 // dsp.MASTER,
                    guard_ms=dsp.GUARD * 1000 // dsp.MASTER, start_guard_ms=500)),
        features=dict(
            caches=dict(detection=det_cfg["cache"], distance=dis_cfg["cache"]),
            fft=dsp.FFT, hop=dsp.HOP, analysis_window="periodic-hann",
            scale="dBFS absolus, plancher -120 dB",
            band_levels=m_det.get("band_levels"),
            logfft=m_det.get("logfft"),
            local_only=True,
            recompute=recompute,
            note="features locales de fenêtre ; descripteurs normalisés par prise refusés"),
        detection=dict(file="detection/model.joblib", model=det_cfg["model"],
                       input=_input_desc(det_cfg), columns=det_cfg.get("columns"),
                       threshold=THRESHOLD, rule="single_window",
                       step=dict(ms=m_det["stride_ms"], samples=m_det["stride_samples"])),
        distance=dict(file="distance/model.joblib", model=dis_cfg["model"],
                      input=_input_desc(dis_cfg), columns=dis_cfg.get("columns"),
                      bounds_m=[ml.DISTANCE_MIN_M, ml.DISTANCE_MAX_M],
                      gate="présence établie par detection",
                      step=dict(ms=m_dis["stride_ms"], samples=m_dis["stride_samples"])),
        training=dict(
            corpus=dict(source=m_det["source"], seed=m_det["seed"],
                        capture_contracts=m_det.get("capture_contracts"),
                        dataset_repo=os.getenv("HF_DATASET"),
                        dataset_revision=os.getenv("HF_REVISION")),
            metrics=["detection/metrics.json", "distance/metrics.json"],
            validation=dict(detection=det_cfg.get("validation"),
                            distance=dis_cfg.get("validation"))),
    )
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    det = joblib.load(tmp / "detection" / "model.joblib")
    dis = joblib.load(tmp / "distance" / "model.joblib")
    wavs = _wav_index(Path(a.data))
    cases = []
    for row in _select_cases(frame):
        wav = wavs.get(str(row.sample_id))
        if wav is None or not wav.is_file():
            raise SystemExit(f"WAV maître introuvable : {row.sample_id} dans {a.data}")
        case = _case(row, dsp.read_wav48(wav), rate, window_ms, recompute, required, det, dis)
        cases.append(case)
        print(f"  parité : {case['sample_id'][:8]} present={case['drone_present']} "
              f"d={case['distance_m']} score={case['expected']['score']:.3f} "
              f"present_pred={case['expected']['present']} "
              f"dist={case['expected']['distance_m']:.2f}")
    fixture = dict(contract=CONTRACT, release=a.release, model_rate_hz=rate,
                   transport_rate_hz=dsp.MASTER, chunk_hint_samples_48k=CHUNK_48K,
                   replay="une session = un extrait (amorce 500 ms + fenêtre) ; "
                          "chaîne par blocs de 50 ms (décimation, passe-plat à 48 kHz), "
                          "fenêtre à window_start_offset dans le flux au taux modèle",
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
    p.add_argument("--detection", type=Path, default=INPUT02 / "output" / DETECTION_RUN)
    p.add_argument("--distance", type=Path, default=INPUT03 / "output" / DISTANCE_RUN)
    p.add_argument("--cache", type=Path, default=None,
                   help="forcer le cache des deux runs (défaut : l'input de chaque run)")
    p.add_argument("--data", type=Path, default=ROOT / "01_ppc_features" / "input" / "data",
                   help="corpus des WAV maîtres, samples/ ou AudioFolder (extraits de parité)")
    p.add_argument("--out", type=Path, default=STAGE / "output")
    p.add_argument("--release", default=None, help="défaut : edth.audio.<taux>k-w<fenêtre_ms>.r2")
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
