"""Étape 1 : WAV maîtres -> fenêtres valides + features + splits groupés.

  python main.py --rate 24000        (défaut ; 16000 et 48000 supportés)

Entrée  input/data/samples/           (posée par l'étape 00)
Sortie  output/windows-<rate>.parquet : une ligne par fenêtre valide ;
        output/manifest-<rate>.json   : provenance, FIR, prise par prise ;
        output/splits.json            : partitions (groupées) — réutilisé tel
        quel s'il est présent dans input/, donc IDENTIQUE pour tous les taux.
La sortie est copiée vers l'entrée des étapes 02 et 03.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from dsp import FLAT_COLS, LOGMEL, N_BANDS, antialias_fir, iter_windows, n_frames, read_wav48, window_features

STAGE = Path(__file__).resolve().parent
INPUT, OUTPUT = STAGE / "input", STAGE / "output"
SEED = 143  # partitions historiques : ne pas changer sans motif


def entries(root):
    """(métadonnées, chemin WAV) : layout samples/ ou AudioFolder CSV."""
    if (root / "samples").is_dir():
        root = root / "samples"
        for m in sorted(root.glob("*/metadata.json")):
            yield json.loads(m.read_text()), m.with_name("audio.wav")
        return
    path = root / "metadata.csv" if (root / "metadata.csv").is_file() else root / "data/metadata.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if Path(row["file_name"]).name != row["file_name"] or Path(row["file_name"]).suffix.lower() != ".wav":
                raise ValueError("attendu un nom de WAV plat (AudioFolder)")
            row["capture_settings"] = json.loads(row["capture_settings"])
            yield row, path.parent / row["file_name"]


def events(rows):
    """Composantes connexes d'intervalles de capture (tolérance 1 s), tous appareils."""
    end, group = -np.inf, -1
    for r in sorted(rows, key=lambda r: (r["start_utc"], r["sample_id"])):
        if r["start_utc"] > end + 1:
            group += 1
        r["group_id"] = f"event-{group:04d}"
        end = max(end, r["start_utc"] + r["duration_s"])


def assign_splits(rows, seed=SEED):
    """Test + trois plis dev équilibrés sur les métadonnées (classes, distances,
    appareils). Échoue explicitement plutôt que de retomber sur du aléatoire."""
    live = [r for r in rows if r["valid_windows"]]
    groups = sorted({r["group_id"] for r in live})
    devices = sorted({r["device_id"] for r in live})
    dists = sorted({r["distance_m"] for r in live if r["drone_present"] and 1 <= r["distance_m"] <= 5})
    if len(groups) < 4 or len(dists) < 2:
        raise ValueError("trop peu de groupes ou de distances pour 3 plis + holdout")
    x = np.zeros((len(groups), 3 + len(dists) + len(devices)))
    for r in live:
        g = x[groups.index(r["group_id"])]
        g[0] += 1
        if not r["drone_present"]:
            g[1] += 1
        if r["distance_m"] in dists:
            g[2] += 1
            g[3 + dists.index(r["distance_m"])] += 1
        g[3 + len(dists) + devices.index(r["device_id"])] += 1
    rng, target = np.random.default_rng(seed), x.sum(axis=0) / 4
    best, chosen = np.inf, None
    for _ in range(8192):
        labels = rng.permutation(np.arange(len(groups)) % 4)
        counts = np.stack([x[labels == k].sum(axis=0) for k in range(4)])
        if np.any(counts[:, 1:3] < 1) or np.any((counts[:, 3:3 + len(dists)] > 0).sum(axis=1) < 2):
            continue
        score = (((counts - target) / np.maximum(target, 1)) ** 2).sum()
        if score < best:
            best, chosen = score, labels
    if chosen is None:
        raise ValueError("aucune partition groupée ne couvre classes et distances partout")
    lookup = dict(zip(groups, chosen.tolist()))
    for r in rows:
        bucket = lookup.get(r["group_id"])
        r["split"] = "excluded" if bucket is None else "test" if bucket == 0 else "dev"
        r["fold"] = bucket - 1 if bucket else -1


def build_rows(root, rate):
    """Valide les métadonnées, lit les WAV stricts, calcule les fenêtres."""
    rows, frames, seen = [], [], set()
    for m, path in entries(root):
        sid = m["sample_id"]
        if not sid or sid in seen:
            raise ValueError(f"identifiant absent ou dupliqué : {sid}")
        seen.add(sid)
        when = datetime.fromisoformat(m["captured_at"].replace("Z", "+00:00"))
        present = str(m["drone_present"]).lower() == "true"
        d = m.get("distance_m")
        d = float(d) if d not in (None, "") else None
        if when.tzinfo is None or not m["device_id"] or present != (d is not None) \
                or (d is not None and not (np.isfinite(d) and d > 0)):
            raise ValueError(f"métadonnées incohérentes : {sid}")
        r = dict(sample_id=sid, device_id=m["device_id"], start_utc=when.timestamp(),
                 drone_present=present, distance_m=d, duration_s=float(m["duration_seconds"]),
                 valid_windows=0)
        rows.append(r)
        if m["mode"] != "hand" or m["capture_settings"].get("capture_contract") != "edth.web-pcm.v1":
            r["reason"] = "hors contrat de capture strict en main"
            continue
        pcm = read_wav48(path)
        r["duration_s"] = len(pcm) / 48000
        r["pcm_sha256"] = hashlib.sha256(pcm.tobytes()).hexdigest()
        wins = [{"start_sample": i, **window_features(w, rate)} for i, w in iter_windows(pcm, rate)]
        r["valid_windows"] = len(wins)
        if wins:
            frames.append(pd.DataFrame(wins).assign(sample_id=sid))
        else:
            r["reason"] = "aucune fenêtre valide"
    if not frames:
        raise ValueError("aucune fenêtre valide dans tout le corpus : vérifier input/data")
    events(rows)
    return rows, frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=INPUT / "data")
    p.add_argument("--rate", type=int, choices=(16000, 24000, 48000), default=24000)
    a = p.parse_args()
    rows, frames = build_rows(a.data, a.rate)
    splits_path = INPUT / "splits.json"
    if splits_path.is_file():
        table = json.loads(splits_path.read_text())
        if any(sid not in {r["sample_id"] for r in rows} for sid in table):
            raise ValueError("input/splits.json référence des prises absentes du corpus")
        print(f"partitions réutilisées : {splits_path}")
    else:
        assign_splits(rows)
        table = {r["sample_id"]: dict(group_id=r["group_id"], split=r["split"], fold=r["fold"])
                 for r in rows if r["valid_windows"]}
        (OUTPUT / "splits.json").parent.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "splits.json").write_text(json.dumps(table, indent=2) + "\n")
        print(f"partitions créées : {OUTPUT / 'splits.json'}")
    for r in rows:
        info = table.get(r["sample_id"])
        if info is None:
            r.update(group_id=None, split="excluded", fold=-1)
        else:
            r.update(group_id=info["group_id"], split=info["split"], fold=info["fold"])
    keys = ["sample_id", "device_id", "group_id", "split", "fold", "drone_present", "distance_m"]
    meta = pd.DataFrame(rows)[keys]
    windows = (pd.concat(frames, ignore_index=True).merge(meta, on="sample_id", validate="many_to_one")
               .sort_values(["sample_id", "start_sample"])[keys + ["start_sample", *FLAT_COLS, LOGMEL]])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parquet = OUTPUT / f"windows-{a.rate}.parquet"
    windows.to_parquet(parquet, index=False)
    fir = antialias_fir(a.rate)
    manifest = dict(rate=a.rate, source=str(a.data.resolve()), seed=SEED,
                    fir=None if fir is None else dict(taps=len(fir), cutoff_hz=11000 if a.rate == 24000 else 7333),
                    logmel=dict(mels=N_BANDS, frames=n_frames(a.rate), fft=2048, hop=1024,
                                layout=f"mel-major : indice = mel * {n_frames(a.rate)} + trame"),
                    takes=sorted(rows, key=lambda r: r["sample_id"]))
    (OUTPUT / f"manifest-{a.rate}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(windows.groupby(["split", "drone_present"]).agg(fenetres=("sample_id", "size"),
                                                         prises=("sample_id", "nunique")))
    for stage in ("02_ppc_classification", "03_ppc_distance"):
        (STAGE.parent / stage / "input").mkdir(parents=True, exist_ok=True)
        for name in (parquet.name, f"manifest-{a.rate}.json", "splits.json"):
            if (OUTPUT / name).is_file():
                copyfile(OUTPUT / name, STAGE.parent / stage / "input" / name)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    copyfile(OUTPUT / "splits.json", splits_path)  # le prochain taux reprendra ces partitions
    print(f"sortie copiée vers 02/input et 03/input ; parquet : {parquet}")


if __name__ == "__main__":
    main()
