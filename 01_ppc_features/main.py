"""Étape 1 : WAV maîtres -> fenêtres valides + features + splits groupés.

  python main.py

Entrée  input/data/samples/           (posée par l'étape 00)
Sortie  deux caches de 50 niveaux de bandes avec manifestes et partitions.
        Pas de 20 ms pour la distance et de 50 ms pour la classification.
        output/splits.json            : partitions (groupées) — réutilisé tel
        quel s'il est présent dans input/, donc IDENTIQUE pour tous les taux.
La sortie est copiée vers l'entrée des étapes 02 et 03.
"""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from dsp import (BAND_DESCRIPTOR_GROUP, BAND_LEVEL_LAYOUTS, CLASSIFICATION_STRIDE_MS,
                 DEFAULT_BAND_GROUP, DEFAULT_CACHE_TAG,
                 FFT, FLAT_COLS, HOP, LOGFFT, MASTER, N_BINS, RATES, STRIDE_MS, WINDOW_MS, antialias_fir, iter_windows,
                 default_window_cache, n_frames, read_wav48, recording_features, resolve_features, stride_samples)
from features import BAND_DESCRIPTOR_NAMES, HANDCRAFTED_COLS

STAGE = Path(__file__).resolve().parent
INPUT, OUTPUT = STAGE / "input", STAGE / "output"
SEED = 143  # partitions historiques : ne pas changer sans motif
SPLIT_CANDIDATES = 8192
PARTITIONS = ("train", "validation", "test")


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
    """Répartit les groupes entre entraînement, validation et test."""
    live = [r for r in rows if r["valid_windows"]]
    groups = sorted({r["group_id"] for r in live})
    devices = sorted({r["device_id"] for r in live})
    dists = sorted({r["distance_m"] for r in live if r["drone_present"] and 1 <= r["distance_m"] <= 5})
    if len(groups) < 4 or len(dists) < 2:
        raise ValueError("trop peu de groupes ou de distances pour train/validation/test")
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
    held_out = len(groups) // 4
    sizes = np.array([len(groups) - 2 * held_out, held_out, held_out])
    target = (sizes / len(groups))[:, None] * x.sum(axis=0)
    rng = np.random.default_rng(seed)
    balanced = np.broadcast_to(np.repeat(np.arange(len(PARTITIONS)), sizes), (SPLIT_CANDIDATES, len(groups)))
    candidates = rng.permuted(balanced, axis=1)
    # Axes : candidat, partition, groupe. La multiplication totalise les métadonnées par partition.
    counts = (candidates[:, None, :] == np.arange(len(PARTITIONS))[None, :, None]) @ x
    has_classes = (counts[:, :, 1:3] >= 1).all(axis=(1, 2))
    has_distances = ((counts[:, :, 3:3 + len(dists)] > 0).sum(axis=2) >= 2).all(axis=1)
    scores = (((counts - target) / np.maximum(target, 1)) ** 2).sum(axis=(1, 2))
    scores[~(has_classes & has_distances)] = np.inf
    best = scores.argmin()
    if not np.isfinite(scores[best]):
        raise ValueError("aucune partition groupée ne couvre classes et distances partout")
    lookup = {group: PARTITIONS[bucket] for group, bucket in zip(groups, candidates[best])}
    for r in rows:
        r["split"] = lookup.get(r["group_id"], "excluded")


def build_rows(root, rate, *, window_ms=WINDOW_MS, stride_ms=STRIDE_MS, include_b48=False,
               band_level_group=None, include_descriptors=False, include_band_descriptors=False,
               export_classification=False):
    """Valide les métadonnées, lit les WAV stricts, calcule les fenêtres."""
    rows, frames, seen = [], [], set()
    classification_frames = []
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
        windows = iter_windows(pcm, rate, window_ms=window_ms, stride_ms=stride_ms)
        wins = recording_features(windows, rate, window_ms=window_ms, include_b48=include_b48,
                                  band_level_group=band_level_group, include_descriptors=include_descriptors,
                                  include_band_descriptors=include_band_descriptors)
        r["valid_windows"] = len(wins)
        if wins:
            frames.append(pd.DataFrame(wins).assign(sample_id=sid))
        else:
            r["reason"] = "aucune fenêtre valide"
        if export_classification:
            class_windows = iter_windows(pcm, rate, window_ms=window_ms,
                                         stride_ms=CLASSIFICATION_STRIDE_MS)
            class_features = recording_features(class_windows, rate, window_ms=window_ms,
                                                band_level_group=DEFAULT_BAND_GROUP)
            if class_features:
                classification_frames.append(pd.DataFrame(class_features).assign(sample_id=sid))
    if not frames:
        raise ValueError("aucune fenêtre valide dans tout le corpus : vérifier input/data")
    events(rows)
    return rows, frames, classification_frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=INPUT / "data")
    p.add_argument("--rate", type=int, choices=RATES, default=MASTER)
    p.add_argument("--window-ms", type=int, default=WINDOW_MS, help="durée d'une fenêtre en ms")
    p.add_argument("--stride-ms", type=float, default=STRIDE_MS, help="pas entre fenêtres en ms")
    p.add_argument("--cache-tag", help="suffixe du cache pour conserver plusieurs essais")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full-spectrum", action="store_true", help="écrire le cache FFT complet historique")
    mode.add_argument("--include-b48", action="store_true", help="ajouter la puissance 4–8 kHz dans un cache séparé")
    mode.add_argument("--band-levels", dest="band_level_group", action="store_const", const="band_levels",
                      help="écrire uniquement les niveaux des 100 bandes FFT")
    mode.add_argument("--band-levels-avg3", dest="band_level_group", action="store_const",
                      const="band_levels_avg3", help="moyenner les niveaux de 200 bandes FFT par groupes de trois")
    mode.add_argument("--band-levels-avg4", dest="band_level_group", action="store_const",
                      const="band_levels_avg4", help="moyenner les niveaux de 200 bandes FFT par groupes de quatre")
    mode.add_argument("--band-levels-avg5", dest="band_level_group", action="store_const",
                      const="band_levels_avg5", help="moyenner les niveaux de 200 bandes FFT par groupes de cinq")
    mode.add_argument("--band-levels-avg6", dest="band_level_group", action="store_const",
                      const="band_levels_avg6", help="moyenner les niveaux de 300 bandes FFT par groupes de six")
    descriptors = p.add_mutually_exclusive_group()
    descriptors.add_argument("--with-descriptors", action="store_true",
                             help="ajouter le RMS et les six descripteurs globaux")
    descriptors.add_argument("--with-band-descriptors", action="store_true",
                             help="ajouter quatre descripteurs spectraux par bande du mode avg4")
    a = p.parse_args()
    if a.cache_tag and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", a.cache_tag):
        p.error("--cache-tag doit contenir uniquement des lettres, chiffres, tirets ou underscores")
    if not (a.band_level_group or a.full_spectrum or a.include_b48):
        a.band_level_group = DEFAULT_BAND_GROUP
    if a.with_descriptors and not a.band_level_group:
        p.error("--with-descriptors nécessite un mode --band-levels")
    if a.with_band_descriptors and a.band_level_group != DEFAULT_BAND_GROUP:
        p.error("--with-band-descriptors nécessite le mode --band-levels-avg4")
    export_classification = (
        (a.rate, a.window_ms, a.stride_ms) == (MASTER, WINDOW_MS, STRIDE_MS)
        and a.band_level_group == DEFAULT_BAND_GROUP
        and not (a.cache_tag or a.with_descriptors or a.with_band_descriptors)
    )
    try:
        frame_count = n_frames(a.rate, a.window_ms)
        step_samples = stride_samples(a.rate, a.stride_ms)
    except ValueError as exc:
        p.error(str(exc))
    rows, frames, classification_frames = build_rows(
        a.data, a.rate, window_ms=a.window_ms, stride_ms=a.stride_ms, include_b48=a.include_b48,
        band_level_group=a.band_level_group, include_descriptors=a.with_descriptors,
        include_band_descriptors=a.with_band_descriptors, export_classification=export_classification,
    )
    if export_classification and not classification_frames:
        raise ValueError("aucune fenêtre valide pour la classification")
    splits_path = INPUT / "splits.json"
    if splits_path.is_file():
        table = json.loads(splits_path.read_text())
        if any(sid not in {r["sample_id"] for r in rows} for sid in table):
            raise ValueError("input/splits.json référence des prises absentes du corpus")
        for entry in table.values():
            if entry["split"] == "dev":
                entry["split"] = "validation" if entry["fold"] == 0 else "train"
            entry.pop("fold", None)
        print(f"partitions réutilisées : {splits_path}")
    else:
        assign_splits(rows)
        table = {r["sample_id"]: dict(group_id=r["group_id"], split=r["split"])
                 for r in rows if r["valid_windows"]}
        print(f"partitions créées : {OUTPUT / 'splits.json'}")
    for r in rows:
        info = table.get(r["sample_id"])
        if info is None:
            r.update(group_id=None, split="excluded")
        else:
            r.update(group_id=info["group_id"], split=info["split"])
    keys = ["sample_id", "device_id", "group_id", "split", "drone_present", "distance_m"]
    meta = pd.DataFrame(rows)[keys]
    band_cols = resolve_features(a.band_level_group) if a.band_level_group else []
    if a.band_level_group:
        feature_cols = list(band_cols)
        count, average = BAND_LEVEL_LAYOUTS[a.band_level_group]
        tag = f"{a.rate}-bands{count}"
        if average > 1:
            tag += f"-avg{average}"
        if a.with_descriptors:
            feature_cols = [*resolve_features("rms,handcrafted"), *band_cols]
            tag += "-descriptors"
        if a.with_band_descriptors:
            feature_cols += resolve_features(BAND_DESCRIPTOR_GROUP)
            tag += "-band-descriptors"
    else:
        feature_cols = list(FLAT_COLS)
        if a.include_b48:
            feature_cols += resolve_features("b48")
        feature_cols.append(LOGFFT)
        tag = f"{a.rate}-b48" if a.include_b48 else str(a.rate)
    id_cols = [*keys, "start_sample"]
    windows = (pd.concat(frames, ignore_index=True).merge(meta, on="sample_id", validate="many_to_one")
               .sort_values(["sample_id", "start_sample"]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_tag = a.cache_tag
    if not cache_tag and (a.window_ms, a.stride_ms) == (WINDOW_MS, STRIDE_MS):
        cache_tag = DEFAULT_CACHE_TAG
    if cache_tag:
        tag += "-" + cache_tag
    parquet = OUTPUT / f"windows-{tag}.parquet"
    windows[id_cols + feature_cols].to_parquet(parquet, index=False)
    (OUTPUT / "splits.json").write_text(json.dumps(table, indent=2) + "\n")
    fir = antialias_fir(a.rate)
    manifest = dict(rate=a.rate, window_ms=a.window_ms, stride_ms=a.stride_ms,
                    stride_samples=step_samples, source=str(a.data.resolve()), seed=SEED,
                    fir=None if fir is None else dict(taps=len(fir), cutoff_hz=11000 if a.rate == 24000 else 7333),
                    takes=sorted(rows, key=lambda r: r["sample_id"]))
    if a.band_level_group:
        manifest["band_levels"] = dict(count=len(band_cols), source_bands=count, average=average,
                                       bandwidth_hz=a.rate * average / (2 * count), scale="dBFS")
        if count % average:
            manifest["band_levels"]["last_bandwidth_hz"] = a.rate * (count % average) / (2 * count)
    if a.with_descriptors:
        manifest.update(handcrafted=HANDCRAFTED_COLS, rms=True)
    if a.with_band_descriptors:
        manifest["per_band_descriptors"] = BAND_DESCRIPTOR_NAMES
    if not a.band_level_group:
        manifest.update(logfft=dict(bins=N_BINS, frames=frame_count, fft=FFT, hop=HOP,
                                    layout=f"indice = bin * {frame_count} + trame"),
                        handcrafted=HANDCRAFTED_COLS, include_b48=a.include_b48)
    manifest_path = OUTPUT / f"manifest-{tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if export_classification:
        classification_path = default_window_cache(OUTPUT, classification=True)
        classification_windows = (pd.concat(classification_frames, ignore_index=True)
                                  .merge(meta, on="sample_id", validate="many_to_one")
                                  .sort_values(["sample_id", "start_sample"]))
        classification_windows[id_cols + band_cols].to_parquet(classification_path, index=False)
        classification_counts = classification_windows["sample_id"].value_counts()
        classification_takes = []
        for take in manifest["takes"]:
            count = int(classification_counts.get(take["sample_id"], 0))
            classification_take = dict(take, valid_windows=count)
            if count:
                classification_take.pop("reason", None)
            else:
                classification_take.setdefault("reason", "aucune fenêtre valide")
            classification_takes.append(classification_take)
        classification_manifest = dict(
            manifest, stride_ms=CLASSIFICATION_STRIDE_MS,
            stride_samples=stride_samples(a.rate, CLASSIFICATION_STRIDE_MS), takes=classification_takes,
        )
        classification_info = (OUTPUT / classification_path.name.replace("windows-", "manifest-")).with_suffix(".json")
        classification_info.write_text(json.dumps(classification_manifest, indent=2) + "\n")
    print(windows.groupby(["split", "drone_present"]).agg(fenetres=("sample_id", "size"),
                                                         prises=("sample_id", "nunique")))
    for stage in ("02_ppc_classification", "03_ppc_distance"):
        (STAGE.parent / stage / "input").mkdir(parents=True, exist_ok=True)
        paths = [parquet, manifest_path, OUTPUT / "splits.json"]
        if export_classification and stage == "02_ppc_classification":
            paths.extend((classification_path, classification_info))
        for path in paths:
            copyfile(path, STAGE.parent / stage / "input" / path.name)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    copyfile(OUTPUT / "splits.json", splits_path)  # le prochain taux reprendra ces partitions
    print(f"sortie copiée vers 02/input et 03/input ; parquet : {parquet}")


if __name__ == "__main__":
    main()
