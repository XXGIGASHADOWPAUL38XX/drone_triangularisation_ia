"""Construit le cache de features et les splits groupés depuis les WAV maîtres.

Emploi :
  python 00_prepare.py --local /home/noe/Projects/EDTH_data-collector/data
  python 00_prepare.py --repo ImAPancake/EDTH_drone_distance_raw --revision <sha>
Options : --rate {16000,24000,48000} (défaut 24000), --out-dir artifacts.

Sorties :
  artifacts/windows-<rate>.parquet : une ligne par fenêtre valide, features + splits ;
  artifacts/windows-<rate>.json    : manifeste (provenance, FIR, prise par prise) ;
  artifacts/splits.json            : partitions groupées, créées au premier appel
                                      puis RÉUTILISÉES pour tous les taux, afin que
                                      chaque expérience compare sur les mêmes folds.

Méthode : coupures numériques >= 40 ms à 48 kHz, garde 500 ms, fenêtres 1 s /
pas 250 ms, niveaux absolus. Groupes = intervalles de capture qui se
chevauchent (tolérance 1 s), tous appareils confondus ; les buckets test/dev
sont équilibrés sur les métadonnées (classes, distances, appareils), jamais
sur les features ni les scores. Les prises hors contrat strict (mode, capture)
sont exclues avec motif, pas corrigées.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from audio import MASTER_RATE, antialias_fir, iter_windows, read_wav48
from features import ALL_COLS, N_BANDS, n_frames, window_features

SEED = 143  # graine des partitions historiques, à ne pas changer sans motif


def entries(root):
    """Itère (métadonnées, chemin WAV) : layout samples/ ou AudioFolder CSV."""
    root = Path(root)
    if (root / 'samples').is_dir():
        for m in sorted((root / 'samples').glob('*/metadata.json')):
            yield json.loads(m.read_text()), m.with_name('audio.wav')
        return
    path = root / 'metadata.csv'
    if not path.is_file():
        path = root / 'data/metadata.csv'
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name = row['file_name']
            if name != Path(name).name or Path(name).suffix.lower() != '.wav':
                raise ValueError('attendu un nom de fichier WAV plat (AudioFolder)')
            row['capture_settings'] = json.loads(row['capture_settings'])
            yield row, path.parent / name


def label(v):
    if str(v).lower() not in ('true', 'false'):
        raise ValueError('étiquette drone_present invalide')
    return str(v).lower() == 'true'


def events(rows):
    """Composantes connexes d'intervalles de capture, tous appareils, + 1 s de tolérance."""
    end, group = -np.inf, -1
    for r in sorted(rows, key=lambda r: (r['start_utc'], r['sample_id'])):
        if r['start_utc'] > end + 1:
            group += 1
        r['group_id'] = f'event-{group:04d}'
        end = max(end, r['start_utc'] + r['duration_s'])


def assign_splits(rows, seed=SEED):
    """Quatre buckets de taille voisine : test + trois plis dev, équilibrés
    sur les comptages par prise (classes, distances, appareils). Échoue
    explicitement plutôt que de retomber sur un découpage aléatoire."""
    live = [r for r in rows if r['valid_windows']]
    groups = sorted({r['group_id'] for r in live})
    devices = sorted({r['device_id'] for r in live})
    distances = sorted({r['distance_m'] for r in live if r['drone_present'] and 1 <= r['distance_m'] <= 5})
    if len(groups) < 4 or len(distances) < 2:
        raise ValueError('trop peu de groupes ou de distances pour 3 plis + holdout')
    x = np.zeros((len(groups), 3 + len(distances) + len(devices)))
    for r in live:
        g = x[groups.index(r['group_id'])]
        g[0] += 1
        if not r['drone_present']:
            g[1] += 1
        if r['distance_m'] in distances:
            g[2] += 1
            g[3 + distances.index(r['distance_m'])] += 1
        g[3 + len(distances) + devices.index(r['device_id'])] += 1
    rng = np.random.default_rng(seed)
    target = x.sum(axis=0) / 4
    best, chosen = np.inf, None
    for _ in range(8192):
        labels = rng.permutation(np.arange(len(groups)) % 4)
        counts = np.stack([x[labels == k].sum(axis=0) for k in range(4)])
        if np.any(counts[:, 1:3] < 1) or np.any((counts[:, 3:3 + len(distances)] > 0).sum(axis=1) < 2):
            continue
        score = (((counts - target) / np.maximum(target, 1)) ** 2).sum()
        if score < best:
            best, chosen = score, labels
    if chosen is None:
        raise ValueError('aucune partition groupée ne couvre classes et distances dans chaque bucket')
    lookup = dict(zip(groups, chosen.tolist()))
    for r in rows:
        bucket = lookup.get(r['group_id'])
        r['split'] = 'excluded' if bucket is None else 'test' if bucket == 0 else 'dev'
        r['fold'] = bucket - 1 if bucket else -1


def build_rows(root, rate):
    """Valide les métadonnées, lit chaque WAV maître strict, calcule les fenêtres."""
    rows, frames, seen = [], [], set()
    for m, path in entries(root):
        sid = m['sample_id']
        if not sid or sid in seen:
            raise ValueError(f'identifiant de prise absent ou dupliqué : {sid}')
        seen.add(sid)
        when = datetime.fromisoformat(m['captured_at'].replace('Z', '+00:00'))
        if when.tzinfo is None or not m['device_id']:
            raise ValueError(f'fuseau ou appareil manquant : {sid}')
        present = label(m['drone_present'])
        d = m.get('distance_m')
        d = float(d) if d not in (None, '') else None
        if present != (d is not None) or (d is not None and not (np.isfinite(d) and d > 0)):
            raise ValueError(f'étiquettes incohérentes : {sid}')
        r = dict(sample_id=sid, device_id=m['device_id'], start_utc=when.timestamp(),
                 drone_present=present, distance_m=d, duration_s=float(m['duration_seconds']),
                 valid_windows=0)
        rows.append(r)
        if m['mode'] != 'hand' or m['capture_settings'].get('capture_contract') != 'edth.web-pcm.v1':
            r['reason'] = 'hors contrat de capture strict en main'
            continue
        pcm = read_wav48(path)
        r['duration_s'] = len(pcm) / MASTER_RATE
        r['pcm_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
        wins = [{'start_sample': i, **window_features(w, rate)} for i, w in iter_windows(pcm, rate)]
        r['valid_windows'] = len(wins)
        if wins:
            frames.append(pd.DataFrame(wins).assign(sample_id=sid))
        else:
            r['reason'] = 'aucune fenêtre valide'
    if not frames:
        raise ValueError('aucune fenêtre valide dans tout le corpus : vérifier la source')
    events(rows)
    return rows, frames


def load_or_create_splits(rows, splits_path):
    """Partitions créées une seule fois, puis réutilisées pour tous les taux."""
    if splits_path.is_file():
        table = json.loads(splits_path.read_text())
        missing = [sid for sid in table if not any(r['sample_id'] == sid for r in rows)]
        if missing:
            raise ValueError(f'splits.json référence des prises absentes du corpus : {missing[:3]}')
    else:
        assign_splits(rows)
        table = {r['sample_id']: dict(group_id=r['group_id'], split=r['split'], fold=r['fold'])
                 for r in rows if r['valid_windows']}
        splits_path.write_text(json.dumps(table, indent=2, allow_nan=False) + '\n')
        print(f'partitions groupées créées : {splits_path}')
    for r in rows:
        info = table.get(r['sample_id'])
        if info is None:
            r.update(group_id=None, split='excluded', fold=-1)
        else:
            r.update(group_id=info['group_id'], split=info['split'], fold=info['fold'])
    return table


def prepare(root, out_dir, rate, source=None):
    rows, frames = build_rows(Path(root), rate)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = out_dir / 'splits.json'
    load_or_create_splits(rows, splits_path)
    keys = ['sample_id', 'device_id', 'group_id', 'split', 'fold', 'drone_present', 'distance_m']
    meta = pd.DataFrame(rows)[keys]
    windows = (pd.concat(frames, ignore_index=True).merge(meta, on='sample_id', validate='many_to_one')
               .sort_values(['sample_id', 'start_sample'])[keys + ['start_sample', *ALL_COLS]])
    parquet = out_dir / f'windows-{rate}.parquet'
    windows.to_parquet(parquet, index=False)
    fir = antialias_fir(rate)
    manifest = dict(
        rate=rate, source=source or {'local': str(Path(root).resolve())},
        grouping='intervalles de capture chevauchants (tolérance 1 s), même campagne ; pas des sessions indépendantes',
        seed=SEED, splits_file=splits_path.name,
        fir=None if fir is None else dict(taps=len(fir), cutoff_hz=(11000 if rate == 24000 else 7333),
                                          sha256=hashlib.sha256(fir.astype('<f8').tobytes()).hexdigest()),
        logmel=dict(mels=N_BANDS, frames=n_frames(rate), fft=2048, hop=1024,
                    layout=f'mel-major : indice = mel * {n_frames(rate)} + trame',
                    scale='dBFS absolus, plancher -120 dB'),
        takes=sorted(rows, key=lambda r: r['sample_id']))
    (out_dir / f'windows-{rate}.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(windows.groupby(['split', 'drone_present'])
          .agg(fenetres=('sample_id', 'size'), prises=('sample_id', 'nunique')))
    return parquet


def resolve_root(args):
    if args.repo:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / '.env', override=False)
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(args.repo, revision=args.revision, repo_type='dataset',
                                      allow_patterns=['data/*.wav', 'data/metadata.csv', 'samples/*']))
        return root, {'repo': args.repo, 'revision': args.revision}
    if args.revision:
        raise SystemExit('--revision exige --repo')
    return Path(args.local), None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--local', type=Path)
    src.add_argument('--repo')
    p.add_argument('--revision')
    p.add_argument('--rate', type=int, choices=(16000, 24000, 48000), default=24000)
    p.add_argument('--out-dir', type=Path, default=Path('artifacts'))
    a = p.parse_args()
    root, source = resolve_root(a)
    prepare(root, a.out_dir, a.rate, source)


if __name__ == '__main__':
    main()
