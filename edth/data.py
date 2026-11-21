"""Offline reference for EDTH mono PCM. No learned transforms or per-take scaling.

features_from_pcm48 expects a whole contiguous capture; do not call per packet.
features_from_pcm24 is the identical last step for an already valid live window.
"""
import argparse
import csv
import hashlib
import json
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RATE = 24000
STEP = 6000
VERSION = 'edth.audio.24k.v2'
# 48 kHz Kaiser-windowed, unity-DC FIR. Causal, initial zero state, phase 0.
FIR = np.sinc(2 * 11000 / 48000 * (np.arange(161) - 80)) * np.kaiser(161, 8.6)
FIR /= FIR.sum()
# Mel-spaced bands including DC and Nyquist. Periodic Hann, density integrated
# over FFT bins with one-sided power normalization and no detrending.
EDGES = 700 * np.expm1(np.linspace(0, np.log1p(12000 / 700), 33))
EDGES[[0, -1]] = [0, 12000]
FREQ = np.fft.rfftfreq(2048, 1 / RATE)
BINS = np.searchsorted(FREQ, EDGES[:-1])
HANN = np.hanning(2049)[:-1]
COLS = [f'f{i}' for i in range(32)]
# Temporal log-mel: 32 mel channels x 22 frames over exactly one valid second.
# FFT 2048 / hop 1024 gives floor((24000-2048)/1024)+1 = 22 frames.
N_MELS, N_FRAMES, FFT_SIZE, HOP = 32, 22, 2048, 1024
LOGMEL_LEN = N_MELS * N_FRAMES
LOGMEL_FIELD = 'logmel'
# 34 boundaries, uniformly spaced in mel, give 32 full triangles (centre =
# middle edge). Triangles are unnormalised, so a full-scale tone reads about
# the same dBFS as the tabular band that contains it.
MEL_EDGES = 700 * np.expm1(np.linspace(0, np.log1p(12000 / 700), N_MELS + 2))
MEL_EDGES[[0, -1]] = [0, 12000]
MEL_CENTERS = MEL_EDGES[1:-1]  # triangle peak = middle boundary
MEL_FILTERS = np.clip(
    np.minimum(
        (FREQ[None, :] - MEL_EDGES[:-2, None]) / np.maximum(MEL_EDGES[1:-1, None] - MEL_EDGES[:-2, None], 1e-12),
        (MEL_EDGES[2:, None] - FREQ[None, :]) / np.maximum(MEL_EDGES[2:, None] - MEL_EDGES[1:-1, None], 1e-12),
    ),
    0,
    None,
)
MEL_FILTERS.setflags(write=False)


def _pcm(a):
    a = np.asarray(a)
    if a.ndim != 1 or a.dtype.kind != 'i' or a.dtype.itemsize != 2:
        raise ValueError('expected mono signed PCM16 array')
    return a


def _db(power):
    return 10 * np.log10(np.maximum(power, 1e-12))


def features_from_pcm24(pcm):
    """Compute dBFS features on exactly 24000 valid 24 kHz S16 samples.

    Returns the 32 tabular band powers, the 4–8 kHz power, RMS, and one
    ``logmel`` vector: 32 mel channels x 22 frames flattened mel-major, so
    index ``m*22 + t`` holds channel ``m`` (low to high) at frame ``t``
    (earliest first). Values are float32 dBFS with a -120 dB floor.
    """
    x = _pcm(pcm).astype('float64') / 32768
    if len(x) != RATE:
        raise ValueError('expected exactly 1 s at 24 kHz')
    frames = np.lib.stride_tricks.sliding_window_view(x, FFT_SIZE)[::HOP]
    if len(frames) != N_FRAMES:  # guards the published logmel layout
        raise ValueError(f'expected {N_FRAMES} frames, got {len(frames)}')
    spectrum = abs(np.fft.rfft(frames * HANN, axis=1)) ** 2
    p = spectrum.mean(axis=0)
    p /= (HANN @ HANN) * FFT_SIZE
    p[1:-1] *= 2
    bands = _db(np.add.reduceat(p, BINS))
    per_frame = spectrum / ((HANN @ HANN) * FFT_SIZE)
    per_frame[:, 1:-1] *= 2
    logmel = _db(per_frame @ MEL_FILTERS.T).T.reshape(-1).astype(np.float32)
    return {**dict(zip(COLS, bands.tolist())), 'b4_8': float(_db(p[(FREQ >= 4000) & (FREQ < 8000)].sum())),
            'rms_db': float(_db(np.mean(x * x))), LOGMEL_FIELD: logmel}


class Decimator48to24:
    """Stateful 48 kHz S16 -> 24 kHz S16. Use one instance per contiguous stream.

    process() only resamples, not QC: apply the raw 48 kHz quality masks before
    accepting output windows. On a genuine stream interruption start anew.
    """
    def __init__(self):
        self.history = np.zeros(len(FIR) - 1, dtype=np.int16)
        self.phase = 0

    def process(self, pcm):
        x = _pcm(pcm)
        if not x.size:
            return np.empty(0, dtype=np.int16)
        padded = np.concatenate((self.history, x))
        filtered = np.convolve(padded, FIR, mode='valid')[self.phase::2]
        self.history = padded[-(len(FIR) - 1):].copy()
        self.phase = (self.phase + len(x)) % 2
        return np.clip(np.rint(filtered), -32768, 32767).astype(np.int16)


def features_from_pcm48(pcm):
    """Yield valid windows from contiguous 48 kHz S16, time on decimated grid.

    Apply a 500 ms guard at startup and after every >=40 ms digital dropout.
    A true discontinuity requires a new call. Never concatenate valid islands.
    """
    x = _pcm(pcm)
    if len(x) < 48000:
        return
    zero = (x == 0)
    changes = np.flatnonzero(np.diff(zero, prepend=False, append=False)).reshape(-1, 2)
    gaps = changes[changes[:, 1] - changes[:, 0] >= 1920]
    y = Decimator48to24().process(x)
    for i in range(RATE // 2, len(y) - RATE + 1, STEP):
        if np.any((gaps[:, 0] < 2 * (i + RATE)) & (gaps[:, 1] + RATE > 2 * i)):
            continue
        yield {'start_sample_24k': i, **features_from_pcm24(y[i:i + RATE])}


def _entries(root):
    if (root / 'samples').is_dir():
        for m in sorted((root / 'samples').glob('*/metadata.json')):
            yield json.loads(m.read_text()), m.with_name('audio.wav')
    else:
        path = root / 'metadata.csv'
        if not path.is_file():
            path = root / 'data/metadata.csv'
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                name = row['file_name']
                if name != Path(name).name or Path(name).suffix.lower() != '.wav':
                    raise ValueError('expected flat AudioFolder WAV filename')
                row['capture_settings'] = json.loads(row['capture_settings'])
                yield row, path.parent / name


def _label(v):
    if str(v).lower() not in ('true', 'false'):
        raise ValueError('invalid drone_present label')
    return str(v).lower() == 'true'


def _events(rows):
    """Connected components of intervals across devices, padded by 1 s."""
    end = -np.inf
    group = -1
    for r in sorted(rows, key=lambda r: (r['start_utc'], r['sample_id'])):
        if r['start_utc'] > end + 1:
            group += 1
        r['group_id'] = f'event-{group:04d}'
        end = max(end, r['start_utc'] + r['duration_s'])


def assign_splits(rows, seed=143):
    """Partition groups, optimizing label counts per take (not window counts).

    Four similarly sized buckets: test and three development folds. Search
    uses metadata counts only, never features or prediction performance.
    Fail explicitly rather than silently reverting to random window splits.
    """
    live = [r for r in rows if r['valid_windows']]
    groups = sorted({r['group_id'] for r in live})
    devices = sorted({r['device_id'] for r in live})
    distances = sorted({r['distance_m'] for r in live if r['drone_present'] and 1 <= r['distance_m'] <= 5})
    if len(groups) < 4 or len(distances) < 2:
        raise ValueError('too few groups or regression distances for 3-fold CV + holdout')
    # Aggregate takes, negatives, in-range positives, distance and device counts.
    x = np.zeros((len(groups), 3 + len(distances) + len(devices)))
    for r in live:
        g = x[groups.index(r['group_id'])]; g[0] += 1
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
        raise ValueError('no grouped partition supports both classes and two distances in every bucket')
    lookup = dict(zip(groups, chosen.tolist()))
    for r in rows:
        bucket = lookup.get(r['group_id'])
        r['split'] = 'excluded' if bucket is None else 'test' if bucket == 0 else 'dev'
        r['fold'] = bucket - 1 if bucket else -1


def prepare(root, out, source=None):
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'output must be a new or empty directory: {out}')
    rows, frames, seen = [], [], set()
    for m, path in _entries(Path(root)):
        sid = m['sample_id']
        if not sid or sid in seen:
            raise ValueError(f'duplicate or missing sample id: {sid}')
        seen.add(sid)
        when = datetime.fromisoformat(m['captured_at'].replace('Z', '+00:00'))
        if when.tzinfo is None or not m['device_id']:
            raise ValueError(f'missing timezone or device: {sid}')
        present = _label(m['drone_present'])
        d = m.get('distance_m'); d = float(d) if d not in (None, '') else None
        if present != (d is not None) or (d is not None and not (np.isfinite(d) and d > 0)):
            raise ValueError(f'inconsistent labels: {sid}')
        r = dict(sample_id=sid, device_id=m['device_id'], start_utc=when.timestamp(),
                 drone_present=present, distance_m=d, duration_s=float(m['duration_seconds']), valid_windows=0)
        rows.append(r)
        if m['mode'] != 'hand' or m['capture_settings'].get('capture_contract') != 'edth.web-pcm.v1':
            r['reason'] = 'outside strict handheld capture contract'; continue
        with wave.open(str(path), 'rb') as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (48000, 1, 2, 'NONE'):
                raise ValueError(f'not mono PCM16 at 48 kHz: {sid}')
            n = w.getnframes(); raw = w.readframes(n)
        if len(raw) != 2*n:
            raise ValueError(f'truncated WAV: {sid}')
        r['duration_s'] = n / 48000
        r['pcm_sha256'] = hashlib.sha256(raw).hexdigest()
        wins = list(features_from_pcm48(np.frombuffer(raw, '<i2')))
        r['valid_windows'] = len(wins)
        if wins:
            frames.append(pd.DataFrame(wins).assign(sample_id=sid))
        else:
            r['reason'] = 'no valid window'
    _events(rows); assign_splits(rows)
    keys = ['sample_id', 'device_id', 'group_id', 'split', 'fold', 'drone_present', 'distance_m']
    windows = pd.concat(frames, ignore_index=True).merge(pd.DataFrame(rows)[keys], on='sample_id', validate='many_to_one')
    windows = windows.sort_values(['sample_id', 'start_sample_24k'])[keys + ['start_sample_24k', *COLS, 'b4_8', 'rms_db', LOGMEL_FIELD]]
    out = Path(out); out.mkdir(exist_ok=True, parents=True)
    windows.to_parquet(out / 'windows.parquet', index=False)
    manifest = dict(version=VERSION, source=source or {'local': str(Path(root).resolve())},
                    grouping='overlapping capture intervals + 1s tolerance, same campaign; not independent sessions',
                    seed=143, fir=FIR.tolist(), band_edges_hz=EDGES.tolist(),
                    logmel=dict(field=LOGMEL_FIELD, dtype='float32', length=LOGMEL_LEN, mels=N_MELS, frames=N_FRAMES,
                                layout='mel-major: index = mel*22 + frame, mel low to high, frame earliest first',
                                fft_size=FFT_SIZE, hop=HOP, window='periodic Hann (no detrending)',
                                power='one-sided |rfft|^2, density-normalised, mean over overlapping frames',
                                filterbank='32 unnormalised triangular filters, 34 boundaries uniform in mel 0-12000 Hz',
                                mel_edges_hz=MEL_EDGES.tolist(), mel_centers_hz=MEL_CENTERS.tolist(),
                                filterbank_sha256=hashlib.sha256(MEL_FILTERS.astype('<f4').tobytes()).hexdigest(),
                                scale='dBFS, same convention as tabular bands, floor -120 dB',
                                replay='features_from_pcm24 on one valid 1 s window; features_from_pcm48 yields it unchanged'),
                    takes=sorted(rows, key=lambda r:r['sample_id']))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument('--local', type=Path); inp.add_argument('--repo')
    parser.add_argument('--revision'); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); source = None; root = args.local
    if args.repo:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / '.env', override=False)
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(args.repo, revision=args.revision, repo_type='dataset', allow_patterns=['data/*.wav', 'data/metadata.csv']))
        source = {'repo': args.repo, 'revision': root.name}
    elif args.revision:
        parser.error('--revision requires --repo')
    frame = prepare(root, args.out, source)
    print(frame.groupby(['split', 'fold', 'drone_present']).agg(windows=('sample_id', 'size'), takes=('sample_id', 'nunique'), groups=('group_id', 'nunique')))


if __name__ == '__main__':
    main()
