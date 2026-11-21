"""Features par fenêtre valide d'une seconde.

Cache complet (une seule passe, la sélection se fait à l'entraînement) :
  - f0..f31 : niveaux dBFS de 32 bandes espacées sur l'échelle mel, jusqu'au
    Nyquist du taux cible ;
  - b4_8   : puissance intégrée 4-8 kHz en dBFS ;
  - rms_db : niveau RMS pleine bande en dBFS ;
  - logmel : spectrogramme log-mel 32 canaux x T trames, aplati mel-major
    (indice = mel * T + trame), float32, plancher -120 dB.

À 24 kHz, le calcul est identique au cache historique pilot-v2 (parité
exacte). Toutes les grandeurs sont absolues, aucune normalisation apprise.
"""
from functools import lru_cache

import numpy as np

from audio import pcm16

N_BANDS = 32
FFT_SIZE, HOP = 2048, 1024
BAND_COLS = [f'f{i}' for i in range(N_BANDS)]
FLAT_COLS = BAND_COLS + ['b4_8', 'rms_db']
LOGMEL = 'logmel'
ALL_COLS = FLAT_COLS + [LOGMEL]
# Groupes nommés sélectionnables à l'entraînement :  --features mel,b48,rms
FEATURE_SETS = {
    'mel': BAND_COLS,
    'b48': ['b4_8'],
    'rms': ['rms_db'],
}


def resolve_features(spec):
    """'mel,b48,rms' -> liste de colonnes plates, dans un ordre fixe."""
    tokens = [t.strip() for t in spec.split(',') if t.strip()]
    unknown = [t for t in tokens if t not in FEATURE_SETS]
    if unknown:
        raise ValueError(f"groupes de features inconnus : {unknown} (choisir parmi {list(FEATURE_SETS)})")
    if not tokens:
        return list(FLAT_COLS)
    cols = []
    for name in ('mel', 'b48', 'rms'):  # ordre stable, indépendant de l'ordre saisi
        if name in tokens:
            cols += FEATURE_SETS[name]
    return cols


def n_frames(rate):
    """Nombre de trames couvrant exactement 1 s (FFT 2048, pas 1024)."""
    return (rate - FFT_SIZE) // HOP + 1


def logmel_len(rate):
    return N_BANDS * n_frames(rate)


@lru_cache(maxsize=None)
def _layout(rate):
    nyquist = rate // 2
    freq = np.fft.rfftfreq(FFT_SIZE, 1 / rate)
    # 33 bords -> 32 bandes intégrées (reduceat), bords extrêmes forcés.
    edges = 700 * np.expm1(np.linspace(0, np.log1p(nyquist / 700), N_BANDS + 1))
    edges[[0, -1]] = [0, nyquist]
    # 34 bords -> 32 triangles mel non normalisés (sommet = bord médian).
    mel_edges = 700 * np.expm1(np.linspace(0, np.log1p(nyquist / 700), N_BANDS + 2))
    mel_edges[[0, -1]] = [0, nyquist]
    filters = np.clip(
        np.minimum(
            (freq[None, :] - mel_edges[:-2, None]) / np.maximum(mel_edges[1:-1, None] - mel_edges[:-2, None], 1e-12),
            (mel_edges[2:, None] - freq[None, :]) / np.maximum(mel_edges[2:, None] - mel_edges[1:-1, None], 1e-12),
        ),
        0, None,
    )
    filters.setflags(write=False)
    hann = np.hanning(FFT_SIZE + 1)[:-1]
    hann.setflags(write=False)
    return (np.searchsorted(freq, edges[:-1]), filters, freq, hann, mel_edges)


def _db(power):
    return 10 * np.log10(np.maximum(power, 1e-12))


def window_features(pcm, rate):
    """Toutes les features d'une fenêtre valide d'exactement 1 s au taux cible."""
    x = pcm16(pcm).astype('float64') / 32768
    if len(x) != rate:
        raise ValueError(f'attendu exactement 1 s ({rate} échantillons), reçu {len(x)}')
    bins, mel_filters, freq, hann, _ = _layout(rate)
    frames = np.lib.stride_tricks.sliding_window_view(x, FFT_SIZE)[::HOP]
    if len(frames) != n_frames(rate):
        raise ValueError(f'attendu {n_frames(rate)} trames, reçu {len(frames)}')
    spectrum = abs(np.fft.rfft(frames * hann, axis=1)) ** 2
    p = spectrum.mean(axis=0)
    p /= (hann @ hann) * FFT_SIZE
    p[1:-1] *= 2  # normalisation unilatérale
    bands = _db(np.add.reduceat(p, bins))
    per_frame = spectrum / ((hann @ hann) * FFT_SIZE)
    per_frame[:, 1:-1] *= 2
    logmel = _db(per_frame @ mel_filters.T).T.reshape(-1).astype(np.float32)
    out = dict(zip(BAND_COLS, bands.tolist()))
    out['b4_8'] = float(_db(p[(freq >= 4000) & (freq < 8000)].sum()))
    out['rms_db'] = float(_db(np.mean(x * x)))
    out[LOGMEL] = logmel
    return out
