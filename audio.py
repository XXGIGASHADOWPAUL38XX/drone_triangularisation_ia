"""Chaîne audio : FIR anti-repliement, décimation, contrôle qualité, fenêtrage.

Les WAV maîtres sont en PCM16 mono 48 kHz. Le taux cible est 16000, 24000
(référence historique : parité exacte avec les features des modèles déjà
entraînés) ou 48000 (aucun filtrage). Fenêtres d'exactement 1 s, pas 250 ms.
Le masque qualité reste calculé à 48 kHz, avant filtrage.
"""
import wave
from pathlib import Path

import numpy as np

MASTER_RATE = 48000
RATES = (16000, 24000, 48000)
DROPOUT_SAMPLES = 1920  # coupure numérique suspecte : suite de zéros >= 40 ms à 48 kHz
GUARD_SAMPLES = 24000  # garde de 500 ms après le démarrage et après chaque coupure


def pcm16(a):
    a = np.asarray(a)
    if a.ndim != 1 or a.dtype.kind != 'i' or a.dtype.itemsize != 2:
        raise ValueError('attendu un tableau mono PCM16 signé')
    return a


def antialias_fir(rate):
    """FIR passe-bas causal symétrique, fenêtre Kaiser. None si rien à filtrer."""
    if rate == MASTER_RATE:
        return None
    if rate not in RATES:
        raise ValueError(f'taux supportés : {RATES}')
    # 24 kHz : conception historique exacte (161 coefficients, coupure 11 kHz),
    # à ne pas modifier pour préserver la parité avec les caches existants.
    cutoff, taps = (11000, 161) if rate == 24000 else (7333, 241)
    fir = np.sinc(2 * cutoff / MASTER_RATE * (np.arange(taps) - taps // 2)) * np.kaiser(taps, 8.6)
    fir /= fir.sum()
    fir.setflags(write=False)
    return fir


class Decimator:
    """48 kHz PCM16 -> taux cible PCM16, état conservé entre les blocs.

    Une instance par flux contigu ; après une vraie rupture, repartir sur une
    nouvelle instance. process() ne fait que rééchantillonner : appliquer les
    masques qualité 48 kHz en amont, jamais ici.
    """
    def __init__(self, rate):
        self.rate = rate
        self.fir = antialias_fir(rate)
        self.ratio = MASTER_RATE // rate
        self.history = np.zeros(0 if self.fir is None else len(self.fir) - 1, dtype=np.int16)
        self.phase = 0

    def process(self, pcm):
        x = pcm16(pcm)
        if self.fir is None:
            return x
        if not x.size:
            return np.empty(0, dtype=np.int16)
        padded = np.concatenate((self.history, x))
        filtered = np.convolve(padded, self.fir, mode='valid')[self.phase::self.ratio]
        self.history = padded[-(len(self.fir) - 1):].copy()
        self.phase = (self.phase + len(x)) % self.ratio
        return np.clip(np.rint(filtered), -32768, 32767).astype(np.int16)


def read_wav48(path):
    """Lit un WAV maître strict : mono PCM16 48 kHz -> tableau int16."""
    with wave.open(str(path), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (48000, 1, 2, 'NONE'):
            raise ValueError(f'pas un WAV mono PCM16 48 kHz : {path}')
        n = w.getnframes()
        raw = w.readframes(n)
    if len(raw) != 2 * n:
        raise ValueError(f'WAV tronqué : {path}')
    return np.frombuffer(raw, '<i2')


def iter_windows(pcm48, rate):
    """Fenêtres valides de 1 s (pas 250 ms) depuis un flux contigu à 48 kHz.

    Rend (index de départ au taux cible, fenêtre PCM16 au taux cible).
    Sont écartées : les 500 premières ms, toute fenêtre chevauchant une suite
    de zéros >= 40 ms et toute fenêtre débutant moins de 500 ms après sa fin.
    """
    x = pcm16(pcm48)
    fir = antialias_fir(rate)
    factor = MASTER_RATE // rate
    if fir is None:
        y = x
    else:
        y = Decimator(rate).process(x)
    if len(y) < rate:
        return
    # Contrôle qualité à 48 kHz, avant filtrage, quel que soit le taux cible.
    zero = (x == 0)
    changes = np.flatnonzero(np.diff(zero, prepend=False, append=False)).reshape(-1, 2)
    gaps = changes[changes[:, 1] - changes[:, 0] >= DROPOUT_SAMPLES]
    for i in range(rate // 2, len(y) - rate + 1, rate // 4):
        if np.any((gaps[:, 0] < factor * (i + rate)) & (gaps[:, 1] + GUARD_SAMPLES > factor * i)):
            continue
        yield i, y[i:i + rate]
