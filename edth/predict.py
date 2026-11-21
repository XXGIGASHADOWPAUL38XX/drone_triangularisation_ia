"""Replay a strict mono PCM16/48 kHz master WAV, not microphone/network capture."""
import argparse
import json
import wave
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import features_from_pcm48


def predict_file(wav_path, device_id, classifier, regressor):
    with wave.open(str(wav_path), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 48000, 'NONE'):
            raise ValueError('Expected a mono PCM16 / 48 kHz master WAV')
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
    features = pd.DataFrame(features_from_pcm48(pcm))
    if features.empty:
        return []
    features['device_id'] = device_id
    probabilities = classifier.predict(features)
    present = probabilities >= 0.5
    distances = np.full(len(features), np.nan)
    if present.any():
        distances[present] = regressor.predict(features.loc[present])
    return [dict(start_s=float(start)/24000, probability=float(p), distance_m=None if not np.isfinite(d) else float(d))
            for start, p, d in zip(features.start_sample_24k, probabilities, distances, strict=True)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True, help='Trusted local release only (joblib executes code)')
    ap.add_argument('--wav', type=Path, required=True)
    ap.add_argument('--device-id', required=True)
    a = ap.parse_args()
    from .release import verify_bundle
    provenance = verify_bundle(a.bundle)
    if a.device_id not in provenance['device_ids']:
        ap.error('Unknown device_id: distance calibration not available')
    models = [joblib.load(a.bundle / task / 'model.joblib') for task in ('classification', 'distance')]
    rows = predict_file(a.wav, a.device_id, *models)
    print(json.dumps({'valid_windows': len(rows), 'windows': rows}, allow_nan=False))


if __name__ == '__main__':
    main()
