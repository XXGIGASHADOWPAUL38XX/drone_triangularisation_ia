from math import gcd
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy.signal import resample_poly
import features as f

SAMPLE_RATE = 16000
STAGE = Path(__file__).resolve().parent
INPUT = STAGE / "input/train"
OUTPUT = STAGE / "output/features.pkl"
CLASSIFICATION_INPUT = STAGE.parent / "02_ppc_classification/input/features.pkl"


def to_audio_dataframe(dataset):
    df = dataset.remove_columns("audio").to_pandas()
    audio_arrays = np.empty(len(df), dtype=object)
    sample_rates = []
    for index, row in enumerate(dataset.select_columns("audio")):
        audio = row["audio"]
        samples = audio.get_all_samples() if audio is not None else None
        audio_arrays[index] = samples.data.T.numpy() if samples is not None else None
        sample_rates.append(samples.sample_rate if samples is not None else None)
    return df.assign(audio=audio_arrays, sample_rate=sample_rates)


def downsample(df, sr=SAMPLE_RATE):
    if sr <= 0:
        raise ValueError("sample rate must be positive")

    def resample(audio, original_rate):
        if audio is None or audio.size == 0 or original_rate == sr:
            return audio
        original_rate = int(original_rate)
        factor = gcd(sr, original_rate)
        return resample_poly(audio, sr // factor, original_rate // factor, axis=0)

    audio = [resample(waveform, rate) for waveform, rate in zip(df["audio"], df["sample_rate"])]
    sample_rates = df["sample_rate"].where(df["audio"].isna(), sr)
    return df.assign(audio=audio, sample_rate=sample_rates)


def window_audio(df, size=200):
    if size <= 0:
        raise ValueError("window size must be positive")

    def split(audio):
        n = len(audio) // size
        return audio[:n * size].reshape(n, size, audio.shape[1])

    windows = df["audio"].map(split)
    columns = ["audio", "sample_rate", "sample_id", "distance_m"]
    full_windows = df.assign(audio=windows).loc[windows.map(len) > 0, columns]
    return full_windows.explode("audio", ignore_index=True)


def remove_null_audio(df):
    return df.loc[df["audio"].map(lambda audio: audio is not None and audio.any())].copy()

def fourier_transform(df):
    fft = df["audio"].map(lambda audio: np.abs(np.fft.rfft(audio[:, 0], axis=0)))
    df.insert(len(df.columns), "fft", fft)
    return df

def add_features(rows):
    raw = np.stack(rows["audio"])[:, :, 0]
    fft = np.stack(rows["fft"])
    hz = np.fft.rfftfreq(raw.shape[1], 1 / rows["sample_rate"].iat[0])
    bands = f.get_multibands(fft, 0.25)
    scores = dict(
        nonzero_count=f.get_abs_amp_nonzero_count(bands),
        avg_frequency=f.get_abs_amp_avg_frequency(bands, hz),
        amp_std=f.get_abs_amp_std(bands),
        entropy=f.get_mb_entropy(bands),
    )
    return rows.assign(
        peak=f.get_abs_amp_max(abs(raw)), zcr=f.get_mb_zcr(raw),
        crest_factor=f.get_mb_crest_factor(raw), centroid=f.get_mb_s_centroid(fft, hz),
        flux=f.get_mb_s_flux(fft), onset=f.get_mb_onset(fft),
        **{f"{name}_{i}": values for name, bands in scores.items()
           for i, values in enumerate(bands, 1)},
    )

def preprocess(dataset):
    raw_audio = to_audio_dataframe(dataset)
    downsampled_audio = downsample(raw_audio)
    no_null_raw_audio = remove_null_audio(downsampled_audio)
    windowed_audio = remove_null_audio(window_audio(no_null_raw_audio))
    fft_audio = fourier_transform(windowed_audio)
    if fft_audio.empty:
        return fft_audio
    return pd.concat(add_features(rows) for _, rows in fft_audio.groupby("sample_id", sort=False))


def main():
    features = preprocess(load_from_disk(str(INPUT)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_INPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_pickle(OUTPUT)
    copyfile(OUTPUT, CLASSIFICATION_INPUT)


if __name__ == "__main__":
    main()
