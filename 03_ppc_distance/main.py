from pathlib import Path

import pandas as pd

STAGE = Path(__file__).resolve().parent
INPUT = STAGE / "input/features.pkl"
OUTPUT = STAGE / "output/distance.pkl"


def preprocess(df):
    df = df.dropna(subset=["distance_m"])
    features = df.drop(
        columns=["audio", "fft", "sample_rate", "sample_id", "drone_present", "distance_m"], errors="ignore"
    )
    return features.assign(distance_m=df["distance_m"])


def main():
    df = preprocess(pd.read_pickle(INPUT))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUTPUT)


if __name__ == "__main__":
    main()
