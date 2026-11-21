from pathlib import Path

import pandas as pd

STAGE = Path(__file__).resolve().parent
INPUT = STAGE / "input/features.pkl"
OUTPUT = STAGE / "output/classification.pkl"


def preprocess(df):
    drone_present = df["distance_m"].notna().astype("int8")
    return df.drop(
        columns=["audio", "fft", "sample_rate", "sample_id", "distance_m"], errors="ignore"
    ).assign(drone_present=drone_present)


def main():
    df = preprocess(pd.read_pickle(INPUT))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUTPUT)


if __name__ == "__main__":
    main()
