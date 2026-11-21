# Drone audio preprocessing

## Run the pipeline

Set `REPO` to the Hugging Face dataset name in the root `.env`. Set `TOKEN` if the dataset requires authentication. Run these commands from the project root using the existing `.venv`.

```sh
.venv/bin/python 00_download_dataset/main.py
.venv/bin/python 01_ppc_features/main.py
.venv/bin/python 02_ppc_classification/main.py
.venv/bin/python 03_ppc_distance/main.py
```

## 00 Download the dataset

Downloads the Hugging Face train split, saves it to `00_download_dataset/output/train`, and copies it to `01_ppc_features/input/train`.

## 01 Extract audio features

Resamples audio to 16 kHz, splits it into non-overlapping windows of 200 samples, and drops silent windows. Computes 22 features from the waveform, the full FFT, and four FFT bands. Saves `01_ppc_features/output/features.pkl` and copies it into the `input/` folders of stages `02` and `03`.

## 02 Prepare classification data

Reads `02_ppc_classification/input/features.pkl`. Saves `02_ppc_classification/output/classification.pkl` with the 22 features and `drone_present`. The label is `0` when `distance_m` is missing and `1` when it is present.

## 03 Prepare distance data

Reads `03_ppc_distance/input/features.pkl` and drops rows with missing `distance_m`. Saves `03_ppc_distance/output/distance.pkl` with the 22 features and the distance in meters.
