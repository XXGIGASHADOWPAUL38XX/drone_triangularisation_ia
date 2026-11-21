# Model trials

The 24 kHz trials used 250 ms windows and a 125 ms step. The native 48 kHz trials vary both settings. Scores give equal weight to each recording.

## Features

- **Relative FFT grid** uses 1,025 FFT bins across four frames. The CNN subtracts the window's `rms_db` from the grid.
- **FFT scalars** use 1,025 FFT bin powers averaged across frames, 4–8 kHz band power, and RMS. All **1,027 columns** are expressed in dBFS.
- **FFT + 20-band features** use the 1,027 FFT scalars plus **86 handcrafted columns** for **1,113 columns** total. Each band contributes active bin count (above 2.5% of the window's FFT peak), average frequency, amplitude standard deviation, and peak amplitude. The six whole-window features are entropy, centroid, flux, onset, zero-crossing rate, and crest factor.
- **Full-spectrum features** use the 1,025 FFT bin powers, RMS, and the six whole-window features for **1,032 columns** total.
- **Full-spectrum + 4–8 kHz** adds one band-power measurement in dBFS to the full-spectrum features for **1,033 columns** total.
- **100 band levels** use **100 columns**, one amplitude level in dBFS per equally spaced FFT band. At 24 kHz, each band spans 120 Hz.
- **200 bands averaged in groups of three** use **67 columns**. Each value is the mean dBFS level of three adjacent 60 Hz bands at 24 kHz, with two bands in the final group.
- **200 bands averaged in groups of four** use **50 columns**. Each value is the mean dBFS level of four adjacent 60 Hz bands at 24 kHz.
- **200 bands averaged in groups of five** use **40 columns**. Each value is the mean dBFS level of five adjacent 60 Hz bands at 24 kHz.
- **300 bands averaged in groups of six** use **50 columns**. Each value is the mean dBFS level of six adjacent 40 Hz bands at 24 kHz.
- **50 band levels + RMS + descriptors** is the default feature set with **57 columns**. It adds RMS, entropy, centroid, flux, onset, zero-crossing rate, and crest factor to the 50 levels from 200 bands averaged in groups of four.
- **50 band levels + per-band descriptors** uses **250 columns**. Each band contributes its averaged level plus entropy, centroid, flux, and onset computed from its FFT bins.

## Drone classification

| Trial | Model | Features | Validation balanced accuracy | Holdout balanced accuracy | Holdout precision | Holdout recall |
|---|---|---|---:|---:|---:|---:|
| [`cnn-24k-single`](02_ppc_classification/output/cnn-24k-single/metrics.json) | CNN | Relative FFT grid | 86.32% | 80.75% | 89.67% | 90.01% |
| [`cnn-24k-b20-fft-only`](02_ppc_classification/output/cnn-24k-b20-fft-only/metrics.json) | CNN | Relative FFT grid | 86.32% | 80.75% | 89.67% | 90.01% |
| [`mlp-24k-b20`](02_ppc_classification/output/mlp-24k-b20/metrics.json) | MLP | FFT + 20-band features | 97.26% | 97.99% | 98.65% | 99.73% |
| [`mlp-24k-full-spectrum`](02_ppc_classification/output/mlp-24k-full-spectrum/metrics.json) | MLP | Full-spectrum features | 97.78% | 97.19% | 98.58% | 98.26% |
| [`mlp-24k-full-spectrum-b48`](02_ppc_classification/output/mlp-24k-full-spectrum-b48/metrics.json) | MLP | Full-spectrum + 4–8 kHz | 98.25% | 96.81% | 98.49% | 97.75% |
| [`mlp-24k-bands100`](02_ppc_classification/output/mlp-24k-bands100/metrics.json) | MLP | 100 band levels | 94.82% | 94.25% | 97.26% | 95.94% |
| [`mlp-24k-bands200-avg3`](02_ppc_classification/output/mlp-24k-bands200-avg3/metrics.json) | MLP | 200 bands averaged in groups of three | 95.47% | 94.30% | 97.14% | 96.40% |
| [`mlp-24k-bands200-avg4`](02_ppc_classification/output/mlp-24k-bands200-avg4/metrics.json) | MLP | 200 bands averaged in groups of four | 95.42% | 94.20% | 97.08% | 96.37% |
| [`mlp-24k-bands200-avg5`](02_ppc_classification/output/mlp-24k-bands200-avg5/metrics.json) | MLP | 200 bands averaged in groups of five | 95.17% | 94.35% | 97.09% | 96.68% |
| [`mlp-24k-bands300-avg6`](02_ppc_classification/output/mlp-24k-bands300-avg6/metrics.json) | MLP | 300 bands averaged in groups of six | 95.33% | 94.48% | 97.25% | 96.47% |
| [`mlp-24k-bands200-avg4-descriptors`](02_ppc_classification/output/mlp-24k-bands200-avg4-descriptors/metrics.json) | MLP | 50 band levels + RMS + descriptors | 97.77% | 96.45% | 98.20% | 97.82% |
| [`mlp-24k-bands200-avg4-band-descriptors`](02_ppc_classification/output/mlp-24k-bands200-avg4-band-descriptors/metrics.json) | MLP | 50 band levels + per-band descriptors | 93.67% | 94.88% | 97.45% | 96.71% |

## Distance regression

| Trial | Model | Features | Evaluation method | Validation MAE | Holdout MAE |
|---|---|---|---|---:|---:|
| [`mlp-24k`](03_ppc_distance/output/mlp-24k/metrics.json) | MLP | FFT scalars | Three-fold grouped CV, then refit | 0.450 m | 0.526 m |
| [`mlp-24k-single`](03_ppc_distance/output/mlp-24k-single/metrics.json) | MLP | FFT scalars | Single grouped validation split | 0.565 m | 0.534 m |
| [`mlp-24k-b20`](03_ppc_distance/output/mlp-24k-b20/metrics.json) | MLP | FFT + 20-band features | Single grouped validation split | 0.682 m | 0.666 m |
| [`mlp-24k-full-spectrum`](03_ppc_distance/output/mlp-24k-full-spectrum/metrics.json) | MLP | Full-spectrum features | Single grouped validation split | 0.573 m | 0.593 m |
| [`mlp-24k-full-spectrum-b48`](03_ppc_distance/output/mlp-24k-full-spectrum-b48/metrics.json) | MLP | Full-spectrum + 4–8 kHz | Single grouped validation split | 0.575 m | 0.606 m |
| [`mlp-24k-bands100`](03_ppc_distance/output/mlp-24k-bands100/metrics.json) | MLP | 100 band levels | Single grouped validation split | 0.559 m | 0.507 m |
| [`mlp-24k-bands200-avg3`](03_ppc_distance/output/mlp-24k-bands200-avg3/metrics.json) | MLP | 200 bands averaged in groups of three | Single grouped validation split | 0.524 m | 0.513 m |
| [`mlp-24k-bands200-avg4`](03_ppc_distance/output/mlp-24k-bands200-avg4/metrics.json) | MLP | 200 bands averaged in groups of four | Single grouped validation split | 0.520 m | 0.443 m |
| [`mlp-24k-bands200-avg5`](03_ppc_distance/output/mlp-24k-bands200-avg5/metrics.json) | MLP | 200 bands averaged in groups of five | Single grouped validation split | 0.486 m | 0.511 m |
| [`mlp-24k-bands300-avg6`](03_ppc_distance/output/mlp-24k-bands300-avg6/metrics.json) | MLP | 300 bands averaged in groups of six | Single grouped validation split | 0.523 m | 0.445 m |
| [`mlp-24k-bands200-avg4-descriptors`](03_ppc_distance/output/mlp-24k-bands200-avg4-descriptors/metrics.json) | MLP | 50 band levels + RMS + descriptors | Single grouped validation split | 0.545 m | 0.476 m |
| [`mlp-24k-bands200-avg4-band-descriptors`](03_ppc_distance/output/mlp-24k-bands200-avg4-band-descriptors/metrics.json) | MLP | 50 band levels + per-band descriptors | Single grouped validation split | 0.631 m | 0.683 m |

## Native 48 kHz window and step sweep

Each MLP uses only the 50 dBFS levels from 200 adjacent bands averaged in groups of four. The input stays at 48 kHz. All configurations use the same 58 recordings and grouped splits.

| Window | Step | Step / window | Windows | Classification validation balanced accuracy | Classification holdout balanced accuracy | Distance validation MAE | Distance holdout MAE | Reports |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 50 ms | 10 ms | 0.20 | 250,382 | 97.36% | 97.54% | 0.586 m | 0.482 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w50-s10/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w50-s10/metrics.json) |
| 50 ms | 20 ms | 0.40 | 125,179 | 97.09% | 98.05% | 0.570 m | 0.457 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w50-s20/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w50-s20/metrics.json) |
| 100 ms | 20 ms | 0.20 | 125,007 | 97.93% | 97.62% | 0.589 m | 0.435 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w100-s20/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w100-s20/metrics.json) |
| 100 ms | 25 ms | 0.25 | 100,008 | 97.81% | 97.00% | 0.589 m | 0.504 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w100-s25/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w100-s25/metrics.json) |
| 100 ms | 33 ms | 0.33 | 75,763 | 97.74% | 97.80% | 0.564 m | 0.479 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w100-s33/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w100-s33/metrics.json) |
| 100 ms | 50 ms | 0.50 | 50,010 | 95.82% | 98.95% | 0.574 m | 0.443 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w100-s50/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w100-s50/metrics.json) |
| 100 ms | 75 ms | 0.75 | 33,337 | 96.38% | 98.51% | 0.572 m | 0.458 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w100-s75/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w100-s75/metrics.json) |
| 100 ms | 100 ms | 1.00 | 25,002 | 94.88% | 97.61% | 0.571 m | 0.497 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w100-s100/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w100-s100/metrics.json) |
| 250 ms | 50 ms | 0.20 | 49,797 | 96.86% | 98.08% | 0.599 m | 0.466 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w250-s50/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w250-s50/metrics.json) |
| 250 ms | 62.5 ms | 0.25 | 39,838 | 95.98% | 97.69% | 0.598 m | 0.492 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w250-s62p5/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w250-s62p5/metrics.json) |
| 250 ms | 82.5 ms | 0.33 | 30,185 | 95.47% | 98.19% | 0.604 m | 0.474 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w250-s82p5/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w250-s82p5/metrics.json) |
| 250 ms | 125 ms | 0.50 | 19,913 | 95.75% | 97.62% | 0.616 m | 0.494 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w250-s125/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w250-s125/metrics.json) |
| 250 ms | 187.5 ms | 0.75 | 13,274 | 95.64% | 97.81% | 0.589 m | 0.489 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w250-s187p5/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w250-s187p5/metrics.json) |
| 250 ms | 250 ms | 1.00 | 9,951 | 95.68% | 98.46% | 0.589 m | 0.501 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w250-s250/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w250-s250/metrics.json) |
| 500 ms | 250 ms | 0.50 | 9,881 | 94.89% | 98.35% | 0.637 m | 0.535 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w500-s250/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w500-s250/metrics.json) |
| 1,000 ms | 500 ms | 0.50 | 4,870 | 93.55% | 97.83% | 0.609 m | 0.518 m | [Classification](02_ppc_classification/output/mlp-48k-bands200-avg4-w1000-s500/metrics.json) · [Distance](03_ppc_distance/output/mlp-48k-bands200-avg4-w1000-s500/metrics.json) |

## Optuna MLP hyperparameter search

Each study used 30 trials at seed 42 and the same 50 grouped dBFS band levels at native 48 kHz. The window and hop were fixed per task. Trial 1 used the existing MLP settings. The best trial was selected on validation and evaluated on holdout.

| Task | Window / hop | Best trial | Learning rate | Dropout | Hidden units | Batch size | Weight decay | Validation (baseline → tuned) | Holdout (baseline → tuned) | All trials |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Classification (balanced accuracy) | 100 / 20 ms | 25 | 0.002091 | 0.20 | 64 | 128 | 0.002364 | 97.93% → 98.21% | 97.62% → 97.60% | [CSV](04_compare/output/optuna/classification/trials.csv) |
| Distance (MAE) | 50 / 25 ms | 11 | 0.0007634 | 0.15 | 16 | 128 | 0.008525 | 0.5587 → 0.5556 m | 0.5039 → 0.4840 m | [CSV](04_compare/output/optuna/distance/trials.csv) |

### Distance holdout MAE at 100 / 20 ms

The study ranked 30 trials by holdout MAE at seed 42. It used the same 50 grouped dBFS band levels at native 48 kHz. The holdout served as the tuning set for this study.

| Trial | Learning rate | Dropout | Hidden units | Batch size | Weight decay | Holdout MAE |
|---|---:|---:|---:|---:|---:|---:|
| 1, baseline and winner | 0.003 | 0.15 | 16 | 128 | 0.01 | **0.434771 m** |
| 18, best new configuration | 0.004083 | 0.10 | 16 | 128 | 0.004354 | 0.440134 m |

[All 30 trials](04_compare/output/optuna-distance-holdout-w100-s20/distance/trials.csv)

## Seven-feature classification at 100 / 20 ms

Both MLPs used the same 125,007 windows and recording-level splits at native 48 kHz. The classifier used RMS, entropy, zero-crossing rate, crest factor, spectral centroid, spectral flux, and onset. The distance rerun reproduced the previous scores.

| Task | Features | Validation | Holdout | Run |
|---|---|---:|---:|---|
| Classification, balanced accuracy | RMS and six descriptors | 85.28% | 92.44% | [Metrics](02_ppc_classification/output/mlp-48k-descriptors-w100-s20/metrics.json) |
| Distance, MAE | 50 grouped dBFS levels | 0.5893 m | 0.4348 m | [Metrics](03_ppc_distance/output/mlp-48k-bands200-avg4-w100-s20-rerun/metrics.json) |

### Seven-feature classification at 100 / 50 ms

At native 48 kHz and the same recording-level splits, the classifier used 50,010 windows and the same RMS and six descriptors.

| Features | Validation balanced accuracy | Holdout balanced accuracy | Run |
|---|---:|---:|---|
| RMS and six descriptors | 85.16% | 93.48% | [Metrics](02_ppc_classification/output/mlp-48k-descriptors-w100-s50/metrics.json) |

The classification default returned to the 50 grouped dBFS band levels at 100 / 50 ms. The rerun reproduced **95.82% validation** and **98.95% holdout balanced accuracy** exactly. [Rerun metrics](02_ppc_classification/output/mlp-48k-bands200-avg4-w100-s50-rerun/metrics.json)
