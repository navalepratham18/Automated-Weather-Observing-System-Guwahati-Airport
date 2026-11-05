# Automated Weather Observing System - Guwahati Airport

This repository contains the reproducible data-processing and machine-learning
workflow for Guwahati Airport meteorological and visibility observations.
It converts source station files into quality-checked, LSTM-ready datasets and
supports visibility prediction experiments.

## Repository contents

```text
Guwahati/
├── 00_source_original/       # Untouched source-data reference
├── Guwahati-RAW-DATA/        # Immutable provider raw files
├── 01_source_selected/       # Selected airport, runway, and sensor files
├── 02_sensor_csv/            # Sensor-level CSV conversions
├── 03_daily_merged/          # One merged file per observation day
├── 04_stage1_preprocessed/   # Initial timestamp and data preparation
├── 05_stage2_flagged/        # Gap, threshold, and quality flags
├── 06_final_cleaned/         # Domain-aware cleaned daily files
├── 07_lstm_ready/            # Model-ready daily files
├── 08_unified_dataset/       # Chronological unified datasets
├── 09_reports/               # Audits, summaries, and run configuration
├── 10_models/                # Trained models, scalers, and metrics
├── notebooks/                # Exploration, training, and evaluation notebooks
└── scripts/                  # Numbered pipeline scripts
```

Each folder contains a local README describing its role and expected contents.

## Processing workflow

Run the pipeline in the following order:

1. Select the required source files.
2. Convert source files into sensor-specific CSV files.
3. Merge sensor files by observation day.
4. Run preprocessing and quality-control stages.
5. Create LSTM-ready daily files.
6. Merge daily files into a chronological unified dataset.
7. Review reports and train or evaluate the models.

The numbered scripts in [`scripts/`](./scripts/) implement these stages:

```text
01_select_source.py
02_convert_his_to_csv.py
03_merge_daily.py
04_stage1_preprocess.py
05_stage2_gap_analysis.py
06_stage3_final.py
07_stage4_mlreadydata.py
08_merge_lstm_ready_dataset.py
09_physics_cloud_correction.py
```

Read [`instructions.md`](./instructions.md) before running the workflow.
Project parameters and expected model inputs are documented in
[`model_input_parameters.md`](./model_input_parameters.md).

## Notebooks

The [`notebooks/`](./notebooks/) directory contains notebooks for:

- Unified-data distribution analysis
- Cloud-visibility LSTM training
- Lucknow-method comparison experiments
- Physics-corrected visibility prediction
- Model evaluation and testing

Keep notebook input paths, target columns, date ranges, and preprocessing
assumptions consistent with the reports and run configuration.

## Version-control policy

Large or provider-controlled files are intentionally excluded from Git:

- Raw and intermediate data folders
- Generated CSV datasets and reports
- Model artifacts
- The local Python virtual environment
- The top-level `train_lucknow_lstm.ipynb`

Folder README files and `.gitkeep` files preserve the project structure without
committing the data itself. Check `.gitignore` before adding new files.

## Reproducibility requirements

Before processing data, record the source, station or airport, runway,
observation interval, date range, target columns, Python version, and package
versions in the run configuration under [`09_reports/`](./09_reports/).
Do not modify the original raw files. Preserve audit reports and investigate
failed files or unexpected sampling changes rather than silently accepting
incomplete output.
