# Guwahati Airport Weather and Visibility Pipeline

This document describes the implementation in this repository. It is not a
generic Lucknow template. The project processes Guwahati Airport station files
for runway `RWY02`, creates quality-controlled LSTM inputs, applies a
physics-based cloud correction, and trains the primary model in
[`notebooks/train_guwahati_lstm_with_physics_correction_V2.ipynb`](./notebooks/train_guwahati_lstm_with_physics_correction_V2.ipynb).

## Project configuration

The current implementation uses:

```text
CITY=Guwahati
AIRPORT=Lokpriya Gopinath Bordoloi International Airport
STATION_CODE=VEGT
RUNWAY=RWY02
OBSERVATION_INTERVAL_SECONDS=15
START_DATE=2024-08-11
END_DATE_EXCLUSIVE=2026-02-01
SOURCE_PERIOD=2024-08-11 through 2026-01-31
```

The pipeline selects only these exact sensor families for `RWY02`:

```text
CLOUD_RWY02_
PTU_RWY02_
PW_RWY02_
VIS_RWY02_
WIND_RWY02_KT_
```

The exact configured names are defined in the scripts. Confirm the normalized
headers in `09_reports/header_manifest.csv` after conversion; do not infer
column names from a different station or runway.

## Important data rule

The original files in `Guwahati-RAW-DATA/` are never edited. Raw, intermediate,
report, dataset, and model contents are ignored by Git. Folder README files and
`.gitkeep` files are tracked so the project structure remains visible without
committing large or provider-controlled data.

## Folder flow

```text
Guwahati-RAW-DATA/       original provider .his files
01_source_selected/      selected RWY02 sensor files
02_sensor_csv/           converted sensor CSV files
03_daily_merged/         one merged file per day
04_stage1_preprocessed/  normalized timestamps and numeric columns
05_stage2_flagged/       PTU gap interpolation and audits
06_final_cleaned/        final daily files and remaining-gap flags
07_lstm_ready/           compact model input files and sensor flags
08_unified_dataset/      chronological unified CSV files
09_reports/              run configuration and audit reports
10_models/               model weights, scalers, metrics, and plots
```

## End-to-end execution

Run these commands from `D:\IMD\Guwahati` in the numbered order:

```cmd
python scripts\01_select_source.py
python scripts\02_convert_his_to_csv.py
python scripts\03_merge_daily.py
python scripts\04_stage1_preprocess.py
python scripts\05_stage2_gap_analysis.py
python scripts\06_stage3_final.py
python scripts\07_stage4_mlreadydata.py
python scripts\08_merge_lstm_ready_dataset.py
python scripts\09_physics_cloud_correction.py
```

The scripts use paths relative to the repository root. They create missing
output directories and write reports under `09_reports/`. Run the first
command only after the raw `.his` files have been placed in
`Guwahati-RAW-DATA/`.

The first script also writes `09_reports/Guwahati_run_config.txt` and
`source_inventory.csv`. Its run configuration currently records the source
provider as a Vaisala AVIMET history export, local `Asia/Kolkata` timestamps,
and blank fields as the source missing-value representation. Verify or update
the download date before a new run.

## Pipeline stages

### 1. Source selection

[`01_select_source.py`](./scripts/01_select_source.py) verifies that the raw
directory exists, rejects empty `.his` files, computes SHA-256 inventory
records, and copies only the configured `RWY02` sensor prefixes into
`01_source_selected/`. It writes `source_inventory.csv` and
`source_selection.csv`.

It does not mix `ALL`, `MID1`, other runway, AWS, or event files into this
project.

### 2. HIS conversion

[`02_convert_his_to_csv.py`](./scripts/02_convert_his_to_csv.py) reads the
tab-separated history files, normalizes headers, retains `Date` and `Time`,
converts measurement values to numeric values where possible, and writes
sensor CSV files under `02_sensor_csv/`. Conversion problems are written to
`conversion_failures.csv`.

### 3. Daily sensor merge

[`03_merge_daily.py`](./scripts/03_merge_daily.py) groups sensor CSV files by
year, month, and day. It strips header whitespace and BOM characters, requires
`Date` and `Time`, prefixes measurement columns with the sensor name, removes
duplicate sensor timestamps, and performs an outer merge on `Date` and `Time`.

Daily files are written under `03_daily_merged/`. Missing sensors are retained
as missing values and reported in `missing_sensors.csv`; merge errors are
written to `merge_failures.csv`.

### 4. Stage 1 preprocessing

[`04_stage1_preprocess.py`](./scripts/04_stage1_preprocess.py) performs the
following operations:

1. Creates `Datetime` from `Date` and `Time` using day-first parsing.
2. Removes invalid timestamps and records the count.
3. Sorts timestamps and removes duplicate timestamps, keeping the first row.
4. Drops `Date`, `Time`, and `PW_Present_Weather_Code` when present.
5. Converts remaining measurement columns to numeric values.
6. Adds `Hour_Sin` and `Hour_Cos` from local observation time.
7. Writes the result to `04_stage1_preprocessed/`.

This implementation does not perform the single-cell or empty-row repair
described in the old template; its repair counts remain zero. Failures are
written to `stage1_failed_files.csv`, and successful-file counts are written
to `stage1_summary.csv`.

### 5. Stage 2 PTU gap analysis

[`05_stage2_gap_analysis.py`](./scripts/05_stage2_gap_analysis.py) processes
the configured PTU columns with:

```text
MAX_GAP_MINUTES=5
MAX_GAP_ROWS=20
PTU temperature threshold=0.5
PTU dew-point threshold=0.5
PTU humidity threshold=5.0
PTU pressure threshold=2.0
```

At the 15-second sampling interval, 20 rows represent five minutes. PTU
missing values are interpolated in time with a maximum limit of three rows and
rounded to two decimals. Original missing runs are still audited, including
runs that were filled.

The stage writes:

- `gap_audit.csv`
- `threshold_audit.csv`
- `flagged_files.csv`
- `ptu_interpolation_summary.csv`
- `stage2_failed_files.csv`

Stage 2 does not delete flagged rows or files.

### 6. Stage 3 final cleaning

[`06_stage3_final.py`](./scripts/06_stage3_final.py) parses and sorts
`Datetime`, converts measurements to numeric values, and records wind gaps
longer than five minutes. It preserves missing wind, rain, and other values;
this implementation does not fill them in Stage 3.

The stage writes cleaned daily files to `06_final_cleaned/` and produces:

- `wind_gap_flags.csv`
- `cleaning_summary.csv`
- `final_flags.csv`
- `stage3_failed_files.csv`

Remaining missing values are intentional audit information, not successful
values manufactured by the pipeline.

### 7. Stage 4 LSTM-ready preparation

[`07_stage4_mlreadydata.py`](./scripts/07_stage4_mlreadydata.py) maps the
runway-specific source columns to a compact, stable model schema. It keeps
valid timestamps, preserves rows with missing targets, and writes daily files
to `07_lstm_ready/`.

The output includes:

```text
Datetime
CLOUD_Cloud_Base_1, CLOUD_Cloud_Base_2, CLOUD_Cloud_Base_3
CLOUD_SITE, CLOUD_OCTA1, CLOUD_OCTA2, CLOUD_OCTA3, CLOUD_OCTA4
CLOUD_Vertical_Visibility
PTU_Pressure_QNH, PTU_Temperature, PTU_DewPoint, PTU_Humidity
PW_Rain_Intensity, PW_Rain_Sum
VIS_Visibility_MOR, VIS_RVR
WIND_Wind_Dir, WIND_Wind_Speed, WIND_Wind_Gust
Hour_Sin, Hour_Cos
cloud imputation and long-gap flags
sensor availability flags
```

The implemented fills are:

- Forward-fill each cloud-base column for at most three rows.
- Forward-fill vertical visibility for at most three rows.
- Mark those fills with `*_IMPUTED`.
- Mark remaining cloud-base and vertical-visibility gaps with long-gap flags.
- Fill cloud octa values with zero.
- Fill rain intensity and rain sum with zero.
- Keep sensor availability flags such as `CLOUD_SENSOR_ON` and
  `VIS_SENSOR_ON`.

The stage writes `dataset_summary.csv`, `uniformity_report.csv`, and
`stage4_failed_files.csv`. It does not remove a daily file merely because
target values or other measurements are missing.

### 8. Unified dataset

[`08_merge_lstm_ready_dataset.py`](./scripts/08_merge_lstm_ready_dataset.py)
reads the daily LSTM-ready CSV files, filters timestamps to
`[2024-08-11, 2026-02-01)`, aligns the configured column order, sorts by
`Datetime`, removes duplicate timestamps, and writes:

```text
08_unified_dataset/
└── guwahati_unified_lstm_dataset_v2_short_cloud_fill_20240811_20260131.csv
```

The corresponding merge summary is
`09_reports/guwahati_unified_lstm_dataset_summary.csv`. The merger requires
`Datetime` in every input file and fails if no rows remain in the configured
date range.

### 9. Physics-based cloud correction

[`09_physics_cloud_correction.py`](./scripts/09_physics_cloud_correction.py)
reads the v2 bounded-cloud-fill unified dataset and writes:

```text
08_unified_dataset/guwahati_unified_lstm_dataset_v3_physics_corrected.csv
```

Its rules are:

1. Recover `CLOUD_Cloud_Base_1` as `15000` feet when the cloud value is
   missing or short-gap-imputed, the cloud sensor is on, rain intensity is
   zero, and `CLOUD_OCTA1` is zero.
2. Re-mask recovered cloud values when visibility is below 2000 m, treating
   this as a fog contradiction.
3. Re-mask cloud-base and vertical-visibility values when the cloud sensor is
   offline.

The script prints counts for recovered clear-sky rows, fog contradictions, and
dead-sensor records. It does not create a separate CSV report.

## Primary model notebook and actual results

The most important notebook is
[`notebooks/train_guwahati_lstm_with_physics_correction_V2.ipynb`](./notebooks/train_guwahati_lstm_with_physics_correction_V2.ipynb).
Use this notebook for the current physics-corrected model results. It reads:

```text
08_unified_dataset/guwahati_unified_lstm_dataset_v3_physics_corrected.csv
```

It writes artifacts under:

```text
10_models/guwahati_physics_corrected/
```

The notebook configuration is:

```text
targets: CLOUD_Cloud_Base_1, VIS_Visibility_MOR
lookback: 96 observations (24 minutes at 15 seconds)
horizon: 1 observation
train/validation/test: 70% / 15% / 15% chronological split
batch size: 256
maximum epochs: 50
random seed: 42
```

The model uses a 64-unit LSTM followed by a 32-unit ReLU head. Feature
medians, feature scaling, and target scaling are fitted from the training
partition only. Sequence splitting is chronological; it is not random.

The recorded test output in the V2 notebook is:

| Target | Samples | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| `CLOUD_Cloud_Base_1` | 403,339 | 86.418915 | 354.882276 | 0.994966 |
| `VIS_Visibility_MOR` | 377,775 | 93.666336 | 192.644781 | 0.995720 |

These metrics are in the original target units after inverse scaling. The
notebook saves `best_guwahati_physics_method.pt`, `scalers.json`,
`test_metrics.csv`, range-based test metrics, predictions, and plots in the
model artifact directory. Re-run the notebook when data, preprocessing, or
model settings change; do not treat the values above as unchanged guarantees.

The selected target `CLOUD_Cloud_Base_1` is different from
`CLOUD_Vertical_Visibility`. The latter remains an input/output feature in
the prepared dataset, but it is not the first target in the current V2
training configuration.

## Reproducibility checklist

Before calling a run complete:

- Confirm the raw source files are unchanged.
- Confirm `source_inventory.csv` and `source_selection.csv` are present.
- Inspect `header_manifest.csv` and verify the normalized column names.
- Check all stage failure reports and explain any non-empty failures.
- Review PTU gap and threshold audits.
- Review wind-gap and final-NaN reports.
- Confirm Stage 4 `dataset_summary.csv` and `uniformity_report.csv`.
- Confirm the unified date range, sorted timestamps, and duplicate count.
- Confirm the physics-corrected dataset was generated from the v2 input.
- Confirm the V2 notebook input path points to the v3 physics-corrected file.
- Confirm model scalers were fitted on training data only.
- Record any changes to runway, dates, source format, thresholds, or fills.

The project is reproducible only when the processing reports and model
configuration are retained alongside the code, even when large data and model
files remain outside Git.
