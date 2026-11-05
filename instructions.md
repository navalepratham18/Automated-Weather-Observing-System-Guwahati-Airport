# Airport Meteorological Data Pipeline: Reusable Instructions

This document is the exact operating procedure for applying the Lucknow airport
weather-data workflow to another airport or city. Replace every value shown in
angle brackets (for example, `<CITY>`) before running anything. Do not silently
change cleaning rules, sampling assumptions, date ranges, target variables, or
model settings. If the new station uses a different convention, record the
change in the project README and in the run log.

The workflow has eight logical phases:

1. Preserve and segregate the source files.
2. Convert raw station files into sensor-specific CSV files.
3. Merge same-day sensor files into one daily file.
4. Run the four cleaning/preparation stages.
5. Merge daily LSTM-ready files into one chronological dataset.
6. Audit the dataset and train/evaluate the LSTM notebooks.

The reference implementation is represented by
[`2024_filesmove.py`](./2024_filesmove.py), [`2024_merge.py`](./2024_merge.py),
[`stage1_preprocess.py`](./stage1_preprocess.py),
[`stage2_gap_analysis.py`](./stage2_gap_analysis.py),
[`stage3_final.py`](./stage3_final.py),
[`stage4_mlreadydata.py`](./stage4_mlreadydata.py), and
[`merge_lstm_ready_dataset.py`](./merge_lstm_ready_dataset.py). These scripts
contain the executable behavior; this document explains how to parameterize and
run that behavior for another airport.

---

## 1. Non-negotiable rules

- Work from copies. Never edit, rename, or overwrite the original `.his` files.
- Keep one city/airport, one runway selection, and one observation period per
  project. Do not mix airports or runways in a single dataset.
- Use a deterministic folder layout and deterministic file ordering.
- Keep the original sensor prefix in every renamed column.
- Treat `Datetime` as the only time key after Stage 1.
- Parse Indian-style dates (`dayfirst=True`) unless the source documentation
  explicitly proves that the source is month-first.
- The reference workflow assumes a **15-second observation interval**.
  Recalculate all row-based thresholds if the new station uses another interval.
- Interpolate only PTU gaps of up to three consecutive rows; never
  interpolate long gaps or target visibility columns.
- Do not delete audit files. Reports are part of the output and are required
  for review and reproducibility.
- Stop on missing required columns, an empty date-filtered result, or an
  unexplained change in sampling frequency. Do not manufacture a successful
  output.

## 2. Required configuration sheet

Create a small configuration record before touching the data. Store it in
`09_reports/<CITY>_run_config.txt` and copy the same values into the project
README.

```text
CITY=<city name>
AIRPORT=<airport name>
IATA_OR_STATION_CODE=<code>
RUNWAY=<runway, for example RWY27>
RAW_SOURCE=<absolute source folder>
OBSERVATION_INTERVAL_SECONDS=15
START_DATE=<YYYY-MM-DD, inclusive>
END_DATE=<YYYY-MM-DD, exclusive>
TARGET_COLUMNS=<comma-separated target columns>
WIND_COLUMNS=<comma-separated wind columns>
RAIN_COLUMNS=<comma-separated rain columns>
PTU_COLUMNS=<comma-separated PTU columns>
```

Also record:

- The source provider and download date.
- The timezone and whether timestamps are local time or UTC.
- The exact sensor prefixes present (`CLOUD`, `PTU`, `PW`, `VIS`, `WIND`, etc.).
- Which runway is being retained and why.
- The expected daily filename pattern.
- Any source-specific missing-value codes such as `-999`, `9999`, or `////`.
- The Python version and package versions used.

## 3. Create the project directory

Use the template tree in [`airport_pipeline_template/`](./airport_pipeline_template/).
Copy it to a city-specific directory at the repository root, for example:

```powershell
Copy-Item -Recurse .\airport_pipeline_template .\<CITY>_airport_pipeline
```

The final city project must look like this:

```text
<CITY>_airport_pipeline/
├── Guwahati-RAW-DATA/        # immutable provider .his files
├── 01_source_selected/       # only the selected runway/sensor files
├── 02_sensor_csv/            # converted sensor-level CSV files
├── 03_daily_merged/          # one merged CSV per day
├── 04_stage1_preprocessed/   # normalized timestamps and basic repairs
├── 05_stage2_flagged/        # gap and threshold audits
├── 06_final_cleaned/         # final domain-aware interpolation
├── 07_lstm_ready/            # daily files with model features
├── 08_unified_dataset/       # one chronological training CSV
├── 09_reports/               # all summaries, flags, and run metadata
├── 10_models/                # model weights, scalers, plots, metrics
├── scripts/                  # city-specific copies/configured scripts
└── notebooks/                # copied training notebooks
```

Use `.gitkeep` files to preserve empty directories. Add raw data and generated
large CSVs to `.gitignore` if they must not be committed; commit the scripts,
configuration, README, and small audit summaries.

## 4. Phase A: preserve and select source files

1. Keep the provider's original directory in the root-level
   `Guwahati-RAW-DATA/` folder.
2. Verify that the copy has the same file count and representative file hashes
   as the source. Do not run cleaning scripts on this directory.
3. Inspect all month/year folders and list the sensor prefixes.
4. Select only the requested runway. For a runway such as `RWY27`, retain:
   `CLOUD_<RUNWAY>`, `PTU_<RUNWAY>`, `PW_<RUNWAY>`, `VIS_<RUNWAY>`, and
   `WIND_<RUNWAY>`.
5. Copy only matching `.his` files into `01_source_selected/`, preserving
   year/month subdirectories and filenames.
6. Record skipped files and the reason in `09_reports/source_selection.csv`.
7. Confirm that every expected day has the required sensors. Missing sensors
   must be reported, not filled by copying another day.

The selection operation is the same idea as `2024_filesmove.py`; replace its
source and target paths and its `TARGET_PREFIXES`. Never use a broad prefix
that could include another runway.

## 5. Phase B: convert `.his` files to CSV

Use the existing conversion helpers (`extractor_2024.py`,
`his_to_excel_converter.py`, and the applicable year-specific extractor) as
reference. Select the converter that matches the provider's `.his` format.

For each source file:

1. Read the header and document the fields before conversion.
2. Preserve `Date` and `Time` exactly long enough to construct a timestamp.
3. Export a CSV with one row per observation.
4. Preserve numeric values as numeric where possible.
5. Preserve blank values as blank; do not replace missing values with zero at
   this phase.
6. Keep the sensor prefix in the output filename and later in every column name.
7. Write conversion errors to `09_reports/conversion_failures.csv`.
8. Compare the input and output row counts for representative files.

Expected sensor-level names are similar to:

```text
CLOUD_<RUNWAY>_<DD>.csv
PTU_<RUNWAY>_<DD>.csv
PW_<RUNWAY>_<DD>.csv
VIS_<RUNWAY>_<DD>.csv
WIND_<RUNWAY>_<DD>.csv
```

Do not proceed until at least one converted day has been manually inspected.

## 6. Phase C: merge sensors into daily files

Use `2024_merge.py` as the reference for same-day merging.

For each year/month/day:

1. Group files by month and the two-digit day suffix.
2. Read each sensor CSV.
3. Strip whitespace and BOM characters from column names.
4. Require `Date` and `Time`; skip and report a file that lacks either.
5. Prefix every non-time column with the sensor name.
6. Remove duplicate rows within a sensor using `Date` and `Time`.
7. Outer-merge all available sensors on `Date` and `Time`; retain missing
   sensor observations as missing values.
8. Sort by date and time.
9. Save one file per day under `03_daily_merged/<Month>/`.
10. Write merge failures and missing-sensor information to `09_reports`.

The daily file must contain one shared time axis and columns such as:
`CLOUD_Cloud_Base_1`, `PTU_Temperature`, `PW_Rain_Sum`,
`VIS_Visibility_MOR`, and `WIND_Wind_Speed`. Adapt names only after inspecting
the actual header. Never assume that another airport uses the same suffixes.

## 7. Phase D: Stage 1 basic preprocessing

Configure `stage1_preprocess.py` with:

```text
INPUT_DIR  = <CITY>_airport_pipeline\03_daily_merged
OUTPUT_DIR = <CITY>_airport_pipeline\04_stage1_preprocessed
```

For every daily CSV, perform these operations in this order:

1. Read with `pd.read_csv(..., low_memory=False)`.
2. Create `Datetime` from `Date + " " + Time`.
3. Parse with `dayfirst=True, errors="coerce"`.
4. Count invalid timestamps, remove only rows with invalid `Datetime`, and
   record the count.
5. Sort by `Datetime`.
6. Remove duplicate timestamps, keeping the first row, and record the count.
7. Set `Datetime` as the index.
8. Drop the original `Date` and `Time` columns.
9. Drop `PW_Present_Weather_Code` if it exists; do not fail if absent.
10. Convert every remaining column with `pd.to_numeric(errors="coerce")`.
11. Repair a single missing cell only when the immediately preceding and
    following values are present; copy the preceding value, and count it.
12. Repair a single completely empty row only when the adjacent rows are
    non-empty; copy the preceding row, and count it.
13. Add `Hour_Sin` and `Hour_Cos` using decimal local hour:
    `sin(2*pi*hour/24)` and `cos(2*pi*hour/24)`, rounded to two decimals.
14. Restore `Datetime` as the first column and save with the same relative
    month/file path.
15. Write `failed_files.csv` if any file fails.

Required Stage 1 checks:

- `Datetime` is present, sorted, and unique.
- The number of rows removed is explained by invalid timestamps or duplicates.
- No unexpected columns disappeared.
- The generated report includes files found, saved, failed, timestamp errors,
  duplicates, repaired cells, and repaired rows.

## 8. Phase E: Stage 2 gap analysis and flags

Configure `stage2_gap_analysis.py` to read Stage 1 and write
`05_stage2_flagged/`.

The reference assumptions are:

- `MAX_GAP_MINUTES = 5`
- `MAX_GAP_ROWS = 20`
- 15-second observations (`20 * 15 seconds = 5 minutes`)
- PTU thresholds:
  - `PTU_Temperature`: `0.5`
  - `PTU_DewPoint`: `0.5`
  - `PTU_Humidity`: `5.0`
  - `PTU_Pressure_QNH`: `2.0`

Recalculate `MAX_GAP_ROWS` as
`floor(MAX_GAP_MINUTES * 60 / OBSERVATION_INTERVAL_SECONDS)` for another
sampling interval. Reconfirm the PTU thresholds with the station's units.

For every PTU column that exists:

1. Parse and sort `Datetime`.
2. Convert values to numeric.
3. Record the original missing runs, then interpolate in time with `limit=3`
   and round to two decimals.
4. Find every original contiguous missing run and retain its audit record,
   including runs that were filled.
5. Record file, variable, start, end, row count, duration, previous value, and
   next value in `gap_audit.csv`.
6. Flag runs longer than the maximum gap.
7. For bounded short gaps, compare next minus previous absolute change with the
   configured threshold.
8. Record threshold breaches in `threshold_audit.csv`.
9. Save processed daily files while preserving relative paths.
10. Write all flagged files to `flagged_files.csv`.

Never turn a flag into a deletion at this stage. Stage 2 is an audit and
intermediate output stage.

## 9. Phase F: Stage 3 final domain-aware cleaning

Configure `stage3_final.py` to read `05_stage2_flagged/` and write
`06_final_cleaned/`.

For every file:

1. Parse, remove invalid timestamps, sort, and set `Datetime` as the index.
2. Convert measurement columns to numeric.
3. For each configured wind column, identify contiguous missing runs.
4. Record every wind run longer than five minutes in `wind_gap_flags.csv`.
5. Preserve wind and rain gaps; do not interpolate them in Stage 3.
6. Keep long and unbounded gaps missing and report them.
7. Do not interpolate target visibility columns here. The explicit cloud-base,
   octa, vertical-visibility, and rain fills above are semantic fills, not
   interpolation; if `CLOUD_Vertical_Visibility` is a model target, document
   that this reference fill is being applied before training.
8. Record per-file wind fills, rain fills, and remaining NaNs in
   `cleaning_summary.csv`.
9. Write files with remaining missing values to `final_flags.csv`.
10. Write failures to `failed_files.csv`.

Missing values and long gaps are preserved rather than invented. Their
locations are recorded in the gap and final-flag reports, and Stage 4 adds
per-row sensor availability flags so downstream users can distinguish an
offline sensor from an online sensor with a missing measurement.

## 10. Phase G: Stage 4 LSTM-ready feature preparation

Configure `stage4_mlreadydata.py` to read `06_final_cleaned/` and write
`07_lstm_ready/`.

For each file:

1. Parse and sort `Datetime`.
2. Preserve rows whose target or other sensor values are missing; do not discard
   a daily file because of missing targets or long gaps.
3. Keep intervals listed in `wind_gap_flags.csv` and record them as audit flags.
4. In v2, forward-fill cloud-base columns only across short gaps of up to
   three rows, and record those rows in `CLOUD_*_IMPUTED`. Do not use `15000`
   as a cloud-base sentinel.
5. Fill cloud octa columns with `0` only when missing.
6. Preserve missing vertical visibility; do not replace it with `15000`.
7. Fill rain intensity and rain sum with `0` only when the source semantics
   confirm that blank means no rain.
8. Count duplicate timestamps, confirm sorted order, count non-interval gaps,
   and count all remaining NaNs.
9. Save the daily file using the Lucknow-compatible compact column structure,
   retaining empty values and adding `*_SENSOR_ON` flags.
10. Save `dataset_summary.csv`, `uniformity_report.csv`, and any failure report.

Do not use these fill rules for unknown columns. Build explicit lists from the
new airport's header and document every semantic decision. The Guwahati output
uses the compact Lucknow-compatible core columns (`CLOUD_*`, `PTU_*`, `PW_*`,
`VIS_*`, `WIND_*`, and time features), followed by
`CLOUD_SENSOR_ON`, `PTU_SENSOR_ON`, `PW_SENSOR_ON`, `VIS_SENSOR_ON`, and
`WIND_SENSOR_ON`. In v2, cloud-base values are forward-filled only for up to
three consecutive rows (45 seconds at the 15-second sampling interval).
`CLOUD_*_IMPUTED` identifies those filled rows and `CLOUD_LONG_GAP` identifies
rows that remain unavailable after the short-gap fill. Missing source months remain absent because the pipeline does
not synthesize observations that are not present in the raw export.

## 11. Phase H: create the unified dataset

Configure `merge_lstm_ready_dataset.py`:

```text
INPUT_DIR  = <CITY>_airport_pipeline\07_lstm_ready
OUTPUT_FILE = <CITY>_airport_pipeline\08_unified_dataset\
              <city>_unified_lstm_dataset_<period>.csv
SUMMARY_FILE = <CITY>_airport_pipeline\09_reports\
               <city>_unified_lstm_dataset_summary.csv
START_DATE = <inclusive start>
END_DATE   = <exclusive end>
```

The merger must:

1. Discover only daily data CSVs, excluding all report and existing unified
   files.
2. Sort files by the date encoded in their filename.
3. Discover the union of all columns while preserving first-seen order.
4. Require `Datetime` in every file.
5. Parse timestamps with `dayfirst=True`.
6. Filter to `[START_DATE, END_DATE)`.
7. Remove duplicate timestamps globally across files.
8. Reindex every file to the union column list.
9. Append rows safely, then sort the completed output by actual timestamp.
10. Save a summary containing discovered/processed/failed files, rows read,
    invalid timestamps, date-filtered rows, duplicate rows removed, rows
    written, columns written, and failures.

After merging, verify that the unified file has one row per expected
timestamp, no accidental month/file ordering, and the expected column set.

## 12. LSTM notebook preparation

Copy the appropriate notebook into `notebooks/` and rename it for the city.
Update all city-specific paths and names; do not leave a Lucknow path in a
new-city notebook.

The reference baseline notebook uses:

```python
TARGET_COLUMNS = ["CLOUD_Vertical_Visibility", "VIS_Visibility_MOR"]
LOOKBACK = 96                 # 96 x 15 seconds = 24 minutes
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
```

The new airport must satisfy these checks before training:

1. Load the unified CSV.
2. Parse `Datetime` and sort it.
3. Drop rows missing the timestamp or any target.
4. Verify every target exists and is numeric.
5. Define feature columns explicitly; include targets only if the notebook's
   stated design requires them as autoregressive inputs.
6. Print dimensions, date range, missingness, descriptive statistics, and
   Pearson correlations between rain parameters and targets.
7. Split chronologically, never randomly, to avoid future leakage.
8. Fit feature and target scalers on the training partition only.
9. Construct sequences with the configured lookback.
10. Train the baseline LSTM and compare it with persistence/naive forecasts.
11. Report MAE, RMSE, and R2 for every target in original physical units.
12. Save model weights, scalers, configuration, predictions, metrics, and plots
    under `10_models/`.

The range-aware notebook may additionally save:
`range_aware_lstm_final.pt`, `range_aware_feature_scaler.joblib`,
`range_aware_residual_scaler.joblib`, and a training configuration containing
feature columns, targets, lookback, horizon, blend weights, and thresholds.
Update those thresholds for the new target distribution rather than copying
Lucknow values blindly.

## 13. Validation checklist (must be completed)

Before declaring the run complete, check every item:

- [ ] Original files remain unchanged in `Guwahati-RAW-DATA/`.
- [ ] Only the selected runway and sensors are in `01_source_selected/`.
- [ ] Conversion failures are empty or explained.
- [ ] Daily merge files have a shared `Date`/`Time` axis.
- [ ] Stage 1 has no invalid timestamps in successful outputs.
- [ ] Duplicate and repair counts are present in the Stage 1 report.
- [ ] Stage 2 contains gap and threshold audits.
- [ ] Long gaps were not interpolated, and short PTU fills are recorded in
  `ptu_interpolation_summary.csv`.
- [ ] Stage 3 contains wind-gap flags and final NaN flags.
- [ ] Stage 4 target rows were not silently invented.
- [ ] Cloud/rain fills match documented sensor semantics.
- [ ] Daily files are sorted and have the expected interval.
- [ ] Unified timestamps are globally sorted and deduplicated.
- [ ] The unified summary reconciles input and output row counts.
- [ ] Notebook paths, city names, targets, features, date range, and output
      artifact directory are city-specific.
- [ ] Scalers were fitted on training data only.
- [ ] Metrics are reported in original units and include a baseline.
- [ ] A run configuration and software environment record are saved.

## 14. Recommended run order

Run from the repository root in a prepared Python environment:

```powershell
python .\<CITY>_airport_pipeline\scripts\01_select_source.py
python .\<CITY>_airport_pipeline\scripts\02_convert_his_to_csv.py
python .\<CITY>_airport_pipeline\scripts\03_merge_daily.py
python .\<CITY>_airport_pipeline\scripts\04_stage1_preprocess.py
python .\<CITY>_airport_pipeline\scripts\05_stage2_gap_analysis.py
python .\<CITY>_airport_pipeline\scripts\06_stage3_final.py
python .\<CITY>_airport_pipeline\scripts\07_stage4_mlreadydata.py
python .\<CITY>_airport_pipeline\scripts\08_merge_lstm_ready_dataset.py
```

If the scripts are copied from the reference files, update paths and column
lists before executing. Run one day first, inspect every report, then run the
full period. Keep the console output or redirect it to
`09_reports/<CITY>_run.log`.

## 15. Troubleshooting decisions

- **No files found:** verify the input root, extension, month folders, and
  runway prefix; do not broaden the prefix without inspection.
- **Missing `Date`/`Time`:** stop and inspect the converter output; do not
  synthesize time from a filename unless the source specification supports it.
- **Unexpected dates:** check timezone, day-first parsing, and year folder
  labels before changing the parser.
- **Many non-interval gaps:** verify that the station's sampling interval is
  actually 15 seconds and that duplicate removal did not collapse legitimate
  observations.
- **Many remaining NaNs:** inspect `final_flags.csv`; long gaps are expected
  to remain missing and must be explained.
- **No rows in unified output:** check date boundaries and timestamp parsing.
- **Model leakage concern:** verify chronological splitting and that scalers
  were not fit on validation or test rows.
- **Different sensor names:** map columns explicitly in the configuration;
  never use fuzzy substring matching for targets.

## 16. Deliverable definition

A city run is complete only when the project contains:

1. A preserved source copy and selection report.
2. Reproducible scripts with city-specific configuration.
3. All stage outputs in separate directories.
4. Audit CSVs for failures, duplicates, gaps, thresholds, repairs, and NaNs.
5. A unified chronological LSTM dataset and merge summary.
6. A notebook with no stale city-specific paths.
7. Model artifacts, metrics, plots, and training configuration.
8. A README describing the airport, period, runway, assumptions, and how to
   reproduce the run.
