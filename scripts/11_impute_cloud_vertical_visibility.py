"""Estimate missing Guwahati cloud vertical visibility from observed predictors.

The original target is never overwritten. The output adds:
  CLOUD_Vertical_Visibility_FILLED
  CLOUD_Vertical_Visibility_MODEL_IMPUTED

Only rows with an original target value train the model. The latest valid
observations are held out chronologically for an honest validation score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "CLOUD_Vertical_Visibility"
FEATURES = [
    "CLOUD_Cloud_Base_1",
    "CLOUD_Cloud_Base_2",
    "CLOUD_Cloud_Base_3",
    "CLOUD_SITE",
    "CLOUD_OCTA1",
    "CLOUD_OCTA2",
    "CLOUD_OCTA3",
    "CLOUD_OCTA4",
    "PTU_Pressure_QNH",
    "PTU_Temperature",
    "PTU_DewPoint",
    "PTU_Humidity",
    "PW_Rain_Intensity",
    "PW_Rain_Sum",
    "VIS_Visibility_MOR",
    "VIS_RVR",
    "WIND_Wind_Dir",
    "WIND_Wind_Speed",
    "WIND_Wind_Gust",
    "Hour_Sin",
    "Hour_Cos",
    "CLOUD_SENSOR_ON",
    "PTU_SENSOR_ON",
    "PW_SENSOR_ON",
    "VIS_SENSOR_ON",
    "WIND_SENSOR_ON",
    "CLOUD_Cloud_Base_1_IMPUTED",
    "CLOUD_Cloud_Base_2_IMPUTED",
    "CLOUD_Cloud_Base_3_IMPUTED",
    "CLOUD_LONG_GAP",
    "CLOUD_VERTICAL_LONG_GAP",
]


def number(value: str) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def read_samples(path: Path) -> tuple[list[list[float]], list[float], list[str], int]:
    samples: list[list[float]] = []
    targets: list[float] = []
    dates: list[str] = []
    missing = 0
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing_features = sorted(set(FEATURES) - set(reader.fieldnames or []))
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")
        if TARGET not in (reader.fieldnames or []):
            raise ValueError(f"Missing target column: {TARGET}")
        for row in reader:
            target = number(row[TARGET])
            if np.isnan(target):
                missing += 1
                continue
            samples.append([number(row[name]) for name in FEATURES])
            targets.append(target)
            dates.append(row["Datetime"])
    return samples, targets, dates, missing


def chronological_split(
    features: np.ndarray, targets: np.ndarray, dates: list[str], validation_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    cutoff = max(1, int(len(targets) * (1 - validation_fraction)))
    cutoff_date = dates[cutoff]
    return (
        features[:cutoff],
        targets[:cutoff],
        features[cutoff:],
        targets[cutoff:],
        cutoff_date,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT
        / "08_unified_dataset"
        / "guwahati_unified_lstm_dataset_v3_physics_corrected.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "08_unified_dataset"
        / "guwahati_unified_lstm_dataset_v3_vertical_visibility_model_imputed.csv",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Number of input data rows already written in --output",
    )
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    if args.resume < 0:
        raise ValueError("--resume must not be negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    print(f"Reading valid training rows from {args.input} ...", flush=True)
    rows, values, dates, missing_count = read_samples(args.input)
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)
    x_train, y_train, x_test, y_test, cutoff_date = chronological_split(
        x, y, dates, args.validation_fraction
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=args.max_iterations,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    print(
        f"Training on {len(y_train):,} valid rows; validating on {len(y_test):,} "
        f"rows from {cutoff_date} ...",
        flush=True,
    )
    model.fit(x_train, np.log1p(np.clip(y_train, 0, None)))
    predictions = np.maximum(0, np.expm1(model.predict(x_test)))
    baseline = np.full_like(y_test, np.median(y_train))
    validation = {
        "validation_start_datetime": cutoff_date,
        "validation_rows": int(len(y_test)),
        "model_mae": float(mean_absolute_error(y_test, predictions)),
        "model_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "median_baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "median_baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
        "model_better_than_median_baseline": bool(
            mean_absolute_error(y_test, predictions)
            < mean_absolute_error(y_test, baseline)
        ),
    }

    print(f"Writing filled dataset to {args.output} ...", flush=True)
    imputed = 0
    fallback = 0
    total_rows = len(rows) + missing_count
    append_mode = args.resume > 0
    with args.input.open("r", newline="", encoding="utf-8-sig") as source, args.output.open(
        "a" if append_mode else "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or []) + [
            f"{TARGET}_FILLED",
            f"{TARGET}_MODEL_IMPUTED",
        ]
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        if not append_mode:
            writer.writeheader()

        chunk_rows: list[dict[str, str]] = []
        processed = 0

        def flush_chunk() -> None:
            nonlocal imputed, fallback
            if not chunk_rows:
                return
            missing_indexes: list[int] = []
            chunk_features: list[list[float]] = []
            for index, row in enumerate(chunk_rows):
                if np.isnan(number(row[TARGET])):
                    missing_indexes.append(index)
                    chunk_features.append([number(row[name]) for name in FEATURES])
                else:
                    row[f"{TARGET}_FILLED"] = row[TARGET]
                    row[f"{TARGET}_MODEL_IMPUTED"] = "0"

            if chunk_features:
                estimates = np.maximum(
                    0,
                    np.expm1(model.predict(np.asarray(chunk_features, dtype=np.float32))),
                )
                for index, estimate in zip(missing_indexes, estimates):
                    estimate = float(estimate)
                    if not np.isfinite(estimate):
                        estimate = float(np.median(y_train))
                        fallback += 1
                    row = chunk_rows[index]
                    row[f"{TARGET}_FILLED"] = f"{estimate:.3f}"
                    row[f"{TARGET}_MODEL_IMPUTED"] = "1"
                    imputed += 1

            for row in chunk_rows:
                writer.writerow(row)
            chunk_rows.clear()

        for row_index, row in enumerate(reader):
            if row_index < args.resume:
                continue
            chunk_rows.append(row)
            if len(chunk_rows) >= args.batch_size:
                flush_chunk()

            processed += 1
            completed = args.resume + processed
            if processed % args.batch_size == 0 or completed == total_rows:
                percent = 100 * completed / total_rows
                print(
                    f"PROGRESS: {completed:,}/{total_rows:,} records "
                    f"({percent:.2f}% complete)",
                    flush=True,
                )

        flush_chunk()

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "target": TARGET,
        "valid_original_rows": int(len(y)),
        "missing_original_rows": int(missing_count),
        "filled_rows": int(imputed),
        "fallback_median_rows": int(fallback),
        "filled_availability_percent": float(100 * (len(y) + imputed) / (len(y) + imputed)),
        "features": FEATURES,
        "validation": validation,
        "warning": (
            "Filled values are model estimates, not measurements. Do not use "
            "model-imputed rows as ground-truth test labels."
        ),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
