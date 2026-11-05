"""Guwahati airport data pipeline.

Run this file from the existing Miniconda environment only after reviewing the
configuration below. The file is intentionally a single, non-training data
pipeline: it preserves the source, converts the tab-separated HIS files,
creates all preparation stages, writes audit reports, and produces the
chronological unified dataset.

Expected packages: pandas and numpy.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ----------------------------- configuration -----------------------------

CITY = "Guwahati"
AIRPORT = "Lokpriya Gopinath Bordoloi International Airport"
STATION_CODE = "VEGT"
RUNWAY = "RWY02"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SOURCE = PROJECT_ROOT / "Guwahati-RAW-DATA"

OBSERVATION_INTERVAL_SECONDS = 15
MAX_GAP_MINUTES = 5
MAX_GAP_ROWS = math.floor(
    MAX_GAP_MINUTES * 60 / OBSERVATION_INTERVAL_SECONDS
)

# The source contains complete RWY02 sensor coverage from 2024-08-11 through
# 2026-01-31. END_DATE is exclusive, as required by instructions.md.
START_DATE = pd.Timestamp("2024-08-11")
END_DATE = pd.Timestamp("2026-02-01")

# These names are produced by normalize_column_name(). Verify them in
# 09_reports/header_manifest.csv after the conversion phase.
TARGET_COLUMNS = [
    f"CLOUD_{RUNWAY}_VERVIS_FT",
    f"VIS_{RUNWAY}_MOR_10A",
]
WIND_COLUMNS = [
    f"WIND_{RUNWAY}_KT_WS10A_KT",
    f"WIND_{RUNWAY}_KT_WD10A",
    f"WIND_{RUNWAY}_KT_GUST_SPEED_KT",
]
RAIN_COLUMNS = [
    f"PW_{RUNWAY}_RAIN_INTENSITY_MM_H",
    f"PW_{RUNWAY}_RAIN_SUM_MM",
]
PTU_THRESHOLDS = {
    f"PTU_{RUNWAY}_TEMP_C": 0.5,
    f"PTU_{RUNWAY}_DEWPOINT_C": 0.5,
    f"PTU_{RUNWAY}_RH": 5.0,
    f"PTU_{RUNWAY}_QNH_AERODROME_HPA": 2.0,
}

# Only these exact prefixes are selected. No ALL, MID1, RWY20, AWS, or event
# files are silently mixed into this runway project.
SENSOR_PREFIXES = [
    f"CLOUD_{RUNWAY}_",
    f"PTU_{RUNWAY}_",
    f"PW_{RUNWAY}_",
    f"VIS_{RUNWAY}_",
    f"WIND_{RUNWAY}_KT_",
]

SEMANTIC_FILL_VALUES = {
    "cloud_base": 15000,
    "cloud_octa": 0,
    "vertical_visibility": 15000,
    "rain": 0,
}

OUTPUT_COLUMNS = [
    "Datetime",
    "CLOUD_Cloud_Base_1",
    "CLOUD_Cloud_Base_2",
    "CLOUD_Cloud_Base_3",
    "CLOUD_SITE",
    "CLOUD_OCTA1",
    "CLOUD_OCTA2",
    "CLOUD_OCTA3",
    "CLOUD_OCTA4",
    "CLOUD_Vertical_Visibility",
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
    "CLOUD_Cloud_Base_1_IMPUTED",
    "CLOUD_Cloud_Base_2_IMPUTED",
    "CLOUD_Cloud_Base_3_IMPUTED",
    "CLOUD_Vertical_Visibility_IMPUTED",
    "CLOUD_LONG_GAP",
    "CLOUD_VERTICAL_LONG_GAP",
    "CLOUD_SENSOR_ON",
    "PTU_SENSOR_ON",
    "PW_SENSOR_ON",
    "VIS_SENSOR_ON",
    "WIND_SENSOR_ON",
]

OUTPUT_MAPPING = {
    "CLOUD_Cloud_Base_1": f"CLOUD_{RUNWAY}_BASE1_FT",
    "CLOUD_Cloud_Base_2": f"CLOUD_{RUNWAY}_BASE2_FT",
    "CLOUD_Cloud_Base_3": f"CLOUD_{RUNWAY}_BASE3_FT",
    "CLOUD_SITE": f"CLOUD_{RUNWAY}_SITE",
    "CLOUD_OCTA1": f"CLOUD_{RUNWAY}_OCTA1",
    "CLOUD_OCTA2": f"CLOUD_{RUNWAY}_OCTA2",
    "CLOUD_OCTA3": f"CLOUD_{RUNWAY}_OCTA3",
    "CLOUD_OCTA4": f"CLOUD_{RUNWAY}_OCTA4",
    "CLOUD_Vertical_Visibility": f"CLOUD_{RUNWAY}_VERVIS_FT",
    "PTU_Pressure_QNH": f"PTU_{RUNWAY}_QNH_AERODROME_HPA",
    "PTU_Temperature": f"PTU_{RUNWAY}_TEMP_C",
    "PTU_DewPoint": f"PTU_{RUNWAY}_DEWPOINT_C",
    "PTU_Humidity": f"PTU_{RUNWAY}_RH",
    "PW_Rain_Intensity": f"PW_{RUNWAY}_RAIN_INTENSITY_MM_H",
    "PW_Rain_Sum": f"PW_{RUNWAY}_RAIN_SUM_MM",
    "VIS_Visibility_MOR": f"VIS_{RUNWAY}_MOR_10A",
    "VIS_RVR": f"VIS_{RUNWAY}_RVR_10A",
    "WIND_Wind_Dir": f"WIND_{RUNWAY}_KT_WD10A",
    "WIND_Wind_Speed": f"WIND_{RUNWAY}_KT_WS10A_KT",
    "WIND_Wind_Gust": f"WIND_{RUNWAY}_KT_GUST_SPEED_KT",
}

SENSOR_SOURCE_COLUMNS = {
    "CLOUD_SENSOR_ON": "CLOUD_",
    "PTU_SENSOR_ON": "PTU_",
    "PW_SENSOR_ON": "PW_",
    "VIS_SENSOR_ON": "VIS_",
    "WIND_SENSOR_ON": "WIND_",
}


# ----------------------------- general helpers -----------------------------

def report_dir() -> Path:
    path = PROJECT_ROOT / "09_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_column_name(value: object) -> str:
    text = str(value).replace("\ufeff", "").replace("Ã‚", "")
    text = text.replace("Â°", "").replace("Â²", "2")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text.upper()


def source_files() -> list[Path]:
    return sorted(RAW_SOURCE.rglob("*.his"))


def selected_source(path: Path) -> bool:
    return any(path.stem.startswith(prefix) for prefix in SENSOR_PREFIXES)


def sensor_name(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def source_relative(path: Path) -> Path:
    return path.relative_to(RAW_SOURCE)


def parse_day(path: Path) -> str:
    match = re.search(r"_(\d{2})$", path.stem)
    if not match:
        raise ValueError(f"Cannot find day suffix in {path.name}")
    return match.group(1)


def iter_csv_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.csv"))


# ----------------------------- stage 4 -------------------------------------

def stage4_lstm_ready() -> None:
    input_root = PROJECT_ROOT / "06_final_cleaned"
    output_root = PROJECT_ROOT / "07_lstm_ready"
    summaries, uniformity, failures = [], [], []
    for path in iter_csv_files(input_root):
        relative = path.relative_to(input_root)
        try:
            source = pd.read_csv(path, parse_dates=["Datetime"], low_memory=False)
            source = source.dropna(subset=["Datetime"]).sort_values("Datetime")
            before = len(source)
            frame = pd.DataFrame({"Datetime": source["Datetime"]})
            for output_column, source_column in OUTPUT_MAPPING.items():
                frame[output_column] = pd.to_numeric(
                    source.get(source_column, pd.Series(index=source.index, dtype="object")),
                    errors="coerce",
                ).to_numpy()
            cloud_base_columns = [
                "CLOUD_Cloud_Base_1",
                "CLOUD_Cloud_Base_2",
                "CLOUD_Cloud_Base_3",
            ]
            for column in cloud_base_columns:
                missing_before_fill = frame[column].isna()
                frame[column] = frame[column].ffill(limit=3)
                frame[f"{column}_IMPUTED"] = (
                    missing_before_fill & frame[column].notna()
                ).astype("int8")
            vertical_visibility = "CLOUD_Vertical_Visibility"
            missing_before_fill = frame[vertical_visibility].isna()
            frame[vertical_visibility] = frame[vertical_visibility].ffill(limit=3)
            frame[f"{vertical_visibility}_IMPUTED"] = (
                missing_before_fill & frame[vertical_visibility].notna()
            ).astype("int8")
            frame["CLOUD_LONG_GAP"] = (
                frame[cloud_base_columns].isna().any(axis=1).astype("int8")
            )
            frame["CLOUD_VERTICAL_LONG_GAP"] = (
                frame[vertical_visibility].isna().astype("int8")
            )
            for column in ["CLOUD_OCTA1", "CLOUD_OCTA2", "CLOUD_OCTA3", "CLOUD_OCTA4"]:
                frame[column] = frame[column].fillna(SEMANTIC_FILL_VALUES["cloud_octa"])
            frame["PW_Rain_Intensity"] = frame["PW_Rain_Intensity"].fillna(SEMANTIC_FILL_VALUES["rain"])
            frame["PW_Rain_Sum"] = frame["PW_Rain_Sum"].fillna(SEMANTIC_FILL_VALUES["rain"])
            for flag, prefix in SENSOR_SOURCE_COLUMNS.items():
                sensor_columns = [column for column in source.columns if column.startswith(f"{prefix}{RUNWAY}_")]
                frame[flag] = source[sensor_columns].notna().any(axis=1).astype("int8") if sensor_columns else 0
            frame["Hour_Sin"] = source.get("Hour_Sin", np.nan)
            frame["Hour_Cos"] = source.get("Hour_Cos", np.nan)
            frame = frame.reindex(columns=OUTPUT_COLUMNS)
            timestamps = frame["Datetime"]
            deltas = timestamps.diff().dropna().dt.total_seconds()
            non_interval = int((deltas != OBSERVATION_INTERVAL_SECONDS).sum())
            duplicates = int(timestamps.duplicated().sum())
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(destination, index=False, date_format="%Y-%m-%d %H:%M:%S")
            summaries.append(
                {
                    "file": str(relative), "rows_read": len(source), "target_rows_removed": 0,
                    "rows_written": len(frame), "duplicate_timestamps": duplicates,
                    "non_interval_gaps": non_interval, "remaining_nan_cells": int(frame.isna().sum().sum()),
                    "sorted": bool(timestamps.is_monotonic_increasing),
                }
            )
            uniformity.append(
                {"file": str(relative), "expected_interval_seconds": OBSERVATION_INTERVAL_SECONDS,
                 "non_interval_gaps": non_interval, "sorted": bool(timestamps.is_monotonic_increasing)}
            )
        except Exception as exc:
            failures.append({"file": str(relative), "error_type": type(exc).__name__, "error": str(exc)})
    write_csv(report_dir() / "dataset_summary.csv", summaries, ["file", "rows_read", "target_rows_removed", "rows_written", "duplicate_timestamps", "non_interval_gaps", "remaining_nan_cells", "sorted"])
    write_csv(report_dir() / "uniformity_report.csv", uniformity, ["file", "expected_interval_seconds", "non_interval_gaps", "sorted"])
    write_csv(report_dir() / "stage4_failed_files.csv", failures, ["file", "error_type", "error"])
    if failures:
        raise RuntimeError(f"{len(failures)} Stage 4 files failed; inspect stage4_failed_files.csv")




if __name__ == '__main__':
    stage4_lstm_ready()
    print('Phase G complete.')
