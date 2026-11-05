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

UNIFIED_COLUMNS = [
    "Datetime",
    "CLOUD_Cloud_Base_1", "CLOUD_Cloud_Base_2", "CLOUD_Cloud_Base_3",
    "CLOUD_SITE", "CLOUD_OCTA1", "CLOUD_OCTA2", "CLOUD_OCTA3", "CLOUD_OCTA4",
    "CLOUD_Vertical_Visibility", "PTU_Pressure_QNH", "PTU_Temperature",
    "PTU_DewPoint", "PTU_Humidity", "PW_Rain_Intensity", "PW_Rain_Sum",
    "VIS_Visibility_MOR", "VIS_RVR", "WIND_Wind_Dir", "WIND_Wind_Speed",
    "WIND_Wind_Gust", "Hour_Sin", "Hour_Cos", "CLOUD_SENSOR_ON",
    "PTU_SENSOR_ON", "PW_SENSOR_ON", "VIS_SENSOR_ON", "WIND_SENSOR_ON",
    "CLOUD_Cloud_Base_1_IMPUTED", "CLOUD_Cloud_Base_2_IMPUTED",
    "CLOUD_Cloud_Base_3_IMPUTED", "CLOUD_Vertical_Visibility_IMPUTED",
    "CLOUD_LONG_GAP", "CLOUD_VERTICAL_LONG_GAP",
]


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


# ----------------------------- phase H -------------------------------------

def merge_unified_dataset() -> None:
    input_root = PROJECT_ROOT / "07_lstm_ready"
    files = iter_csv_files(input_root)
    frames = []
    failures = []
    rows_read = invalid_timestamps = filtered_rows = 0
    for path in files:
        relative = path.relative_to(input_root)
        try:
            frame = pd.read_csv(path, low_memory=False)
            if "Datetime" not in frame:
                raise ValueError("missing Datetime")
            rows_read += len(frame)
            frame["Datetime"] = pd.to_datetime(frame["Datetime"], dayfirst=False, errors="coerce")
            invalid_timestamps += int(frame["Datetime"].isna().sum())
            frame = frame.dropna(subset=["Datetime"])
            frame = frame[(frame["Datetime"] >= START_DATE) & (frame["Datetime"] < END_DATE)]
            filtered_rows += len(frame)
            frames.append(frame)
        except Exception as exc:
            failures.append({"file": str(relative), "error_type": type(exc).__name__, "error": str(exc)})
    if not frames:
        raise RuntimeError("No rows remain for the configured unified date range.")
    combined = pd.concat(
        [frame.reindex(columns=UNIFIED_COLUMNS) for frame in frames],
        ignore_index=True,
    )
    before = len(combined)
    combined = combined.sort_values("Datetime").drop_duplicates("Datetime", keep="first")
    duplicate_rows = before - len(combined)
    output = PROJECT_ROOT / "08_unified_dataset" / f"{CITY.lower()}_unified_lstm_dataset_v2_short_cloud_fill_20240811_20260131.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, date_format="%d-%m-%Y %H:%M:%S")
    write_csv(
        report_dir() / f"{CITY.lower()}_unified_lstm_dataset_summary.csv",
        [{
            "discovered_files": len(files), "processed_files": len(frames), "failed_files": len(failures),
            "rows_read": rows_read, "invalid_timestamps": invalid_timestamps,
            "date_filtered_rows": filtered_rows, "duplicate_rows_removed": duplicate_rows,
            "rows_written": len(combined), "columns_written": len(combined.columns),
            "start_date": START_DATE.date(), "end_date_exclusive": END_DATE.date(),
            "failures": " | ".join(item["file"] for item in failures),
        }],
        ["discovered_files", "processed_files", "failed_files", "rows_read", "invalid_timestamps", "date_filtered_rows", "duplicate_rows_removed", "rows_written", "columns_written", "start_date", "end_date_exclusive", "failures"],
    )
    if failures:
        write_csv(report_dir() / "unified_failed_files.csv", failures, ["file", "error_type", "error"])
        raise RuntimeError(f"{len(failures)} files failed unified merge; inspect unified_failed_files.csv")




if __name__ == '__main__':
    merge_unified_dataset()
    print('Phase H complete.')
