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


# ----------------------------- stage 3 -------------------------------------

def missing_runs(values: pd.Series) -> list[tuple[int, int]]:
    missing = values.isna().to_numpy()
    runs = []
    start = None
    for index, is_missing in enumerate(missing):
        if is_missing and start is None:
            start = index
        if start is not None and (not is_missing or index == len(missing) - 1):
            end = index - 1 if not is_missing else index
            runs.append((start, end))
            start = None
    return runs


def bounded_interpolate(frame: pd.DataFrame, columns: list[str]) -> int:
    filled = 0
    for column in columns:
        if column not in frame:
            continue
        before = int(frame[column].isna().sum())
        values = frame[column].interpolate(method="time", limit_area="inside")
        for start, end in missing_runs(frame[column]):
            row_count = end - start + 1
            bounded = start > 0 and end + 1 < len(frame)
            if row_count <= MAX_GAP_ROWS and bounded:
                frame.iloc[start : end + 1, frame.columns.get_loc(column)] = values.iloc[
                    start : end + 1
                ]
        filled += before - int(frame[column].isna().sum())
    return filled


def stage3_final_cleaning() -> None:
    input_root = PROJECT_ROOT / "05_stage2_flagged"
    output_root = PROJECT_ROOT / "06_final_cleaned"
    wind_flags, summaries, final_flags, failures = [], [], [], []
    for path in iter_csv_files(input_root):
        relative = path.relative_to(input_root)
        try:
            frame = pd.read_csv(path, parse_dates=["Datetime"], low_memory=False)
            frame = frame.dropna(subset=["Datetime"]).sort_values("Datetime").set_index("Datetime")
            for column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            for column in WIND_COLUMNS:
                if column not in frame:
                    continue
                for start, end in missing_runs(frame[column]):
                    if end - start + 1 > MAX_GAP_ROWS:
                        wind_flags.append(
                            {
                                "file": str(relative), "variable": column,
                                "start": str(frame.index[start]), "end": str(frame.index[end]),
                                "row_count": end - start + 1,
                            }
                        )
            wind_filled = 0
            rain_filled = 0
            remaining = int(frame.isna().sum().sum())
            if remaining:
                final_flags.append({"file": str(relative), "remaining_nan_cells": remaining})
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.reset_index().to_csv(destination, index=False, date_format="%Y-%m-%d %H:%M:%S")
            summaries.append(
                {
                    "file": str(relative), "wind_values_filled": wind_filled,
                    "rain_values_filled": rain_filled, "remaining_nan_cells": remaining,
                }
            )
        except Exception as exc:
            failures.append({"file": str(relative), "error_type": type(exc).__name__, "error": str(exc)})
    write_csv(report_dir() / "wind_gap_flags.csv", wind_flags, ["file", "variable", "start", "end", "row_count"])
    write_csv(report_dir() / "cleaning_summary.csv", summaries, ["file", "wind_values_filled", "rain_values_filled", "remaining_nan_cells"])
    write_csv(report_dir() / "final_flags.csv", final_flags, ["file", "remaining_nan_cells"])
    write_csv(report_dir() / "stage3_failed_files.csv", failures, ["file", "error_type", "error"])
    if failures:
        raise RuntimeError(f"{len(failures)} Stage 3 files failed; inspect stage3_failed_files.csv")




if __name__ == '__main__':
    stage3_final_cleaning()
    print('Phase F complete.')

