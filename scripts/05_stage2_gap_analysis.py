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


# ----------------------------- stage 2 -------------------------------------

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


def stage2_gap_analysis() -> None:
    input_root = PROJECT_ROOT / "04_stage1_preprocessed"
    output_root = PROJECT_ROOT / "05_stage2_flagged"
    gap_rows, threshold_rows, flagged, summaries, failures = [], [], [], [], []
    for path in iter_csv_files(input_root):
        relative = path.relative_to(input_root)
        try:
            frame = pd.read_csv(path, parse_dates=["Datetime"], low_memory=False).sort_values("Datetime")
            frame = frame.set_index("Datetime")
            file_flagged = False
            ptu_interpolated = 0
            for column, threshold in PTU_THRESHOLDS.items():
                if column not in frame:
                    continue
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
                before = int(frame[column].isna().sum())
                original_values = frame[column].copy()
                frame[column] = frame[column].interpolate(
                    method="time",
                    limit=3,
                ).round(2)
                ptu_interpolated += before - int(frame[column].isna().sum())
                for start, end in missing_runs(original_values):
                    previous = frame[column].iloc[start - 1] if start > 0 else np.nan
                    following = frame[column].iloc[end + 1] if end + 1 < len(frame) else np.nan
                    duration = (end - start + 1) * OBSERVATION_INTERVAL_SECONDS
                    gap_rows.append(
                        {
                            "file": str(relative), "variable": column,
                            "start": str(frame.index[start]), "end": str(frame.index[end]),
                            "row_count": end - start + 1, "duration_seconds": duration,
                            "previous_value": previous, "next_value": following,
                        }
                    )
                    if end - start + 1 > MAX_GAP_ROWS:
                        file_flagged = True
                    elif pd.notna(previous) and pd.notna(following) and abs(following - previous) > threshold:
                        threshold_rows.append(
                            {
                                "file": str(relative), "variable": column,
                                "start": str(frame.index[start]), "end": str(frame.index[end]),
                                "row_count": end - start + 1, "absolute_change": abs(following - previous),
                                "threshold": threshold,
                            }
                        )
                        file_flagged = True
            if file_flagged:
                flagged.append({"file": str(relative), "reason": "Long PTU gap or threshold breach"})
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.reset_index().to_csv(destination, index=False, date_format="%Y-%m-%d %H:%M:%S")
            summaries.append(
                {
                    "file": str(relative),
                    "ptu_interpolated_cells": ptu_interpolated,
                }
            )
        except Exception as exc:
            failures.append({"file": str(relative), "error_type": type(exc).__name__, "error": str(exc)})
    write_csv(report_dir() / "gap_audit.csv", gap_rows, ["file", "variable", "start", "end", "row_count", "duration_seconds", "previous_value", "next_value"])
    write_csv(report_dir() / "threshold_audit.csv", threshold_rows, ["file", "variable", "start", "end", "row_count", "absolute_change", "threshold"])
    write_csv(report_dir() / "flagged_files.csv", flagged, ["file", "reason"])
    write_csv(
        report_dir() / "ptu_interpolation_summary.csv",
        summaries,
        ["file", "ptu_interpolated_cells"],
    )
    write_csv(report_dir() / "stage2_failed_files.csv", failures, ["file", "error_type", "error"])
    if failures:
        raise RuntimeError(f"{len(failures)} Stage 2 files failed; inspect stage2_failed_files.csv")




if __name__ == '__main__':
    stage2_gap_analysis()
    print('Phase E complete.')

