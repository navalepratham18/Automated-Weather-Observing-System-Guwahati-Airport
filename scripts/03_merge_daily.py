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


# ----------------------------- phase C -------------------------------------

def merge_daily_files() -> None:
    input_root = PROJECT_ROOT / "02_sensor_csv"
    output_root = PROJECT_ROOT / "03_daily_merged"
    grouped: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for path in iter_csv_files(input_root):
        relative = path.relative_to(input_root)
        grouped[(relative.parent, parse_day(path))].append(path)

    failures = []
    missing = []
    expected = {prefix.rstrip("_") for prefix in SENSOR_PREFIXES}
    for (parent, day), paths in sorted(grouped.items(), key=lambda item: str(item[0])):
        merged = None
        available = set()
        try:
            for path in sorted(paths):
                frame = pd.read_csv(path, low_memory=False)
                frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
                if not {"Date", "Time"}.issubset(frame.columns):
                    raise ValueError("missing Date or Time")
                prefix = sensor_name(path)
                available.add(prefix)
                frame = frame.drop_duplicates(["Date", "Time"], keep="first")
                rename = {
                    column: f"{prefix}_{normalize_column_name(column)}"
                    for column in frame.columns
                    if column not in {"Date", "Time"}
                }
                frame = frame.rename(columns=rename)
                merged = frame if merged is None else merged.merge(
                    frame, on=["Date", "Time"], how="outer"
                )
            if merged is None:
                continue
            merged["_sort_datetime"] = pd.to_datetime(
                merged["Date"].astype(str) + " " + merged["Time"].astype(str),
                errors="coerce",
                dayfirst=False,
            )
            merged = merged.sort_values("_sort_datetime").drop(columns="_sort_datetime")
            destination = output_root / parent / f"{parent.name}_{day}.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(destination, index=False)
            for sensor in sorted(expected - available):
                missing.append(
                    {"file": str(destination.relative_to(output_root)), "missing_sensor": sensor}
                )
        except Exception as exc:
            failures.append(
                {
                    "directory": str(parent),
                    "day": day,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    write_csv(report_dir() / "merge_failures.csv", failures, ["directory", "day", "error_type", "error"])
    write_csv(report_dir() / "missing_sensors.csv", missing, ["file", "missing_sensor"])
    if failures:
        raise RuntimeError(f"{len(failures)} daily merges failed; inspect merge_failures.csv")




if __name__ == '__main__':
    merge_daily_files()
    print('Phase C complete.')

