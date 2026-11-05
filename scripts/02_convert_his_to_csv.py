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


def read_his(path: Path) -> pd.DataFrame:
    # All selected files are tab-separated with a one-line "History file"
    # marker followed by the header. Keep blank fields as missing values.
    frame = None
    last_decode_error = None
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                next(handle)
                header = next(handle).rstrip("\r\n")
            expected_fields = len(header.split("\t"))

            def repair_extra_fields(fields: list[str]) -> list[str]:
                if len(fields) <= expected_fields:
                    return fields
                return fields[: expected_fields - 1] + [
                    "\t".join(fields[expected_fields - 1 :])
                ]

            frame = pd.read_csv(
                path,
                sep="\t",
                skiprows=1,
                dtype=str,
                keep_default_na=False,
                na_values=[""],
                engine="python",
                encoding=encoding,
                on_bad_lines=repair_extra_fields,
            )
            break
        except UnicodeDecodeError as exc:
            last_decode_error = exc
    if frame is None:
        raise RuntimeError(f"Unable to decode {path.name}") from last_decode_error

    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame]
    if "CREATEDATE" not in frame.columns:
        raise ValueError(f"{path.name} has no CREATEDATE column")
    created = frame.pop("CREATEDATE")
    timestamps = pd.to_datetime(created, errors="coerce")
    frame.insert(0, "Date", timestamps.dt.strftime("%Y-%m-%d"))
    frame.insert(1, "Time", timestamps.dt.strftime("%H:%M:%S"))
    frame = frame.loc[timestamps.notna()].copy()
    return frame


def convert_his_to_csv() -> None:
    output_root = PROJECT_ROOT / "02_sensor_csv"
    failures = []
    manifest = []
    for path in sorted((PROJECT_ROOT / "01_source_selected").rglob("*.his")):
        relative = path.relative_to(PROJECT_ROOT / "01_source_selected")
        destination = output_root / relative.with_suffix(".csv")
        try:
            frame = read_his(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(destination, index=False)
            manifest.extend(
                {
                    "source_file": str(relative),
                    "output_file": str(destination.relative_to(output_root)),
                    "column_position": index,
                    "source_column": column,
                    "normalized_column": (
                        column
                        if column in {"Date", "Time"}
                        else f"{sensor_name(path)}_{normalize_column_name(column)}"
                    ),
                }
                for index, column in enumerate(frame.columns)
            )
        except Exception as exc:
            failures.append(
                {"file": str(relative), "error_type": type(exc).__name__, "error": str(exc)}
            )
    write_csv(
        report_dir() / "conversion_failures.csv",
        failures,
        ["file", "error_type", "error"],
    )
    write_csv(
        report_dir() / "header_manifest.csv",
        manifest,
        ["source_file", "output_file", "column_position", "source_column", "normalized_column"],
    )
    if failures:
        raise RuntimeError(f"{len(failures)} HIS files failed conversion; inspect conversion_failures.csv")




if __name__ == '__main__':
    convert_his_to_csv()
    print('Phase B complete.')

