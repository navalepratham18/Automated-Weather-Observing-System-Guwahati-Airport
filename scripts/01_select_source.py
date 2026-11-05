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
    "CLOUD_Vertical_Visibility",
    "VIS_Visibility_MOR",
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


def ensure_layout() -> None:
    for name in [
        "01_source_selected",
        "02_sensor_csv",
        "03_daily_merged",
        "04_stage1_preprocessed",
        "05_stage2_flagged",
        "06_final_cleaned",
        "07_lstm_ready",
        "08_unified_dataset",
        "09_reports",
        "10_models",
        "scripts",
        "notebooks",
    ]:
        (PROJECT_ROOT / name).mkdir(parents=True, exist_ok=True)


def verify_raw_source() -> None:
    if not RAW_SOURCE.is_dir():
        raise RuntimeError(f"Raw source directory does not exist: {RAW_SOURCE}")

    source_files_only = sorted(RAW_SOURCE.rglob("*.his"))
    if not source_files_only:
        raise RuntimeError(f"No HIS files found in {RAW_SOURCE}")

    empty_files = [path for path in source_files_only if path.stat().st_size == 0]
    if empty_files:
        raise RuntimeError(
            f"{len(empty_files)} source HIS files are empty: "
            f"{empty_files[0]}"
        )

    verification = [
        {
            "relative_path": str(path.relative_to(RAW_SOURCE)),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in source_files_only
    ]
    write_csv(
        report_dir() / "source_inventory.csv",
        verification,
        ["relative_path", "sha256", "size_bytes"],
    )


# ----------------------------- phase A -------------------------------------

def select_source_files() -> None:
    selected_root = PROJECT_ROOT / "01_source_selected"
    rows = []
    for path in source_files():
        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Source file is empty and cannot be selected: {path}"
            )
        relative = source_relative(path)
        if selected_source(path):
            target = selected_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            rows.append(
                {
                    "source": str(relative),
                    "selected": True,
                    "reason": f"Exact selected sensor prefix for {RUNWAY}",
                }
            )
        else:
            rows.append(
                {
                    "source": str(relative),
                    "selected": False,
                    "reason": "Not one of the configured runway sensor prefixes",
                }
            )
    write_csv(
        report_dir() / "source_selection.csv",
        rows,
        ["source", "selected", "reason"],
    )


def write_run_config() -> None:
    write_text(
        report_dir() / f"{CITY}_run_config.txt",
        "\n".join(
            [
                f"CITY={CITY}",
                f"AIRPORT={AIRPORT}",
                f"IATA_OR_STATION_CODE={STATION_CODE}",
                f"RUNWAY={RUNWAY}",
                f"RAW_SOURCE={RAW_SOURCE}",
                f"OBSERVATION_INTERVAL_SECONDS={OBSERVATION_INTERVAL_SECONDS}",
                f"START_DATE={START_DATE.date()}",
                f"END_DATE={END_DATE.date()}",
                f"TARGET_COLUMNS={','.join(TARGET_COLUMNS)}",
                f"WIND_COLUMNS={','.join(WIND_COLUMNS)}",
                f"RAIN_COLUMNS={','.join(RAIN_COLUMNS)}",
                f"PTU_COLUMNS={','.join(PTU_THRESHOLDS)}",
                "SOURCE_PROVIDER=Vaisala AVIMET history export; verify with station documentation",
                "DOWNLOAD_DATE=unknown; fill before execution",
                "TIMEZONE=Asia/Kolkata; timestamps appear to be local station time; verify before execution",
                "MISSING_VALUE_CODES=blank fields; control/status fields are not numeric measurements",
                "MODEL_TRAINING=not performed by this file",
            ]
        )
        + "\n",
    )



def main() -> None:
    ensure_layout()
    write_run_config()
    verify_raw_source()
    select_source_files()
    print('Phase A complete.')


if __name__ == '__main__':
    main()
