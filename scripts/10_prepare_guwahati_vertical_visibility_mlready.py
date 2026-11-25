"""
Stage 4: Guwahati ML-ready data preparation.

Mirrors the Lucknow stage4_mlreadydata.py workflow:
- parse and sort Datetime
- preserve every valid-timestamp row
- fill missing primary target with 15000 (no cloud)
- fill cloud bases with 15000 when no cloud height is reported
- fill octas with 0
- fill vertical visibility with 15000
- fill rain columns with 0
- remove duplicate timestamps
- generate uniformity and dataset summary reports

Primary target: CLOUD_Vertical_Visibility
Secondary target: VIS_Visibility_MOR

This script does not train or run an ML/DL model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "08_unified_dataset"

TARGET_COL = "CLOUD_Vertical_Visibility"
SECONDARY_TARGET_COL = "VIS_Visibility_MOR"
NO_CLOUD_HEIGHT = 15000

CLOUD_BASE_COLS = [
    "CLOUD_Cloud_Base_1",
    "CLOUD_Cloud_Base_2",
    "CLOUD_Cloud_Base_3",
]

OCTA_COLS = [
    "CLOUD_OCTA1",
    "CLOUD_OCTA2",
    "CLOUD_OCTA3",
    "CLOUD_OCTA4",
]

RAIN_COLS = [
    "PW_Rain_Intensity",
    "PW_Rain_Sum",
]

VV_COL = "CLOUD_Vertical_Visibility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Guwahati dataset for LSTM training using Lucknow Stage-4 logic."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATASET_DIR / "guwahati_unified_lstm_dataset_v3_physics_corrected.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "07_lstm_ready" / "vertical_visibility_experiment",
    )
    parser.add_argument("--progress-every", type=int, default=250000)
    return parser.parse_args()


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    seconds = df["Datetime"].dt.hour * 3600 + df["Datetime"].dt.minute * 60 + df["Datetime"].dt.second
    angle = 2 * np.pi * seconds / 86400
    df["Hour_Sin"] = np.sin(angle)
    df["Hour_Cos"] = np.cos(angle)
    return df


def build_summary(
    df_before: pd.DataFrame,
    df: pd.DataFrame,
    rows_before: int,
    duplicate_removed: int,
    target_filled: int,
) -> dict:
    diff = df["Datetime"].diff().dt.total_seconds()
    non15 = int(diff.dropna()[~diff.dropna().isin([15.0])].count())

    return {
        "Rows_Before": rows_before,
        "Duplicate_Rows_Removed": duplicate_removed,
        "Missing_Target_Rows_Filled": target_filled,
        "Rows_Final": len(df),
        "Duplicate_Datetime": int(df["Datetime"].duplicated().sum()),
        "Sorted": bool(df["Datetime"].is_monotonic_increasing),
        "Non15Sec_Intervals": non15,
        "Remaining_NaNs": int(df.isna().sum().sum()),
    }


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GUWAHATI STAGE 4 ML-READY DATA")
    print("=" * 80)
    print("Input:", args.input)
    print("Output:", args.output_dir)

    df = pd.read_csv(args.input, low_memory=False)

    if "Datetime" not in df.columns:
        raise ValueError("Missing required column: Datetime")

    required_target_cols = [TARGET_COL]
    missing_targets = [column for column in required_target_cols if column not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing primary target column: {missing_targets}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Datetime"]).copy()
    df = df.sort_values("Datetime")

    rows_before = len(df)

    duplicate_removed = int(df["Datetime"].duplicated().sum())
    df = df.drop_duplicates(subset=["Datetime"], keep="first")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    target_filled = int(df[TARGET_COL].isna().sum())
    df[TARGET_COL] = df[TARGET_COL].fillna(NO_CLOUD_HEIGHT)

    if SECONDARY_TARGET_COL in df.columns:
        df[SECONDARY_TARGET_COL] = pd.to_numeric(df[SECONDARY_TARGET_COL], errors="coerce")

    for column in CLOUD_BASE_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(NO_CLOUD_HEIGHT)

    for column in OCTA_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    if VV_COL in df.columns:
        df[VV_COL] = pd.to_numeric(df[VV_COL], errors="coerce").fillna(NO_CLOUD_HEIGHT)

    for column in RAIN_COLS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in [
        "CLOUD_Cloud_Base_1",
        "CLOUD_Cloud_Base_2",
        "CLOUD_Cloud_Base_3",
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
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = add_datetime_features(df)

    output_file = args.output_dir / "guwahati_mlready.csv"
    df_report = df.copy()
    df_report["Datetime"] = df_report["Datetime"].dt.strftime("%d-%m-%Y %H:%M:%S")
    df_report.to_csv(output_file, index=False)

    summary = build_summary(
        df,
        df,
        rows_before,
        duplicate_removed,
        target_filled,
    )
    pd.DataFrame([summary]).to_csv(
        args.output_dir / "dataset_summary.csv",
        index=False,
    )

    uniformity = {
        "File": "guwahati_mlready.csv",
        "Duplicate_Datetime": summary["Duplicate_Datetime"],
        "Sorted": summary["Sorted"],
        "Non15Sec_Intervals": summary["Non15Sec_Intervals"],
        "Remaining_NaNs": summary["Remaining_NaNs"],
    }
    pd.DataFrame([uniformity]).to_csv(
        args.output_dir / "uniformity_report.csv",
        index=False,
    )

    print("Rows before:", rows_before)
    print("Duplicates removed:", duplicate_removed)
    print("Missing primary target rows filled:", target_filled)
    print("Rows final:", len(df))
    print("Remaining NaNs:", summary["Remaining_NaNs"])
    print("Non-15-second intervals:", summary["Non15Sec_Intervals"])
    print("Saved:", output_file)
    print("=" * 80)
    print("STAGE 4 COMPLETE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
