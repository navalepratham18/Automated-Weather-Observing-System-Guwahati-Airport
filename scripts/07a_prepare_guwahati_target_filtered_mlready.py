import os
import numpy as np
import pandas as pd

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "06_final_cleaned")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "07_lstm_ready",
    "target_filtered_experiment",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =====================================================
# CONFIGURATION
# =====================================================

TARGET_COL = "VIS_Visibility_MOR"

WIND_COLS = [
    "WIND_Wind_Dir",
    "WIND_Wind_Speed",
    "WIND_Wind_Gust"
]

RAIN_COLS = [
    "PW_Rain_Intensity",
    "PW_Rain_Sum"
]

CLOUD_BASE_COLS = [
    "CLOUD_Cloud_Base_1",
    "CLOUD_Cloud_Base_2",
    "CLOUD_Cloud_Base_3"
]

OCTA_COLS = [
    "CLOUD_OCTA1",
    "CLOUD_OCTA2",
    "CLOUD_OCTA3",
    "CLOUD_OCTA4"
]

VV_COL = "CLOUD_Vertical_Visibility"

NO_CLOUD_HEIGHT = 15000

# =====================================================
# REPORTS
# =====================================================

dataset_summary = []
uniformity_report = []
failed_files = []

# =====================================================
# FIND FILES
# =====================================================

all_files = []

for root, dirs, files in os.walk(INPUT_DIR):

    for f in files:

        if (
            f.lower().endswith(".csv")
            and
            f not in [
                "cleaning_summary.csv",
                "final_flags.csv",
                "wind_gap_flags.csv",
                "failed_files.csv"
            ]
        ):

            all_files.append(
                os.path.join(root, f)
            )

all_files.sort()

print(
    "\nFILES FOUND:",
    len(all_files)
)

# =====================================================
# LOAD WIND FLAGS
# =====================================================

wind_flag_file = os.path.join(
    BASE_DIR,
    "09_reports",
    "wind_gap_flags.csv"
)

if os.path.exists(
    wind_flag_file
):

    wind_flags = pd.read_csv(
        wind_flag_file
    )

    wind_flags["start"] = (
        pd.to_datetime(
            wind_flags["start"],
            errors="coerce"
        )
    )

    wind_flags["end"] = (
        pd.to_datetime(
            wind_flags["end"],
            errors="coerce"
        )
    )

else:

    wind_flags = pd.DataFrame()

# =====================================================
# PROCESS FILES
# =====================================================

for file_no, file_path in enumerate(
    all_files,
    start=1
):

    file_name = os.path.basename(
        file_path
    )

    print("\n" + "=" * 80)
    print(
        f"[{file_no}/{len(all_files)}]"
    )
    print(file_name)
    print(
        f"Progress : "
        f"{100*file_no/len(all_files):.2f}%"
    )
    print("=" * 80)

    try:

        # =============================================
        # LOAD
        # =============================================

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        df["Datetime"] = pd.to_datetime(
            df["Datetime"],
            dayfirst=True,
            errors="coerce"
        )

        df = df.dropna(
            subset=["Datetime"]
        )

        df = df.sort_values(
            "Datetime"
        )

        rows_before = len(df)

        # =============================================
        # REMOVE LONG WIND GAP ROWS
        # =============================================

        rows_removed = 0

        relative_path = os.path.relpath(file_path, INPUT_DIR)

        if not wind_flags.empty:

            file_flags = wind_flags[
                wind_flags["file"]
                ==
                relative_path
            ]

            for _, row in (
                file_flags.iterrows()
            ):

                start_time = row["start"]

                end_time = row["end"]

                remove_mask = (

                    (df["Datetime"]
                     >= start_time)
                    &
                    (df["Datetime"]
                     <= end_time)

                )

                removed = int(
                    remove_mask.sum()
                )

                rows_removed += (
                    removed
                )

                df = df.loc[
                    ~remove_mask
                ]

        # =============================================
        # REMOVE MISSING TARGETS
        # =============================================

        before = len(df)

        df = df.dropna(
            subset=[TARGET_COL]
        )

        target_removed = (
            before
            -
            len(df)
        )

        # =============================================
        # FILL CLOUD BASES
        # =============================================

        for col in CLOUD_BASE_COLS:

            if col in df.columns:

                df[col] = (

                    df[col]
                    .fillna(
                        NO_CLOUD_HEIGHT
                    )

                )

        # =============================================
        # FILL OCTAS
        # =============================================

        for col in OCTA_COLS:

            if col in df.columns:

                df[col] = (
                    df[col]
                    .fillna(0)
                )

        # =============================================
        # FILL VV
        # =============================================

        if VV_COL in df.columns:

            df[VV_COL] = (

                df[VV_COL]
                .fillna(
                    NO_CLOUD_HEIGHT
                )

            )

        # =============================================
        # FILL RAIN
        # =============================================

        for col in RAIN_COLS:

            if col in df.columns:

                df[col] = (
                    df[col]
                    .fillna(0)
                )

        # =============================================
        # UNIFORMITY CHECKS
        # =============================================

        duplicates = int(

            df["Datetime"]
            .duplicated()
            .sum()

        )

        is_sorted = (

            df["Datetime"]
            .is_monotonic_increasing

        )

        diff = (

            df["Datetime"]
            .diff()
            .dt.total_seconds()

        )

        non15_intervals = int(

            diff[
                ~diff.isin(
                    [15, np.nan]
                )
            ]
            .count()

        )

        remaining_nans = int(

            df.isna()
            .sum()
            .sum()

        )

        # =============================================
        # REPORTS
        # =============================================

        dataset_summary.append({

            "File":
                file_name,

            "Rows_Before":
                rows_before,

            "Wind_Gap_Rows_Removed":
                rows_removed,

            "Target_Rows_Removed":
                target_removed,

            "Rows_Final":
                len(df),

            "Remaining_NaNs":
                remaining_nans
        })

        uniformity_report.append({

            "File":
                file_name,

            "Duplicate_Datetime":
                duplicates,

            "Sorted":
                is_sorted,

            "Non15Sec_Intervals":
                non15_intervals,

            "Remaining_NaNs":
                remaining_nans
        })

        # =============================================
        # SAVE
        # =============================================

        out_file = os.path.join(
            OUTPUT_DIR,
            relative_path
        )

        os.makedirs(
            os.path.dirname(
                out_file
            ),
            exist_ok=True
        )

        df["Datetime"] = (
            df["Datetime"]
            .dt.strftime(
                "%d-%m-%Y %H:%M:%S"
            )
        )

        df.to_csv(
            out_file,
            index=False
        )

        # =============================================
        # PROGRESS
        # =============================================

        print(
            "Wind Rows Removed :",
            rows_removed
        )

        print(
            "Target Removed    :",
            target_removed
        )

        print(
            "Remaining NaNs    :",
            remaining_nans
        )

        print(
            "Bad Intervals     :",
            non15_intervals
        )

        print(
            "Saved Successfully"
        )

    except Exception as e:

        failed_files.append({

            "File":
                file_name,

            "Path":
                file_path,

            "Error":
                str(e)
        })

        print(
            "FAILED:",
            file_name
        )

        print(
            "ERROR:",
            str(e)
        )

# =====================================================
# SAVE REPORTS
# =====================================================

pd.DataFrame(
    dataset_summary
).to_csv(

    os.path.join(
        OUTPUT_DIR,
        "dataset_summary.csv"
    ),

    index=False
)

pd.DataFrame(
    uniformity_report
).to_csv(

    os.path.join(
        OUTPUT_DIR,
        "uniformity_report.csv"
    ),

    index=False
)

if len(failed_files) > 0:

    pd.DataFrame(
        failed_files
    ).to_csv(

        os.path.join(
            OUTPUT_DIR,
            "failed_files.csv"
        ),

        index=False
    )

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 80)
print("STAGE 4 COMPLETE")
print("=" * 80)

print(
    "Files Processed :",
    len(all_files)
)

print(
    "Files Failed :",
    len(failed_files)
)

print("\nOutput Folder:")
print(OUTPUT_DIR)
print("=" * 80)