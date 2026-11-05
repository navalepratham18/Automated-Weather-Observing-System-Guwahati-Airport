"""Guwahati airport physics-based cloud correction pipeline.

Runs after 08_merge_lstm_ready_dataset.py to recover "Clear Sky" 
observations masked by NaNs, without statistical hallucination.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths based on the Guwahati pipeline structure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "08_unified_dataset" / "guwahati_unified_lstm_dataset_v2_bounded_cloud_fill_20240811_20260131.csv"
OUTPUT_FILE = PROJECT_ROOT / "08_unified_dataset" / "guwahati_unified_lstm_dataset_v3_physics_corrected.csv"

# Configuration
MAX_CEILING_FT = 15000  # Semantic fill value for clear skies[cite: 22]

def apply_physics_rules():
    print(f"Loading {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Stage 4 forward-filled up to 3 gaps and marked them as IMPUTED=1[cite: 22].
    # We target rows where Cloud_Base_1 is genuinely missing (NaN or previously imputed).
    missing_cloud_mask = df["CLOUD_Cloud_Base_1"].isna() | (df["CLOUD_Cloud_Base_1_IMPUTED"] == 1)
    
    # ---------------------------------------------------------
    # RULE 1: Clear Sky Recovery (The primary fix)
    # If the cloud sensor is actively powered/transmitting (ON) 
    # AND there is no rain falling (Rain == 0), the missing cloud 
    # base is a valid Clear Sky (NCD).
    # ---------------------------------------------------------
    clear_sky_mask = (
        missing_cloud_mask & 
        (df["CLOUD_SENSOR_ON"] == 1) & 
        (df["PW_Rain_Intensity"] == 0) & 
        (df["CLOUD_OCTA1"] == 0)
    )
    
    # Encode physical reality instead of NaN
    df.loc[clear_sky_mask, "CLOUD_Cloud_Base_1"] = MAX_CEILING_FT
    
    # Unmask the loss function: This is real data now
    df.loc[clear_sky_mask, "CLOUD_Cloud_Base_1_IMPUTED"] = 0
    df.loc[clear_sky_mask, "CLOUD_LONG_GAP"] = 0

    # ---------------------------------------------------------
    # RULE 2: The Visibility Contradiction (Failsafe)
    # If visibility is severely restricted (e.g., dense fog < 2000m)
    # but the sensor reports no clouds/VV, the sensor lens is likely
    # obscured or failing. Force this back to a masked state.
    # ---------------------------------------------------------
    fog_mask = (df["VIS_Visibility_MOR"] < 2000) & (df["CLOUD_SENSOR_ON"] == 1)
    
    # Re-mask Cloud Base
    df.loc[clear_sky_mask & fog_mask, "CLOUD_Cloud_Base_1"] = np.nan
    df.loc[clear_sky_mask & fog_mask, "CLOUD_Cloud_Base_1_IMPUTED"] = 1
    
    # Re-mask Vertical Visibility
    missing_vv_mask = df["CLOUD_Vertical_Visibility"].isna() | (df["CLOUD_Vertical_Visibility_IMPUTED"] == 1)
    df.loc[missing_vv_mask & fog_mask, "CLOUD_Vertical_Visibility"] = np.nan
    df.loc[missing_vv_mask & fog_mask, "CLOUD_Vertical_Visibility_IMPUTED"] = 1

    # ---------------------------------------------------------
    # RULE 3: Dead Sensor Heartbeat
    # If the sensor is fully offline, ensure it remains strictly masked.
    # ---------------------------------------------------------
    dead_sensor_mask = (df["CLOUD_SENSOR_ON"] == 0)
    df.loc[dead_sensor_mask, "CLOUD_Cloud_Base_1"] = np.nan
    df.loc[dead_sensor_mask, "CLOUD_Cloud_Base_1_IMPUTED"] = 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n================ PHYSICS CORRECTION SUMMARY ================")
    print(f"Clear Sky observations recovered : {clear_sky_mask.sum()} rows")
    print(f"Fog contradictions masked        : {(clear_sky_mask & fog_mask).sum()} rows")
    print(f"Dead sensor records retained     : {dead_sensor_mask.sum()} rows")
    print(f"Saved optimized dataset to       : {OUTPUT_FILE.name}")
    print("============================================================\n")

if __name__ == "__main__":
    apply_physics_rules()