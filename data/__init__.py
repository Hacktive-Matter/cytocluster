from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re

metadata_columns = [
    "stab_conc_uM",
    "stab_name",
    "mt_conc_uM",
    "actin_conc_uM",
    "channel",
    "trial",
    "time_PT",
    "spatial_x",
    "spatial_y",
]

exclude_columns = [
    "Filename",
    "Channel",
    "Flags",
] + metadata_columns


def get_data(channel: Optional[int] = None) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv("data/BARCODE_Higher_filament_930am.csv")

    metadata_df = pd.DataFrame(
        df["Filename"].apply(extract_metadata_from_filename).tolist()
    )
    results_df = pd.concat([df, metadata_df], axis=1)

    if channel is not None:
        results_df = results_df[results_df["channel"] == channel]

    feature_columns = [col for col in results_df.columns if col not in exclude_columns]

    # Extract features and metadata separately
    features_df = results_df[feature_columns].copy()

    metadata_only_df = results_df[metadata_columns].copy()

    # Remove any rows where metadata extraction failed (all NaN metadata)
    valid_rows = ~metadata_only_df.isna().all(axis=1)
    features_df = features_df[valid_rows]
    metadata_only_df = metadata_only_df[valid_rows]

    # Scale only the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    results_df[feature_columns] = X_scaled

    return results_df, X_scaled


def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extracts metadata from a filename and returns it as a dictionary.

    Args:
        filename (str): The original filename string.

    Returns:
        dict: Dictionary containing extracted metadata, or empty dict if no match.
    """
    # Regex for the first format (with MT and Actin concentrations)
    pattern1 = re.compile(
        r"(\d+\.?\d*)uM_([A-Z]+)_(\d+\.?\d*)uM-MT_(\d+\.?\d*)uM-A_C=(\d+)_trial-(\d+)_PT-(\d+)_tile_(\d+)_(\d+)\.tif"
    )

    # Regex for the second format (without MT and Actin concentrations)
    pattern2 = re.compile(
        r"(\d+\.?\d*)mM-([A-Z]+)_trial-(\d+)_C=(\d+)_PT-(\d+)_tile_(\d+)_(\d+)\.tif"
    )

    match1 = pattern1.match(filename)
    match2 = pattern2.match(filename)

    if match1:
        # Extract parts from the first format
        parts = match1.groups()
        stab_conc, stab_name, mt_conc, actin_conc, channel, trial, time, x, y = parts

        return {
            "stab_conc_uM": float(stab_conc),
            "stab_name": stab_name,
            "mt_conc_uM": float(mt_conc),
            "actin_conc_uM": float(actin_conc),
            "channel": int(channel),
            "trial": int(trial),
            "time_PT": int(time),
            "spatial_x": int(x),
            "spatial_y": int(y),
        }

    elif match2:
        # Extract parts from the second format
        parts = match2.groups()
        stab_conc_mM, stab_name, trial, channel, time, x, y = parts

        # Convert stabilizer concentration from mM to uM
        stab_conc_uM = float(stab_conc_mM) * 1000
        stab_conc_uM = int(stab_conc_uM) if stab_conc_uM.is_integer() else stab_conc_uM

        return {
            "stab_conc_uM": stab_conc_uM,
            "stab_name": stab_name,
            "mt_conc_uM": 3.19,  # Default value
            "actin_conc_uM": 2.62,  # Default value
            "channel": int(channel),
            "trial": int(trial),
            "time_PT": int(time),
            "spatial_x": int(x),
            "spatial_y": int(y),
        }
    else:
        # Return empty dict if no match (you could also return None)
        return {}
