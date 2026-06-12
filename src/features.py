"""
features.py
============
Temporal feature engineering: rolling statistics, time-since-last-observation,
and ICU length-of-stay features.

CRITICAL: all rolling windows look **backwards only** — we shift by 1 before
computing the rolling aggregate to ensure no data leakage from the current hour.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# Vital / lab columns for which we compute rolling features
ROLLING_COLS = ["HR", "O2Sat", "Temp", "SBP", "Resp", "Glucose", "Lactate"]

# Sparse lab columns for time-since-last-observation
SPARSE_LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

WINDOW = 6  # 6-hour rolling window


def _rolling_features_for_group(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute backward-looking rolling features for a single patient.
    """
    grp = grp.sort_values("ICULOS").copy()

    for col in ROLLING_COLS:
        if col not in grp.columns:
            continue
        # Shift by 1 so current hour is excluded from the window
        shifted = grp[col].shift(1)
        grp[f"{col}_roll_mean"] = shifted.rolling(window=WINDOW, min_periods=1).mean()
        grp[f"{col}_roll_std"] = shifted.rolling(window=WINDOW, min_periods=1).std().fillna(0)
        grp[f"{col}_roll_min"] = shifted.rolling(window=WINDOW, min_periods=1).min()
        grp[f"{col}_roll_max"] = shifted.rolling(window=WINDOW, min_periods=1).max()

    return grp


def _time_since_last_obs(grp: pd.DataFrame) -> pd.DataFrame:
    """
    For each sparse lab column, compute hours since the last non-NaN observation.
    Uses the *original* missingness indicator to determine observation times.
    """
    grp = grp.sort_values("ICULOS").copy()

    for col in SPARSE_LAB_COLS:
        missing_flag = f"{col}_missing"
        feat_name = f"{col}_time_since"

        if missing_flag not in grp.columns:
            # If no missingness flag exists, skip
            continue

        # observed = 1 where the value was actually measured (missing == 0)
        observed = (grp[missing_flag] == 0).astype(float)
        # cumulative count of non-observations since last observation
        # reset counter when a new observation appears
        groups = observed.cumsum()
        time_since = grp.groupby(groups).cumcount()
        # where the value was just observed, time_since = 0
        grp[feat_name] = time_since.values

    return grp


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all temporal feature engineering steps.

    1. Rolling mean, std, min, max for key vitals/labs (window = 6 hours).
    2. Time-since-last-observation for sparse lab values.
    3. Ensure ICULOS is present as a feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with patient_id, ICULOS, and clinical columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered feature columns.
    """
    print("[Features] Computing rolling statistics (window=6, backward-only) ...")
    frames = []
    for pid, grp in tqdm(df.groupby("patient_id"), desc="Rolling features"):
        grp = _rolling_features_for_group(grp)
        frames.append(grp)
    df = pd.concat(frames, ignore_index=True)

    print("[Features] Computing time-since-last-observation for sparse labs ...")
    frames = []
    for pid, grp in tqdm(df.groupby("patient_id"), desc="Time-since features"):
        grp = _time_since_last_obs(grp)
        frames.append(grp)
    df = pd.concat(frames, ignore_index=True)

    # Ensure ICULOS is present
    if "ICULOS" not in df.columns:
        raise ValueError("ICULOS column not found in DataFrame.")

    n_new = sum(
        1 for c in df.columns if c.endswith(("_roll_mean", "_roll_std",
                                              "_roll_min", "_roll_max",
                                              "_time_since"))
    )
    print(f"[Features] Done. Added {n_new} new temporal features.\n")
    return df
