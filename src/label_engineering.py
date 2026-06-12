"""
label_engineering.py
=====================
Create the 6-hour-ahead prediction label (``label_6h``) and perform a
patient-level train / test split so no patient appears in both sets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_6h_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each patient:
      - Find T, the first hour where SepsisLabel == 1.
      - Set label_6h = 1 for hours [T-6, T-1]  (the 6-hour prediction window).
      - Set label_6h = 0 for hours before T-6.
      - **Drop** all rows at hour T and after (post-diagnosis data is unusable).
      - For patients who never develop sepsis, label_6h = 0 everywhere.

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined DataFrame with 'patient_id', 'ICULOS', and 'SepsisLabel'.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new 'label_6h' column and post-diagnosis rows removed.
    """
    out_frames = []

    for pid, grp in df.groupby("patient_id"):
        grp = grp.sort_values("ICULOS").copy()

        sepsis_rows = grp[grp["SepsisLabel"] == 1]

        if len(sepsis_rows) == 0:
            # Patient never develops sepsis
            grp["label_6h"] = 0
            out_frames.append(grp)
            continue

        # T = first hour of recorded sepsis
        T_idx = sepsis_rows.index[0]
        T_pos = grp.index.get_loc(T_idx)  # positional index within group

        # Drop rows at T and after (post-diagnosis)
        grp = grp.iloc[:T_pos].copy()

        if len(grp) == 0:
            # Sepsis label was on the very first row — nothing to keep
            continue

        # Assign label_6h: 1 for the last 6 rows (T-6 to T-1), 0 otherwise
        grp["label_6h"] = 0
        window_start = max(0, len(grp) - 6)
        grp.iloc[window_start:, grp.columns.get_loc("label_6h")] = 1

        out_frames.append(grp)

    result = pd.concat(out_frames, ignore_index=True)

    n_pos = int(result["label_6h"].sum())
    n_neg = len(result) - n_pos
    print(f"\n[Label Engineering] label_6h created")
    print(f"  Positive (1) : {n_pos:,}  ({n_pos / len(result) * 100:.2f}%)")
    print(f"  Negative (0) : {n_neg:,}  ({n_neg / len(result) * 100:.2f}%)")
    print(f"  Total rows   : {len(result):,}")
    print(f"  Patients     : {result['patient_id'].nunique():,}\n")

    return result


def patient_level_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the data at the **patient** level so that no patient appears in
    both the training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'patient_id' and 'label_6h' columns.
    test_size : float
        Fraction of patients to put in the test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    df_train, df_test : pd.DataFrame
    """
    patient_ids = np.array(df["patient_id"].unique())

    # Determine per-patient label for stratification (1 if patient has any label_6h=1)
    patient_labels = (
        df.groupby("patient_id")["label_6h"]
        .max()
        .reindex(patient_ids)
        .values
    )

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=patient_labels,
    )

    df_train = df[df["patient_id"].isin(train_ids)].copy()
    df_test = df[df["patient_id"].isin(test_ids)].copy()

    print(f"[Split] Patient-level 80/20 split (random_state={random_state})")
    print(f"  Train patients : {len(train_ids):,}  |  rows : {len(df_train):,}")
    print(f"  Test  patients : {len(test_ids):,}  |  rows : {len(df_test):,}")

    # Verify no overlap
    overlap = set(train_ids) & set(test_ids)
    assert len(overlap) == 0, f"Data leakage! {len(overlap)} patients in both sets."
    print("  [OK] No patient overlap between train and test.\n")

    return df_train, df_test
