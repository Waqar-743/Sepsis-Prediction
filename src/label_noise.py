"""
label_noise.py
===============
Detect and remove likely mislabelled samples in the training set using
cleanlab's CleanLearning with a RandomForestClassifier as the base model.
"""

import numpy as np
import pandas as pd
from cleanlab.classification import CleanLearning
from sklearn.ensemble import RandomForestClassifier


def detect_and_remove_noise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
):
    """
    Use cleanlab to find label issues in the training data and return a
    boolean mask of clean (non-noisy) samples.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels (0 / 1).
    random_state : int
        Random seed for the base classifier.

    Returns
    -------
    clean_mask : np.ndarray[bool]
        True for samples that are NOT flagged as noisy.
    n_removed : int
        Number of samples identified as noisy.
    """
    print("[Label Noise] Running cleanlab noise detection ...")

    base_clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )

    cl = CleanLearning(clf=base_clf, seed=random_state)

    # find_label_issues returns a DataFrame with a boolean 'is_label_issue' column
    label_issues = cl.find_label_issues(X=X_train, labels=y_train)

    noisy_mask = label_issues["is_label_issue"].values
    clean_mask = ~noisy_mask
    n_removed = int(noisy_mask.sum())
    pct = n_removed / len(y_train) * 100

    print(f"[Label Noise] Detected {n_removed:,} noisy samples ({pct:.2f}%)")
    print(f"[Label Noise] Keeping {clean_mask.sum():,} clean samples.\n")

    return clean_mask, n_removed


def clean_training_data(
    df_train: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label_6h",
    random_state: int = 42,
):
    """
    Convenience wrapper: detect noisy labels and return a filtered DataFrame.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training DataFrame (must include feature_cols and label_col).
    feature_cols : list
        Column names used as features.
    label_col : str
        Name of the label column.
    random_state : int
        Random seed.

    Returns
    -------
    df_clean : pd.DataFrame
        Training DataFrame with noisy samples removed.
    n_removed : int
        Number of removed samples.
    """
    X = df_train[feature_cols].values
    y = df_train[label_col].values.astype(int)

    clean_mask, n_removed = detect_and_remove_noise(X, y, random_state)

    df_clean = df_train[clean_mask].copy().reset_index(drop=True)
    return df_clean, n_removed
