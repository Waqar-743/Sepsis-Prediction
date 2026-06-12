"""
preprocessing.py
=================
Imputation (forward-fill + median), missingness indicators, and
StandardScaler normalisation — all fitted on training data only.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Columns that are static demographics / identifiers (not imputed the same way)
STATIC_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
ID_COLS = ["patient_id"]
LABEL_COLS = ["SepsisLabel", "label_6h"]

# All 34 physiological + lab columns
VITAL_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]
LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
NUMERIC_COLS = VITAL_COLS + LAB_COLS + STATIC_COLS


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every numeric clinical column, add a binary flag ``<col>_missing``
    that equals 1 where the original value was NaN.
    """
    for col in VITAL_COLS + LAB_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(np.int8)
    return df


def forward_fill_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each patient, forward-fill NaN values to simulate real-time
    availability of the most recent measurement.
    """
    cols_to_fill = [c for c in VITAL_COLS + LAB_COLS if c in df.columns]
    df[cols_to_fill] = df.groupby("patient_id")[cols_to_fill].transform(
        lambda s: s.ffill()
    )
    return df


def compute_train_medians(df_train: pd.DataFrame) -> pd.Series:
    """
    Compute column medians on the training set for later median imputation.
    """
    cols = [c for c in NUMERIC_COLS if c in df_train.columns]
    return df_train[cols].median()


def median_fill(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """
    Fill remaining NaN values with pre-computed training-set medians.
    """
    for col in medians.index:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(medians[col])
    return df


def fit_scaler(df_train: pd.DataFrame, feature_cols: list) -> StandardScaler:
    """
    Fit a StandardScaler on training features ONLY.
    """
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler, feature_cols: list
) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to the given DataFrame.
    """
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def preprocess(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    label_col: str = "label_6h",
):
    """
    Full preprocessing pipeline.

    1. Add missingness indicators (before any imputation).
    2. Forward-fill per patient.
    3. Median-fill with training medians.
    4. Normalize numeric features with a scaler fit on train only.

    Returns
    -------
    df_train, df_test : preprocessed DataFrames
    scaler            : fitted StandardScaler
    feature_cols      : list of feature column names to use for modelling
    """
    print("[Preprocessing] Adding missingness indicators ...")
    df_train = add_missingness_indicators(df_train.copy())
    df_test = add_missingness_indicators(df_test.copy())

    print("[Preprocessing] Forward-filling per patient ...")
    df_train = forward_fill_per_patient(df_train)
    df_test = forward_fill_per_patient(df_test)

    print("[Preprocessing] Computing training-set medians & filling ...")
    medians = compute_train_medians(df_train)
    df_train = median_fill(df_train, medians)
    df_test = median_fill(df_test, medians)

    # Determine feature columns (everything except IDs, labels, patient_id)
    exclude = set(ID_COLS + LABEL_COLS + ["SepsisLabel"])
    feature_cols = [
        c for c in df_train.columns
        if c not in exclude and c != "patient_id"
    ]

    # Identify numeric feature columns for scaling
    numeric_features = [
        c for c in feature_cols
        if df_train[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8]
    ]

    print("[Preprocessing] Fitting scaler on training data ...")
    scaler = fit_scaler(df_train, numeric_features)
    df_train = apply_scaler(df_train, scaler, numeric_features)
    df_test = apply_scaler(df_test, scaler, numeric_features)

    # Final NaN safety net — fill any remaining NaN with 0
    df_train[feature_cols] = df_train[feature_cols].fillna(0)
    df_test[feature_cols] = df_test[feature_cols].fillna(0)

    print(f"[Preprocessing] Done. Feature count: {len(feature_cols)}")
    return df_train, df_test, scaler, feature_cols
