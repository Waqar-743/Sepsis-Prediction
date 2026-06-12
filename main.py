"""
main.py
========
End-to-end pipeline for Early Sepsis Onset Prediction.
Runs all modules in sequence and saves results to outputs/.

Usage:
    python main.py
"""

import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.load_data import load_all_patients
from src.label_engineering import create_6h_labels, patient_level_split
from src.features import engineer_features
from src.preprocessing import preprocess
from src.label_noise import clean_training_data
from src.model import train_xgboost, train_lightgbm
from src.calibration import calibrate_model, get_calibrated_predictions
from src.evaluate import evaluate_all

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
DATA_DIR = "data/training_setA"


def main():
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.random.seed(RANDOM_STATE)

    # ==================================================================
    # STEP 1: Load data
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 1 / 8 — Loading data")
    print("=" * 60)
    df = load_all_patients(DATA_DIR)

    # ==================================================================
    # STEP 2: Label engineering (6-hour-ahead labels)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 2 / 8 — Label engineering")
    print("=" * 60)
    df = create_6h_labels(df)

    # ==================================================================
    # STEP 3: Patient-level train / test split
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 3 / 8 — Patient-level train/test split")
    print("=" * 60)
    df_train, df_test = patient_level_split(df, test_size=0.2, random_state=RANDOM_STATE)

    # ==================================================================
    # STEP 4: Feature engineering
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 4 / 8 — Feature engineering")
    print("=" * 60)
    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    # ==================================================================
    # STEP 5: Preprocessing (imputation, normalization)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 5 / 8 — Preprocessing")
    print("=" * 60)
    df_train, df_test, scaler, feature_cols = preprocess(df_train, df_test, label_col="label_6h")

    # ==================================================================
    # STEP 6: Label noise detection and removal (training set only)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 6 / 8 — Label noise detection (cleanlab)")
    print("=" * 60)
    df_train_clean, n_removed = clean_training_data(
        df_train, feature_cols, label_col="label_6h", random_state=RANDOM_STATE
    )

    # Prepare arrays
    X_train = df_train_clean[feature_cols].values
    y_train = df_train_clean["label_6h"].values.astype(int)
    X_test = df_test[feature_cols].values
    y_test = df_test["label_6h"].values.astype(int)

    print(f"  Final training set : {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Test set           : {X_test.shape[0]:,} samples")

    # ==================================================================
    # STEP 7: Model training + calibration
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 7 / 8 — Model training & calibration")
    print("=" * 60)

    # --- XGBoost ---
    xgb_model = train_xgboost(X_train, y_train, random_state=RANDOM_STATE, output_dir=OUTPUT_DIR)
    print("[Calibration] Calibrating XGBoost ...")
    xgb_model.set_params(early_stopping_rounds=None)
    xgb_calibrated = calibrate_model(xgb_model, X_train, y_train)
    xgb_probs = get_calibrated_predictions(xgb_calibrated, X_test)

    # --- LightGBM ---
    lgbm_model = train_lightgbm(X_train, y_train, random_state=RANDOM_STATE, output_dir=OUTPUT_DIR)
    print("[Calibration] Calibrating LightGBM ...")
    lgbm_calibrated = calibrate_model(lgbm_model, X_train, y_train)
    lgbm_probs = get_calibrated_predictions(lgbm_calibrated, X_test)

    # ==================================================================
    # STEP 8: Evaluation
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 8 / 8 — Evaluation")
    print("=" * 60)

    prob_dict = {
        "XGBoost": xgb_probs,
        "LightGBM": lgbm_probs,
    }

    results = evaluate_all(y_test, prob_dict, output_dir=OUTPUT_DIR)

    # ==================================================================
    # Save summary
    # ==================================================================
    elapsed = time.time() - start
    summary_path = os.path.join(OUTPUT_DIR, "results.txt")
    with open(summary_path, "w") as f:
        f.write("Early Sepsis Onset Prediction — Results Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Random state          : {RANDOM_STATE}\n")
        f.write(f"Training samples      : {X_train.shape[0]:,}\n")
        f.write(f"Test samples          : {X_test.shape[0]:,}\n")
        f.write(f"Features              : {X_train.shape[1]}\n")
        f.write(f"Noisy labels removed  : {n_removed:,}\n\n")
        for name, res in results.items():
            f.write(f"--- {name} ---\n")
            f.write(f"  AUROC : {res['auroc']:.4f}\n")
            f.write(f"  AUPRC : {res['auprc']:.4f}\n\n")
        f.write(f"Total runtime: {elapsed:.1f}s\n")

    print(f"\n{'='*55}")
    print(f"  Pipeline complete!  ({elapsed:.1f}s)")
    print(f"  Results saved to {summary_path}")
    print(f"  Plots  saved to {OUTPUT_DIR}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
