"""
model.py
=========
Train an XGBoost baseline and a LightGBM advanced model for sepsis
prediction.  Both handle class imbalance via scale_pos_weight and use
early stopping on a held-out validation portion of the training set.
"""

import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """Count(negatives) / Count(positives)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    output_dir: str = "outputs",
):
    """
    Train an XGBoost classifier with early stopping.

    Returns
    -------
    model : xgb.XGBClassifier
        Trained XGBoost model.
    """
    print("[Model] Training XGBoost baseline ...")

    spw = _compute_scale_pos_weight(y_train)
    print(f"  scale_pos_weight = {spw:.2f}")

    # Hold out 15% of training data for early stopping validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=random_state,
        stratify=y_train,
    )

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=random_state,
        n_jobs=-1,
        use_label_encoder=False,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # Save model
    path = os.path.join(output_dir, "xgboost_model.joblib")
    joblib.dump(model, path)
    print(f"  Saved to {path}\n")

    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    output_dir: str = "outputs",
):
    """
    Train a LightGBM classifier with early stopping.

    Returns
    -------
    model : lgb.LGBMClassifier
        Trained LightGBM model.
    """
    print("[Model] Training LightGBM advanced model ...")

    spw = _compute_scale_pos_weight(y_train)
    print(f"  scale_pos_weight = {spw:.2f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=random_state,
        stratify=y_train,
    )

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        metric="average_precision",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = model.best_iteration_
    print(f"  Best iteration: {best_iter}")

    path = os.path.join(output_dir, "lightgbm_model.joblib")
    joblib.dump(model, path)
    print(f"  Saved to {path}\n")

    return model
