"""
calibration.py
===============
Calibrate model probability outputs using isotonic regression via
scikit-learn's CalibratedClassifierCV.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(model, X_train, y_train, method: str = "isotonic", cv: int = 5):
    """
    Wrap a trained model with CalibratedClassifierCV and re-fit on training data
    to produce well-calibrated probability estimates.

    Parameters
    ----------
    model : estimator
        A trained classifier that implements predict_proba.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    method : str
        Calibration method ('isotonic' or 'sigmoid').
    cv : int
        Number of cross-validation folds for calibration.

    Returns
    -------
    calibrated : CalibratedClassifierCV
        Calibrated classifier.
    """
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def get_calibrated_predictions(calibrated_model, X: np.ndarray) -> np.ndarray:
    """
    Return calibrated probability predictions for the positive class.

    Parameters
    ----------
    calibrated_model : CalibratedClassifierCV
        A fitted calibrated classifier.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        Calibrated probabilities for class 1.
    """
    return calibrated_model.predict_proba(X)[:, 1]
