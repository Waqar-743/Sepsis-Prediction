"""
evaluate.py
=============
Comprehensive evaluation: confusion matrix, classification report,
AUROC, AUPRC, calibration curves, and threshold analysis.
All plots are saved to the outputs/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve


def _save_confusion_matrix(
    y_true, y_pred, model_name: str, output_dir: str = "outputs"
):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Sepsis", "Sepsis"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {path}")
    return path


def _save_calibration_curve(
    y_true,
    prob_dict: dict,
    output_dir: str = "outputs",
):
    """
    Plot calibration curves for one or more models on the same figure.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    prob_dict : dict[str, np.ndarray]
        Mapping from model name → predicted probabilities.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    for name, probs in prob_dict.items():
        fraction_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, fraction_pos, marker="o", label=name)

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(output_dir, "calibration_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved calibration curve -> {path}")
    return path


def threshold_analysis(y_true, y_prob, model_name: str):
    """
    Print precision, recall, F1 at several clinical thresholds.
    """
    thresholds = [0.3, 0.4, 0.5, 0.6]
    print(f"\n  Threshold Analysis — {model_name}")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*44}")
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        print(f"  {t:>10.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")
    print()


def evaluate_model(
    y_true,
    y_prob,
    model_name: str,
    output_dir: str = "outputs",
    threshold: float = 0.5,
):
    """
    Full evaluation for a single model.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    model_name : str
        Human-readable model name for printing and filenames.
    output_dir : str
        Directory to save plots.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict with auroc, auprc, and paths to saved plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    y_pred = (y_prob >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    print(f"\n{'='*55}")
    print(f"  Evaluation — {model_name}")
    print(f"{'='*55}")
    print(f"  AUROC  : {auroc:.4f}")
    print(f"  AUPRC  : {auprc:.4f}")
    print(f"\n  Classification Report (threshold={threshold}):")
    print(classification_report(y_true, y_pred, target_names=["No Sepsis", "Sepsis"]))

    cm_path = _save_confusion_matrix(y_true, y_pred, model_name, output_dir)
    threshold_analysis(y_true, y_prob, model_name)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix_path": cm_path,
    }


def evaluate_all(
    y_true,
    prob_dict: dict,
    output_dir: str = "outputs",
    threshold: float = 0.5,
):
    """
    Evaluate multiple models and produce a combined calibration curve.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    prob_dict : dict[str, np.ndarray]
        Mapping model_name → predicted probabilities.
    output_dir : str
        Where to save plots.
    threshold : float
        Default classification threshold.

    Returns
    -------
    dict[str, dict] — per-model results.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, probs in prob_dict.items():
        results[name] = evaluate_model(y_true, probs, name, output_dir, threshold)

    # Combined calibration curve
    cal_path = _save_calibration_curve(y_true, prob_dict, output_dir)
    for name in results:
        results[name]["calibration_curve_path"] = cal_path

    return results
