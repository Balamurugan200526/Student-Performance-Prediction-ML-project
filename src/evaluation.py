"""
evaluation.py
--------------
Provides detailed evaluation of a trained classifier: accuracy,
precision, recall, F1-score, confusion matrix, classification report,
and (where applicable) multi-class ROC curves.

Each metric is explained in the accompanying README/report in simple
language; this module focuses on computing and visualizing them.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_test, y_pred) -> dict:
    """
    accuracy   -> overall % of correct predictions
    precision  -> of predicted "Pass"/"Fail"/etc, how many were correct
    recall     -> of actual "Pass"/"Fail"/etc, how many were found
    f1_score   -> harmonic mean of precision & recall (balances both)
    (all averaged as 'weighted' to account for class imbalance)
    """
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    for k, v in metrics.items():
        print(f"[evaluation] {k}: {v:.4f}")
    return metrics


def plot_confusion_matrix(y_test, y_pred, class_names, out_dir):
    """
    Confusion matrix: rows = actual class, columns = predicted class.
    The diagonal shows correct predictions; off-diagonal cells show
    exactly which classes get confused with each other (e.g., how many
    'Average' students were mis-predicted as 'Pass').
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[evaluation] Saved: {path}")
    return cm


def print_classification_report(y_test, y_pred, class_names):
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print("[evaluation] Classification Report:\n", report)
    return report


def plot_roc_curves(model, X_test, y_test, class_names, out_dir):
    """
    ROC (Receiver Operating Characteristic) curve: plots True Positive
    Rate vs False Positive Rate at different classification thresholds
    for EACH class (one-vs-rest), since this is a multi-class problem.
    AUC (Area Under Curve) close to 1.0 = excellent separability;
    0.5 = no better than random guessing.
    """
    if not hasattr(model, "predict_proba"):
        print("[evaluation] Model has no predict_proba(); skipping ROC curve.")
        return None

    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    y_score = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "roc_curve.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[evaluation] Saved: {path}")
    return path


def full_evaluation(model, X_test, y_test, class_names, out_dir):
    """Run every evaluation step and return a metrics dict."""
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, class_names, out_dir)
    print_classification_report(y_test, y_pred, class_names)
    plot_roc_curves(model, X_test, y_test, class_names, out_dir)
    return metrics
