"""Shared evaluation helpers.

Both train_models.py (which produces the README table) and app.py (which shows
live results) import from here, so the numbers in the report and in the app are
computed by exactly the same code.

Averaging choice: the target is imbalanced (Graduate >> Dropout > Enrolled), so
Precision / Recall / F1 are MACRO-averaged (every class counts equally) and AUC
is one-vs-rest macro. Weighted averaging would track accuracy and hide how badly
the small "Enrolled" class is handled.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_auc(y_true, y_proba, n_classes):
    """AUC for binary or multi-class (OvR, macro). Returns NaN if undefined,
    e.g. when an uploaded test file does not contain every class."""
    try:
        if n_classes == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro",
            labels=list(range(n_classes)),
        )
    except ValueError:
        return float("nan")


def evaluate_model(model, X, y_true, n_classes):
    """Predict with an ALREADY-FITTED model and return (metrics dict, y_pred)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": compute_auc(y_true, y_proba, n_classes),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred


def confusion(y_true, y_pred, n_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def report_dict(y_true, y_pred, class_names):
    return classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
