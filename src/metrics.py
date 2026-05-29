"""Evaluation metrics for binary MBTI dimension classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_score_vector(model, X):
    """
    Return a 1-D score vector suitable for ROC-AUC computation.

    Uses predict_proba for models that support it, otherwise decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Compute all evaluation metrics for a binary classification task."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def metrics_to_row(dimension: str, dim_label: str, model_name: str, metrics: dict) -> dict:
    """Flatten metrics dict into a CSV-friendly row."""
    return {
        "dimension": dimension,
        "dim_label": dim_label,
        "model": model_name,
        "macro_f1": round(metrics["macro_f1"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
    }
