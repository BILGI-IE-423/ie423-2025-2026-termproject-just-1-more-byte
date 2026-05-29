"""Interpretability analysis helpers for MBTI dimension classification."""

import os

import numpy as np
import pandas as pd

from src.config import DIM_FEATURE_TITLES, DIM_LABELS, DIMENSIONS
from src.models import load_model
from src.paths import TABLES_DIR


def load_model_results() -> pd.DataFrame:
    """Load the full model results summary table."""
    path = os.path.join(TABLES_DIR, "model_results_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results not found at {path}. Run scripts/evaluation/07_evaluate_models.py first."
        )
    return pd.read_csv(path)


def get_best_models_per_dimension(results_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the best-performing model for each MBTI dimension by macro F1."""
    if results_df is None:
        results_df = load_model_results()
    return (
        results_df.sort_values("macro_f1", ascending=False)
        .groupby("dimension", as_index=False)
        .first()
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )


def extract_lr_coefficients(dimension: str) -> np.ndarray:
    """Extract logistic regression coefficients for a dimension."""
    model = load_model(dimension, "logistic_regression")
    return model.coef_.flatten()


def get_top_lr_features(
    feature_names: list[str],
    coefficients: np.ndarray,
    top_n: int = 15,
) -> tuple[list[str], list[float], list[str]]:
    """
    Return top positive and negative LR features.

    Positive coefficients favor class 1; negative coefficients favor class 0.
    """
    pos_idx = np.argsort(coefficients)[-top_n:][::-1]
    neg_idx = np.argsort(coefficients)[:top_n]

    features = [feature_names[i] for i in pos_idx] + [feature_names[i] for i in neg_idx]
    values = [coefficients[i] for i in pos_idx] + [coefficients[i] for i in neg_idx]
    directions = ["positive"] * top_n + ["negative"] * top_n
    return features, values, directions


def build_top_features_summary(
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Build a CSV-friendly summary of top LR features across all dimensions."""
    rows = []
    for dimension in DIMENSIONS:
        coefficients = extract_lr_coefficients(dimension)
        pos_idx = np.argsort(coefficients)[-top_n:][::-1]
        neg_idx = np.argsort(coefficients)[:top_n]
        pos_features = [feature_names[i] for i in pos_idx]
        pos_values = [coefficients[i] for i in pos_idx]
        neg_features = [feature_names[i] for i in neg_idx]
        neg_values = [coefficients[i] for i in neg_idx]

        for rank, (feature, value) in enumerate(zip(pos_features, pos_values), start=1):
            rows.append({
                "dimension": dimension,
                "dim_label": DIM_LABELS[dimension],
                "comparison": DIM_FEATURE_TITLES[dimension],
                "feature": feature,
                "coefficient": round(value, 6),
                "direction": "positive",
                "rank": rank,
            })

        for rank, (feature, value) in enumerate(zip(neg_features, neg_values), start=1):
            rows.append({
                "dimension": dimension,
                "dim_label": DIM_LABELS[dimension],
                "comparison": DIM_FEATURE_TITLES[dimension],
                "feature": feature,
                "coefficient": round(value, 6),
                "direction": "negative",
                "rank": rank,
            })

    return pd.DataFrame(rows)


def build_coefficient_heatmap_matrix(
    feature_names: list[str],
    top_word_count: int = 25,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a normalized coefficient matrix for cross-dimension feature heatmap.

    Rows: important terms; columns: dimensions; values: column-normalized coefficients.
    """
    coef_by_dim = {dim: extract_lr_coefficients(dim) for dim in DIMENSIONS}

    # Select words with highest absolute coefficient in any dimension
    max_abs = np.zeros(len(feature_names))
    for coef in coef_by_dim.values():
        max_abs = np.maximum(max_abs, np.abs(coef))

    top_indices = np.argsort(max_abs)[-top_word_count:][::-1]
    top_words = [feature_names[i] for i in top_indices]

    matrix = {}
    for dim in DIMENSIONS:
        values = coef_by_dim[dim][top_indices]
        max_val = np.max(np.abs(values)) or 1.0
        matrix[DIM_LABELS[dim]] = values / max_val

    return pd.DataFrame(matrix, index=top_words), top_words
