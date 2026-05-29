"""
08_interpretability_analysis.py
-------------------------------
IE 423 — Term Project
Script 8: Research-quality interpretability visualizations.

Generates core storytelling figures in visuals/figures/ and archives
supplementary plots in visuals/figures/archive/.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import DIMENSIONS
from src.data import load_processed_data, load_split_indices
from src.features import load_tfidf_artifacts
from src.figure_catalog import organize_figures
from src.interpretability import (
    build_coefficient_heatmap_matrix,
    build_top_features_summary,
    extract_lr_coefficients,
    get_best_models_per_dimension,
    load_model_results,
)
from src.metrics import get_score_vector
from src.models import load_model
from src.paths import FIGURES_ARCHIVE_DIR, FIGURES_DIR, TABLES_DIR, ensure_dirs
from src.plots import (
    plot_best_model_overview,
    plot_class_imbalance_overview,
    plot_dimension_predictability_ranking,
    plot_feature_heatmap,
    plot_roc_summary_grid,
    plot_top_lr_features,
)

ensure_dirs()

print("Loading data and model results...")
df = load_processed_data()
results_df = load_model_results()
_, test_idx = load_split_indices()
_, X_test, vectorizer = load_tfidf_artifacts()
feature_names = vectorizer.get_feature_names_out().tolist()
best_models = get_best_models_per_dimension(results_df)

# --- Part 1: Dimension predictability ranking ---
print("\n[Part 1] Dimension predictability ranking...")
path = plot_dimension_predictability_ranking(best_models)
print(f"  Saved: {path}")

# --- Part 2: Top LR features (core: T/F; supplementary: other dims → archive) ---
print("\n[Part 2] Top feature importance (Logistic Regression)...")
for dimension in DIMENSIONS:
    coefficients = extract_lr_coefficients(dimension)
    output_dir = FIGURES_DIR if dimension == "dim_TF" else FIGURES_ARCHIVE_DIR
    path = plot_top_lr_features(feature_names, coefficients, dimension, output_dir=output_dir)
    label = "core" if dimension == "dim_TF" else "archive"
    print(f"  Saved ({label}): {path}")

top_features_df = build_top_features_summary(feature_names)
top_features_path = os.path.join(TABLES_DIR, "top_features_summary.csv")
top_features_df.to_csv(top_features_path, index=False)
print(f"  Saved: {top_features_path}")

# --- Part 3: ROC summary grid ---
print("\n[Part 3] ROC summary grid...")
roc_data = []
for _, row in best_models.iterrows():
    dimension = row["dimension"]
    model_name = row["model"]
    model = load_model(dimension, model_name)
    y_test = df.iloc[test_idx][dimension].values
    y_score = get_score_vector(model, X_test)
    roc_data.append({
        "dimension": dimension,
        "dim_label": row["dim_label"],
        "model_name": model_name,
        "y_true": y_test,
        "y_score": y_score,
    })

dim_order = {d: i for i, d in enumerate(DIMENSIONS)}
roc_data.sort(key=lambda x: dim_order[x["dimension"]])
path = plot_roc_summary_grid(roc_data)
print(f"  Saved: {path}")

# --- Part 4: Class imbalance ---
print("\n[Part 4] Class imbalance overview...")
path = plot_class_imbalance_overview(df)
print(f"  Saved: {path}")

# --- Part 5: Best model summary (preferred over macro_f1_comparison) ---
print("\n[Part 5] Best model summary...")
best_summary = best_models[[
    "dimension", "dim_label", "model", "macro_f1", "accuracy", "roc_auc"
]].copy()
best_summary["model"] = best_summary["model"].str.replace("_", " ").str.title()
best_summary.columns = [
    "dimension", "dim_label", "best_model", "macro_f1", "accuracy", "roc_auc"
]
best_summary_path = os.path.join(TABLES_DIR, "best_models_summary.csv")
best_summary.to_csv(best_summary_path, index=False)
print(f"  Saved: {best_summary_path}")

path = plot_best_model_overview(best_models)
print(f"  Saved: {path}")

# --- Part 6: Feature heatmap ---
print("\n[Part 6] Cross-dimension feature heatmap...")
coef_matrix, _ = build_coefficient_heatmap_matrix(feature_names)
path = plot_feature_heatmap(coef_matrix)
print(f"  Saved: {path}")

# --- Part 7: Archive any remaining non-core figures ---
print("\n[Part 7] Organizing figures (core vs archive)...")
organize_figures(verbose=True)

print(f"\nDone. Core figures in {FIGURES_DIR}; archive in {FIGURES_ARCHIVE_DIR}.")
