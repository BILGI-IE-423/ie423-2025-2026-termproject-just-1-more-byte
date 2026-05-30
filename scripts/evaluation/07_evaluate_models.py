"""
07_evaluate_models.py
---------------------
IE 423 — Term Project
Script 7: Evaluate trained models on the held-out test set.

This script:
- Computes accuracy, precision, recall, macro F1, and ROC-AUC for all models
- Saves results tables
- Saves ONE core confusion matrix (T/F, Logistic Regression)
- Generates model comparison figures from real test-set metrics
- Detailed interpretability plots are handled by stage 08
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import DIM_LABELS, DIMENSIONS
from src.data import load_processed_data, load_split_indices
from src.features import load_tfidf_artifacts
from src.metrics import compute_metrics, get_score_vector, metrics_to_row
from src.models import get_models, load_model
from src.paths import TABLES_DIR, ensure_dirs
from src.plots import (
    plot_confusion_matrix,
    plot_model_comparison_overview,
    plot_svm_vs_logreg_comparison,
)

ensure_dirs()

# Core showcase confusion matrix
CORE_CM_DIMENSION = "dim_TF"
CORE_CM_MODEL = "logistic_regression"

# --- 1. Load data and features ---
print("Loading data, features, and models...")
df = load_processed_data()
_, test_idx = load_split_indices()
X_train, X_test, vectorizer = load_tfidf_artifacts()

models = get_models()
result_rows = []

# --- 2. Evaluate each model × dimension (metrics only) ---
for dimension in DIMENSIONS:
    dim_label = DIM_LABELS[dimension]
    y_test = df.iloc[test_idx][dimension].values

    print(f"\n--- Evaluating {dim_label} ({dimension}) ---")

    for model_name in models:
        model = load_model(dimension, model_name)
        y_pred = model.predict(X_test)
        y_score = get_score_vector(model, X_test)

        metrics = compute_metrics(y_test, y_pred, y_score)
        row = metrics_to_row(dimension, dim_label, model_name, metrics)
        result_rows.append(row)

        print(
            f"  {model_name:25s}  macro_f1={row['macro_f1']:.4f}  "
            f"acc={row['accuracy']:.4f}  roc_auc={row['roc_auc']:.4f}"
        )

# --- 3. Save results tables ---
results_df = pd.DataFrame(result_rows).sort_values(
    ["dimension", "macro_f1"], ascending=[True, False]
)

summary_path = os.path.join(TABLES_DIR, "model_results_summary.csv")
results_df.to_csv(summary_path, index=False)
print(f"\n[OK] Model results summary saved to: {summary_path}")

pivot_path = os.path.join(TABLES_DIR, "model_results_by_dimension.csv")
results_df.to_csv(pivot_path, index=False)
print(f"[OK] Model results by dimension saved to: {pivot_path}")

best_models = (
    results_df.sort_values("macro_f1", ascending=False)
    .groupby("dimension")
    .first()
    .reset_index()
)
best_path = os.path.join(TABLES_DIR, "best_models_per_dimension.csv")
best_models.to_csv(best_path, index=False)
print(f"[OK] Best models per dimension saved to: {best_path}")

print("\nBest model per dimension (by macro F1):")
for _, row in best_models.iterrows():
    print(f"  {row['dim_label']:5s}  {row['model']:25s}  macro_f1={row['macro_f1']:.4f}")

# --- 4. Core confusion matrix only ---
print(f"\nSaving core confusion matrix ({CORE_CM_DIMENSION}, {CORE_CM_MODEL})...")
model = load_model(CORE_CM_DIMENSION, CORE_CM_MODEL)
y_test = df.iloc[test_idx][CORE_CM_DIMENSION].values
y_pred = model.predict(X_test)
cm_path = plot_confusion_matrix(y_test, y_pred, CORE_CM_DIMENSION, CORE_CM_MODEL)
print(f"[OK] Saved: {cm_path}")

# --- 5. Model comparison figures (from evaluation metrics) ---
print("\nGenerating model comparison figures from test-set results...")
comparison_path = plot_model_comparison_overview(results_df)
print(f"[OK] Saved: {comparison_path}")

svm_path = plot_svm_vs_logreg_comparison(results_df)
print(f"[OK] Saved: {svm_path}")

print(f"\nDone. Metrics saved to {TABLES_DIR}.")
print("Run scripts/visualization/08_interpretability_analysis.py for interpretability figures.")
