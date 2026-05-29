"""
06_train_models.py
------------------
IE 423 — Term Project
Script 6: Train classifiers for each MBTI dimension.

This script:
- Loads TF-IDF features and train/test split
- Trains Logistic Regression, Linear SVM, and Random Forest
- Saves trained models and a training summary table
"""

import os
import sys
import time

import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import DIM_LABELS, DIMENSIONS
from src.data import load_processed_data, load_split_indices
from src.features import load_tfidf_artifacts
from src.models import get_models, save_model
from src.paths import TABLES_DIR, ensure_dirs

ensure_dirs()

# --- 1. Load data and features ---
print("Loading data and TF-IDF features...")
df = load_processed_data()
train_idx, test_idx = load_split_indices()
X_train, X_test, vectorizer = load_tfidf_artifacts()

print(f"  Train samples: {len(train_idx)}")
print(f"  Test samples : {len(test_idx)}")
print(f"  Feature dim  : {X_train.shape[1]}")

models = get_models()
training_rows = []

# --- 2. Train models for each dimension ---
for dimension in DIMENSIONS:
    dim_label = DIM_LABELS[dimension]
    y_train = df.iloc[train_idx][dimension].values

    print(f"\n--- Training models for {dim_label} ({dimension}) ---")
    print(f"  Class balance (train): {dict(pd.Series(y_train).value_counts().sort_index())}")

    for model_name, model in models.items():
        print(f"  Training {model_name}...", end=" ")
        start = time.time()

        model.fit(X_train, y_train)
        elapsed = time.time() - start

        path = save_model(model, dimension, model_name)
        print(f"done ({elapsed:.1f}s) -> {path}")

        training_rows.append({
            "dimension": dimension,
            "dim_label": dim_label,
            "model": model_name,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "class_0_count": int((y_train == 0).sum()),
            "class_1_count": int((y_train == 1).sum()),
            "fit_time_sec": round(elapsed, 2),
        })

# --- 3. Save training summary ---
summary_df = pd.DataFrame(training_rows)
summary_path = os.path.join(TABLES_DIR, "training_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n[OK] Training summary saved to: {summary_path}")

print("\nDone.")
