"""
04_build_tfidf.py
-----------------
IE 423 — Term Project
Script 4: Build TF-IDF features for MBTI dimension classification.

This script:
- Loads the cleaned dataset
- Creates a stratified train/test split (stratified on MBTI type)
- Fits TF-IDF vectorizer on training texts only
- Saves sparse feature matrices and the fitted vectorizer
"""

import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TEST_SIZE
from src.data import create_stratified_split, load_processed_data
from src.features import build_tfidf_features, save_tfidf_artifacts
from src.paths import (
    SPLITS_DIR,
    TFIDF_VECTORIZER_PATH,
    X_TEST_PATH,
    X_TRAIN_PATH,
    ensure_dirs,
)

ensure_dirs()

# --- 1. Load data ---
print("Loading processed data...")
df = load_processed_data()
print(f"Loaded {df.shape[0]} rows")

# --- 2. Stratified train/test split ---
print(f"\nCreating stratified train/test split (test_size={TEST_SIZE})...")
train_idx, test_idx = create_stratified_split(df)
print(f"  Train size: {len(train_idx)}")
print(f"  Test size : {len(test_idx)}")
print(f"[OK] Split indices saved to: {SPLITS_DIR}")

# --- 3. Build TF-IDF features ---
print(f"\nBuilding TF-IDF features (max_features={TFIDF_MAX_FEATURES})...")

texts_train = df.iloc[train_idx]["clean_posts"].tolist()
texts_test = df.iloc[test_idx]["clean_posts"].tolist()

X_train, X_test, vectorizer = build_tfidf_features(
    texts_train,
    texts_test,
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
)

print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape : {X_test.shape}")
print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# --- 4. Save artifacts ---
save_tfidf_artifacts(X_train, X_test, vectorizer)
print(f"\n[OK] TF-IDF vectorizer saved to: {TFIDF_VECTORIZER_PATH}")
print(f"[OK] X_train saved to: {X_TRAIN_PATH}")
print(f"[OK] X_test saved to: {X_TEST_PATH}")

print("\nDone.")
