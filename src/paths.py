"""Centralized relative paths for the project."""

import os

# Data paths
RAW_DATA = os.path.join("data", "raw", "mbti_1.csv")
PROCESSED_DATA = os.path.join("data", "processed", "mbti_cleaned.csv")
SPLITS_DIR = os.path.join("data", "processed", "splits")
X_TRAIN_PATH = os.path.join("data", "processed", "X_train.npz")
X_TEST_PATH = os.path.join("data", "processed", "X_test.npz")

# Artifact paths
FIGURES_DIR = os.path.join("visuals", "figures")
FIGURES_ARCHIVE_DIR = os.path.join("visuals", "figures", "archive")
TABLES_DIR = os.path.join("visuals", "tables")
MODELS_DIR = os.path.join("models")

# Split index files
TRAIN_INDICES_PATH = os.path.join(SPLITS_DIR, "train_indices.npy")
TEST_INDICES_PATH = os.path.join(SPLITS_DIR, "test_indices.npy")

# Model artifacts
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")


def ensure_dirs():
    """Create output directories if they do not exist."""
    for path in [FIGURES_DIR, FIGURES_ARCHIVE_DIR, TABLES_DIR, MODELS_DIR, SPLITS_DIR]:
        os.makedirs(path, exist_ok=True)
