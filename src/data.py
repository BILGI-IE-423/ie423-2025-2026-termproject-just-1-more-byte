"""Data loading and train/test splitting utilities."""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.paths import (
    PROCESSED_DATA,
    TEST_INDICES_PATH,
    TRAIN_INDICES_PATH,
    SPLITS_DIR,
)


def load_processed_data() -> pd.DataFrame:
    """Load the cleaned MBTI dataset."""
    if not os.path.exists(PROCESSED_DATA):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA}. "
            "Run scripts/preprocessing/02_preprocess_data.py first."
        )
    return pd.read_csv(PROCESSED_DATA)


def create_stratified_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a stratified train/test split on MBTI type.

    Returns index arrays so the same split is reused across all dimensions.
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["type"],
    )

    np.save(TRAIN_INDICES_PATH, train_idx)
    np.save(TEST_INDICES_PATH, test_idx)

    return train_idx, test_idx


def load_split_indices() -> tuple[np.ndarray, np.ndarray]:
    """Load previously saved train/test split indices."""
    if not os.path.exists(TRAIN_INDICES_PATH) or not os.path.exists(TEST_INDICES_PATH):
        raise FileNotFoundError(
            "Split indices not found. Run scripts/feature_engineering/04_build_tfidf.py first."
        )
    return np.load(TRAIN_INDICES_PATH), np.load(TEST_INDICES_PATH)
