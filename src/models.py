"""Model factory for MBTI dimension classification."""

import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.config import LOGREG_MAX_ITER, RANDOM_STATE, RF_N_ESTIMATORS
from src.paths import MODELS_DIR


def get_models() -> dict:
    """Return a dict of configured classifiers with balanced class weights."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=LOGREG_MAX_ITER,
            random_state=RANDOM_STATE,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def model_path(dimension: str, model_name: str) -> str:
    """Build the file path for a saved model."""
    return os.path.join(MODELS_DIR, f"{dimension}_{model_name}.joblib")


def save_model(model, dimension: str, model_name: str) -> str:
    """Save a trained model and return its path."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = model_path(dimension, model_name)
    joblib.dump(model, path)
    return path


def load_model(dimension: str, model_name: str):
    """Load a previously trained model."""
    path = model_path(dimension, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
