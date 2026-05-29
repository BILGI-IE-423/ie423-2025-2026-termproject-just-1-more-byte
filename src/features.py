"""Feature engineering utilities."""

import joblib
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
from src.paths import TFIDF_VECTORIZER_PATH, X_TRAIN_PATH, X_TEST_PATH


def build_tfidf_features(
    texts_train: list[str],
    texts_test: list[str],
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
) -> tuple:
    """
    Fit TF-IDF on training texts and transform both train and test sets.

    Returns (X_train, X_test, vectorizer).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range,
    )
    X_train = vectorizer.fit_transform(texts_train)
    X_test = vectorizer.transform(texts_test)
    return X_train, X_test, vectorizer


def save_tfidf_artifacts(X_train, X_test, vectorizer) -> None:
    """Persist TF-IDF matrices and fitted vectorizer."""
    save_npz(X_TRAIN_PATH, X_train)
    save_npz(X_TEST_PATH, X_test)
    joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)


def load_tfidf_artifacts() -> tuple:
    """Load saved TF-IDF matrices and vectorizer."""
    X_train = load_npz(X_TRAIN_PATH)
    X_test = load_npz(X_TEST_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    return X_train, X_test, vectorizer
