"""Project-wide configuration constants."""

RANDOM_STATE = 42
TEST_SIZE = 0.2

DIMENSIONS = ["dim_IE", "dim_NS", "dim_TF", "dim_JP"]
DIM_LABELS = {
    "dim_IE": "I/E",
    "dim_NS": "N/S",
    "dim_TF": "T/F",
    "dim_JP": "J/P",
}

# Human-readable labels for interpretability plots
DIM_FEATURE_TITLES = {
    "dim_IE": "Introvert vs Extrovert",
    "dim_NS": "Intuitive vs Sensing",
    "dim_TF": "Thinking vs Feeling",
    "dim_JP": "Judging vs Perceiving",
}

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 1)

# Model hyperparameters
LOGREG_MAX_ITER = 1000
RF_N_ESTIMATORS = 100
