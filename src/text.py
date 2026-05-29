"""Text cleaning and linguistic helper functions."""

import re

# MBTI dimension label maps used across EDA and visualization
DIMENSION_LABELS = {
    "dim_IE": ("Introvert (I)", "Extrovert (E)"),
    "dim_NS": ("Intuitive (N)", "Sensing (S)"),
    "dim_TF": ("Thinking (T)", "Feeling (F)"),
    "dim_JP": ("Judging (J)", "Perceiving (P)"),
}

MBTI_PATTERN = (
    r"\b(INTJ|INTP|ENTJ|ENTP|INFJ|INFP|ENFJ|ENFP|"
    r"ISTJ|ISFJ|ESTJ|ESFJ|ISTP|ISFP|ESTP|ESFP)\b"
)


def clean_text(text: str) -> str:
    """Remove URLs, MBTI type mentions, special chars, and extra whitespace."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(MBTI_PATTERN, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def type_token_ratio(text: str) -> float:
    """Unique words / total words — a measure of vocabulary diversity."""
    words = text.split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)
