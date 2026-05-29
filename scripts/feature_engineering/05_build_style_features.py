"""
05_build_style_features.py
--------------------------
IE 423 — Term Project
(FUTURE) Script for Research Question 3: style-only features.

Research Question 3:
    Can personality traits still be predicted when topical vocabulary is removed?

Planned approach:
    1. Identify high-IDF content/topic words from the TF-IDF vocabulary
    2. Remove topical vocabulary from cleaned text, retaining stylistic signals
    3. Engineer style-only features:
       - Punctuation usage patterns
       - Sentence and word length statistics
       - Vocabulary richness (type-token ratio)
       - Function word frequencies
    4. Build a separate feature matrix from style-only representations
    5. Re-run the same modeling/evaluation pipeline (scripts 06-07)
    6. Compare macro F1 against the baseline TF-IDF results

This script is a placeholder. Implementation is deferred to a later phase.
"""

import sys

print("=" * 60)
print("  RQ3 Placeholder: Style-Only Feature Engineering")
print("=" * 60)
print()
print("This script is not yet implemented.")
print("See docstring in 05_build_style_features.py for the planned approach.")
print()
print("To run the baseline pipeline, use:")
print("  python scripts/run_pipeline.py --from 06")
print()
sys.exit(0)
