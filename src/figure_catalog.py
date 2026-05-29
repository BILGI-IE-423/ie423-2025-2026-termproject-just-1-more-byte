"""Core figure catalog and archive organization."""

import os
import shutil

from src.paths import FIGURES_ARCHIVE_DIR, FIGURES_DIR, ensure_dirs

# Primary figures kept in visuals/figures for storytelling / GitHub Pages
CORE_FIGURES = frozenset({
    "fig1_type_distribution.png",
    "class_imbalance_overview.png",
    "dimension_predictability_ranking.png",
    "roc_summary_grid.png",
    "top_features_dim_TF.png",
    "feature_heatmap.png",
    "best_model_overview.png",
    "confusion_matrix_dim_TF_logistic_regression.png",
})


def organize_figures(verbose: bool = True) -> list[str]:
    """
    Move non-core figures from visuals/figures to visuals/figures/archive.

    Preserves all files; nothing is deleted permanently.
    """
    ensure_dirs()
    os.makedirs(FIGURES_ARCHIVE_DIR, exist_ok=True)

    moved = []
    for entry in sorted(os.listdir(FIGURES_DIR)):
        if entry == "archive":
            continue

        src = os.path.join(FIGURES_DIR, entry)
        if not os.path.isfile(src):
            continue

        if entry in CORE_FIGURES:
            continue

        dest = os.path.join(FIGURES_ARCHIVE_DIR, entry)
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(src, dest)
        moved.append(entry)
        if verbose:
            print(f"  Archived: {entry}")

    if verbose:
        core_count = len(CORE_FIGURES)
        print(f"\n[OK] {len(moved)} figure(s) moved to archive.")
        print(f"     {core_count} core figures remain in visuals/figures/")

    return moved
