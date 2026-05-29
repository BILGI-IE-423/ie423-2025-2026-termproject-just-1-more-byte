"""Visualization helpers for model evaluation and interpretability."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, roc_curve

from src.config import DIM_FEATURE_TITLES, DIM_LABELS, DIMENSIONS
from src.paths import FIGURES_DIR

# Shared research-style palette
COLORS = {
    "primary": "#2C5282",
    "secondary": "#4A5568",
    "positive": "#DD8452",
    "negative": "#4C72B0",
    "easiest": "#38A169",
    "hardest": "#E53E3E",
    "neutral": "#718096",
    "grid": "#E2E8F0",
    "class_a": "#4C72B0",
    "class_b": "#DD8452",
}


def setup_plot_style() -> None:
    """Apply consistent publication-style defaults."""
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": "#1A202C",
        "text.color": "#1A202C",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
    })


def _save_fig(fig, filename: str, output_dir: str = FIGURES_DIR) -> str:
    """Save figure and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_confusion_matrix(
    y_true, y_pred, dimension: str, model_name: str, output_dir: str = FIGURES_DIR
) -> str:
    """Plot and save a confusion matrix."""
    setup_plot_style()
    dim_label = DIM_LABELS[dimension]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {dim_label} ({model_name.replace('_', ' ').title()})")
    filename = f"confusion_matrix_{dimension}_{model_name}.png"
    return _save_fig(fig, filename, output_dir)


def plot_roc_curve(y_true, y_score, dimension: str, model_name: str) -> str:
    """Plot and save an ROC curve."""
    setup_plot_style()
    dim_label = DIM_LABELS[dimension]
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_title(f"ROC Curve — {dim_label} ({model_name.replace('_', ' ').title()})")
    filename = f"roc_curve_{dimension}_{model_name}.png"
    return _save_fig(fig, filename)


def plot_macro_f1_comparison(results_df: pd.DataFrame) -> str:
    """Grouped bar chart of macro F1 across dimensions and models."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = results_df.pivot(index="dim_label", columns="model", values="macro_f1")
    pivot.plot(kind="bar", ax=ax, rot=0, colormap="Set2")
    ax.set_title("Macro F1 Comparison by Dimension and Model")
    ax.set_xlabel("MBTI Dimension")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.legend(title="Model", fontsize=9)
    plt.tight_layout()
    return _save_fig(fig, "macro_f1_comparison.png")


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    dimension: str,
    model_name: str,
    top_n: int = 20,
) -> str:
    """Plot top-N feature importances or coefficients."""
    setup_plot_style()
    dim_label = DIM_LABELS[dimension]
    indices = np.argsort(np.abs(importances))[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in top_values]
    ax.barh(top_features, top_values, color=colors)
    ax.set_title(
        f"Top {top_n} Features — {dim_label} ({model_name.replace('_', ' ').title()})"
    )
    ax.set_xlabel("Coefficient / Importance")
    ax.axvline(0, color=COLORS["secondary"], linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    filename = f"feature_importance_{dimension}_{model_name}.png"
    return _save_fig(fig, filename)


def plot_dimension_predictability_ranking(best_models_df: pd.DataFrame) -> str:
    """Horizontal bar chart ranking dimensions by best macro F1."""
    setup_plot_style()
    df = best_models_df.sort_values("macro_f1", ascending=True).copy()
    easiest = df.iloc[-1]["dimension"]
    hardest = df.iloc[0]["dimension"]

    colors = [
        COLORS["easiest"] if row["dimension"] == easiest
        else COLORS["hardest"] if row["dimension"] == hardest
        else COLORS["primary"]
        for _, row in df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["dim_label"], df["macro_f1"], color=colors, height=0.6, edgecolor="white")

    for bar, score in zip(bars, df["macro_f1"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.set_xlim(0, min(1.0, df["macro_f1"].max() + 0.12))
    ax.set_xlabel("Macro F1 (Best Model)")
    ax.set_title("Predictability of MBTI Dimensions")
    ax.axvline(df["macro_f1"].mean(), color=COLORS["neutral"], linestyle="--",
               linewidth=1, label=f"Mean: {df['macro_f1'].mean():.3f}")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLORS["easiest"], label=f"Easiest: {DIM_LABELS[easiest]}"),
        Patch(facecolor=COLORS["hardest"], label=f"Hardest: {DIM_LABELS[hardest]}"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    return _save_fig(fig, "dimension_predictability_ranking.png")


def plot_top_lr_features(
    feature_names: list[str],
    coefficients: np.ndarray,
    dimension: str,
    top_n: int = 15,
    output_dir: str = FIGURES_DIR,
) -> str:
    """Diverging horizontal bar chart of top positive and negative LR coefficients."""
    setup_plot_style()
    pos_idx = np.argsort(coefficients)[-top_n:][::-1]
    neg_idx = np.argsort(coefficients)[:top_n][::-1]
    selected = list(neg_idx) + list(pos_idx)

    labels = [feature_names[i] for i in selected]
    values = [coefficients[i] for i in selected]
    colors = [COLORS["negative"] if v < 0 else COLORS["positive"] for v in values]

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.axvline(0, color=COLORS["secondary"], linewidth=1)
    ax.set_xlabel("Logistic Regression Coefficient")
    title = DIM_FEATURE_TITLES[dimension]
    ax.set_title(f"Most Predictive Terms for {title}")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLORS["positive"], label="Favors second trait"),
        Patch(facecolor=COLORS["negative"], label="Favors first trait"),
    ], loc="lower right", frameon=True)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    return _save_fig(fig, f"top_features_{dimension}.png", output_dir)


def plot_roc_summary_grid(
    roc_data: list[dict],
) -> str:
    """
    2x2 ROC grid for the best model per dimension.

    roc_data: list of dicts with keys dimension, dim_label, model_name, y_true, y_score
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, entry in zip(axes, roc_data):
        fpr, tpr, _ = roc_curve(entry["y_true"], entry["y_score"])
        auc = roc_auc_score(entry["y_true"], entry["y_score"])
        model_label = entry["model_name"].replace("_", " ").title()

        ax.plot(fpr, tpr, color=COLORS["primary"], linewidth=2,
                label=f"{model_label} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
        ax.set_title(f"{entry['dim_label']} — {DIM_FEATURE_TITLES[entry['dimension']]}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", frameon=True)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle("ROC Curves — Best Model per Dimension", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save_fig(fig, "roc_summary_grid.png")


def plot_class_imbalance_overview(df: pd.DataFrame) -> str:
    """Grouped bar chart of class counts per MBTI dimension."""
    setup_plot_style()
    from src.text import DIMENSION_LABELS

    dim_labels = [DIM_LABELS[d] for d in DIMENSIONS]
    class_0_counts = []
    class_1_counts = []
    trait_0_labels = []
    trait_1_labels = []

    for dim in DIMENSIONS:
        counts = df[dim].value_counts().sort_index()
        class_0_counts.append(counts[0])
        class_1_counts.append(counts[1])
        trait_0_labels.append(DIMENSION_LABELS[dim][0].split()[0])  # e.g. "Introvert"
        trait_1_labels.append(DIMENSION_LABELS[dim][1].split()[0])  # e.g. "Extrovert"

    x = np.arange(len(DIMENSIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars0 = ax.bar(x - width / 2, class_0_counts, width, color=COLORS["class_a"], edgecolor="white")
    bars1 = ax.bar(x + width / 2, class_1_counts, width, color=COLORS["class_b"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)

    for i, (b0, b1) in enumerate(zip(bars0, bars1)):
        ax.text(b0.get_x() + b0.get_width() / 2, b0.get_height() + 80,
                f"{trait_0_labels[i]}\n{int(b0.get_height()):,}",
                ha="center", va="bottom", fontsize=8)
        ax.text(b1.get_x() + b1.get_width() / 2, b1.get_height() + 80,
                f"{trait_1_labels[i]}\n{int(b1.get_height()):,}",
                ha="center", va="bottom", fontsize=8)

    ax.set_title("Class Imbalance Across MBTI Dimensions")
    ax.set_xlabel("MBTI Dimension")
    ax.set_ylabel("Number of Users")
    ax.set_ylim(0, max(class_0_counts + class_1_counts) * 1.15)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    return _save_fig(fig, "class_imbalance_overview.png")


def plot_best_model_overview(best_models_df: pd.DataFrame) -> str:
    """Summary card combining metrics table and macro F1 bars."""
    setup_plot_style()
    df = best_models_df.sort_values("macro_f1", ascending=False).copy()
    df["model_display"] = df["model"].str.replace("_", " ").str.title()

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

    # Left: metrics table
    ax_table = fig.add_subplot(gs[0])
    ax_table.axis("off")
    table_data = df[["dim_label", "model_display", "macro_f1", "accuracy", "roc_auc"]].copy()
    table_data.columns = ["Dimension", "Best Model", "Macro F1", "Accuracy", "ROC-AUC"]
    for col in ["Macro F1", "Accuracy", "ROC-AUC"]:
        table_data[col] = table_data[col].map(lambda x: f"{x:.3f}")

    table = ax_table.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F7FAFC" if row % 2 == 0 else "white")
    ax_table.set_title("Best Model per Dimension", pad=20)

    # Right: macro F1 bars
    ax_bar = fig.add_subplot(gs[1])
    colors = sns.color_palette("Blues_d", len(df))[::-1]
    bars = ax_bar.barh(df["dim_label"], df["macro_f1"], color=colors, height=0.55)
    for bar, score in zip(bars, df["macro_f1"]):
        ax_bar.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", fontsize=10)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xlabel("Macro F1")
    ax_bar.set_title("Performance Ranking")
    ax_bar.invert_yaxis()
    sns.despine(ax=ax_bar, left=True, bottom=False)

    fig.suptitle("Model Winners Summary", fontsize=15, fontweight="bold")
    return _save_fig(fig, "best_model_overview.png")


def plot_feature_heatmap(coef_matrix: pd.DataFrame) -> str:
    """Heatmap of normalized LR coefficients across dimensions."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, max(6, len(coef_matrix) * 0.28)))
    sns.heatmap(
        coef_matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Normalized Coefficient", "shrink": 0.8},
    )
    ax.set_title("Cross-Dimension Feature Importance Heatmap")
    ax.set_xlabel("MBTI Dimension")
    ax.set_ylabel("Term")
    plt.tight_layout()
    return _save_fig(fig, "feature_heatmap.png")


def extract_feature_importances(model, feature_names: list[str]) -> np.ndarray | None:
    """Extract coefficients or feature importances from a fitted model."""
    if hasattr(model, "coef_"):
        return model.coef_.flatten()
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return None
