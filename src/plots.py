"""Visualization helpers — cinematic editorial style aligned with the website."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve

from src.chart_style import (
    AXES_BG,
    CHART_BG,
    DIMENSION_PALETTES,
    GRID_COLOR,
    NEUTRAL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    add_figure_title,
    build_heatmap_cmap,
    build_sequential_cmap,
    dim_color,
    setup_cinematic_style,
    style_axes,
    trait_colors,
)
from src.config import DIM_FEATURE_TITLES, DIM_LABELS, DIMENSIONS
from src.paths import FIGURES_DIR


def _save_fig(fig, filename: str, output_dir: str = FIGURES_DIR) -> str:
    """Save figure with website-matched background."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=CHART_BG, edgecolor="none")
    plt.close(fig)
    return path


def setup_plot_style() -> None:
    """Apply cinematic editorial defaults (alias for chart_style)."""
    setup_cinematic_style()


def plot_confusion_matrix(
    y_true, y_pred, dimension: str, model_name: str, output_dir: str = FIGURES_DIR
) -> str:
    """Confusion matrix with dimension-specific sequential palette."""
    setup_cinematic_style()
    dim_label = DIM_LABELS[dimension]
    t0, t1 = trait_colors(dimension)
    cmap = build_sequential_cmap(dimension)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, colorbar=True, cmap=cmap,
    )
    for text in disp.text_.ravel():
        text.set_color(TEXT_PRIMARY)
        text.set_fontsize(11)
        text.set_fontweight("600")
    ax.set_title(
        f"Confusion Matrix · {dim_label}\n{model_name.replace('_', ' ').title()}",
        fontsize=12, fontweight="600", color=TEXT_PRIMARY, pad=12,
    )
    ax.set_xlabel("Predicted", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("Actual", fontsize=10, color=TEXT_SECONDARY)
    style_axes(ax, grid_axis=None)
    plt.tight_layout()
    filename = f"confusion_matrix_{dimension}_{model_name}.png"
    return _save_fig(fig, filename, output_dir)


def plot_roc_curve(y_true, y_score, dimension: str, model_name: str) -> str:
    """Single ROC curve with dimension accent."""
    setup_cinematic_style()
    dim_label = DIM_LABELS[dimension]
    color = dim_color(dimension)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"AUC = {auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.08, color=color)
    ax.plot([0, 1], [0, 1], color=NEUTRAL["reference"], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title(
        f"ROC · {dim_label} · {model_name.replace('_', ' ').title()}",
        fontsize=12, fontweight="600",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=True)
    style_axes(ax, grid_axis="both")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    filename = f"roc_curve_{dimension}_{model_name}.png"
    return _save_fig(fig, filename)


def plot_macro_f1_comparison(results_df: pd.DataFrame) -> str:
    """Grouped bar chart — macro F1 by dimension for all models (legacy alias)."""
    return plot_model_comparison_overview(results_df)


def plot_model_comparison_overview(results_df: pd.DataFrame) -> str:
    """Grouped bar chart — macro F1 across dimensions for LR, SVM, and Random Forest."""
    setup_cinematic_style()
    from src.chart_style import MODEL_COLORS, add_figure_title

    model_order = ["logistic_regression", "linear_svm", "random_forest"]
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "linear_svm": "Linear SVM",
        "random_forest": "Random Forest",
    }
    dim_order = ["I/E", "N/S", "T/F", "J/P"]

    pivot = results_df.pivot(index="dim_label", columns="model", values="macro_f1")
    pivot = pivot.reindex(index=dim_order, columns=model_order)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(AXES_BG)

    x = np.arange(len(dim_order))
    width = 0.24

    for i, model in enumerate(model_order):
        offset = (i - 1) * width
        alpha = 0.88 if model == "random_forest" else 0.95
        ax.bar(
            x + offset,
            pivot[model].values,
            width,
            label=model_labels[model],
            color=MODEL_COLORS[i],
            edgecolor="white",
            linewidth=1.1,
            alpha=alpha,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dim_order)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 0.9)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
    )
    style_axes(ax)
    add_figure_title(
        fig,
        "Model Performance Across Dimensions",
        "Macro F1 · Logistic Regression · Linear SVM · Random Forest",
    )
    plt.subplots_adjust(top=0.82, bottom=0.2)
    return _save_fig(fig, "model_comparison_overview.png")


def plot_svm_vs_logreg_comparison(results_df: pd.DataFrame) -> str:
    """Side-by-side macro F1 — Logistic Regression vs Linear SVM per dimension."""
    setup_cinematic_style()
    from src.chart_style import MODEL_COLORS, add_figure_title

    models = ["logistic_regression", "linear_svm"]
    model_labels = {
        "logistic_regression": "Logistic Regression",
        "linear_svm": "Linear SVM",
    }
    dim_order = ["I/E", "N/S", "T/F", "J/P"]

    pivot = results_df.pivot(index="dim_label", columns="model", values="macro_f1")
    pivot = pivot.reindex(index=dim_order, columns=models)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(AXES_BG)

    x = np.arange(len(dim_order))
    width = 0.34

    lr_vals = pivot["logistic_regression"].values
    svm_vals = pivot["linear_svm"].values

    ax.bar(
        x - width / 2,
        lr_vals,
        width,
        label=model_labels["logistic_regression"],
        color=MODEL_COLORS[0],
        edgecolor="white",
        linewidth=1.1,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        svm_vals,
        width,
        label=model_labels["linear_svm"],
        color=MODEL_COLORS[1],
        edgecolor="white",
        linewidth=1.1,
        alpha=0.92,
        zorder=3,
    )

    for i, (lr, svm) in enumerate(zip(lr_vals, svm_vals)):
        delta = lr - svm
        ax.text(
            x[i],
            max(lr, svm) + 0.018,
            f"Δ {delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=TEXT_MUTED,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dim_order)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 0.92)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
    )
    style_axes(ax)
    add_figure_title(
        fig,
        "Linear SVM vs Logistic Regression",
        "Macro F1 by dimension · held-out test set",
    )
    plt.subplots_adjust(top=0.82, bottom=0.22)
    return _save_fig(fig, "svm_vs_logreg_comparison.png")


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    dimension: str,
    model_name: str,
    top_n: int = 20,
) -> str:
    """Top-N features with dimension trait colors."""
    setup_cinematic_style()
    t0, t1 = trait_colors(dimension)
    dim_label = DIM_LABELS[dimension]
    indices = np.argsort(np.abs(importances))[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]
    colors = [t1 if v > 0 else t0 for v in top_values]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(top_features, top_values, color=colors, height=0.65, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Top {top_n} Features · {dim_label}", fontweight="600")
    ax.set_xlabel("Coefficient / Importance")
    ax.axvline(0, color=NEUTRAL["line"], linewidth=0.9, alpha=0.6)
    style_axes(ax)
    plt.tight_layout()
    filename = f"feature_importance_{dimension}_{model_name}.png"
    return _save_fig(fig, filename)


def plot_dimension_predictability_ranking(best_models_df: pd.DataFrame) -> str:
    """Horizontal bar ranking — each dimension uses its identity color."""
    setup_cinematic_style()
    df = best_models_df.sort_values("macro_f1", ascending=True).copy()
    colors = [dim_color(row["dimension"], "primary") for _, row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(
        df["dim_label"], df["macro_f1"], color=colors,
        height=0.58, edgecolor="white", linewidth=1.2,
    )

    for bar, (_, row) in zip(bars, df.iterrows()):
        score = row["macro_f1"]
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center", ha="left", fontsize=10, fontweight="600", color=TEXT_PRIMARY,
        )
        ax.text(
            0.015, bar.get_y() + bar.get_height() / 2,
            DIMENSION_PALETTES[row["dimension"]]["mood"],
            va="center", ha="left", fontsize=7.5, color=TEXT_MUTED, style="italic",
        )

    mean_f1 = df["macro_f1"].mean()
    ax.axvline(mean_f1, color=NEUTRAL["mean"], linestyle="--", linewidth=1, alpha=0.85)
    ax.text(
        mean_f1 + 0.008, 0.02, f"mean {mean_f1:.3f}",
        fontsize=8, color=TEXT_MUTED, transform=ax.get_xaxis_transform(),
    )

    ax.set_xlim(0, min(1.0, df["macro_f1"].max() + 0.14))
    ax.set_xlabel("Macro F1 · Best Model")
    ax.set_title("Which dimensions leave the clearest linguistic traces?", fontweight="600", pad=16)

    legend_handles = [
        Patch(facecolor=dim_color(d, "primary"), edgecolor="white", label=DIM_LABELS[d])
        for d in DIMENSIONS
    ]
    ax.legend(handles=legend_handles, loc="lower right", title="Dimension identity", title_fontsize=8)
    style_axes(ax)
    sns.despine(ax=ax, left=True, bottom=False)
    plt.tight_layout()
    return _save_fig(fig, "dimension_predictability_ranking.png")


def plot_top_lr_features(
    feature_names: list[str],
    coefficients: np.ndarray,
    dimension: str,
    top_n: int = 15,
    output_dir: str = FIGURES_DIR,
) -> str:
    """Diverging editorial bar chart — emotionally meaningful T/F contrast."""
    setup_cinematic_style()
    t0, t1 = trait_colors(dimension)
    pos_idx = np.argsort(coefficients)[-top_n:][::-1]
    neg_idx = np.argsort(coefficients)[:top_n][::-1]
    selected = list(neg_idx) + list(pos_idx)

    labels = [feature_names[i] for i in selected]
    values = [coefficients[i] for i in selected]
    colors = [t0 if v < 0 else t1 for v in values]

    from src.text import DIMENSION_LABELS
    trait_0, trait_1 = DIMENSION_LABELS[dimension]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, height=0.72, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, fontfamily="sans-serif")
    ax.axvline(0, color=NEUTRAL["line"], linewidth=1, alpha=0.7)
    ax.set_xlabel("Logistic Regression Coefficient", labelpad=10)
    title = DIM_FEATURE_TITLES[dimension]
    ax.set_title(
        f"Language signals · {title}",
        fontsize=13, fontweight="600", pad=16,
    )
    ax.text(
        0.5, 1.01,
        "Words that pull toward each pole of the dimension",
        transform=ax.transAxes, ha="center", fontsize=9, color=TEXT_MUTED, style="italic",
    )

    ax.legend(handles=[
        Patch(facecolor=t0, label=trait_0.split(" (")[0]),
        Patch(facecolor=t1, label=trait_1.split(" (")[0]),
    ], loc="lower right", frameon=True)
    style_axes(ax)
    sns.despine(ax=ax, left=True, bottom=False)
    plt.tight_layout()
    return _save_fig(fig, f"top_features_{dimension}.png", output_dir)


def plot_roc_summary_grid(roc_data: list[dict]) -> str:
    """2×2 ROC grid — dimension-colored curves, minimal clutter."""
    setup_cinematic_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))
    axes = axes.flatten()

    for ax, entry in zip(axes, roc_data):
        dim = entry["dimension"]
        color = dim_color(dim)
        fpr, tpr, _ = roc_curve(entry["y_true"], entry["y_score"])
        auc = roc_auc_score(entry["y_true"], entry["y_score"])
        model_label = entry["model_name"].replace("_", " ").title()

        ax.plot(fpr, tpr, color=color, linewidth=2.4, solid_capstyle="round")
        ax.fill_between(fpr, tpr, alpha=0.07, color=color)
        ax.plot([0, 1], [0, 1], color=NEUTRAL["reference"], linestyle="--", linewidth=0.9, alpha=0.65)

        ax.set_title(
            f"{entry['dim_label']} · {DIM_FEATURE_TITLES[dim].split(' vs ')[0]}",
            fontsize=11, fontweight="600", color=TEXT_PRIMARY, pad=10,
        )
        ax.text(
            0.97, 0.08, f"{model_label}\nAUC {auc:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=TEXT_SECONDARY,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=AXES_BG, edgecolor=NEUTRAL["line"], alpha=0.9),
        )
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        style_axes(ax, grid_axis="both")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    add_figure_title(fig, "ROC Curves · Best Model per Dimension", "Discriminative power across linguistic dimensions")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, "roc_summary_grid.png")


def plot_class_imbalance_overview(df: pd.DataFrame) -> str:
    """Grouped bars — each dimension uses its own trait pair colors."""
    setup_cinematic_style()
    from src.text import DIMENSION_LABELS

    dim_labels = [DIM_LABELS[d] for d in DIMENSIONS]
    class_0_counts, class_1_counts = [], []
    trait_0_labels, trait_1_labels = [], []
    colors_0, colors_1 = [], []

    for dim in DIMENSIONS:
        counts = df[dim].value_counts().sort_index()
        class_0_counts.append(counts[0])
        class_1_counts.append(counts[1])
        trait_0_labels.append(DIMENSION_LABELS[dim][0].split(" (")[0])
        trait_1_labels.append(DIMENSION_LABELS[dim][1].split(" (")[0])
        c0, c1 = trait_colors(dim)
        colors_0.append(c0)
        colors_1.append(c1)

    x = np.arange(len(DIMENSIONS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars0 = ax.bar(
        x - width / 2, class_0_counts, width,
        color=colors_0, edgecolor="white", linewidth=1, label="First trait",
    )
    bars1 = ax.bar(
        x + width / 2, class_1_counts, width,
        color=colors_1, edgecolor="white", linewidth=1, label="Second trait",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontweight="600")

    for i, (b0, b1) in enumerate(zip(bars0, bars1)):
        for bar, label in [(b0, trait_0_labels[i]), (b1, trait_1_labels[i])]:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
                f"{label}\n{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY,
            )

    ax.set_title("Class balance across dimensions", fontweight="600", pad=16)
    ax.set_xlabel("MBTI Dimension")
    ax.set_ylabel("Number of users")
    ax.set_ylim(0, max(class_0_counts + class_1_counts) * 1.12)
    style_axes(ax)
    sns.despine(ax=ax, left=True, bottom=False)
    plt.tight_layout()
    return _save_fig(fig, "class_imbalance_overview.png")


def plot_best_model_overview(best_models_df: pd.DataFrame) -> str:
    """Summary card — dimension-colored performance bars + refined table."""
    setup_cinematic_style()
    df = best_models_df.sort_values("macro_f1", ascending=False).copy()
    df["model_display"] = df["model"].str.replace("_", " ").str.title()

    fig = plt.figure(figsize=(12.5, 6.2), facecolor=CHART_BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28)

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
    table.set_fontsize(9.5)
    table.scale(1.05, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:
            cell.set_facecolor(NEUTRAL["table_header"])
            cell.set_text_props(color="white", fontweight="600")
        else:
            dim_key = df.iloc[row - 1]["dimension"]
            if col == 0:
                cell.set_text_props(color=dim_color(dim_key), fontweight="600")
            cell.set_facecolor(NEUTRAL["table_stripe"] if row % 2 == 0 else AXES_BG)

    ax_table.set_title("Best model per dimension", fontsize=12, fontweight="600", pad=18, color=TEXT_PRIMARY)

    ax_bar = fig.add_subplot(gs[1])
    colors = [dim_color(row["dimension"]) for _, row in df.iterrows()]
    bars = ax_bar.barh(df["dim_label"], df["macro_f1"], color=colors, height=0.55, edgecolor="white", linewidth=1)
    for bar, score in zip(bars, df["macro_f1"]):
        ax_bar.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=9.5, fontweight="600", color=TEXT_PRIMARY,
        )
    ax_bar.set_xlim(0, 1.06)
    ax_bar.set_xlabel("Macro F1")
    ax_bar.set_title("Performance ranking", fontsize=12, fontweight="600", pad=12)
    ax_bar.invert_yaxis()
    style_axes(ax_bar)
    sns.despine(ax=ax_bar, left=True, bottom=False)

    add_figure_title(fig, "Model winners summary")
    return _save_fig(fig, "best_model_overview.png")


def plot_feature_heatmap(coef_matrix: pd.DataFrame) -> str:
    """Cross-dimension heatmap — cinematic muted diverging palette."""
    setup_cinematic_style()
    cmap = build_heatmap_cmap()

    fig, ax = plt.subplots(figsize=(8.5, max(6.5, len(coef_matrix) * 0.32)))
    sns.heatmap(
        coef_matrix,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        linecolor=CHART_BG,
        cbar_kws={
            "label": "Normalized coefficient",
            "shrink": 0.72,
            "aspect": 28,
        },
        xticklabels=[DIM_LABELS.get(c, c) for c in coef_matrix.columns],
        yticklabels=True,
    )
    ax.set_title("Cross-dimension lexical signals", fontweight="600", pad=16, fontsize=13)
    ax.set_xlabel("MBTI Dimension", labelpad=8)
    ax.set_ylabel("Term", labelpad=8)
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=8.5)
    plt.tight_layout()
    return _save_fig(fig, "feature_heatmap.png")


def extract_feature_importances(model, feature_names: list[str]) -> np.ndarray | None:
    """Extract coefficients or feature importances from a fitted model."""
    if hasattr(model, "coef_"):
        return model.coef_.flatten()
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    return None
