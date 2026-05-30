"""
03_basic_eda.py
---------------
IE 423 — Term Project
Script 3: Exploratory Data Analysis on the preprocessed MBTI dataset.

This script:
- Loads the cleaned dataset
- Produces class distribution plots
- Plots word count distributions per dimension (I/E, N/S, T/F, J/P)
- Generates a vocabulary richness comparison
- Saves core EDA figure to visuals/figures/
- Saves supplementary EDA figures to visuals/figures/archive/
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data import load_processed_data
from src.paths import FIGURES_ARCHIVE_DIR, FIGURES_DIR, TABLES_DIR, ensure_dirs
from src.text import DIMENSION_LABELS, type_token_ratio

ensure_dirs()

# --- Load data ---
print("Loading data...")

df = load_processed_data()
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

dim_labels = ["I/E", "N/S", "T/F", "J/P"]

# --- Figure 1: MBTI Type Distribution (cinematic editorial) ---

from src.chart_style import (
    CHART_BG,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    mbti_type_color,
    setup_cinematic_style,
    style_axes,
)

setup_cinematic_style()
type_counts = df["type"].value_counts().sort_values(ascending=False)
bar_colors = [mbti_type_color(t) for t in type_counts.index]

fig, ax = plt.subplots(figsize=(14, 5.5), facecolor=CHART_BG)
bars = ax.bar(
    type_counts.index, type_counts.values,
    color=bar_colors, edgecolor="white", linewidth=0.8, width=0.78,
)

for bar, val in zip(bars, type_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
        str(val), ha="center", va="bottom", fontsize=7.5, color=TEXT_SECONDARY,
    )

ax.set_title(
    "Distribution of personality types in the corpus",
    fontsize=13, fontweight="600", color=TEXT_PRIMARY, pad=16,
)
ax.set_xlabel("MBTI Type", fontsize=10, color=TEXT_SECONDARY)
ax.set_ylabel("Number of users", fontsize=10, color=TEXT_SECONDARY)
ax.set_ylim(0, type_counts.max() * 1.12)
ax.tick_params(axis="x", labelsize=8.5, rotation=0)
style_axes(ax)
plt.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, "fig1_type_distribution.png"),
    dpi=160, bbox_inches="tight", facecolor=CHART_BG, edgecolor="none",
)
plt.close()
print("[OK] Saved: visuals/figures/fig1_type_distribution.png")

# --- Figure 2: Class Balance per Dimension ---

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, (col, (label_0, label_1)), dim_label in zip(axes, DIMENSION_LABELS.items(), dim_labels):
    counts = df[col].value_counts().sort_index()
    total = counts.sum()
    labels = [
        f"{label_0}\n{counts[0]} ({counts[0]/total*100:.1f}%)",
        f"{label_1}\n{counts[1]} ({counts[1]/total*100:.1f}%)",
    ]
    ax.pie(counts.values, labels=labels, autopct=None,
           colors=["#4C72B0", "#DD8452"], startangle=90,
           textprops={"fontsize": 9})
    ax.set_title(f"Dimension: {dim_label}", fontsize=12, fontweight="bold")

fig.suptitle("Class Balance per MBTI Dimension", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_ARCHIVE_DIR, "fig2_dimension_balance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: visuals/figures/archive/fig2_dimension_balance.png")

# --- Figure 3: Word Count Distribution by I/E ---

fig, ax = plt.subplots(figsize=(10, 5))

for val, label, color in [(0, "Introvert (I)", "#4C72B0"), (1, "Extrovert (E)", "#DD8452")]:
    subset = df[df["dim_IE"] == val]["word_count"]
    ax.hist(subset, bins=40, alpha=0.6, label=f"{label} (n={len(subset)})",
            color=color, edgecolor="white", linewidth=0.5)
    ax.axvline(subset.mean(), color=color, linestyle="--", linewidth=1.5,
               label=f"Mean ({label[:1]}): {subset.mean():.0f}")

ax.set_title("Word Count Distribution: Introvert vs. Extrovert",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Total Word Count (all posts)", fontsize=11)
ax.set_ylabel("Number of Users", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_ARCHIVE_DIR, "fig3_wordcount_IE.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: visuals/figures/archive/fig3_wordcount_IE.png")

# --- Figure 4: Word Count Boxplots ---

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for ax, (col, (label_0, label_1)), dim_label in zip(axes, DIMENSION_LABELS.items(), dim_labels):
    label_map = {0: label_0, 1: label_1}
    data = [df[df[col] == v]["word_count"].values for v in [0, 1]]
    bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([label_map[0], label_map[1]], fontsize=9)
    ax.set_title(f"Dimension: {dim_label}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Word Count" if dim_label == "I/E" else "", fontsize=10)

fig.suptitle("Word Count Distributions by Each MBTI Dimension", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_ARCHIVE_DIR, "fig4_wordcount_boxplots.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: visuals/figures/archive/fig4_wordcount_boxplots.png")

# --- Figure 5: Vocabulary Richness ---

df["ttr"] = df["clean_posts"].apply(type_token_ratio)

ttr_by_type = df.groupby("type")["ttr"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(ttr_by_type.index, ttr_by_type.values,
       color=sns.color_palette("Greens_d", len(ttr_by_type)))
ax.axhline(ttr_by_type.mean(), color="red", linestyle="--", linewidth=1.5,
           label=f"Overall mean TTR: {ttr_by_type.mean():.3f}")
ax.set_title("Average Vocabulary Richness (Type-Token Ratio) by MBTI Type",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("MBTI Type", fontsize=11)
ax.set_ylabel("Mean Type-Token Ratio (TTR)", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_ARCHIVE_DIR, "fig5_vocabulary_richness.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: visuals/figures/archive/fig5_vocabulary_richness.png")

# --- Summary statistics table ---

summary = df.groupby("type")[["word_count", "char_count", "avg_word_len", "ttr"]].mean().round(2)
summary.columns = ["Avg Word Count", "Avg Char Count", "Avg Word Length", "Avg TTR"]
summary.to_csv(os.path.join(TABLES_DIR, "linguistic_summary_by_type.csv"))
print("[OK] Saved: visuals/tables/linguistic_summary_by_type.csv")
print(summary.to_string())

print("\nDone. Core figure in visuals/figures/; supplementary figures in archive.")
