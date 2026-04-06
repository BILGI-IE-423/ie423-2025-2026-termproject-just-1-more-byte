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
- Saves all figures to outputs/figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# --- Paths ---
PROCESSED_DATA_PATH = os.path.join("data", "processed", "mbti_cleaned.csv")
FIGURES_PATH        = os.path.join("outputs", "figures")
TABLES_PATH         = os.path.join("outputs", "tables")

os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(TABLES_PATH, exist_ok=True)

# --- Load data ---
print("Loading data...")

df = pd.read_csv(PROCESSED_DATA_PATH)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")


# --- Figure 1: MBTI Type Distribution ---

type_counts = df["type"].value_counts().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(type_counts.index, type_counts.values,
              color=sns.color_palette("Blues_d", len(type_counts)))

for bar, val in zip(bars, type_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            str(val), ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_title("Distribution of MBTI Personality Types in Dataset",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("MBTI Type", fontsize=11)
ax.set_ylabel("Number of Users", fontsize=11)
ax.set_ylim(0, type_counts.max() * 1.15)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_PATH, "fig1_type_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: outputs/figures/fig1_type_distribution.png")

# --- Figure 2: Class Balance per Dimension ---

dims = {
    "dim_IE": ("Introvert (I)", "Extrovert (E)"),
    "dim_NS": ("Intuitive (N)", "Sensing (S)"),
    "dim_TF": ("Thinking (T)", "Feeling (F)"),
    "dim_JP": ("Judging (J)", "Perceiving (P)"),
}
dim_labels = ["I/E", "N/S", "T/F", "J/P"]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, (col, (label_0, label_1)), dim_label in zip(axes, dims.items(), dim_labels):
    counts = df[col].value_counts().sort_index()
    total  = counts.sum()
    labels = [f"{label_0}\n{counts[0]} ({counts[0]/total*100:.1f}%)",
              f"{label_1}\n{counts[1]} ({counts[1]/total*100:.1f}%)"]
    ax.pie(counts.values, labels=labels, autopct=None,
           colors=["#4C72B0", "#DD8452"], startangle=90,
           textprops={"fontsize": 9})
    ax.set_title(f"Dimension: {dim_label}", fontsize=12, fontweight="bold")

fig.suptitle("Class Balance per MBTI Dimension", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_PATH, "fig2_dimension_balance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: outputs/figures/fig2_dimension_balance.png")

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
fig.savefig(os.path.join(FIGURES_PATH, "fig3_wordcount_IE.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: outputs/figures/fig3_wordcount_IE.png")

# --- Figure 4: Word Count Boxplots ---

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

dim_info = [
    ("dim_IE", {0: "Introvert (I)", 1: "Extrovert (E)"}),
    ("dim_NS", {0: "Intuitive (N)", 1: "Sensing (S)"}),
    ("dim_TF", {0: "Thinking (T)", 1: "Feeling (F)"}),
    ("dim_JP", {0: "Judging (J)",   1: "Perceiving (P)"}),
]

for ax, (col, label_map), dim_label in zip(axes, dim_info, dim_labels):
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
fig.savefig(os.path.join(FIGURES_PATH, "fig4_wordcount_boxplots.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: outputs/figures/fig4_wordcount_boxplots.png")

# --- Figure 5: Vocabulary Richness ---

def type_token_ratio(text):
    """Unique words / total words — a measure of vocabulary diversity."""
    words = text.split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

df["ttr"] = df["clean_posts"].apply(type_token_ratio)

ttr_by_type = df.groupby("type")["ttr"].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(ttr_by_type.index, ttr_by_type.values,
              color=sns.color_palette("Greens_d", len(ttr_by_type)))
ax.axhline(ttr_by_type.mean(), color="red", linestyle="--", linewidth=1.5,
           label=f"Overall mean TTR: {ttr_by_type.mean():.3f}")
ax.set_title("Average Vocabulary Richness (Type-Token Ratio) by MBTI Type",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("MBTI Type", fontsize=11)
ax.set_ylabel("Mean Type-Token Ratio (TTR)", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_PATH, "fig5_vocabulary_richness.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: outputs/figures/fig5_vocabulary_richness.png")

# --- Summary statistics table ---

summary = df.groupby("type")[["word_count", "char_count", "avg_word_len", "ttr"]].mean().round(2)
summary.columns = ["Avg Word Count", "Avg Char Count", "Avg Word Length", "Avg TTR"]
summary.to_csv(os.path.join(TABLES_PATH, "linguistic_summary_by_type.csv"))
print("[OK] Saved: outputs/tables/linguistic_summary_by_type.csv")
print(summary.to_string())

print("\nDone. All figures saved to:", FIGURES_PATH)
