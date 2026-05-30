"""
Cinematic editorial chart styling — aligned with the website identity.

Strict dimension color system: each MBTI pair owns a distinct visual language.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Website-aligned surfaces ---
CHART_BG = "#F4F2EF"
AXES_BG = "#FAF9F7"
TEXT_PRIMARY = "#1A1A1F"
TEXT_SECONDARY = "#5C5C66"
TEXT_MUTED = "#8A8A94"
GRID_COLOR = "#DDD9D4"
SPINE_COLOR = "#C8C4BE"

# --- Strict dimension identity palettes ---
DIMENSION_PALETTES: dict[str, dict[str, str]] = {
    "dim_IE": {
        "primary": "#2D7878",
        "trait_0": "#2D7878",   # Introvert — deep teal
        "trait_1": "#8AAEB5",   # Extrovert — cool silver-blue
        "accent": "#5A9E9E",
        "muted": "#C5DDE0",
        "mood": "reflective teal",
    },
    "dim_NS": {
        "primary": "#64508C",
        "trait_0": "#64508C",   # Intuitive — soft violet
        "trait_1": "#486E58",   # Sensing — earthy green
        "accent": "#7A6AA8",
        "muted": "#D4CCE8",
        "mood": "abstract vs grounded",
    },
    "dim_TF": {
        "primary": "#5B7A94",
        "trait_0": "#5B7A94",   # Thinking — icy blue
        "trait_1": "#C46B6B",   # Feeling — warm coral
        "accent": "#6B7280",
        "muted": "#D4DCE4",
        "mood": "logic vs emotion",
    },
    "dim_JP": {
        "primary": "#4A4F54",
        "trait_0": "#4A4F54",   # Judging — graphite
        "trait_1": "#D4A054",   # Perceiving — warm amber
        "accent": "#9CA3AF",
        "muted": "#E0E2E4",
        "mood": "order vs flow",
    },
}

# Neutral accents for cross-dimension elements
NEUTRAL = {
    "line": "#9CA3AF",
    "mean": "#8A8A94",
    "reference": "#B8B4AE",
    "table_header": "#4A5568",
    "table_stripe": "#F0EEEB",
}

# Model comparison (when needed) — muted, non-dimension
MODEL_COLORS = ["#5B7A94", "#64508C", "#486E58"]


def dim_color(dimension: str, key: str = "primary") -> str:
    """Return a palette color for a dimension."""
    return DIMENSION_PALETTES[dimension][key]


def trait_colors(dimension: str) -> tuple[str, str]:
    """Return (trait_0, trait_1) colors for a dimension pair."""
    p = DIMENSION_PALETTES[dimension]
    return p["trait_0"], p["trait_1"]


def mbti_type_color(type_str: str) -> str:
    """Blend dimension trait colors from a four-letter MBTI type."""
    letter_map = {
        "I": ("dim_IE", "trait_0"),
        "E": ("dim_IE", "trait_1"),
        "N": ("dim_NS", "trait_0"),
        "S": ("dim_NS", "trait_1"),
        "T": ("dim_TF", "trait_0"),
        "F": ("dim_TF", "trait_1"),
        "J": ("dim_JP", "trait_0"),
        "P": ("dim_JP", "trait_1"),
    }
    rgbs = []
    for ch in type_str.upper():
        if ch in letter_map:
            dim, key = letter_map[ch]
            rgbs.append(mcolors.to_rgb(DIMENSION_PALETTES[dim][key]))
    if not rgbs:
        return NEUTRAL["line"]
    avg = tuple(sum(c[i] for c in rgbs) / len(rgbs) for i in range(3))
    return mcolors.to_hex(avg)


def build_tf_diverging_cmap() -> LinearSegmentedColormap:
    """Thinking (cool) ↔ Feeling (warm) for coefficient / heatmap visuals."""
    return LinearSegmentedColormap.from_list(
        "cinematic_tf",
        [DIMENSION_PALETTES["dim_TF"]["trait_0"], AXES_BG, DIMENSION_PALETTES["dim_TF"]["trait_1"]],
        N=256,
    )


def build_sequential_cmap(dimension: str) -> LinearSegmentedColormap:
    """Soft sequential cmap for confusion matrices per dimension."""
    p = DIMENSION_PALETTES[dimension]
    return LinearSegmentedColormap.from_list(
        f"cinematic_{dimension}",
        [CHART_BG, p["muted"], p["primary"]],
        N=256,
    )


def build_heatmap_cmap() -> LinearSegmentedColormap:
    """Cross-dimension heatmap — muted cinematic diverging."""
    return LinearSegmentedColormap.from_list(
        "cinematic_heatmap",
        [
            DIMENSION_PALETTES["dim_TF"]["trait_0"],
            "#E8E6E3",
            DIMENSION_PALETTES["dim_TF"]["trait_1"],
        ],
        N=256,
    )


def setup_cinematic_style() -> None:
    """Apply editorial defaults matching the website."""
    plt.rcParams.update({
        "figure.facecolor": CHART_BG,
        "axes.facecolor": AXES_BG,
        "savefig.facecolor": CHART_BG,
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 0.8,
        "axes.titleweight": "600",
        "axes.titlesize": 13,
        "axes.titlepad": 14,
        "axes.labelsize": 10,
        "axes.labelweight": "500",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": GRID_COLOR,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica Neue", "Helvetica"],
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.55,
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def style_axes(ax, *, grid_axis: str | None = "y") -> None:
    """Minimal editorial axes — hide clutter, soft grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="-", alpha=0.45, color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)


def style_legend(ax, **kwargs) -> None:
    """Refined legend defaults."""
    defaults = {"frameon": True, "fancybox": False, "edgecolor": GRID_COLOR, "framealpha": 0.95}
    defaults.update(kwargs)
    leg = ax.get_legend()
    if leg:
        leg.set_frame_on(defaults.pop("frameon", True))


def add_figure_title(fig, title: str, subtitle: str | None = None) -> None:
    """Editorial figure title with optional subtitle."""
    fig.suptitle(title, fontsize=14, fontweight="600", color=TEXT_PRIMARY, y=0.98)
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=9.5, color=TEXT_MUTED, style="italic")
