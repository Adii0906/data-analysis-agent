"""
tools/viz_tools.py  —  chart generation with absolute output paths
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from langchain.tools import tool

from exceptions import VisualizationError

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TMP_DIR  = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
PAL      = ["#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4"]
BG       = "#0A0E1A"
SURF     = "#111827"
GRID_C   = "#1F2D45"
TEXT_C   = "#F1F5F9"
MUTED_C  = "#64748B"


def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":            return pd.read_csv(p)
    elif ext in (".xlsx",".xls"): return pd.read_excel(p)
    elif ext == ".json":          return pd.read_json(p)
    elif ext == ".parquet":       return pd.read_parquet(p)
    return pd.read_csv(p)


def _style(ax, title: str):
    ax.set_facecolor(SURF)
    ax.figure.patch.set_facecolor(BG)
    ax.set_title(title, color=TEXT_C, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors=MUTED_C, labelsize=8)
    ax.xaxis.label.set_color(MUTED_C)
    ax.yaxis.label.set_color(MUTED_C)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_C)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


def _grid_fig(n_plots: int, cols: int = 3):
    rows = max(1, (n_plots + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.patch.set_facecolor(BG)
    axes = np.array(axes).flatten()
    return fig, axes


@tool
def plot_distributions(file_path: str) -> str:
    """Histogram + KDE distribution plots for all numeric columns. Returns JSON with path."""
    try:
        df       = _load_df(file_path)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return json.dumps({"status": "skipped", "reason": "No numeric columns"})

        fig, axes = _grid_fig(len(num_cols))
        for i, col in enumerate(num_cols):
            ax = axes[i]; ax.set_facecolor(SURF)
            sns.histplot(df[col].dropna(), ax=ax, kde=True,
                         color=PAL[i % len(PAL)], edgecolor="none", alpha=0.8)
            _style(ax, col)
        for j in range(len(num_cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Variable Distributions", color=TEXT_C, fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = TMP_DIR / "distributions.png"
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
        plt.close()
        return json.dumps({"status": "success", "path": str(out)})
    except Exception as e:
        raise VisualizationError(str(e), {"file_path": file_path}, e)


@tool
def plot_correlation_heatmap(file_path: str) -> str:
    """Correlation heatmap for numeric columns. Returns JSON with path."""
    try:
        df     = _load_df(file_path)
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return json.dumps({"status": "skipped", "reason": "Need ≥2 numeric columns"})

        size = max(7, len(num_df.columns))
        fig, ax = plt.subplots(figsize=(size, size * 0.75))
        fig.patch.set_facecolor(BG); ax.set_facecolor(SURF)

        mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
        sns.heatmap(num_df.corr(), mask=mask, ax=ax,
                    cmap=sns.diverging_palette(220, 20, as_cmap=True),
                    annot=True, fmt=".2f", annot_kws={"size": 8, "color": TEXT_C},
                    linewidths=0.4, linecolor=GRID_C,
                    cbar_kws={"shrink": 0.75})
        ax.set_title("Correlation Matrix", color=TEXT_C, fontsize=13, fontweight="bold", pad=12)
        ax.tick_params(colors=MUTED_C, labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color=MUTED_C)
        plt.setp(ax.get_yticklabels(), color=MUTED_C)
        cbar = ax.collections[0].colorbar
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED_C)

        plt.tight_layout()
        out = TMP_DIR / "correlation_heatmap.png"
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
        plt.close()
        return json.dumps({"status": "success", "path": str(out)})
    except Exception as e:
        raise VisualizationError(str(e), {"file_path": file_path}, e)


@tool
def plot_boxplots(file_path: str) -> str:
    """Box plots for outlier analysis. Returns JSON with path."""
    try:
        df       = _load_df(file_path)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return json.dumps({"status": "skipped", "reason": "No numeric columns"})

        fig, axes = _grid_fig(len(num_cols))
        for i, col in enumerate(num_cols):
            ax = axes[i]; ax.set_facecolor(SURF)
            ax.boxplot(df[col].dropna(), patch_artist=True, notch=False,
                       boxprops=dict(facecolor=PAL[i % len(PAL)], alpha=0.7),
                       medianprops=dict(color=TEXT_C, linewidth=2),
                       whiskerprops=dict(color=GRID_C),
                       capprops=dict(color=GRID_C),
                       flierprops=dict(marker="o", color=PAL[(i+1)%len(PAL)], markersize=3))
            _style(ax, col)
        for j in range(len(num_cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Outlier Analysis — Box Plots", color=TEXT_C, fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = TMP_DIR / "boxplots.png"
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
        plt.close()
        return json.dumps({"status": "success", "path": str(out)})
    except Exception as e:
        raise VisualizationError(str(e), {"file_path": file_path}, e)


@tool
def plot_categorical_bars(file_path: str) -> str:
    """Horizontal bar charts for top-10 values of categorical columns. Returns JSON with path."""
    try:
        df       = _load_df(file_path)
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()[:6]
        if not cat_cols:
            return json.dumps({"status": "skipped", "reason": "No categorical columns"})

        fig, axes = _grid_fig(len(cat_cols), cols=2)
        for i, col in enumerate(cat_cols):
            ax = axes[i]; ax.set_facecolor(SURF)
            vc = df[col].value_counts().head(10)
            colors = [PAL[j % len(PAL)] for j in range(len(vc))]
            ax.barh(vc.index.astype(str)[::-1], vc.values[::-1],
                    color=colors[::-1], edgecolor="none")
            _style(ax, col)
        for j in range(len(cat_cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Categorical Analysis", color=TEXT_C, fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = TMP_DIR / "categorical_bars.png"
        plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=BG)
        plt.close()
        return json.dumps({"status": "success", "path": str(out)})
    except Exception as e:
        raise VisualizationError(str(e), {"file_path": file_path}, e)