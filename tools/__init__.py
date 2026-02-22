"""tools/__init__.py — export all LangChain tools."""

from .data_tools import (
    inspect_schema,
    detect_missing_values,
    handle_missing_values,
    compute_statistics,
)
from .viz_tools import (
    plot_distributions,
    plot_correlation_heatmap,
    plot_boxplots,
    plot_categorical_bars,
)
from .report_tool import generate_html_report

ALL_TOOLS = [
    inspect_schema,
    detect_missing_values,
    handle_missing_values,
    compute_statistics,
    plot_distributions,
    plot_correlation_heatmap,
    plot_boxplots,
    plot_categorical_bars,
    generate_html_report,
]

__all__ = ["ALL_TOOLS"] + [t.name for t in ALL_TOOLS]