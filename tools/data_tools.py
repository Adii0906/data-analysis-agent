"""
tools/data_tools.py  —  fixed tool signatures & absolute paths
"""

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.tools import tool

from exceptions import (
    CleaningError,
    DataLoadError,
    FileOperationError,
    MissingValueError,
    SchemaInspectionError,
)

logger = logging.getLogger(__name__)

# Always resolve relative to THIS file so paths work regardless of cwd
BASE_DIR = Path(__file__).resolve().parent.parent
TMP_DIR  = BASE_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    try:
        if p.suffix == ".csv":
            return pd.read_csv(p)
        elif p.suffix in (".xlsx", ".xls"):
            return pd.read_excel(p)
        elif p.suffix == ".json":
            return pd.read_json(p)
        elif p.suffix == ".parquet":
            return pd.read_parquet(p)
        else:
            raise DataLoadError(f"Unsupported file type: {p.suffix}", {"path": str(p)})
    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"Failed to load {path}: {e}", {"path": path}, e)


def _save_df(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    try:
        if p.suffix == ".csv":
            df.to_csv(p, index=False)
        elif p.suffix in (".xlsx", ".xls"):
            df.to_excel(p, index=False)
        elif p.suffix == ".parquet":
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)
    except Exception as e:
        raise FileOperationError(f"Failed to save dataframe to {path}", {"path": path}, e)




@tool
def inspect_schema(file_path: str) -> str:
    """Inspects dataset schema: columns, dtypes, shape, sample rows. Returns JSON summary."""
    try:
        df = _load_df(file_path)
        schema = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "non_null_count": int(df[col].count()),
                    "unique_values": int(df[col].nunique()),
                }
                for col in df.columns
            ],
            "sample": df.head(3).to_dict(orient="records"),
            "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        }
        return json.dumps(schema, default=str)
    except Exception as e:
        raise SchemaInspectionError(f"Schema inspection failed: {e}", {"file_path": file_path}, e)


@tool
def detect_missing_values(file_path: str) -> str:
    """Detects missing values per column. Returns count, percentage, and affected columns as JSON."""
    try:
        df = _load_df(file_path)
        missing = df.isnull().sum()
        total   = len(df)
        report  = {
            "total_rows": total,
            "total_missing_cells": int(missing.sum()),
            "columns_with_missing": [
                {
                    "column":      col,
                    "missing_count": int(missing[col]),
                    "missing_pct": round(missing[col] / total * 100, 2),
                    "dtype":       str(df[col].dtype),
                }
                for col in df.columns if missing[col] > 0
            ],
        }
        return json.dumps(report)
    except Exception as e:
        raise MissingValueError(f"Missing value detection failed: {e}", {"file_path": file_path}, e)


@tool
def handle_missing_values(file_path: str, strategy: str) -> str:
    """
    Handles missing values. strategy: 'mean' | 'median' | 'drop'.
    Saves cleaned file in place and returns a summary JSON.
    """
    valid = {"mean", "median", "drop"}
    if strategy not in valid:
        raise CleaningError(f"Invalid strategy '{strategy}'. Choose from {valid}", {"strategy": strategy})
    try:
        df     = _load_df(file_path)
        before = int(df.isnull().sum().sum())

        if strategy == "drop":
            df = df.dropna()
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            fill_val = df[num_cols].mean() if strategy == "mean" else df[num_cols].median()
            df[num_cols] = df[num_cols].fillna(fill_val)
            for col in df.select_dtypes(exclude=[np.number]).columns:
                if df[col].isnull().any():
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")

        after = int(df.isnull().sum().sum())
        _save_df(df, file_path)
        return json.dumps({
            "status":          "success",
            "strategy_used":   strategy,
            "missing_before":  before,
            "missing_after":   after,
            "rows_remaining":  len(df),
            "file_path":       file_path,
        })
    except CleaningError:
        raise
    except Exception as e:
        raise CleaningError(f"Cleaning failed: {e}", {"file_path": file_path, "strategy": strategy}, e)


@tool
def compute_statistics(file_path: str) -> str:
    """Computes descriptive stats, correlations, skewness, kurtosis, and IQR outlier counts."""
    try:
        df     = _load_df(file_path)
        num_df = df.select_dtypes(include=[np.number])

        describe     = json.loads(num_df.describe().to_json())
        correlations = json.loads(num_df.corr().to_json()) if len(num_df.columns) > 1 else {}
        skewness     = {c: round(float(num_df[c].skew()), 4) for c in num_df.columns}
        kurtosis     = {c: round(float(num_df[c].kurt()), 4) for c in num_df.columns}

        outliers = {}
        for col in num_df.columns:
            q1, q3 = num_df[col].quantile(0.25), num_df[col].quantile(0.75)
            iqr = q3 - q1
            outliers[col] = int(((num_df[col] < q1 - 1.5 * iqr) | (num_df[col] > q3 + 1.5 * iqr)).sum())

        cat_summary = {}
        for col in df.select_dtypes(exclude=[np.number]).columns:
            cat_summary[col] = json.loads(df[col].value_counts().head(5).to_json())

        return json.dumps({
            "descriptive_stats":   describe,
            "correlations":        correlations,
            "skewness":            skewness,
            "kurtosis":            kurtosis,
            "outlier_counts_iqr":  outliers,
            "categorical_top5":    cat_summary,
        }, default=str)
    except Exception as e:
        from exceptions import ToolExecutionError
        raise ToolExecutionError(f"Statistics computation failed: {e}", {"file_path": file_path}, e)