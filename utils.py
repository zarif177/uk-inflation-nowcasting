"""
Shared utilities used across estimation and evaluation scripts.

Author: Zarif Khan, University of Bath, April 2026.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path: str | Path = "data/dataset_9variables.csv") -> pd.DataFrame:
    """Load the nine-variable monthly dataset, parsing dates and setting a monotonic index."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def hp_filter(series: pd.Series, lamb: float = 129_600) -> pd.Series:
    """HP filter (monthly default smoothing parameter 129,600, per Ravn and Uhlig 2002)."""
    from statsmodels.tsa.filters.hp_filter import hpfilter

    cycle, _ = hpfilter(series.dropna(), lamb=lamb)
    return cycle


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def significance_stars(p_value: float) -> str:
    """Return conventional significance markers."""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""
