"""
Rolling-origin out-of-sample evaluation.

Computes RMSE, MAE, and Diebold-Mariano test statistics for each candidate model
over the January 2019 to December 2024 evaluation window.

Requirements
------------
pandas, numpy, scipy, statsmodels

Usage
-----
python 04_rolling_origin_evaluation.py --nowcasts outputs/nowcasts_var.csv

Author: Zarif Khan, University of Bath, April 2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """
    Diebold-Mariano test on two sequences of one-step-ahead forecast errors.

    Null hypothesis: the two forecasts have equal predictive accuracy
    under squared-error loss.

    Returns (DM_statistic, two-sided p-value).
    Uses the Harvey, Leybourne and Newbold (1997) small-sample correction.
    """
    d = err_a**2 - err_b**2
    T = len(d)
    mean_d = np.mean(d)

    # Long-run variance with bandwidth h-1 (Newey-West)
    gamma_0 = np.var(d, ddof=0)
    lrv = gamma_0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        lrv += 2 * gamma_k

    dm = mean_d / np.sqrt(lrv / T)
    # Harvey-Leybourne-Newbold small-sample correction
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_corrected = dm * correction
    p_value = 2 * (1 - stats.t.cdf(abs(dm_corrected), df=T - 1))
    return float(dm_corrected), float(p_value)


def evaluate(nowcasts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE, MAE for each nowcast column against 'realised'."""
    realised = nowcasts_df["realised"].values
    rows = []
    for col in nowcasts_df.columns:
        if col in {"date", "realised"}:
            continue
        preds = nowcasts_df[col].values
        err = realised - preds
        rmse_val = float(np.sqrt(np.mean(err**2)))
        mae_val = float(np.mean(np.abs(err)))
        rows.append({"model": col, "RMSE": rmse_val, "MAE": mae_val})
    return pd.DataFrame(rows).sort_values("RMSE")


def dm_matrix(nowcasts_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Diebold-Mariano table for every pair of nowcast columns."""
    realised = nowcasts_df["realised"].values
    model_cols = [c for c in nowcasts_df.columns if c not in {"date", "realised"}]
    rows = []
    for i, a in enumerate(model_cols):
        for b in model_cols[i + 1:]:
            err_a = realised - nowcasts_df[a].values
            err_b = realised - nowcasts_df[b].values
            stat, p = diebold_mariano(err_a, err_b)
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            rows.append({"A": a, "B": b, "DM_stat": stat, "p_value": p, "sig": sig})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nowcasts", type=Path, default=Path("outputs/nowcasts_var.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.nowcasts, parse_dates=["date"])

    summary = evaluate(df)
    print("\n=== Accuracy summary ===")
    print(summary.to_string(index=False))

    dm_df = dm_matrix(df)
    print("\n=== Diebold-Mariano pairwise tests ===")
    print(dm_df.to_string(index=False))

    summary.to_csv("outputs/accuracy_summary.csv", index=False)
    dm_df.to_csv("outputs/dm_tests.csv", index=False)
    print("\nSaved to outputs/accuracy_summary.csv and outputs/dm_tests.csv")


if __name__ == "__main__":
    main()
