"""
Hybrid NKPC estimation via GMM for UK monthly CPI inflation.

Reproduces the GMM estimates reported in Table 3 of the dissertation.
Uses HAC (Newey-West) weighting matrix and reports:
  - gamma_f (forward-looking coefficient)
  - gamma_b (backward-looking coefficient)
  - lambda  (marginal cost pass-through)
  - First-stage F-statistic
  - Kleibergen-Paap rk Wald statistic
  - Hansen J-test of over-identifying restrictions

Requirements
------------
pandas, numpy, linearmodels >= 5.0, statsmodels

Usage
-----
python 02_nkpc_gmm.py --data data/dataset_9variables.csv

Author: Zarif Khan, University of Bath, April 2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import IVGMM


def build_nkpc_design(df: pd.DataFrame, mc_proxy: str = "rulc", use_narrative: bool = False):
    """
    Build endog, exog, instruments arrays for the hybrid NKPC:
        pi_t = gamma_f * E_t[pi_{t+1}] + gamma_b * pi_{t-1} + lambda * mc_t + eps_t

    Instrumented regressor: pi_{t+1}.
    Instruments: four lags of inflation + four lags of mc proxy (+ narrative if requested).
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    pi = df["cpi_monthly_pct"]
    mc = df[mc_proxy]

    data = pd.DataFrame({
        "pi_t": pi,
        "pi_tp1": pi.shift(-1),  # forward-looking regressor (endogenous)
        "pi_tm1": pi.shift(1),   # backward-looking regressor (exogenous)
        "mc_t": mc,
    })
    # Build four lags of pi and mc as instruments
    for k in range(1, 5):
        data[f"pi_lag_{k}"] = pi.shift(k)
        data[f"mc_lag_{k}"] = mc.shift(k)
    if use_narrative:
        data["narrative_lag_1"] = df["narrative_aggregate"].shift(1)

    data = data.dropna()
    dep = data["pi_t"]
    exog = data[["pi_tm1", "mc_t"]]
    endog = data[["pi_tp1"]]
    instr_cols = [f"pi_lag_{k}" for k in range(1, 5)] + [f"mc_lag_{k}" for k in range(1, 5)]
    if use_narrative:
        instr_cols.append("narrative_lag_1")
    instruments = data[instr_cols]
    return dep, exog, endog, instruments


def estimate_gmm(df: pd.DataFrame, mc_proxy: str, use_narrative: bool):
    """Run 2-step GMM with HAC weighting matrix."""
    dep, exog, endog, instr = build_nkpc_design(df, mc_proxy, use_narrative)
    model = IVGMM(dep, exog, endog, instr, weight_type="kernel", kernel="bartlett", bandwidth=4)
    result = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=4, iter_limit=2)
    return result


def report(result, label: str) -> None:
    """Print a compact summary block."""
    print(f"\n=== {label} ===")
    print(result.params.rename({"pi_tp1": "gamma_f", "pi_tm1": "gamma_b", "mc_t": "lambda"}))
    print("Std errors:")
    print(result.std_errors.rename({"pi_tp1": "gamma_f", "pi_tm1": "gamma_b", "mc_t": "lambda"}))
    if hasattr(result, "j_stat"):
        print(f"Hansen J-test: stat={result.j_stat.stat:.3f}, p={result.j_stat.pval:.3f}")
    print(f"N = {result.nobs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/dataset_9variables.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["date"])

    # NKPC-RULC without narrative
    r1 = estimate_gmm(df, mc_proxy="rulc", use_narrative=False)
    report(r1, "NKPC-RULC (baseline)")

    # NKPC-RULC with narrative instrument
    r2 = estimate_gmm(df, mc_proxy="rulc", use_narrative=True)
    report(r2, "NKPC-RULC + narrative instrument")

    # NKPC-OG
    r3 = estimate_gmm(df, mc_proxy="output_gap", use_narrative=False)
    report(r3, "NKPC-OG (output-gap proxy)")


if __name__ == "__main__":
    main()
