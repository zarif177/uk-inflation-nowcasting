"""
Stationarity Tests for UK Macroeconomic and Narrative Variables
================================================================

Implements three unit root tests used in Section 4.4 of the dissertation:

    1. Augmented Dickey-Fuller (ADF)
    2. Phillips-Perron (PP)
    3. Elliott-Rothenberg-Stock DF-GLS

Tests are run on all nine variables in the dataset at both levels and
first differences, following the standard protocol in Enders (2014).

Usage:
    python stationarity_tests.py
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron, DFGLS


DATA_PATH = "data/dataset_9variables.csv"

VARIABLES = [
    "cpi_monthly_pct",
    "bank_rate",
    "d_bank_rate",
    "output_gap",
    "rulc",
    "narrative_aggregate",
    "energy_subindex",
    "wage_subindex",
    "supply_subindex",
]


def run_all_tests(series: pd.Series, name: str) -> dict:
    """Run ADF, PP, and DF-GLS tests on a single series.

    Returns a dict with test statistics and p-values for each test.
    Uses constant-only regression (no trend).
    """
    series = series.dropna()

    # ADF
    adf_result = adfuller(series, autolag="AIC", regression="c")

    # Phillips-Perron
    pp = PhillipsPerron(series, trend="c")

    # DF-GLS (Elliott-Rothenberg-Stock)
    dfgls = DFGLS(series, trend="c")

    return {
        "variable": name,
        "n_obs": len(series),
        "adf_stat": adf_result[0],
        "adf_pvalue": adf_result[1],
        "pp_stat": pp.stat,
        "pp_pvalue": pp.pvalue,
        "dfgls_stat": dfgls.stat,
        "dfgls_pvalue": dfgls.pvalue,
    }


def summarise_results(df: pd.DataFrame) -> pd.DataFrame:
    """Add significance markers (*/**/***) at 10/5/1% levels."""

    def marker(p: float) -> str:
        if p < 0.01:
            return "***"
        if p < 0.05:
            return "**"
        if p < 0.10:
            return "*"
        return ""

    for test in ["adf", "pp", "dfgls"]:
        df[f"{test}_sig"] = df[f"{test}_pvalue"].apply(marker)
    return df


def main() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date")
    results = []

    for var in VARIABLES:
        if var not in data.columns:
            print(f"  skipping {var} (not in dataset)")
            continue
        results.append(run_all_tests(data[var], var))

    df = pd.DataFrame(results)
    df = summarise_results(df)

    print("\nStationarity test results (levels):")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
