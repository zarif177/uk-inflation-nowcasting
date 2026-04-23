"""
Frequentist VAR and Bayesian VAR with Minnesota priors for UK CPI nowcasting.

Estimates both bivariate and trivariate specifications and produces rolling-origin
out-of-sample nowcasts for the January 2019 to December 2024 evaluation window.

Requirements
------------
pandas, numpy, statsmodels >= 0.14 (for VAR), pymc >= 5 (for BVAR)

Usage
-----
python 03_var_bvar_estimation.py --data data/dataset_9variables.csv

Author: Zarif Khan, University of Bath, April 2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


VAR_LAGS = 12  # selected by SBIC/HQ for monthly data
EVAL_START = "2019-01-01"
EVAL_END = "2024-12-31"


def fit_frequentist_var(df: pd.DataFrame, variables: list[str], lags: int = VAR_LAGS):
    """Fit a frequentist VAR(p) by OLS."""
    model = VAR(df[variables].dropna())
    result = model.fit(maxlags=lags, ic=None)
    return result


def minnesota_prior_var(df: pd.DataFrame, variables: list[str], lags: int = 6,
                       lambda1: float = 0.1, lambda2: float = 0.5):
    """
    Bayesian VAR with Minnesota prior, solved analytically via dummy observations
    (Banbura, Giannone and Reichlin 2010; Cimadomo et al. 2022).

    Prior on coefficient of variable j in equation i at lag l:
        mean  = 1 if (i==j and l==1) else 0
        var   = (lambda1 / l)^2                    for own lags
              = (lambda1 * lambda2 / l * sig_i/sig_j)^2  for cross lags
    """
    Y = df[variables].dropna().values
    n, k = Y.shape

    # AR(1) residual std deviations for prior scaling
    sigmas = []
    for j in range(k):
        y_j = Y[:, j]
        rho = np.corrcoef(y_j[:-1], y_j[1:])[0, 1]
        eps = y_j[1:] - rho * y_j[:-1]
        sigmas.append(np.std(eps, ddof=1))
    sigmas = np.array(sigmas)

    # Build dummy observations that implement the Minnesota prior
    # For each coef c_{ij,l}, add a synthetic observation that pulls c toward prior mean
    # (see Banbura et al. 2010 Appendix)
    # For brevity this is a schematic implementation; a production version uses
    # the full Karlsson (2013) handbook algebra with Normal-Inverse-Wishart hyperpriors.
    # The pedagogical purpose here is to document the pipeline, not re-implement
    # a fully-tuned BVAR library.

    # Return the OLS fit augmented with the dummies.
    # ... (see paper for derivation)
    raise NotImplementedError(
        "Use bvartools (R) or pymc BVAR sampler for production estimation. "
        "This stub documents the Minnesota-prior choice; full code available in "
        "the accompanying notebook notebooks/03_bvar_estimation.ipynb."
    )


def rolling_origin_forecast(df: pd.DataFrame, variables: list[str], lags: int,
                            eval_start: str = EVAL_START,
                            eval_end: str = EVAL_END) -> pd.Series:
    """Generate one-step-ahead nowcasts for the evaluation window."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    df.set_index("date", inplace=True)

    eval_idx = pd.date_range(eval_start, eval_end, freq="MS")
    preds = []
    for date in eval_idx:
        train = df.loc[df.index < date, variables].dropna()
        if len(train) <= lags + 5:
            preds.append(np.nan)
            continue
        model = VAR(train).fit(lags)
        next_values = model.forecast(train.values[-lags:], steps=1)
        preds.append(next_values[0, 0])  # position 0 = pi (target)
    return pd.Series(preds, index=eval_idx, name="nowcast")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/dataset_9variables.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["date"])

    # Bivariate VAR: (pi, d_bank_rate)
    bivar = rolling_origin_forecast(df, ["cpi_monthly_pct", "d_bank_rate"], lags=VAR_LAGS)

    # Trivariate VAR adds narrative aggregate
    trivar = rolling_origin_forecast(
        df, ["cpi_monthly_pct", "d_bank_rate", "narrative_aggregate"], lags=VAR_LAGS
    )

    # Realised target
    realised = df.set_index("date")["cpi_monthly_pct"].loc[EVAL_START:EVAL_END]

    print(f"\nBivariate VAR:  RMSE={rmse(realised.loc[bivar.index], bivar):.3f}  "
          f"MAE={mae(realised.loc[bivar.index], bivar):.3f}")
    print(f"Trivariate VAR: RMSE={rmse(realised.loc[trivar.index], trivar):.3f}  "
          f"MAE={mae(realised.loc[trivar.index], trivar):.3f}")

    # Persist nowcasts for downstream evaluation and plotting
    out = pd.DataFrame({
        "date": bivar.index,
        "realised": realised.loc[bivar.index].values,
        "var_bivariate_nowcast": bivar.values,
        "var_trivariate_nowcast": trivar.values,
    })
    out.to_csv("outputs/nowcasts_var.csv", index=False)
    print("Wrote outputs/nowcasts_var.csv")


if __name__ == "__main__":
    main()
