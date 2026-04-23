"""
Produces the empirical figures from the dissertation (Figures 1, 2, 5, 6).

Requirements
------------
pandas, numpy, matplotlib

Usage
-----
python 05_generate_figures.py --data data/dataset_9variables.csv

Author: Zarif Khan, University of Bath, April 2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (9, 5),
})


def figure_1_macro(df: pd.DataFrame, out: Path) -> None:
    """Figure 1: UK Macroeconomic Variables (2010-2024)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    df = df.set_index("date")

    axes[0, 0].plot(df.index, df["cpi_monthly_pct"], color="#1f4e79")
    axes[0, 0].set_title("CPI Inflation (monthly %)")
    axes[0, 1].plot(df.index, df["bank_rate"], color="#8b2a3a")
    axes[0, 1].set_title("Bank of England Bank Rate (%)")
    axes[1, 0].plot(df.index, df["rulc"], color="#2E7D32")
    axes[1, 0].set_title("Real Unit Labour Costs (% deviation)")
    axes[1, 1].plot(df.index, df["output_gap"], color="#c77700")
    axes[1, 1].set_title("Output Gap (%)")

    # Shade COVID-19 and energy crisis
    for ax in axes.ravel():
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-01"),
                   color="red", alpha=0.15, label="COVID-19")
        ax.axvspan(pd.Timestamp("2022-02-01"), pd.Timestamp("2023-06-01"),
                   color="orange", alpha=0.15, label="Energy crisis")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def figure_2_narrative(df: pd.DataFrame, out: Path) -> None:
    """Figure 2: Narrative Inflation Indices from BoE MPC minutes."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    df = df.set_index("date")
    ax.plot(df.index, df["narrative_aggregate"], label="Aggregate", linewidth=2, color="#1f4e79")
    ax.plot(df.index, df["energy_subindex"], label="Energy", alpha=0.7, color="#c77700")
    ax.plot(df.index, df["wage_subindex"], label="Wage", alpha=0.7, color="#2E7D32")
    ax.plot(df.index, df["supply_subindex"], label="Supply chain", alpha=0.7, color="#8b2a3a")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Polarity (P(hawkish) - P(dovish))")
    ax.set_title("Granular Narrative Inflation Indices from BoE MPC Minutes (2010-2024)")
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def figure_5_nowcasts(nowcasts_path: Path, out: Path) -> None:
    """Figure 5: Actual vs Nowcast Inflation (Jan 2019 to Dec 2024)."""
    if not nowcasts_path.exists():
        print(f"Skipping Figure 5: {nowcasts_path} does not exist. "
              f"Run 03_var_bvar_estimation.py first.")
        return
    df = pd.read_csv(nowcasts_path, parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(df.index, df["realised"], color="black", linewidth=2, label="Realised")
    for col in df.columns:
        if col == "realised":
            continue
        ax.plot(df.index, df[col], linestyle="--", alpha=0.8, label=col.replace("_", " ").title())
    ax.set_ylabel("Monthly CPI inflation (%)")
    ax.set_title("Actual vs Nowcast Inflation (January 2019 to December 2024)")
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-01"),
               color="red", alpha=0.12)
    ax.axvspan(pd.Timestamp("2022-02-01"), pd.Timestamp("2023-06-01"),
               color="orange", alpha=0.12)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def figure_6_cumulative_errors(nowcasts_path: Path, out: Path) -> None:
    """Figure 6: Cumulative Squared Forecast Errors."""
    if not nowcasts_path.exists():
        print(f"Skipping Figure 6: {nowcasts_path} does not exist.")
        return
    df = pd.read_csv(nowcasts_path, parse_dates=["date"]).set_index("date")
    realised = df["realised"].values
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for col in df.columns:
        if col == "realised":
            continue
        sq_err = (realised - df[col].values) ** 2
        ax.plot(df.index, np.cumsum(sq_err), label=col.replace("_", " ").title())
    ax.set_ylabel("Cumulative squared forecast error")
    ax.set_title("Cumulative Squared Forecast Errors by Model")
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/dataset_9variables.csv"))
    parser.add_argument("--nowcasts", type=Path, default=Path("outputs/nowcasts_var.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, parse_dates=["date"])

    figure_1_macro(df, args.out_dir / "figure1_uk_macro_variables.png")
    figure_2_narrative(df, args.out_dir / "figure2_narrative_indices.png")
    figure_5_nowcasts(args.nowcasts, args.out_dir / "figure5_actual_vs_nowcast.png")
    figure_6_cumulative_errors(args.nowcasts, args.out_dir / "figure6_cumulative_errors.png")


if __name__ == "__main__":
    main()
