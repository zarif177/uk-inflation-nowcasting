# Improving UK Inflation Nowcasting Through Textual Features

**BSc Economics Dissertation — University of Bath**
**Author:** Zarif Khan
**Supervisor:** Dr Imran Shah
**Submitted:** 24 April 2026

A comparison of the hybrid New Keynesian Phillips Curve (NKPC), frequentist Vector Autoregression (VAR), and Bayesian VAR (BVAR), augmented with FinBERT-derived sentiment from Bank of England MPC minutes, for nowcasting UK CPI inflation.

---

## Overview

This repository contains the full replication materials for the dissertation, including the dataset, estimation code, FinBERT pipeline, and figures. The study compares four nowcasting frameworks for UK monthly CPI inflation over the January 2019 to December 2024 evaluation window:

1. **Hybrid New Keynesian Phillips Curve (NKPC)** estimated by GMM with RULC and output gap as alternative marginal cost proxies
2. **Frequentist Vector Autoregression (VAR)** in bivariate and trivariate specifications
3. **Bayesian VAR (BVAR)** with Minnesota priors
4. **Random walk benchmark**

A monthly Narrative Inflation Index is constructed by applying FinBERT to 140 Bank of England Monetary Policy Committee minutes (2010–2024) and used to augment both structural and reduced-form frameworks. Three thematic sub-indices (energy, wage, supply chain) decompose the aggregate signal.

## Key findings

- The frequentist bivariate VAR achieves the lowest RMSE, an 18.1% improvement over the random walk
- The Narrative Inflation Index significantly improves the NKPC (DM = 2.387, p = 0.017) but adds no significant value to the frequentist VAR
- The BVAR with Minnesota priors significantly outperforms the NKPC while nearly matching the random walk
- RULC outperforms the output gap as the NKPC marginal cost proxy, validating Batini, Jackson and Nickell (2005)

## Repository structure

```
.
├── README.md                         — this file
├── requirements.txt                  — Python dependencies
├── .gitignore
├── data/
│   └── dataset_9variables.csv        — monthly dataset, January 2010 to December 2024
│
├── code/
│   ├── finbert_pipeline.py           — FinBERT sentiment extraction from MPC minutes
│   ├── stationarity_tests.py         — ADF, PP, ERS/DFGLS unit root tests
│   ├── nkpc_estimation.py            — GMM estimation of hybrid NKPC
│   ├── var_bvar_estimation.py        — Frequentist VAR and BVAR estimation
│   ├── rolling_evaluation.py         — Rolling-origin forecast evaluation + Diebold-Mariano
│   ├── generate_figures.py           — Reproduce Figures 1, 2, 5, 6 of the dissertation
│   └── utils.py                      — Shared helpers
│
└── figures/
    ├── figure1_uk_macro_variables.png
    ├── figure2_narrative_indices.png
    ├── figure3_methodology_overview.png
    ├── figure4_finbert_pipeline.png
    ├── figure5_actual_vs_nowcast.png
    ├── figure6_cumulative_errors.png
    └── figure7_contributions_schematic.png
```

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/zarif177/uk-inflation-nowcasting.git
cd uk-inflation-nowcasting
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Reproduce the core results

```bash
# Create the output folder first time
mkdir -p outputs

# Step 1: check stationarity of all variables
python code/stationarity_tests.py

# Step 2: estimate the hybrid NKPC via GMM
python code/nkpc_estimation.py --data data/dataset_9variables.csv

# Step 3: run the frequentist VAR + BVAR rolling-origin nowcasts
python code/var_bvar_estimation.py --data data/dataset_9variables.csv

# Step 4: evaluate accuracy (RMSE, MAE, Diebold-Mariano)
python code/rolling_evaluation.py --nowcasts outputs/nowcasts_var.csv

# Step 5: regenerate the empirical figures
python code/generate_figures.py --data data/dataset_9variables.csv
```

The full FinBERT pipeline (`code/finbert_pipeline.py`) requires the raw MPC minutes corpus, which is not redistributed here but can be downloaded from the [Bank of England website](https://www.bankofengland.co.uk/monetary-policy/monetary-policy-committee). The resulting narrative indices are already embedded in `data/dataset_9variables.csv` so the pipeline only needs to be re-run if you want to regenerate them from scratch.

## Data

The dataset (`data/dataset_9variables.csv`) covers January 2010 to December 2024 at monthly frequency (180 observations) and contains nine variables:

| Variable | Source | Description |
|---|---|---|
| `date` | — | Month-year (YYYY-MM-01) |
| `cpi_monthly_pct` | ONS (2024), series D7G7 | Monthly percentage change in CPI |
| `bank_rate` | BoE (2024) | Official Bank Rate, level |
| `d_bank_rate` | BoE (2024) | First difference of Bank Rate |
| `output_gap` | ONS (2024), series K222 | HP-filtered log-deviation of Index of Production |
| `rulc` | ONS (2024) | HP-filtered cyclical component of whole-economy unit labour costs |
| `narrative_aggregate` | 140 MPC minutes, FinBERT | Mean polarity score across all inflation-related paragraphs |
| `energy_subindex` | 140 MPC minutes, FinBERT | Polarity score for energy-related paragraphs |
| `wage_subindex` | 140 MPC minutes, FinBERT | Polarity score for wage-related paragraphs |
| `supply_subindex` | 140 MPC minutes, FinBERT | Polarity score for supply chain-related paragraphs |

## FinBERT pipeline

The `code/finbert_pipeline.py` script applies the `ProsusAI/finbert` model from Hugging Face to tokenised MPC minutes, computes a polarity score (P(positive) − P(negative)) for every inflation-related paragraph, and aggregates the scores to monthly frequency with forward-fill for non-meeting months. Keyword filters identify energy, wage, and supply chain paragraphs for the three thematic sub-indices. The full pipeline is documented in Section 3.5 of the dissertation and illustrated in Figure 4.

## Citation

```
Khan, Z. (2026) 'Improving UK Inflation Nowcasting Through Textual Features:
A Comparison of the New Keynesian Phillips Curve and Vector Autoregression Models'.
BSc Economics dissertation, University of Bath.
```


## Contact

For questions about this work, please contact the author via the University of Bath Department of Economics.
