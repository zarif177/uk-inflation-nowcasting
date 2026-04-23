"""
FinBERT pipeline for constructing the Narrative Inflation Index from Bank of England MPC minutes.

Applies the ProsusAI/finbert model from Hugging Face to tokenised MPC minutes and
produces:
  (i) a monthly aggregate narrative index (polarity in [-1, +1]),
  (ii) three thematic sub-indices (energy, wage, supply chain).

Requirements
------------
transformers >= 4.30
torch >= 2.0
pandas
numpy
beautifulsoup4
requests

Usage
-----
python 01_finbert_pipeline.py \\
    --minutes_dir path/to/mpc_minutes/ \\
    --output data/narrative_indices.csv

Author: Zarif Khan, University of Bath, April 2026.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 512

# Keyword filters for the four indices.
# Aggregate: any paragraph containing inflation-related vocabulary.
# Thematic sub-indices: paragraphs matching topic-specific keywords.
KEYWORDS = {
    "aggregate": [
        "inflation", "CPI", "price", "prices", "cost", "costs",
    ],
    "energy": [
        "oil", "gas", "electricity", "energy", "Ofgem", "utility", "utilities",
    ],
    "wage": [
        "wage", "wages", "pay", "earnings", "employment", "vacancies",
        "AWE", "labour market",
    ],
    "supply": [
        "shipping", "freight", "exchange rate", "import", "imports",
        "supply chain", "bottleneck",
    ],
}


def load_model(model_name: str = MODEL_NAME):
    """Load tokeniser + classifier from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, strip, drop empty fragments."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 30]


def filter_paragraphs(paragraphs: list[str], keywords: list[str]) -> list[str]:
    """Return paragraphs that match any keyword (case-insensitive)."""
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, keywords)) + r")\b", re.IGNORECASE)
    return [p for p in paragraphs if pattern.search(p)]


@torch.no_grad()
def polarity_score(paragraph: str, tokenizer, model) -> float:
    """Return P(positive) - P(negative) for a single paragraph, with truncation to 512 tokens."""
    inputs = tokenizer(
        paragraph,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
    )
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    # ProsusAI/finbert returns [positive, negative, neutral] - check config.id2label
    id2label = model.config.id2label
    p_pos = probs[[i for i, lbl in id2label.items() if lbl.lower() == "positive"][0]]
    p_neg = probs[[i for i, lbl in id2label.items() if lbl.lower() == "negative"][0]]
    return float(p_pos - p_neg)


def score_minute(
    text: str,
    tokenizer,
    model,
    keyword_set: dict[str, list[str]] = KEYWORDS,
) -> dict[str, float]:
    """Return a dict of {index_name: mean_polarity} for one MPC minute."""
    paragraphs = split_paragraphs(text)
    results: dict[str, float] = {}
    for index_name, keywords in keyword_set.items():
        filtered = filter_paragraphs(paragraphs, keywords)
        if not filtered:
            results[index_name] = np.nan
            continue
        scores = [polarity_score(p, tokenizer, model) for p in filtered]
        results[index_name] = float(np.mean(scores))
    return results


def process_corpus(minutes_dir: Path) -> pd.DataFrame:
    """Iterate over MPC minutes files (assumes one file per release, filename YYYY-MM-DD.txt)."""
    tokenizer, model = load_model()
    rows = []
    for fp in sorted(minutes_dir.glob("*.txt")):
        date = pd.to_datetime(fp.stem)
        text = fp.read_text(encoding="utf-8")
        scores = score_minute(text, tokenizer, model)
        rows.append({"release_date": date, **scores})
    df = pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)
    return df


def aggregate_monthly(df_release: pd.DataFrame) -> pd.DataFrame:
    """Convert release-date scores to monthly index with forward-fill for non-meeting months."""
    df = df_release.copy()
    df["month"] = df["release_date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month").mean(numeric_only=True).reset_index()

    # Build continuous monthly index, forward-fill gaps
    full_range = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
    monthly = monthly.set_index("month").reindex(full_range).ffill().rename_axis("date").reset_index()
    return monthly


def main() -> None:
    parser = argparse.ArgumentParser(description="FinBERT pipeline for UK MPC minutes.")
    parser.add_argument("--minutes_dir", type=Path, required=True, help="Folder of MPC minutes .txt files.")
    parser.add_argument("--output", type=Path, default=Path("narrative_indices.csv"), help="Output CSV.")
    args = parser.parse_args()

    releases = process_corpus(args.minutes_dir)
    monthly = aggregate_monthly(releases)
    monthly.to_csv(args.output, index=False)
    print(f"Wrote {len(monthly)} monthly observations to {args.output}.")


if __name__ == "__main__":
    main()
