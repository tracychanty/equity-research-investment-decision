from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent

# Load data

benchmark = pd.read_csv(
    BASE_DIR / "industry_benchmarking_output.csv"
)

pred = pd.read_csv(
    BASE_DIR / "quantile_predictions_return_20d.csv"
)

ml_data = pd.read_csv(
    BASE_DIR / "ml_dataset_enhanced.csv"
)

# Date formatting

benchmark["earnings_date"] = pd.to_datetime(
    benchmark["earnings_date"]
)

pred["earnings_date"] = pd.to_datetime(
    pred["earnings_date"]
)

ml_data["earnings_date"] = pd.to_datetime(
    ml_data["earnings_date"]
)

# Merge actual 20-day returns

actual_returns = ml_data[
    [
        "ticker",
        "earnings_date",
        "return_20d",
        "market_cap",
        "sector",
        "industry"
    ]
].drop_duplicates()

portfolio_data = benchmark.merge(
    actual_returns,
    on=[
        "ticker",
        "earnings_date"
    ],
    how="left",
    suffixes=(
        "",
        "_raw"
    )
)

# Create quarter

portfolio_data["quarter"] = (
    portfolio_data["earnings_date"]
    .dt.to_period("Q")
    .astype(str)
)

# Clean key fields

portfolio_data["predicted_return"] = portfolio_data["predicted_return"].fillna(
    portfolio_data["predicted_return"].median()
)

portfolio_data["interval_width"] = portfolio_data["interval_width"].replace(
    0,
    np.nan
)

portfolio_data["interval_width"] = portfolio_data["interval_width"].fillna(
    portfolio_data["interval_width"].median()
)

portfolio_data["return_20d"] = portfolio_data["return_20d"].fillna(
    portfolio_data["return_20d"].median()
)

portfolio_data["market_cap"] = portfolio_data["market_cap"].fillna(
    portfolio_data["market_cap"].median()
)

# Forecast confidence

portfolio_data["forecast_confidence"] = (
    1 /
    (
        portfolio_data["interval_width"].abs() +
        1e-6
    )
)

# Downside risk

portfolio_data["downside_risk"] = (
    portfolio_data["predicted_return"] -
    portfolio_data["lower_bound"]
)

# Recommendation score

portfolio_data["recommendation_score"] = (
    0.40 * portfolio_data["overall_score"] +
    0.30 * portfolio_data["predicted_return"].rank(pct=True) +
    0.20 * portfolio_data["forecast_confidence"].rank(pct=True) -
    0.10 * portfolio_data["downside_risk"].rank(pct=True)
)

# Filter test period

portfolio_data = portfolio_data[
    portfolio_data["earnings_date"] >= "2022-01-01"
].copy()

# Remove extreme market caps and invalid rows

portfolio_data = portfolio_data[
    portfolio_data["market_cap"] > 0
].copy()

# Rank within each quarter

portfolio_data["quarter_rank"] = (
    portfolio_data
    .groupby("quarter")["recommendation_score"]
    .rank(
        ascending=False,
        method="first"
    )
)

portfolio_data["is_top10"] = portfolio_data["quarter_rank"] <= 10

# Save full candidate universe

portfolio_data.to_csv(
    BASE_DIR / "portfolio_candidate_universe.csv",
    index=False
)

# Save top 10 candidates per quarter

top10 = portfolio_data[
    portfolio_data["is_top10"]
].copy()

top10 = top10.sort_values(
    [
        "quarter",
        "quarter_rank"
    ]
)

top10.to_csv(
    BASE_DIR / "portfolio_top10_candidates.csv",
    index=False
)

# Summary

summary = (
    portfolio_data
    .groupby("quarter")
    .agg(
        total_candidates=("ticker", "count"),
        avg_predicted_return=("predicted_return", "mean"),
        avg_actual_return=("return_20d", "mean"),
        avg_score=("recommendation_score", "mean")
    )
    .reset_index()
)

summary.to_csv(
    BASE_DIR / "portfolio_candidate_summary.csv",
    index=False
)

print("Portfolio preparation completed.")
print("Saved: portfolio_candidate_universe.csv")
print("Saved: portfolio_top10_candidates.csv")
print("Saved: portfolio_candidate_summary.csv")

print("\nCandidate universe shape:")
print(portfolio_data.shape)

print("\nTop 10 preview:")
print(
    top10[
        [
            "quarter",
            "ticker",
            "sector",
            "predicted_return",
            "return_20d",
            "overall_score",
            "recommendation_score",
            "quarter_rank"
        ]
    ].head(20)
)

print("\nQuarter summary:")
print(summary)