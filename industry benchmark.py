from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3

BASE_DIR = Path(__file__).parent

# Load data

df = pd.read_csv(BASE_DIR / "ml_dataset_enhanced.csv")
pred = pd.read_csv(BASE_DIR / "quantile_predictions_return_20d.csv")

df["earnings_date"] = pd.to_datetime(df["earnings_date"])
pred["earnings_date"] = pd.to_datetime(pred["earnings_date"])

# Keep latest record per ticker

latest = (
    df.sort_values("earnings_date")
    .groupby("ticker", as_index=False)
    .tail(1)
)

pred_latest = (
    pred.sort_values("earnings_date")
    .groupby("ticker", as_index=False)
    .tail(1)
)

# Merge predictions

benchmark = latest.merge(
    pred_latest[
        [
            "ticker",
            "predicted_return",
            "lower_bound",
            "upper_bound",
            "interval_width"
        ]
    ],
    on="ticker",
    how="left"
)

# Clean missing prediction values

benchmark["interval_width"] = benchmark["interval_width"].replace(0, np.nan)

benchmark["predicted_return"] = benchmark["predicted_return"].fillna(
    benchmark["predicted_return"].median()
)

benchmark["lower_bound"] = benchmark["lower_bound"].fillna(
    benchmark["lower_bound"].median()
)

benchmark["upper_bound"] = benchmark["upper_bound"].fillna(
    benchmark["upper_bound"].median()
)

benchmark["interval_width"] = benchmark["interval_width"].fillna(
    benchmark["interval_width"].median()
)

# Build raw component scores

benchmark["sentiment_score"] = benchmark[
    [
        "net_sentiment",
        "mgmt_sentiment",
        "qa_sentiment",
        "confidence_score"
    ]
].mean(axis=1)

benchmark["financial_quality_score"] = benchmark[
    [
        "gross_margin",
        "operating_margin",
        "profit_margin",
        "return_on_equity",
        "return_on_assets"
    ]
].mean(axis=1)

benchmark["valuation_score"] = -benchmark["forward_pe"]

benchmark["leverage_score"] = -benchmark["debt_to_equity"]

benchmark["risk_score_combined"] = benchmark[
    [
        "risk_score",
        "uncertainty_score",
        "volatility_30d"
    ]
].mean(axis=1)

benchmark["forecast_score"] = (
    benchmark["predicted_return"] /
    (benchmark["interval_width"].abs() + 1e-6)
)

benchmark["confidence_adjusted_forecast"] = (
    benchmark["predicted_return"] *
    (1 / (benchmark["interval_width"].abs() + 1e-6))
)

# Sector percentile scores

benchmark["sentiment_pct"] = benchmark.groupby("sector")["sentiment_score"].rank(
    pct=True
)

benchmark["financial_pct"] = benchmark.groupby("sector")["financial_quality_score"].rank(
    pct=True
)

benchmark["valuation_pct"] = benchmark.groupby("sector")["valuation_score"].rank(
    pct=True
)

benchmark["leverage_pct"] = benchmark.groupby("sector")["leverage_score"].rank(
    pct=True
)

benchmark["forecast_pct"] = benchmark.groupby("sector")["forecast_score"].rank(
    pct=True
)

benchmark["risk_pct"] = benchmark.groupby("sector")["risk_score_combined"].rank(
    pct=True
)

# Overall score

benchmark["overall_score"] = (
    0.25 * benchmark["sentiment_pct"] +
    0.25 * benchmark["financial_pct"] +
    0.20 * benchmark["forecast_pct"] +
    0.10 * benchmark["valuation_pct"] +
    0.10 * benchmark["leverage_pct"] +
    0.10 * (1 - benchmark["risk_pct"])
)

benchmark["overall_rank_in_sector"] = benchmark.groupby("sector")["overall_score"].rank(
    ascending=False,
    method="dense"
)

benchmark["peer_count"] = benchmark.groupby("sector")["ticker"].transform("count")

benchmark["rank_percentile"] = 1 - (
    (benchmark["overall_rank_in_sector"] - 1) /
    benchmark["peer_count"]
)

# Recommendation

def assign_recommendation(row):
    if row["overall_score"] >= 0.70 and row["predicted_return"] > 0:
        return "Buy"
    if row["overall_score"] >= 0.45:
        return "Hold"
    return "Avoid"

benchmark["recommendation"] = benchmark.apply(
    assign_recommendation,
    axis=1
)

# Add sector averages for dashboard comparison

sector_avg = benchmark.groupby("sector").agg(
    sector_avg_sentiment=("sentiment_score", "mean"),
    sector_avg_financial_quality=("financial_quality_score", "mean"),
    sector_avg_forecast=("predicted_return", "mean"),
    sector_avg_risk=("risk_score_combined", "mean"),
    sector_avg_overall_score=("overall_score", "mean")
).reset_index()

benchmark = benchmark.merge(
    sector_avg,
    on="sector",
    how="left"
)

# Differences from sector average

benchmark["sentiment_vs_sector"] = (
    benchmark["sentiment_score"] -
    benchmark["sector_avg_sentiment"]
)

benchmark["financial_vs_sector"] = (
    benchmark["financial_quality_score"] -
    benchmark["sector_avg_financial_quality"]
)

benchmark["forecast_vs_sector"] = (
    benchmark["predicted_return"] -
    benchmark["sector_avg_forecast"]
)

benchmark["risk_vs_sector"] = (
    benchmark["risk_score_combined"] -
    benchmark["sector_avg_risk"]
)

# Select final output

output_cols = [
    "ticker",
    "sector",
    "industry",
    "earnings_date",

    "sentiment_score",
    "financial_quality_score",
    "valuation_score",
    "leverage_score",
    "forecast_score",
    "risk_score_combined",

    "predicted_return",
    "lower_bound",
    "upper_bound",
    "interval_width",

    "sentiment_pct",
    "financial_pct",
    "forecast_pct",
    "valuation_pct",
    "leverage_pct",
    "risk_pct",

    "overall_score",
    "overall_rank_in_sector",
    "peer_count",
    "rank_percentile",
    "recommendation",

    "sector_avg_sentiment",
    "sector_avg_financial_quality",
    "sector_avg_forecast",
    "sector_avg_risk",
    "sector_avg_overall_score",

    "sentiment_vs_sector",
    "financial_vs_sector",
    "forecast_vs_sector",
    "risk_vs_sector"
]

benchmark_output = benchmark[output_cols].copy()

benchmark_output = benchmark_output.sort_values(
    [
        "sector",
        "overall_rank_in_sector"
    ]
)

# Save CSV

benchmark_output.to_csv(
    BASE_DIR / "industry_benchmarking_output.csv",
    index=False
)

# Save top 10 per sector

top10_sector = (
    benchmark_output
    .sort_values(["sector", "overall_rank_in_sector"])
    .groupby("sector")
    .head(10)
)

top10_sector.to_csv(
    BASE_DIR / "industry_top10_by_sector.csv",
    index=False
)

# Save recommendation summary

recommendation_summary = (
    benchmark_output
    .groupby(["sector", "recommendation"])
    .size()
    .reset_index(name="count")
)

recommendation_summary.to_csv(
    BASE_DIR / "industry_recommendation_summary.csv",
    index=False
)

# Save SQLite database

conn = sqlite3.connect(
    BASE_DIR / "equity_research_outputs.sqlite"
)

benchmark_output.to_sql(
    "industry_benchmarking",
    conn,
    if_exists="replace",
    index=False
)

top10_sector.to_sql(
    "industry_top10_by_sector",
    conn,
    if_exists="replace",
    index=False
)

recommendation_summary.to_sql(
    "industry_recommendation_summary",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Industry benchmarking completed.")
print("Saved: industry_benchmarking_output.csv")
print("Saved: industry_top10_by_sector.csv")
print("Saved: industry_recommendation_summary.csv")
print("Saved: equity_research_outputs.sqlite")

print("\nOutput shape:")
print(benchmark_output.shape)

print("\nRecommendation counts:")
print(benchmark_output["recommendation"].value_counts())

print("\nTop 20 companies:")
print(
    benchmark_output[
        [
            "ticker",
            "sector",
            "industry",
            "overall_score",
            "overall_rank_in_sector",
            "recommendation",
            "predicted_return"
        ]
    ].head(20)
)