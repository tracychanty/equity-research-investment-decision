from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent

# =========================
# Step 1: Load data
# =========================
fundamentals = pd.read_csv(BASE_DIR / "fundamentals.csv")
sentiment = pd.read_csv(BASE_DIR / "sentiment_features.csv")
prices = pd.read_csv(BASE_DIR / "stock_prices_updated.csv")

fundamentals["ticker"] = fundamentals["ticker"].astype(str).str.upper().str.strip()
sentiment["ticker"] = sentiment["ticker"].astype(str).str.upper().str.strip()
prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()

sentiment["earnings_date"] = pd.to_datetime(sentiment["earnings_date"], errors="coerce")
prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")

prices = prices.sort_values(["ticker", "Date"])

print("fundamentals:", fundamentals.shape)
print("sentiment:", sentiment.shape)
print("prices:", prices.shape)

# =========================
# Step 2: Market features
# =========================
prices["daily_return"] = prices.groupby("ticker")["close"].pct_change()

prices["momentum_5d"] = prices.groupby("ticker")["close"].pct_change(5)
prices["momentum_10d"] = prices.groupby("ticker")["close"].pct_change(10)
prices["momentum_30d"] = prices.groupby("ticker")["close"].pct_change(30)
prices["momentum_60d"] = prices.groupby("ticker")["close"].pct_change(60)

prices["volatility_10d"] = (
    prices.groupby("ticker")["daily_return"]
    .rolling(10)
    .std()
    .reset_index(level=0, drop=True)
)

prices["volatility_30d"] = (
    prices.groupby("ticker")["daily_return"]
    .rolling(30)
    .std()
    .reset_index(level=0, drop=True)
)

prices["volatility_60d"] = (
    prices.groupby("ticker")["daily_return"]
    .rolling(60)
    .std()
    .reset_index(level=0, drop=True)
)

prices["momentum_spread"] = prices["momentum_30d"] - prices["momentum_60d"]
prices["volatility_spread"] = prices["volatility_10d"] - prices["volatility_60d"]

prices["risk_adjusted_momentum_30d"] = (
    prices["momentum_30d"] / prices["volatility_30d"].replace(0, np.nan)
)

market_features = prices[
    [
        "ticker",
        "Date",
        "close",
        "momentum_5d",
        "momentum_10d",
        "momentum_30d",
        "momentum_60d",
        "volatility_10d",
        "volatility_30d",
        "volatility_60d",
        "momentum_spread",
        "volatility_spread",
        "risk_adjusted_momentum_30d",
    ]
].dropna()

# =========================
# Step 3: Align market features with earnings date
# =========================
sentiment_sorted = sentiment.sort_values(["earnings_date", "ticker"]).reset_index(drop=True)
market_sorted = market_features.sort_values(["Date", "ticker"]).reset_index(drop=True)

sentiment_with_market = pd.merge_asof(
    sentiment_sorted,
    market_sorted,
    left_on="earnings_date",
    right_on="Date",
    by="ticker",
    direction="backward"
)

print("sentiment + market:", sentiment_with_market.shape)

# =========================
# Step 4: Merge fundamentals
# =========================
ml_dataset = sentiment_with_market.merge(
    fundamentals,
    on="ticker",
    how="left"
)

print("After fundamentals merge:", ml_dataset.shape)

# =========================
# Step 5: Advanced financial features
# =========================

# log market cap
ml_dataset["log_market_cap"] = np.log1p(ml_dataset["market_cap"])

# valuation gap: negative value may mean forward PE is lower than trailing PE
ml_dataset["valuation_gap"] = ml_dataset["forward_pe"] - ml_dataset["pe_ratio"]

# EPS revision proxy: forward EPS minus trailing EPS
ml_dataset["eps_revision"] = ml_dataset["eps_forward"] - ml_dataset["eps_trailing"]

# price location within 52-week range
ml_dataset["price_position_52w"] = (
    (ml_dataset["close"] - ml_dataset["fifty_two_wk_low"])
    / (ml_dataset["fifty_two_wk_high"] - ml_dataset["fifty_two_wk_low"])
)

# margin strength: average of major margin metrics
ml_dataset["margin_strength"] = ml_dataset[
    ["gross_margin", "operating_margin", "profit_margin"]
].mean(axis=1)

# profitability score: ROE + ROA
ml_dataset["profitability_score"] = ml_dataset[
    ["return_on_equity", "return_on_assets"]
].mean(axis=1)

# leverage risk proxy
ml_dataset["leverage_risk"] = ml_dataset["debt_to_equity"]

# sentiment difference between management and Q&A
ml_dataset["mgmt_qa_sentiment_gap"] = (
    ml_dataset["mgmt_sentiment"] - ml_dataset["qa_sentiment"]
)

# positive/negative balance
ml_dataset["sentiment_balance"] = (
    ml_dataset["positive_ratio"] - ml_dataset["negative_ratio"]
)

ml_dataset["mgmt_sentiment_balance"] = (
    ml_dataset["mgmt_positive_ratio"] - ml_dataset["mgmt_negative_ratio"]
)

ml_dataset["qa_sentiment_balance"] = (
    ml_dataset["qa_positive_ratio"] - ml_dataset["qa_negative_ratio"]
)

# =========================
# Step 6: Select final columns
# =========================
final_columns = [
    "ticker",
    "earnings_date",
    "q",

    # sentiment features
    "net_sentiment",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "sentiment_volatility",
    "mgmt_sentiment",
    "mgmt_positive_ratio",
    "mgmt_negative_ratio",
    "mgmt_neutral_ratio",
    "mgmt_sentiment_volatility",
    "qa_sentiment",
    "qa_positive_ratio",
    "qa_negative_ratio",
    "qa_neutral_ratio",
    "qa_sentiment_volatility",
    "confidence_score",
    "risk_score",
    "uncertainty_score",
    "cost_cutting_score",
    "sentiment_surprise",
    "mgmt_qa_sentiment_gap",
    "sentiment_balance",
    "mgmt_sentiment_balance",
    "qa_sentiment_balance",

    # market features
    "close",
    "momentum_5d",
    "momentum_10d",
    "momentum_30d",
    "momentum_60d",
    "volatility_10d",
    "volatility_30d",
    "volatility_60d",
    "momentum_spread",
    "volatility_spread",
    "risk_adjusted_momentum_30d",

    # financial features
    "sector",
    "industry",
    "market_cap",
    "log_market_cap",
    "pe_ratio",
    "forward_pe",
    "valuation_gap",
    "eps_trailing",
    "eps_forward",
    "eps_revision",
    "revenue",
    "gross_margin",
    "operating_margin",
    "profit_margin",
    "margin_strength",
    "debt_to_equity",
    "leverage_risk",
    "return_on_equity",
    "return_on_assets",
    "profitability_score",
    "beta",
    "fifty_two_wk_high",
    "fifty_two_wk_low",
    "price_position_52w",

    # labels
    "label_5d",
    "label_10d",
    "label_20d",
    "return_5d",
    "return_10d",
    "return_20d",
]

ml_dataset = ml_dataset[final_columns]

# =========================
# Step 7: Clean infinities and missing values
# =========================
ml_dataset = ml_dataset.replace([np.inf, -np.inf], np.nan)

print("\nMissing values before cleaning:")
print(ml_dataset.isna().mean().sort_values(ascending=False).head(25))

ml_dataset = ml_dataset.dropna(subset=["label_5d", "label_10d", "label_20d"])

numeric_cols = ml_dataset.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    ml_dataset[col] = ml_dataset[col].fillna(ml_dataset[col].median())

# fill categorical missing
ml_dataset["sector"] = ml_dataset["sector"].fillna("Unknown")
ml_dataset["industry"] = ml_dataset["industry"].fillna("Unknown")

print("\nFinal missing values:", ml_dataset.isna().sum().sum())
print("Final dataset shape:", ml_dataset.shape)

# =========================
# Step 8: Save
# =========================
output_path = BASE_DIR / "ml_dataset_enhanced.csv"
ml_dataset.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")

print("\nLabel distribution:")
print(ml_dataset["label_5d"].value_counts(normalize=True))
print(ml_dataset["label_10d"].value_counts(normalize=True))
print(ml_dataset["label_20d"].value_counts(normalize=True))

print("\nDate range:")
print(ml_dataset["earnings_date"].min())
print(ml_dataset["earnings_date"].max())

print("\nNumber of tickers:")
print(ml_dataset["ticker"].nunique())

print("\nColumns:")
print(ml_dataset.columns.tolist())

print("\nPreview:")
print(ml_dataset.head())