from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent

# Load candidate universe

candidates = pd.read_csv(
    BASE_DIR / "portfolio_candidate_universe.csv"
)

# Keep top 10 candidates per quarter

top10 = candidates[
    candidates["is_top10"] == True
].copy()

top10["market_cap"] = top10["market_cap"].replace(
    0,
    np.nan
)

top10["market_cap"] = top10["market_cap"].fillna(
    top10["market_cap"].median()
)

# Create portfolio weights

all_weights = []

for quarter, group in top10.groupby("quarter"):

    group = group.copy()

    n = len(group)

    if n == 0:
        continue

    # Equal weight portfolio

    equal_weight = group.copy()

    equal_weight["portfolio_type"] = "Equal_Weight"

    equal_weight["weight"] = 1 / n

    all_weights.append(equal_weight)

    # Market cap weight portfolio

    market_cap_weight = group.copy()

    market_cap_weight["portfolio_type"] = "Market_Cap_Weight"

    market_cap_weight["weight"] = (
        market_cap_weight["market_cap"] /
        market_cap_weight["market_cap"].sum()
    )

    all_weights.append(market_cap_weight)

    # ML score weight portfolio

    ml_weight = group.copy()

    score = ml_weight["recommendation_score"].clip(
        lower=0
    )

    if score.sum() == 0:
        ml_weight["weight"] = 1 / n
    else:
        ml_weight["weight"] = score / score.sum()

    ml_weight["portfolio_type"] = "ML_Score_Weight"

    all_weights.append(ml_weight)

portfolio_weights = pd.concat(
    all_weights,
    ignore_index=True
)

# Validate weights

weight_check = (
    portfolio_weights
    .groupby(
        [
            "quarter",
            "portfolio_type"
        ]
    )["weight"]
    .sum()
    .reset_index()
)

# Save outputs

portfolio_weights.to_csv(
    BASE_DIR / "portfolio_weights.csv",
    index=False
)

weight_check.to_csv(
    BASE_DIR / "portfolio_weight_check.csv",
    index=False
)

print("Portfolio construction completed.")
print("Saved: portfolio_weights.csv")
print("Saved: portfolio_weight_check.csv")

print("\nWeight check:")
print(weight_check)

print("\nPortfolio preview:")
print(
    portfolio_weights[
        [
            "quarter",
            "ticker",
            "sector",
            "portfolio_type",
            "recommendation_score",
            "market_cap",
            "weight",
            "return_20d"
        ]
    ].head(30)
)