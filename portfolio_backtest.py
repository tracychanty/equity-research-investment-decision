from pathlib import Path
import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

BASE_DIR = Path(__file__).parent

# Load data

data = pd.read_csv(
    BASE_DIR / "portfolio_candidate_universe.csv"
)

data["earnings_date"] = pd.to_datetime(
    data["earnings_date"]
)

# Optional classification probabilities

prob_file = BASE_DIR / "predictions_test.csv"

if prob_file.exists():
    prob = pd.read_csv(prob_file)

    prob["earnings_date"] = pd.to_datetime(
        prob["earnings_date"]
    )

    possible_prob_cols = [
        "predicted_probability",
        "prob_up",
        "prediction_probability",
        "y_proba",
        "proba",
        "probability"
    ]

    prob_col = None

    for col in possible_prob_cols:
        if col in prob.columns:
            prob_col = col
            break

    if prob_col is not None:
        prob = prob[
            [
                "ticker",
                "earnings_date",
                prob_col
            ]
        ].rename(
            columns={
                prob_col: "prob_up"
            }
        )

        data = data.merge(
            prob,
            on=[
                "ticker",
                "earnings_date"
            ],
            how="left"
        )
    else:
        data["prob_up"] = 0.5
else:
    data["prob_up"] = 0.5

data["prob_up"] = data["prob_up"].fillna(0.5)

# Clean numeric columns

numeric_cols = [
    "predicted_return",
    "lower_bound",
    "upper_bound",
    "interval_width",
    "return_20d",
    "market_cap",
    "overall_score",
    "recommendation_score",
    "prob_up"
]

for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

data = data.dropna(
    subset=[
        "quarter",
        "ticker",
        "sector",
        "predicted_return",
        "return_20d",
        "market_cap",
        "overall_score"
    ]
)

data["interval_width"] = data["interval_width"].replace(
    0,
    np.nan
)

data["interval_width"] = data["interval_width"].fillna(
    data["interval_width"].median()
)

data["market_cap"] = data["market_cap"].replace(
    0,
    np.nan
)

data["market_cap"] = data["market_cap"].fillna(
    data["market_cap"].median()
)

# Advanced signals

data["forecast_confidence"] = (
    1 /
    (
        data["interval_width"].abs() +
        1e-6
    )
)

data["downside_risk"] = (
    data["predicted_return"] -
    data["lower_bound"]
)

data["sector_avg_forecast"] = (
    data
    .groupby(
        [
            "quarter",
            "sector"
        ]
    )["predicted_return"]
    .transform("mean")
)

data["expected_alpha"] = (
    data["predicted_return"] -
    data["sector_avg_forecast"]
)

data["prob_adjusted_return"] = (
    data["predicted_return"] *
    data["prob_up"]
)

data["alpha_prob_score"] = (
    data["expected_alpha"] *
    data["prob_up"]
)

data["return_to_uncertainty"] = (
    data["predicted_return"] /
    (
        data["interval_width"].abs() +
        1e-6
    )
)

data["alpha_to_uncertainty"] = (
    data["expected_alpha"] /
    (
        data["interval_width"].abs() +
        1e-6
    )
)

# Rank-based final score

rank_cols = [
    "expected_alpha",
    "prob_adjusted_return",
    "forecast_confidence",
    "overall_score",
    "return_to_uncertainty"
]

for col in rank_cols:
    data[col + "_rank"] = (
        data
        .groupby("quarter")[col]
        .rank(pct=True)
    )

data["downside_risk_rank"] = (
    data
    .groupby("quarter")["downside_risk"]
    .rank(pct=True)
)

data["final_score"] = (
    0.30 * data["expected_alpha_rank"] +
    0.25 * data["prob_adjusted_return_rank"] +
    0.15 * data["forecast_confidence_rank"] +
    0.15 * data["overall_score_rank"] +
    0.10 * data["return_to_uncertainty_rank"] -
    0.05 * data["downside_risk_rank"]
)

# Portfolio settings

top_n_min = 8
top_n_max = 15
max_position_weight = 0.20
max_sector_weight = 0.40
transaction_cost_rate = 0.001
risk_aversion = 5.0

all_weights = []

# Weight optimizer

def optimize_weights(group):
    group = group.copy()

    n = len(group)

    expected = group["prob_adjusted_return"].values

    risk = group["interval_width"].abs().values

    risk = np.where(
        risk <= 0,
        np.nanmedian(risk),
        risk
    )

    covariance = np.diag(
        risk ** 2
    )

    if not CVXPY_AVAILABLE:
        raw = group["final_score"].clip(lower=0).values

        if raw.sum() == 0:
            return np.repeat(1 / n, n)

        return raw / raw.sum()

    w = cp.Variable(n)

    objective = cp.Maximize(
        expected @ w -
        risk_aversion * cp.quad_form(
            w,
            covariance
        )
    )

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_position_weight
    ]

    for sector in group["sector"].unique():
        idx = np.where(
            group["sector"].values == sector
        )[0]

        constraints.append(
            cp.sum(w[idx]) <= max_sector_weight
        )

    problem = cp.Problem(
        objective,
        constraints
    )

    try:
        problem.solve(
            solver=cp.SCS,
            verbose=False
        )

        weights = np.array(w.value).flatten()

        if weights is None or np.isnan(weights).any():
            raise ValueError

        weights = np.maximum(weights, 0)

        weights = weights / weights.sum()

        return weights

    except Exception:
        raw = group["final_score"].clip(lower=0).values

        if raw.sum() == 0:
            return np.repeat(1 / n, n)

        return raw / raw.sum()

# Build portfolios

for quarter, group in data.groupby("quarter"):

    group = group.copy()

    eligible = group[
        (
            group["predicted_return"] > 0
        )
        &
        (
            group["prob_up"] >= 0.50
        )
    ].copy()

    if len(eligible) < top_n_min:
        eligible = group.copy()

    top_n = min(
        top_n_max,
        max(
            top_n_min,
            int(len(eligible) * 0.10)
        )
    )

    selected = (
        eligible
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    if len(selected) == 0:
        continue

    n = len(selected)

    # Equal weight

    equal = selected.copy()
    equal["portfolio_type"] = "Equal_Weight"
    equal["weight"] = 1 / n
    all_weights.append(equal)

    # Market cap weight

    mc = selected.copy()
    mc["portfolio_type"] = "Market_Cap_Weight"

    raw_mc = mc["market_cap"].clip(lower=0)

    if raw_mc.sum() == 0:
        mc["weight"] = 1 / n
    else:
        mc["weight"] = raw_mc / raw_mc.sum()

    mc["weight"] = mc["weight"].clip(
        upper=max_position_weight
    )

    mc["weight"] = mc["weight"] / mc["weight"].sum()

    all_weights.append(mc)

    # ML score weight

    ml = selected.copy()
    ml["portfolio_type"] = "ML_Score_Weight"

    raw_ml = (
        ml["final_score"].clip(lower=0) ** 2
    )

    if raw_ml.sum() == 0:
        ml["weight"] = 1 / n
    else:
        ml["weight"] = raw_ml / raw_ml.sum()

    ml["weight"] = ml["weight"].clip(
        upper=max_position_weight
    )

    ml["weight"] = ml["weight"] / ml["weight"].sum()

    all_weights.append(ml)

    # Optimized portfolio

    opt = selected.copy()
    opt["portfolio_type"] = "ML_Optimized"
    opt["weight"] = optimize_weights(opt)
    all_weights.append(opt)

    # Long-short portfolio

    long_side = (
        group
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(10)
        .copy()
    )

    short_side = (
        group
        .sort_values(
            "final_score",
            ascending=True
        )
        .head(10)
        .copy()
    )

    if len(long_side) > 0 and len(short_side) > 0:
        long_side["portfolio_type"] = "Long_Short"
        short_side["portfolio_type"] = "Long_Short"

        long_side["weight"] = 0.5 / len(long_side)
        short_side["weight"] = -0.5 / len(short_side)

        long_short = pd.concat(
            [
                long_side,
                short_side
            ],
            ignore_index=True
        )

        all_weights.append(long_short)

portfolio_weights = pd.concat(
    all_weights,
    ignore_index=True
)

# Turnover

turnover_records = []

for portfolio_type, group in portfolio_weights.groupby("portfolio_type"):

    group = group.sort_values(
        [
            "quarter",
            "ticker"
        ]
    )

    prev_weights = None

    for quarter, qdata in group.groupby("quarter"):

        current_weights = qdata.set_index("ticker")["weight"]

        if prev_weights is None:
            turnover = np.abs(current_weights).sum()
        else:
            all_tickers = current_weights.index.union(
                prev_weights.index
            )

            current_aligned = current_weights.reindex(
                all_tickers
            ).fillna(0)

            previous_aligned = prev_weights.reindex(
                all_tickers
            ).fillna(0)

            turnover = 0.5 * np.abs(
                current_aligned -
                previous_aligned
            ).sum()

        turnover_records.append({
            "quarter": quarter,
            "portfolio_type": portfolio_type,
            "turnover": turnover
        })

        prev_weights = current_weights

turnover = pd.DataFrame(
    turnover_records
)

turnover["transaction_cost"] = (
    turnover["turnover"] *
    transaction_cost_rate
)

# Returns

portfolio_weights["weighted_return"] = (
    portfolio_weights["weight"] *
    portfolio_weights["return_20d"]
)

portfolio_weights["weighted_predicted_return"] = (
    portfolio_weights["weight"] *
    portfolio_weights["predicted_return"]
)

returns = (
    portfolio_weights
    .groupby(
        [
            "quarter",
            "portfolio_type"
        ]
    )
    .agg(
        gross_return=("weighted_return", "sum"),
        predicted_portfolio_return=("weighted_predicted_return", "sum"),
        avg_predicted_return=("predicted_return", "mean"),
        avg_actual_return=("return_20d", "mean"),
        avg_prob_up=("prob_up", "mean"),
        avg_final_score=("final_score", "mean"),
        number_of_holdings=("ticker", "count")
    )
    .reset_index()
)

returns = returns.merge(
    turnover,
    on=[
        "quarter",
        "portfolio_type"
    ],
    how="left"
)

returns["net_return"] = (
    returns["gross_return"] -
    returns["transaction_cost"]
)

# Full universe benchmark

universe = (
    data
    .groupby("quarter")
    .agg(
        gross_return=("return_20d", "mean"),
        predicted_portfolio_return=("predicted_return", "mean"),
        avg_predicted_return=("predicted_return", "mean"),
        avg_actual_return=("return_20d", "mean"),
        avg_prob_up=("prob_up", "mean"),
        avg_final_score=("final_score", "mean"),
        number_of_holdings=("ticker", "count")
    )
    .reset_index()
)

universe["portfolio_type"] = "Full_Universe"
universe["turnover"] = 0
universe["transaction_cost"] = 0
universe["net_return"] = universe["gross_return"]

universe = universe[
    returns.columns
]

returns = pd.concat(
    [
        returns,
        universe
    ],
    ignore_index=True
)

returns = returns.sort_values(
    [
        "quarter",
        "portfolio_type"
    ]
)

# Wide tables

net_wide = returns.pivot(
    index="quarter",
    columns="portfolio_type",
    values="net_return"
).reset_index()

net_wide.columns.name = None

cumulative = net_wide.copy()

for col in cumulative.columns:
    if col != "quarter":
        cumulative[col] = (
            1 + cumulative[col]
        ).cumprod()

drawdown = cumulative.copy()

for col in drawdown.columns:
    if col != "quarter":
        running_max = drawdown[col].cummax()

        drawdown[col] = (
            drawdown[col] /
            running_max
        ) - 1

# Save

portfolio_weights.to_csv(
    BASE_DIR / "v3_portfolio_weights.csv",
    index=False
)

returns.to_csv(
    BASE_DIR / "v3_portfolio_returns_long.csv",
    index=False
)

net_wide.to_csv(
    BASE_DIR / "v3_portfolio_net_returns_wide.csv",
    index=False
)

cumulative.to_csv(
    BASE_DIR / "v3_portfolio_cumulative_returns.csv",
    index=False
)

drawdown.to_csv(
    BASE_DIR / "v3_portfolio_drawdowns.csv",
    index=False
)

print("V3 optimized portfolio backtest completed.")

print("\nNet returns:")
print(net_wide)

print("\nCumulative returns:")
print(cumulative)

print("\nDrawdowns:")
print(drawdown)

print("\nFinal cumulative values:")

for col in cumulative.columns:
    if col != "quarter":
        print(
            col,
            round(cumulative[col].iloc[-1], 4)
        )