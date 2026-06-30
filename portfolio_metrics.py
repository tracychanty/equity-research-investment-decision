from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent

# Load data

returns = pd.read_csv(
    BASE_DIR / "v3_portfolio_net_returns_wide.csv"
)

cumulative = pd.read_csv(
    BASE_DIR / "v3_portfolio_cumulative_returns.csv"
)

drawdowns = pd.read_csv(
    BASE_DIR / "v3_portfolio_drawdowns.csv"
)

# Settings

periods_per_year = 4
risk_free_rate = 0.00
main_strategy = "ML_Optimized"
benchmark = "Full_Universe"

portfolio_cols = [
    col for col in returns.columns
    if col != "quarter"
]

# Helper functions

def cumulative_return(r):
    return (1 + r).prod() - 1


def cagr(r):
    total_return = cumulative_return(r)

    return (
        (1 + total_return)
        ** (periods_per_year / len(r))
        - 1
    )


def annualized_volatility(r):
    return r.std() * np.sqrt(periods_per_year)


def sharpe_ratio(r):
    excess_return = cagr(r) - risk_free_rate
    vol = annualized_volatility(r)

    if vol == 0 or np.isnan(vol):
        return np.nan

    return excess_return / vol


def downside_deviation(r):
    downside = np.minimum(r, 0)

    downside_std = np.sqrt(
        np.mean(downside ** 2)
    ) * np.sqrt(periods_per_year)

    return downside_std


def sortino_ratio(r):
    downside_vol = downside_deviation(r)

    if downside_vol == 0 or np.isnan(downside_vol):
        return np.nan

    return (cagr(r) - risk_free_rate) / downside_vol


def max_drawdown(dd):
    return dd.min()


def calmar_ratio(r, max_dd):
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan

    return cagr(r) / abs(max_dd)


def win_rate(r):
    return (r > 0).mean()


def gain_loss_ratio(r):
    gains = r[r > 0]
    losses = r[r < 0]

    if len(losses) == 0:
        return np.nan

    avg_gain = gains.mean() if len(gains) > 0 else 0
    avg_loss = abs(losses.mean())

    if avg_loss == 0:
        return np.nan

    return avg_gain / avg_loss


def value_at_risk(r, level=0.05):
    return np.quantile(r, level)


def conditional_value_at_risk(r, level=0.05):
    var = value_at_risk(r, level)

    tail_losses = r[r <= var]

    if len(tail_losses) == 0:
        return np.nan

    return tail_losses.mean()


def omega_ratio(r, threshold=0):
    gains = r[r > threshold] - threshold
    losses = threshold - r[r < threshold]

    if losses.sum() == 0:
        return np.nan

    return gains.sum() / losses.sum()


# Portfolio metrics

metrics = []

for portfolio in portfolio_cols:

    r = pd.to_numeric(
        returns[portfolio],
        errors="coerce"
    ).dropna()

    dd = pd.to_numeric(
        drawdowns[portfolio],
        errors="coerce"
    ).dropna()

    max_dd = max_drawdown(dd)

    metrics.append({
        "Portfolio": portfolio,
        "Is Main Strategy": portfolio == main_strategy,
        "Cumulative Return": cumulative_return(r),
        "CAGR": cagr(r),
        "Annualized Volatility": annualized_volatility(r),
        "Sharpe Ratio": sharpe_ratio(r),
        "Sortino Ratio": sortino_ratio(r),
        "Maximum Drawdown": max_dd,
        "Calmar Ratio": calmar_ratio(r, max_dd),
        "Win Rate": win_rate(r),
        "Gain Loss Ratio": gain_loss_ratio(r),
        "Omega Ratio": omega_ratio(r),
        "VaR 95": value_at_risk(r, 0.05),
        "CVaR 95": conditional_value_at_risk(r, 0.05),
        "Average Quarterly Return": r.mean(),
        "Median Quarterly Return": r.median(),
        "Best Quarter": r.max(),
        "Worst Quarter": r.min(),
        "Number of Quarters": len(r)
    })

metrics_df = pd.DataFrame(metrics)

# Active metrics versus benchmark

if benchmark in returns.columns:

    benchmark_returns = pd.to_numeric(
        returns[benchmark],
        errors="coerce"
    )

    active_rows = []

    for portfolio in portfolio_cols:

        if portfolio == benchmark:
            continue

        portfolio_returns = pd.to_numeric(
            returns[portfolio],
            errors="coerce"
        )

        active = (
            portfolio_returns -
            benchmark_returns
        ).dropna()

        tracking_error = active.std() * np.sqrt(periods_per_year)

        active_return = active.mean() * periods_per_year

        information_ratio = (
            active_return / tracking_error
            if tracking_error != 0
            else np.nan
        )

        active_rows.append({
            "Portfolio": portfolio,
            "Annualized Active Return vs Benchmark": active_return,
            "Tracking Error vs Benchmark": tracking_error,
            "Information Ratio vs Benchmark": information_ratio
        })

    active_df = pd.DataFrame(active_rows)

    metrics_df = metrics_df.merge(
        active_df,
        on="Portfolio",
        how="left"
    )

# Active metrics versus Equal Weight

if "Equal_Weight" in returns.columns:

    equal_returns = pd.to_numeric(
        returns["Equal_Weight"],
        errors="coerce"
    )

    active_equal_rows = []

    for portfolio in portfolio_cols:

        if portfolio == "Equal_Weight":
            continue

        portfolio_returns = pd.to_numeric(
            returns[portfolio],
            errors="coerce"
        )

        active = (
            portfolio_returns -
            equal_returns
        ).dropna()

        tracking_error = active.std() * np.sqrt(periods_per_year)

        active_return = active.mean() * periods_per_year

        information_ratio = (
            active_return / tracking_error
            if tracking_error != 0
            else np.nan
        )

        active_equal_rows.append({
            "Portfolio": portfolio,
            "Annualized Active Return vs Equal Weight": active_return,
            "Tracking Error vs Equal Weight": tracking_error,
            "Information Ratio vs Equal Weight": information_ratio
        })

    active_equal_df = pd.DataFrame(active_equal_rows)

    metrics_df = metrics_df.merge(
        active_equal_df,
        on="Portfolio",
        how="left"
    )

# Ranking

metrics_df["Rank by CAGR"] = metrics_df["CAGR"].rank(
    ascending=False,
    method="dense"
)

metrics_df["Rank by Sharpe"] = metrics_df["Sharpe Ratio"].rank(
    ascending=False,
    method="dense"
)

metrics_df["Rank by Sortino"] = metrics_df["Sortino Ratio"].rank(
    ascending=False,
    method="dense"
)

metrics_df["Rank by Max Drawdown"] = metrics_df["Maximum Drawdown"].rank(
    ascending=False,
    method="dense"
)

metrics_df["Rank by Information Ratio"] = metrics_df[
    "Information Ratio vs Benchmark"
].rank(
    ascending=False,
    method="dense"
)

metrics_df = metrics_df.sort_values(
    [
        "Is Main Strategy",
        "Rank by CAGR"
    ],
    ascending=[
        False,
        True
    ]
)

# Save raw metrics

metrics_df.to_csv(
    BASE_DIR / "portfolio_performance_metrics.csv",
    index=False
)

# Report version

report_metrics = metrics_df.copy()

percent_cols = [
    "Cumulative Return",
    "CAGR",
    "Annualized Volatility",
    "Maximum Drawdown",
    "Win Rate",
    "VaR 95",
    "CVaR 95",
    "Average Quarterly Return",
    "Median Quarterly Return",
    "Best Quarter",
    "Worst Quarter",
    "Annualized Active Return vs Benchmark",
    "Tracking Error vs Benchmark",
    "Annualized Active Return vs Equal Weight",
    "Tracking Error vs Equal Weight"
]

for col in percent_cols:
    if col in report_metrics.columns:
        report_metrics[col] = (
            report_metrics[col] * 100
        ).round(2)

ratio_cols = [
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Gain Loss Ratio",
    "Omega Ratio",
    "Information Ratio vs Benchmark",
    "Information Ratio vs Equal Weight"
]

for col in ratio_cols:
    if col in report_metrics.columns:
        report_metrics[col] = report_metrics[col].round(3)

report_metrics.to_csv(
    BASE_DIR / "portfolio_performance_metrics_report.csv",
    index=False
)

# Main strategy summary

main_summary = report_metrics[
    report_metrics["Portfolio"] == main_strategy
].copy()

main_summary.to_csv(
    BASE_DIR / "main_strategy_summary.csv",
    index=False
)

print("Enhanced portfolio performance metrics completed.")
print("Saved: portfolio_performance_metrics.csv")
print("Saved: portfolio_performance_metrics_report.csv")
print("Saved: main_strategy_summary.csv")

print("\nReport Metrics:")
print(report_metrics)

print("\nMain Strategy Summary:")
print(main_summary)