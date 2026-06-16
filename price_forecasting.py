from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# Load data

BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR / "ml_dataset_enhanced.csv")
df["earnings_date"] = pd.to_datetime(df["earnings_date"])

# Add interaction features

eps = 1e-6

df["confidence_risk_ratio"] = df["confidence_score"] / (df["risk_score"] + eps)
df["sentiment_momentum_30d"] = df["net_sentiment"] * df["momentum_30d"]
df["sentiment_momentum_60d"] = df["net_sentiment"] * df["momentum_60d"]
df["sentiment_volatility_30d"] = df["net_sentiment"] * df["volatility_30d"]
df["confidence_uncertainty_gap"] = df["confidence_score"] - df["uncertainty_score"]
df["risk_uncertainty_combo"] = df["risk_score"] + df["uncertainty_score"]
df["quality_valuation_ratio"] = df["profitability_score"] / (df["pe_ratio"].abs() + eps)
df["margin_leverage_ratio"] = df["margin_strength"] / (df["debt_to_equity"].abs() + eps)

# One-hot encoding

df = pd.get_dummies(
    df,
    columns=["sector", "industry"],
    drop_first=False
)

print("\nDataset shape:")
print(df.shape)

# Load selected features

selected = pd.read_csv(BASE_DIR / "selected_features.csv")

feature_col = selected.columns[0]

selected_features = (
    selected[feature_col]
    .dropna()
    .astype(str)
    .tolist()
)

extra_features = [
    "confidence_risk_ratio",
    "sentiment_momentum_30d",
    "sentiment_momentum_60d",
    "sentiment_volatility_30d",
    "confidence_uncertainty_gap",
    "risk_uncertainty_combo",
    "quality_valuation_ratio",
    "margin_leverage_ratio"
]

features = [
    f for f in selected_features
    if f in df.columns
]

features = features + [
    f for f in extra_features
    if f in df.columns
]

features = list(dict.fromkeys(features))

print("\nFeatures used:")
print(len(features))

# Train validation test split

train_full = df[df["earnings_date"] < "2022-01-01"]
test = df[df["earnings_date"] >= "2022-01-01"]

train = train_full[train_full["earnings_date"] < "2021-01-01"]
val = train_full[train_full["earnings_date"] >= "2021-01-01"]

print("\nTrain:")
print(train.shape)

print("\nValidation:")
print(val.shape)

print("\nTest:")
print(test.shape)

# Model settings

targets = [
    "return_5d",
    "return_10d",
    "return_20d"
]

param_grid = [
    {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": 6
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4
    }
]

all_results = []

for target in targets:

    print("\n" + "-" * 60)
    print("Target:", target)

    X_train = train[features]
    y_train = train[target]

    X_val = val[features]
    y_val = val[target]

    X_train_full = train_full[features]
    y_train_full = train_full[target]

    X_test = test[features]
    y_test = test[target]

    best_params = None
    best_val_mae = np.inf

    # Tune median model

    for params in param_grid:

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=0.50,
            random_state=42,
            verbose=-1,
            force_col_wise=True,
            **params
        )

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)

        val_mae = mean_absolute_error(
            y_val,
            val_pred
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params

    print("Best params:")
    print(best_params)

    # Baseline

    baseline_pred = np.repeat(
        y_train_full.mean(),
        len(y_test)
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_pred
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_pred
        )
    )

    # Final quantile models

    median_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.50,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
        **best_params
    )

    lower_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.05,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
        **best_params
    )

    upper_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.95,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
        **best_params
    )

    median_model.fit(X_train_full, y_train_full)
    lower_model.fit(X_train_full, y_train_full)
    upper_model.fit(X_train_full, y_train_full)

    pred_median = median_model.predict(X_test)
    pred_lower = lower_model.predict(X_test)
    pred_upper = upper_model.predict(X_test)

    mae = mean_absolute_error(y_test, pred_median)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred_median
        )
    )

    r2 = r2_score(y_test, pred_median)

    coverage = np.mean(
        (y_test >= pred_lower)
        &
        (y_test <= pred_upper)
    )

    print("Baseline MAE:", round(baseline_mae, 6))
    print("Model MAE:", round(mae, 6))
    print("Baseline RMSE:", round(baseline_rmse, 6))
    print("Model RMSE:", round(rmse, 6))
    print("R2:", round(r2, 6))
    print("Coverage:", round(coverage, 4))

    all_results.append({
        "Target": target,
        "Baseline_MAE": baseline_mae,
        "Model_MAE": mae,
        "Baseline_RMSE": baseline_rmse,
        "Model_RMSE": rmse,
        "R2": r2,
        "Coverage": coverage,
        "Best_Params": str(best_params)
    })

    predictions = pd.DataFrame({
        "ticker": test["ticker"],
        "earnings_date": test["earnings_date"],
        "actual_return": y_test,
        "predicted_return": pred_median,
        "lower_bound": pred_lower,
        "upper_bound": pred_upper,
        "interval_width": pred_upper - pred_lower
    })

    predictions.to_csv(
        BASE_DIR / f"quantile_predictions_{target}.csv",
        index=False
    )

    importance = pd.DataFrame({
        "feature": features,
        "importance": median_model.feature_importances_
    }).sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        BASE_DIR / f"quantile_feature_importance_{target}.csv",
        index=False
    )

    joblib.dump(
        median_model,
        BASE_DIR / f"quantile_median_model_{target}.pkl"
    )

    joblib.dump(
        lower_model,
        BASE_DIR / f"quantile_lower_model_{target}.pkl"
    )

    joblib.dump(
        upper_model,
        BASE_DIR / f"quantile_upper_model_{target}.pkl"
    )

# Save summary

summary = pd.DataFrame(all_results)

summary.to_csv(
    BASE_DIR / "quantile_forecast_summary_tuned.csv",
    index=False
)

print("\nFinal Summary")
print(summary)

print("\nFiles saved:")
print("quantile_forecast_summary_tuned.csv")
print("quantile_predictions_return_5d.csv")
print("quantile_predictions_return_10d.csv")
print("quantile_predictions_return_20d.csv")
print("quantile_feature_importance_return_5d.csv")
print("quantile_feature_importance_return_10d.csv")
print("quantile_feature_importance_return_20d.csv")