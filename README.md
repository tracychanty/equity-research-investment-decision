## Data Setup

Data files are not stored in this repository due to size constraints.

Download from: [Google Drive](https://drive.google.com/drive/folders/1NYfYCJvgmE7Gay2oawZWb8ZWBTFLSW38?usp=drive_link)

Place all files in the correct local folders before running any notebooks:

#### Phase 1 inputs/outputs — place in `Data/`
- `cleaned_transcripts_final.pkl` — main Phase 1 output (14,738 rows → 13,803 after dedup)
- `stock_prices_updated.csv` — full price history (2,194 tickers)
- `sp500_prices.csv` — S&P 500 benchmark
- `fundamentals.csv` — financial fundamentals (2,186 tickers)

#### Phase 2 outputs — place in `Outputs/sentiment_extraction/`
- `sentiment_features.csv` — main Phase 2 output, Phase 3 input (13,803 rows, 32 columns)
- `sentiment_scores_raw.csv` — FinBERT full transcript scores
- `mgmt_scores_raw.csv` — FinBERT management remarks scores
- `qa_scores_raw.csv` — FinBERT Q&A section scores

#### Phase 3 outputs — place in `Outputs/feature_engineering/`
- `ml_dataset_enhanced.csv` — feature-engineered dataset (13,803 rows, 68 columns)

#### Phase 4 outputs — place in `Outputs/classification/`
- `selected_features.csv` — final 48 features selected after correlation filter and VIF
- `cv_results.csv` — 5-fold TimeSeriesSplit CV AUC scores for all models
- `test_results.csv` — evaluation metrics on 2022–2023 test set (all models)
- `robustness_10d.csv` — robustness check results for 10-day horizon
- `robustness_20d.csv` — robustness check results for 20-day horizon
- `robustness_summary.csv` — AUC comparison across 5d / 10d / 20d horizons
- `tuning_summary.csv` — hyperparameter tuning summary (Random Forest)
- `tuning_results.csv` — full RandomizedSearchCV CV results (50 iterations)
- `predictions_test.csv` — predicted probabilities and labels for all models on test set
- `final_model_comparison.csv` — complete model comparison table including tuned RF
- `best_model.pkl` — final model: Random Forest (Baseline), test AUC = 0.5330
- `scaler.pkl` — StandardScaler fit on train set (for Logistic Regression / Phase 5)
- `final_model_name.txt` — name of selected final model

#### Phase 5 outputs — place in `Outputs/price_forecasting/`
- `quantile_forecast_summary_tuned.csv` — MAE, RMSE, R², coverage across all 3 horizons
- `quantile_predictions_return_5d.csv` — median, lower (5th), upper (95th) predictions
- `quantile_predictions_return_10d.csv` — median, lower (5th), upper (95th) predictions
- `quantile_predictions_return_20d.csv` — median, lower (5th), upper (95th) predictions
- `quantile_feature_importance_return_5d.csv` — LightGBM feature importances (5d)
- `quantile_feature_importance_return_10d.csv` — LightGBM feature importances (10d)
- `quantile_feature_importance_return_20d.csv` — LightGBM feature importances (20d)
- `quantile_median_model_return_5d.pkl` — trained median LightGBM model (5d)
- `quantile_median_model_return_10d.pkl` — trained median LightGBM model (10d)
- `quantile_median_model_return_20d.pkl` — trained median LightGBM model (20d)
- `quantile_lower_model_return_5d.pkl` — trained 5th percentile model (5d)
- `quantile_lower_model_return_10d.pkl` — trained 5th percentile model (10d)
- `quantile_lower_model_return_20d.pkl` — trained 5th percentile model (20d)
- `quantile_upper_model_return_5d.pkl` — trained 95th percentile model (5d)
- `quantile_upper_model_return_10d.pkl` — trained 95th percentile model (10d)
- `quantile_upper_model_return_20d.pkl` — trained 95th percentile model (20d)

#### Phase 6 outputs — place in `Outputs/shap_explainability/`
- `shap_feature_importance.csv` — top 20 features by mean |SHAP| (Table 1)
- `sentiment_feature_importance.csv` — top 10 sentiment features by mean |SHAP| (Table 2)
- `shap_group_summary.csv` — feature group contributions: Financial 40.8%, Market 32.2%, Sentiment 23.5%, Sector 3.4% (Table 3)
- `shap_values_test.csv` — raw SHAP values matrix (1,000 × 48)
- `shap_global_bar.png` — Figure 1: global feature importance bar chart
- `shap_beeswarm.png` — Figure 2: beeswarm plot showing feature impact direction
- `shap_group_contribution.png` — Figure 3: feature group contribution chart
- `shap_dependence_net_sentiment.png` — Figure 4: net sentiment dependence plot
- `shap_dependence_sentiment_surprise.png` — Figure 5: sentiment surprise dependence plot
- `shap_dependence_cost_cutting_score.png` — Figure 6: cost cutting score dependence plot
- `shap_waterfall_confident_up.png` — Figure 7: local explanation, confident UP (P=0.624)
- `shap_waterfall_confident_down.png` — Figure 8: local explanation, confident DOWN (P=0.282)
- `shap_waterfall_borderline.png` — Figure 9: local explanation, borderline (P=0.500)
  
## Phase Structure
- Phase 1: `data_preparation.ipynb` ✅ Complete
- Phase 2: `sentiment_extraction.ipynb` ✅ Complete
- Phase 3: `feature.py` ✅ Complete
- Phase 4: `classification.ipynb` ✅ Complete
- Phase 5: Price Forecasting ✅ Complete
- Phase 6: Explainability (SHAP) ✅ Complete
- Phase 7: Industry Benchmarking
- Phase 8: Portfolio Construction & Recommendation
- Phase 9: Equity Research Dashboard
