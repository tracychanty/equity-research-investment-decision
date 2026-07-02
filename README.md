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

#### Phase 7 outputs — place in `Outputs/industry_benchmarking/`
- `industry_benchmarking_output.csv` — composite scores and rankings for all tickers (34 columns)
- `industry_top10_by_sector.csv` — top 10 ranked tickers per sector by overall score
- `industry_recommendation_summary.csv` — Buy / Hold / Avoid counts by sector
- `equity_research_outputs.sqlite` — SQLite database with all three tables

#### Phase 8 outputs — place in `Outputs/portfolio_construction/`
- `portfolio_candidate_universe.csv` — full candidate universe with scores and predicted returns
- `portfolio_candidate_summary.csv` — summary statistics of candidate universe
- `portfolio_top10_candidates.csv` — top 10 candidates per quarter
- `portfolio_weights.csv` — portfolio weights per quarter (equal weight and market cap weight)
- `portfolio_weight_check.csv` — weight validation checks
- `v3_portfolio_net_returns_wide.csv` — quarterly net returns wide format (all strategies)
- `v3_portfolio_returns_long.csv` — quarterly returns long format
- `v3_portfolio_cumulative_returns.csv` — cumulative returns over backtest period
- `v3_portfolio_drawdowns.csv` — drawdown series per strategy
- `v3_portfolio_weights.csv` — final optimised portfolio weights
- `portfolio_performance_metrics.csv` — Sharpe ratio, CAGR, max drawdown, volatility
- `portfolio_performance_metrics_report.csv` — formatted performance report
- `main_strategy_summary.csv` — ML_Optimized strategy summary statistics

#### Stretch Goal 1 outputs — place in `Outputs/event_study/`
- `events_prepared.csv` — 13,760 valid earnings events with estimation and event windows
- `prices_with_returns.csv` — daily stock and market returns aligned for event study
- `capm_estimates.csv` — CAPM alpha, beta, R² per event (estimation window [-120, -10])
- `abnormal_returns.csv` — AR and CAR across 6 windows per event with sentiment groups
- `significance_summary.csv` — combined Q4 vs Q1 significance: t-test and Mann-Whitney
- `one_sample_tests.csv` — one-sample t-test results per quartile group
- `welch_ttest_results.csv` — Welch's two-sample t-test Q4 vs Q1 per window
- `mann_whitney_tests.csv` — Mann-Whitney U non-parametric test results
- `sector_car_summary.csv` — CAR by sector and sentiment quartile group
- `sector_significance.csv` — t-test significance per sector (CAR_0_1 and CAR_minus1_1)
- `sector_sentiment.csv` — average sentiment scores by sector
- `monotonic_relationship.csv` — CAR progression across Q1 to Q4 quartiles
- `plots/fig1_car_event_time.png` — cumulative abnormal returns Q1 vs Q4 (day -1 to +10)
- `plots/fig2_sector_heatmap.png` — sector × window heatmap of Q4 - Q1 CAR difference
- `plots/fig3_surprise_vs_car.png` — sentiment surprise vs CAR[0,1] scatter (r = 0.057)
- `plots/fig4_ar_distribution.png` — AR distribution Q1 vs Q4 for AR[0] and CAR[0,1]
- `plots/fig5_sector_significance.png` — sector bar chart coloured by significance level
- `plots/fig6_shap_event_bridge.png` — SHAP ML importance vs event study economic impact
- `plots/fig7_monotonic_relationship.png` — CAR bar chart Q1 to Q4 for CAR[0,1] and CAR[-1,1]

## Phase Structure
- Phase 1: `data_preparation.ipynb` ✅ Complete
- Phase 2: `sentiment_extraction.ipynb` ✅ Complete
- Phase 3: `feature.py` ✅ Complete
- Phase 4: `classification.ipynb` ✅ Complete
- Phase 5: Price Forecasting ✅ Complete
- Phase 6: Explainability (SHAP) ✅ Complete
- Phase 7: Industry Benchmarking ✅ Complete
- Phase 8: Portfolio Construction & Recommendation ✅ Complete
- Phase 9: Equity Research Dashboard
- Stretch Goal 1: `event_study.ipynb` ✅ Complete
