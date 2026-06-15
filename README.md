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
- 
## Phase Structure
- Phase 1: `data_preparation.ipynb` ✅ Complete
- Phase 2: `sentiment_extraction.ipynb` ✅ Complete
- Phase 3: `feature.py` ✅ Complete
- Phase 4: `classification.ipynb` ✅ Complete
- Phase 5: Price Forecasting
- Phase 6: Explainability (SHAP)
- Phase 7: Industry Benchmarking
- Phase 8: Portfolio Construction & Recommendation
- Phase 9: Equity Research Dashboard
