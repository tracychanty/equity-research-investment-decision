## Data Setup

Data files are not stored in this repository due to size constraints.

Download from: [Google Drive](https://drive.google.com/drive/folders/1NYfYCJvgmE7Gay2oawZWb8ZWBTFLSW38?usp=drive_link)

### Phase 1 inputs/outputs — place in `Data/`
- `cleaned_transcripts_final.pkl` — main Phase 1 output (14,738 rows → 13,803 after dedup)
- `stock_prices_updated.csv` — full price history (2,194 tickers)
- `sp500_prices.csv` — S&P 500 benchmark
- `fundamentals.csv` — financial fundamentals (2,186 tickers)

### Phase 2 outputs — place in `Outputs/sentiment_extraction/`
- `sentiment_features.csv` — main Phase 2 output, Phase 3 input (13,803 rows, 32 columns)
- `sentiment_scores_raw.csv` — FinBERT full transcript scores
- `mgmt_scores_raw.csv` — FinBERT management remarks scores
- `qa_scores_raw.csv` — FinBERT Q&A section scores

## Phase Structure
- Phase 1: `data_preparation.ipynb` ✅ Complete
- Phase 2: Sentiment Extraction (FinBERT pipeline) ✅ Complete
- Phase 3: ML Prediction & Price Forecasting
- Phase 4: Explainability (SHAP)
- Phase 5: Industry Benchmarking
- Phase 6: Portfolio Construction & Recommendation
- Phase 7: Equity Research Dashboard
