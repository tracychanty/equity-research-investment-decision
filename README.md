**AI-Powered Equity Research & Investment Decision Platform**

Tracy Chan ([tsz.y.chan@mail.mcgill.ca](mailto:tsz.y.chan@mail.mcgill.ca)) · Yanxin Li ([yanxin.li@mail.mcgill.ca](mailto:yanxin.li@mail.mcgill.ca))

**Live dashboard:** [investor-copilot.streamlit.app](https://investor-copilot.streamlit.app/)

---

## Overview

Investor Copilot turns unstructured earnings-call transcripts into an explainable, end-to-end equity research pipeline. It combines FinBERT-based sentiment extraction, financial fundamentals, and market data to predict post-earnings stock direction and return ranges. It then explains the model predictions with SHAP, benchmarks companies against sector peers, and converts the signal into a backtested portfolio strategy. All outputs are delivered through a live multi-page Streamlit dashboard.

The goal is to give retail investors, students, and early-career professionals access to qualitative-signal research that is usually reserved for institutional analysts, without requiring their budget or headcount.

## Why This Exists

Public companies release a large amount of qualitative information each quarter through earnings calls, including tone, management confidence, and hedging language. Prior research shows that this language can move stock prices beyond what is explained by reported financial numbers alone (Loughran & McDonald, 2011; Ke, Kelly & Xiu, 2019). But interpreting that information at scale is expensive, time-consuming, and difficult to do consistently.

Investor Copilot automates that scoring process, connects the signal to predicted market reaction, and explains why the model reaches its conclusions instead of operating as a black box.

## Project Pipeline

| # | Phase | Summary |
|---|-------|---------|
| 1 | Data Preparation | Clean, join, and label raw transcript and price data |
| 2 | Sentiment Extraction | Extract transcript sentiment using FinBERT and split management vs. Q&A |
| 3 | Feature Engineering | Build the full feature set from sentiment, fundamentals, and market data |
| 4 | Classification | Predict post-earnings stock direction |
| 5 | Price Forecasting | Quantile regression for expected return ranges |
| 6 | SHAP Explainability | Explain model predictions globally and locally |
| 7 | Industry Benchmarking | Score companies against sector peers |
| 8 | Portfolio Construction & Backtest | Turn signals into a tested investment strategy |
| 9 | Dashboard | Integrate pipeline outputs into one live analyst-facing tool |
| S1 | Event Study *(stretch goal)* | Test whether sentiment is associated with abnormal returns around earnings |
| S2 | RAG Financial Chatbot *(stretch goal, local-only)* | Create a local RAG-based financial Q&A tool for uploaded filings |

## Data



**Train/test split** (chronological, to avoid look-ahead bias): 2019–2021 train (10,831 rows, 2,052 tickers) · 2022–2023 test (3,673 rows, 1,031 tickers).

**Sources:** Motley Fool earnings call transcripts (Kaggle), yfinance daily OHLCV, yfinance company fundamentals, S&P 500 (`^GSPC`) as market benchmark.

## Data

| Stage | Rows | Tickers | Notes |
|---|---:|---:|---|
| Raw Motley Fool transcripts | 18,755 | 2,876 | 2017–2023 transcript universe |
| After price join | 14,857 | — | Some transcript tickers had no matching price history |
| After ticker standardization + 2-quarter minimum | 14,738 | 2,078 | `cleaned_transcripts_final.csv` |
| After duplicate removal | **13,803** | **2,078** | Final modeling sample used in all downstream phases |

**Train/test split** (chronological, to avoid look-ahead bias): 2019–2021 train (10,831 rows; 2,052 tickers) · 2022–2023 test (3,673 rows; 1,031 tickers).

**Sources:** Motley Fool earnings-call transcripts (Kaggle), yfinance daily OHLCV, yfinance company fundamentals, and the S&P 500 (`^GSPC`) as the market benchmark.

## Phase Highlights

### Sentiment Extraction (Phase 2)
- Applied FinBERT separately to management remarks and Q&A sections
- Rebuilt the speaker-level parser after an initial version failed on Motley Fool transcripts (0% parse rate), reaching 99.8% coverage on management text and 97.6% on Q&A
- Produced **32 sentiment features**, including transcript-level tone, management/Q&A split sentiment, keyword-based confidence/risk/uncertainty scores, and `sentiment_surprise` (quarter-over-quarter tone change)
  
### Feature Engineering (Phase 3)
- Built a 68-column enhanced dataset (`ml_dataset_enhanced.csv`) from sentiment, fundamentals, and market data
- Reduced the feature set to **48 final features** through correlation filtering (dropped 13 at r > 0.85), VIF filtering (dropped 4 at VIF > 20), and 5-fold cross-validation
- Retained a balanced set of sentiment, financial, and market features for downstream modeling
  
### Classification (Phase 4)
- Compared five models: Dummy, Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Random Forest was selected as the final classifier** with the highest test AUC-ROC (**0.5330**) and accuracy of **52.35%**
- Tested hyperparameter tuning, but retained the untuned Random Forest because it outperformed the tuned version on the test set
- AUC was used as the primary selection metric because downstream portfolio ranking depends more on relative ordering than on a single classification threshold
  
| Model | AUC-ROC | Accuracy | F1 |
|---|---|---|---|
| **Random Forest** *(selected; best AUC)* | **0.5330** | **52.35%** | **0.5430** |
| Random Forest (Tuned) | 0.5301 | 52.41% | 0.5373 |
| Logistic Regression | 0.5285 | 52.20% | 0.4961 |
| LightGBM | 0.5195 | 52.02% | 0.5700 |
| XGBoost | 0.5190 | 51.25% | 0.5603 |
| Dummy (baseline) | 0.5000 | 51.34% | 0.6785 |

### Price Forecasting (Phase 5)
- Built LightGBM quantile regression models (α = 0.05 / 0.50 / 0.95) for 5-, 10-, and 20-day return horizons
- Achieved **R² of 0.003–0.026** and **MAE of 8.0–11.2%** across horizons
- Forecast outputs were used primarily as a **ranking signal** for portfolio construction rather than as precise point estimates
  
### SHAP Explainability (Phase 6)
- Applied TreeExplainer to the final Random Forest model on a **1,000-row sample** of the test set (27.2% of 3,673 rows, capped for computational cost)
- Feature-group contributions were **Financial 40.8%**, **Market 32.2%**, **Sentiment 23.6%**, and **Sector 3.4%**
- The top overall feature was `price_position_52w`, while `qa_sentiment` was the most important sentiment feature, indicating that Q&A tone carried more signal than management remarks
  
### Industry Benchmarking (Phase 7)
- Benchmarked companies across six sector-relative dimensions: Sentiment (25%), Financial Quality (25%), Forecast (20%), Valuation (10%), Leverage (10%), Safety (10%)
- Combined these into a composite `overall_score` used to generate **Buy (≥0.70) / Hold (≥0.45) / Avoid** recommendations
- Framed recommendations in relative sector context rather than as absolute market-wide scores

### Portfolio Construction & Backtest (Phase 8)
- Tested six strategies: `ML_Optimized` (featured, mean-variance optimized), `Full_Universe` (passive benchmark), `Equal_Weight`, `Market_Cap_Weight`, `ML_Score_Weight`, and `Long_Short`
- Applied portfolio constraints of 20% max position weight, 40% max sector weight, 10bps transaction cost, risk-aversion coefficient 5.0, and 8–15 holdings per quarter
- Achieved a **1.552 Sharpe ratio** with the featured `ML_Optimized` strategy, clearing the project’s ≥1.0 target but trailing `Equal_Weight` (1.916)
- Excluded `Long_Short` from the core recommendation despite its higher paper Sharpe (3.41), because it was based on only 5 quarters, showed unrealistic stability, and requires shorting capability
  
### Event Study — Stretch Goal (S1)
- Estimated CAPM-adjusted abnormal returns using an estimation window of [−120, −10] and multiple event windows around earnings
- Evaluated **13,760 valid earnings events**, ranked into sentiment quartiles from Q1 (most negative) to Q4 (most positive)
- Q4 outperformed Q1 by **+1.07pp in CAR[0,+1]**, significant at **p < 0.001** in both Welch’s t-test and Mann-Whitney U
- The effect was strongest around the earnings date and faded over longer windows, with sector-level significance in Technology, Industrials, Basic Materials, Real Estate, and Utilities
  
### RAG Financial Chatbot — Stretch Goal (S2)
- Built a local retrieval-augmented assistant for natural-language Q&A over uploaded financial filings
- Implemented PDF/TXT ingestion, chunking, FAISS retrieval, source-cited answers, and hallucination guardrails
- Runs locally through Ollama (`qwen2.5:3b`) and is not yet integrated into the hosted dashboard

### Dashboard (Phase 9)
- Integrated the pipeline into a live Streamlit dashboard that turns model outputs into an analyst-facing workflow
- Connected each page directly to pipeline CSV outputs so results update consistently across the app
- Made the following pages available:
  1. **Company Overview** — single-ticker sentiment, fundamentals, and latest Buy/Hold/Avoid signal
  2. **Peer Benchmarking** — sector-relative comparison, rankings, and Buy/Hold/Avoid recommendations
  3. **Investment Recommendations** — portfolio holdings, weights, strategy comparison, and backtest results
  4. **Explainability & Economic Evidence** — SHAP drivers and event-study validation
  5. **Model Validation** — classification, forecasting, and robustness metrics
  6. **AI Research Assistant** — local RAG chatbot for uploaded financial documents
     
## Success Metrics vs. Proposal

| Metric | Target | Actual | Status |
|---|---|---|---|
| Sentiment coverage | ≥ 95% | 100% core scores · 97.7% on `qa_sentiment` | ✅ Achieved |
| Prediction AUC / Accuracy | AUC ≥ 0.65 · Acc ≥ 60% | AUC 0.533 · Accuracy 52.4% | ❌ Not met |
| Price range MAE | ≤ 5% | 8.0–11.2% across horizons | ❌ Not met |
| SHAP coverage | 100% of predictions | 27.2% of test set (1,000 / 3,673) | ⚠️ Partial |
| Backtest Sharpe ratio | ≥ 1.0 or beat Equal-Weight | 1.552 — clears the floor, trails Equal-Weight (1.916) | ⚠️ Partial |

## What Holds Up

- Near-complete transcript coverage enabled broad and consistent sentiment analysis
- Q&A sentiment and `sentiment_surprise` emerged as the most informative sentiment features
- The event study found a statistically significant short-window sentiment–return relationship around earnings
- SHAP validated the model’s main drivers and made sampled predictions interpretable, transparent, and auditable
- The featured portfolio strategy cleared the project’s Sharpe ratio floor
- The dashboard successfully integrated the pipeline into one working analyst-facing tool

## Limitations

- Directional classification performance was modest (AUC 0.53), limiting standalone predictive strength
- Price forecasting error remained above the original MAE target
- SHAP explanations were generated for a sample of the test set rather than all predictions
- The `ML_Optimized` portfolio did not outperform the simpler `Equal_Weight` benchmark
- The RAG assistant remains local-only and is not yet integrated into the hosted dashboard
  
## Future Work

- Extend SHAP explanations from the 1,000-row sample to the full test set
- Test richer forecasting features and sequence-based models to improve predictive accuracy
- Strengthen portfolio construction with turnover-aware optimization and a full covariance-based risk model
- Formally benchmark dashboard query latency against the <5 second target
- Integrate the RAG assistant into the hosted dashboard when deployment and API budget allow
  
## Tech Stack

- **NLP / ML:** FinBERT for transcript sentiment extraction; scikit-learn, XGBoost, and LightGBM for predictive modeling; SHAP (TreeExplainer) for explainability
- **Portfolio Optimization:** cvxpy for mean-variance portfolio construction
- **RAG Pipeline:** LangChain, FAISS, and Ollama (`qwen2.5:3b`) for local document retrieval and question answering
- **Data Processing:** pandas, numpy, and yfinance
- **Dashboard / Visualization:** Streamlit and Plotly
- **Development Environments:** Jupyter, Google Colab (T4 GPU), and Kaggle Notebooks
- **Deployment:** Streamlit Community Cloud
  
## Repository Structure

```
Outputs/
├── feature_engineering/
│   └── ml_dataset_enhanced.csv
├── classification/
│   ├── final_model_comparison.csv
│   └── robustness_summary.csv
├── price_forecasting/
│   └── quantile_forecast_summary_tuned.csv
├── shap_explainability/
│   ├── shap_values_test.csv
│   ├── shap_feature_importance.csv
│   ├── shap_group_summary.csv
│   └── sentiment_feature_importance.csv
├── industry_benchmarking/
│   └── industry_benchmarking_output.csv
├── event_study/
│   ├── abnormal_returns.csv
│   ├── significance_summary.csv
│   ├── monotonic_relationship.csv
│   └── sector_car_summary.csv
└── portfolio_construction/
    ├── v3_portfolio_weights.csv
    ├── v3_portfolio_net_returns_wide.csv
    ├── v3_portfolio_cumulative_returns.csv
    ├── portfolio_performance_metrics_report.csv
    └── main_strategy_summary.csv
```

## References

- Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35–65.
- Ke, B., Kelly, X., & Xiu, D. (2019). Predicting returns with text data. *NBER Working Paper No. 26186*.
- Phan, T. (2024). Sentiment-semantic word vectors: A new method to estimate management sentiment. *Swiss Journal of Economics and Statistics*, 160, 1–19.
