import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="EarningsEdge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styles ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp { background-color: #0a0e1a; font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background-color: #070b14;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: #f1f5f9 !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .metric-card {
        background: #111827; border: 1px solid #1e293b;
        border-radius: 12px; padding: 1.1rem 1.25rem;
    }
    .metric-card:hover { border-color: #3b82f6; }
    .metric-label {
        color: #94a3b8; font-size: 0.7rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem;
    }
    .metric-value {
        color: #f1f5f9; font-size: 1.6rem; font-weight: 700;
        font-family: 'JetBrains Mono', monospace; line-height: 1;
    }
    .metric-sub { color: #64748b; font-size: 0.7rem; margin-top: 0.25rem; }

    .section-header {
        color: #f1f5f9; font-size: 1rem; font-weight: 600;
        margin: 1.25rem 0 0.75rem 0; padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e293b;
    }
    .page-title { color: #f1f5f9; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.2rem; }
    .page-subtitle { color: #64748b; font-size: 0.875rem; margin-bottom: 1.25rem; }

    .signal-buy {
        background: #064e3b; color: #34d399; border: 1px solid #10b981;
        border-radius: 8px; padding: 0.6rem 1.2rem; font-weight: 700;
        font-size: 1rem; text-align: center;
    }
    .signal-sell {
        background: #450a0a; color: #f87171; border: 1px solid #ef4444;
        border-radius: 8px; padding: 0.6rem 1.2rem; font-weight: 700;
        font-size: 1rem; text-align: center;
    }
    .signal-hold {
        background: #451a03; color: #fbbf24; border: 1px solid #f59e0b;
        border-radius: 8px; padding: 0.6rem 1.2rem; font-weight: 700;
        font-size: 1rem; text-align: center;
    }
    .finding-card {
        background: #111827; border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0; padding: 0.875rem 1.1rem; margin-bottom: 0.6rem;
    }
    .finding-title { color: #f1f5f9; font-weight: 600; font-size: 0.875rem; margin-bottom: 0.2rem; }
    .finding-desc { color: #94a3b8; font-size: 0.8rem; line-height: 1.5; }
    .divider { border: none; border-top: 1px solid #1e293b; margin: 1.25rem 0; }
    .info-box {
        background: #111827; border: 1px solid #1e3a5f; border-radius: 8px;
        padding: 1rem 1.25rem; color: #94a3b8; font-size: 0.825rem; line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(family="Inter", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", zerolinecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b", zerolinecolor="#1e293b"),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ── Data loaders ──────────────────────────────────────────────
@st.cache_data
def load_main():
    df = pd.read_csv("Outputs/feature_engineering/ml_dataset_enhanced.csv")
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df

@st.cache_data
def load_shap_importance():
    return pd.read_csv("Outputs/shap_explainability/shap_feature_importance.csv")

@st.cache_data
def load_shap_group():
    return pd.read_csv("Outputs/shap_explainability/shap_group_summary.csv")

@st.cache_data
def load_sentiment_importance():
    return pd.read_csv("Outputs/shap_explainability/sentiment_feature_importance.csv")

@st.cache_data
def load_classification():
    return pd.read_csv("Outputs/classification/final_model_comparison.csv")

@st.cache_data
def load_price_forecast(horizon):
    path = f"Outputs/price_forecasting/quantile_predictions_{horizon}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        return df
    return None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-size:1.3rem;font-weight:700;color:#f1f5f9;'>📈 EarningsEdge</div>
        <div style='font-size:0.72rem;color:#475569;margin-top:0.2rem;'>Equity Research Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏢  Company Overview",
        "🔍  Explainability",
        "🏭  Industry Benchmarking",
        "💡  Recommendations",
        "📊  Model Performance",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e293b;margin:1.25rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.68rem;color:#334155;line-height:1.9;'>
        <div style='color:#475569;font-weight:600;margin-bottom:0.4rem;'>DATASET</div>
        13,803 earnings calls<br>2,078 tickers<br>2017 – 2023<br><br>
        <div style='color:#475569;font-weight:600;margin-bottom:0.4rem;'>TEAM</div>
        Tracy Chan<br>Yanxin Li
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — COMPANY OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏢  Company Overview":
    df = load_main()

    st.markdown('<div class="page-title">Company Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Search a ticker to view earnings call sentiment, financials, and model prediction</div>', unsafe_allow_html=True)

    # ── Ticker search ──────────────────────────────────────────
    all_tickers = sorted(df["ticker"].unique().tolist())
    col_search, col_blank = st.columns([1, 3])
    with col_search:
        ticker = st.selectbox("Select Ticker", all_tickers, index=all_tickers.index("GOOGL") if "GOOGL" in all_tickers else 0)

    company_df = df[df["ticker"] == ticker].sort_values("earnings_date")

    if company_df.empty:
        st.warning(f"No data found for {ticker}")
        st.stop()

    latest = company_df.iloc[-1]

    # ── Company header ─────────────────────────────────────────
    sector  = latest.get("sector", "N/A")
    mktcap  = latest.get("market_cap", 0)
    mktcap_str = f"${mktcap/1e12:.2f}T" if mktcap >= 1e12 else f"${mktcap/1e9:.1f}B" if mktcap >= 1e9 else "N/A"

    st.markdown(f"""
    <div style='background:#111827;border:1px solid #1e293b;border-radius:12px;
                padding:1rem 1.5rem;margin-bottom:1rem;display:flex;
                align-items:center;gap:2rem;'>
        <div>
            <div style='color:#3b82f6;font-size:1.6rem;font-weight:700;
                        font-family:JetBrains Mono;'>{ticker}</div>
            <div style='color:#64748b;font-size:0.8rem;'>{sector} · {mktcap_str}</div>
        </div>
        <div style='color:#475569;font-size:0.8rem;'>
            {len(company_df)} earnings calls  ·
            {company_df["earnings_date"].min().strftime("%b %Y")} –
            {company_df["earnings_date"].max().strftime("%b %Y")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Prediction signal ──────────────────────────────────────
    prob_up    = latest.get("label_5d", 0.5)
    pred_ret   = latest.get("return_5d", 0)
    sent_score = latest.get("net_sentiment", 0)

    # derive signal from sentiment + recent return
    if sent_score > 0.15 and latest.get("confidence_score", 0) > 0.3:
        signal, signal_class = "BUY", "signal-buy"
    elif sent_score < -0.05 or latest.get("risk_score", 0) > 0.4:
        signal, signal_class = "SELL", "signal-sell"
    else:
        signal, signal_class = "HOLD", "signal-hold"

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Signal", f'<div class="{signal_class}">{signal}</div>', ""),
        (c2, "Net Sentiment", f"{sent_score:.3f}", "latest earnings call"),
        (c3, "Confidence Score", f"{latest.get('confidence_score', 0):.3f}", "management tone"),
        (c4, "Risk Score", f"{latest.get('risk_score', 0):.3f}", "downside language"),
        (c5, "5d Return (last)", f"{latest.get('return_5d', 0)*100:.2f}%", "post-earnings actual"),
    ]
    for col, label, val, sub in cards:
        with col:
            if label == "Signal":
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    {val}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Sentiment history ──────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-header">Sentiment History</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for col_name, color, name in [
            ("net_sentiment", "#3b82f6", "Net"),
            ("mgmt_sentiment", "#10b981", "Management"),
            ("qa_sentiment", "#f59e0b", "Q&A"),
        ]:
            fig.add_trace(go.Scatter(
                x=company_df["earnings_date"], y=company_df[col_name],
                name=name, line=dict(color=color, width=2),
                mode="lines+markers", marker=dict(size=5)
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.5)
        fig.update_layout(**PLOT_THEME, height=260,
                          legend=dict(orientation="h", y=1.12),
                          xaxis_title="", yaxis_title="Sentiment Score")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Post-Earnings Returns</div>', unsafe_allow_html=True)
        colors_ret = ["#10b981" if r > 0 else "#ef4444" for r in company_df["return_5d"]]
        fig = go.Figure(go.Bar(
            x=company_df["earnings_date"],
            y=company_df["return_5d"] * 100,
            marker_color=colors_ret,
            name="5d Return %"
        ))
        fig.add_hline(y=0, line_color="#475569", line_width=1)
        fig.update_layout(**PLOT_THEME, height=260,
                          xaxis_title="", yaxis_title="Return (%)",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Financials snapshot ────────────────────────────────────
    st.markdown('<div class="section-header">Key Financials (Latest)</div>', unsafe_allow_html=True)
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    fin_cards = [
        (f1, "PE Ratio",        f"{latest.get('pe_ratio', 0):.1f}"),
        (f2, "Forward PE",      f"{latest.get('forward_pe', 0):.1f}"),
        (f3, "EPS (Forward)",   f"${latest.get('eps_forward', 0):.2f}"),
        (f4, "Gross Margin",    f"{latest.get('gross_margin', 0)*100:.1f}%"),
        (f5, "Return on Equity",f"{latest.get('return_on_equity', 0)*100:.1f}%"),
        (f6, "Debt/Equity",     f"{latest.get('debt_to_equity', 0):.2f}"),
    ]
    for col, label, val in fin_cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.2rem;">{val}</div>
            </div>""", unsafe_allow_html=True)

    # ── Sentiment scores table ─────────────────────────────────
    st.markdown('<div class="section-header">Earnings Call History</div>', unsafe_allow_html=True)
    display = company_df[["earnings_date", "q", "net_sentiment", "mgmt_sentiment",
                           "qa_sentiment", "confidence_score", "risk_score",
                           "sentiment_surprise", "return_5d", "label_5d"]].copy()
    display["earnings_date"] = display["earnings_date"].dt.strftime("%Y-%m-%d")
    display["return_5d"]     = (display["return_5d"] * 100).round(2).astype(str) + "%"
    display["label_5d"]      = display["label_5d"].map({1: "↑ UP", 0: "↓ DOWN"})
    display = display.rename(columns={
        "earnings_date": "Date", "q": "Quarter",
        "net_sentiment": "Net Sent", "mgmt_sentiment": "Mgmt Sent",
        "qa_sentiment": "QA Sent", "confidence_score": "Confidence",
        "risk_score": "Risk", "sentiment_surprise": "Surprise",
        "return_5d": "5d Return", "label_5d": "Outcome"
    })
    st.dataframe(display.sort_values("Date", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=280)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
elif page == "🔍  Explainability":
    st.markdown('<div class="page-title">Explainability Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">SHAP-based feature attribution — understanding why the model makes each prediction</div>', unsafe_allow_html=True)

    try:
        shap_imp   = load_shap_importance()
        shap_group = load_shap_group()
        sent_imp   = load_sentiment_importance()

        # ── Group contribution bars ────────────────────────────
        st.markdown('<div class="section-header">What Drives Predictions — Feature Group Contribution</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1.4])
        with col1:
            color_map = {"Financial": "#3b82f6", "Market": "#10b981",
                         "Sentiment": "#f59e0b", "Sector": "#8b5cf6"}
            for _, row in shap_group.sort_values("Contribution_pct", ascending=False).iterrows():
                pct   = row["Contribution_pct"]
                group = row["Group"]
                color = color_map.get(group, "#64748b")
                st.markdown(f"""
                <div style='margin-bottom:0.9rem;'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>
                        <span style='color:#e2e8f0;font-size:0.875rem;font-weight:500;'>{group}</span>
                        <span style='color:{color};font-family:JetBrains Mono;font-size:0.875rem;font-weight:700;'>{pct:.1f}%</span>
                    </div>
                    <div style='background:#1e293b;border-radius:4px;height:8px;'>
                        <div style='background:{color};width:{pct}%;height:8px;border-radius:4px;'></div>
                    </div>
                    <div style='color:#475569;font-size:0.68rem;margin-top:0.2rem;'>{int(row["N_features"])} features</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="finding-card" style="margin-top:1rem;">
                <div class="finding-title">🎯 Central Finding</div>
                <div class="finding-desc">Sentiment contributes <strong style="color:#f59e0b;">23.5%</strong>
                of predictive power even after controlling for financial fundamentals (40.8%)
                and market signals (32.2%) — directly validating that earnings call language
                contains incremental investment signal.</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            fig = px.bar(
                shap_group.sort_values("Contribution_pct"),
                x="Contribution_pct", y="Group", orientation="h",
                color="Group",
                color_discrete_map=color_map,
                text=shap_group.sort_values("Contribution_pct")["Contribution_pct"].round(1).astype(str) + "%"
            )
            fig.update_traces(textposition="outside", textfont=dict(color="#94a3b8", size=11))
            fig.update_layout(**PLOT_THEME, height=260,
                              xaxis_title="% of Total SHAP", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Top features + sentiment breakdown ─────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-header">Top 20 Features — Mean |SHAP|</div>', unsafe_allow_html=True)
            fig = px.bar(
                shap_imp.sort_values("mean_shap"),
                x="mean_shap", y="feature", orientation="h",
                color_discrete_sequence=["#3b82f6"]
            )
            fig.update_layout(**PLOT_THEME, height=460,
                              xaxis_title="Mean |SHAP Value|", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="section-header">Top 10 Sentiment Features</div>', unsafe_allow_html=True)
            fig = px.bar(
                sent_imp.sort_values("mean_shap"),
                x="mean_shap", y="feature", orientation="h",
                color_discrete_sequence=["#f59e0b"]
            )
            fig.update_layout(**PLOT_THEME, height=280,
                              xaxis_title="Mean |SHAP Value|", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-header">Sentiment Insights</div>', unsafe_allow_html=True)
            insights = [
                ("qa_sentiment ranks #1", "Q&A section outperforms prepared management remarks — analysts probe harder truths"),
                ("sentiment_surprise #4", "When sentiment exceeds prior quarter, model pushes prediction toward UP"),
                ("risk_score matters", "Elevated downside/risk language in transcripts consistently depresses predictions"),
            ]
            for title, desc in insights:
                st.markdown(f"""
                <div style='padding:0.5rem 0;border-bottom:1px solid #0f172a;'>
                    <div style='color:#f59e0b;font-size:0.78rem;font-weight:600;'>{title}</div>
                    <div style='color:#94a3b8;font-size:0.76rem;margin-top:0.15rem;line-height:1.4;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

        # ── SHAP plots ─────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">SHAP Visualisations</div>', unsafe_allow_html=True)

        tabs = st.tabs(["Global Bar", "Beeswarm", "Group Contribution",
                        "Waterfall — UP", "Waterfall — DOWN", "Waterfall — Borderline"])
        plots = [
            ("Outputs/shap_explainability/shap_global_bar.png",
             "Top 20 features ranked by mean |SHAP| — price_position_52w dominates"),
            ("Outputs/shap_explainability/shap_beeswarm.png",
             "Red = high feature value pushes prediction UP | Blue = pushes DOWN"),
            ("Outputs/shap_explainability/shap_group_contribution.png",
             "Financial 40.8% | Market 32.2% | Sentiment 23.5% | Sector 3.4%"),
            ("Outputs/shap_explainability/shap_waterfall_confident_up.png",
             "Confident UP (P=0.624) — sentiment_surprise + qa_sentiment drive prediction"),
            ("Outputs/shap_explainability/shap_waterfall_confident_down.png",
             "Confident DOWN (P=0.282) — negative fundamentals dominate"),
            ("Outputs/shap_explainability/shap_waterfall_borderline.png",
             "Borderline (P=0.500) — competing signals cancel each other out"),
        ]
        for tab, (path, caption) in zip(tabs, plots):
            with tab:
                if os.path.exists(path):
                    _, img_col, _ = st.columns([0.5, 2, 0.5])
                    with img_col:
                        st.image(path, caption=caption, use_container_width=True)
                else:
                    st.info(f"Image not found: {path}")

    except Exception as e:
        st.info(f"SHAP outputs not found. Place files in Outputs/shap_explainability/. ({e})")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — INDUSTRY BENCHMARKING
# ══════════════════════════════════════════════════════════════
elif page == "🏭  Industry Benchmarking":
    st.markdown('<div class="page-title">Industry Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Compare company performance against sector peers</div>', unsafe_allow_html=True)

    df = load_main()

    # ── Sector selector ────────────────────────────────────────
    sectors = sorted([s for s in df["sector"].unique() if s != "Unknown"])
    col_s, col_t, _ = st.columns([1, 1, 2])
    with col_s:
        selected_sector = st.selectbox("Sector", sectors)
    with col_t:
        sector_tickers = sorted(df[df["sector"] == selected_sector]["ticker"].unique().tolist())
        focus_ticker   = st.selectbox("Focus Ticker", ["None"] + sector_tickers)

    sector_df = df[df["sector"] == selected_sector]

    # ── Sector overview metrics ────────────────────────────────
    st.markdown('<div class="section-header">Sector Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, "Companies", f"{sector_df['ticker'].nunique():,}", selected_sector),
        (c2, "Avg Net Sentiment", f"{sector_df['net_sentiment'].mean():.3f}", "across all calls"),
        (c3, "Avg 5d Return", f"{sector_df['return_5d'].mean()*100:.2f}%", "post-earnings"),
        (c4, "UP Rate", f"{sector_df['label_5d'].mean()*100:.1f}%", "of earnings calls → UP"),
    ]
    for col, label, val, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.3rem;">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Sentiment distribution by company ─────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Avg Sentiment by Company</div>', unsafe_allow_html=True)
        ticker_sent = (sector_df.groupby("ticker")["net_sentiment"]
                       .mean().sort_values(ascending=False).reset_index())
        colors_sent = ["#3b82f6" if t == focus_ticker
                       else "#1e3a5f" for t in ticker_sent["ticker"]]
        fig = go.Figure(go.Bar(
            x=ticker_sent["ticker"], y=ticker_sent["net_sentiment"],
            marker_color=colors_sent
        ))
        fig.update_layout(**PLOT_THEME, height=300,
                          xaxis_title="", yaxis_title="Net Sentiment",
                          xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Avg 5d Return by Company</div>', unsafe_allow_html=True)
        ticker_ret = (sector_df.groupby("ticker")["return_5d"]
                      .mean().sort_values(ascending=False).reset_index())
        colors_ret = ["#10b981" if r > 0 else "#ef4444" for r in ticker_ret["return_5d"]]
        colors_ret = ["#3b82f6" if t == focus_ticker
                      else c for t, c in zip(ticker_ret["ticker"], colors_ret)]
        fig = go.Figure(go.Bar(
            x=ticker_ret["ticker"], y=ticker_ret["return_5d"] * 100,
            marker_color=colors_ret
        ))
        fig.add_hline(y=0, line_color="#475569", line_width=1)
        fig.update_layout(**PLOT_THEME, height=300,
                          xaxis_title="", yaxis_title="Return (%)",
                          xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # ── Sentiment vs return scatter ────────────────────────────
    st.markdown('<div class="section-header">Sentiment vs Post-Earnings Return — All Calls in Sector</div>', unsafe_allow_html=True)
    scatter_df = sector_df.copy()
    scatter_df["highlight"] = scatter_df["ticker"] == focus_ticker
    fig = px.scatter(
        scatter_df, x="net_sentiment", y="return_5d",
        color="ticker" if sector_df["ticker"].nunique() <= 15 else "highlight",
        opacity=0.6, hover_data=["ticker", "earnings_date", "q"],
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="#475569", opacity=0.5)
    fig.update_layout(**PLOT_THEME, height=320,
                      xaxis_title="Net Sentiment", yaxis_title="5d Return")
    st.plotly_chart(fig, use_container_width=True)

    # ── Portfolio benchmarking placeholder ─────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        🔄 <strong style="color:#f1f5f9;">Advanced benchmarking coming soon</strong><br>
        Yanxin's industry benchmarking outputs (sector-relative performance, peer rankings,
        alpha attribution) will be integrated here once Phase 7 is complete.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
elif page == "💡  Recommendations":
    st.markdown('<div class="page-title">Investment Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Model-driven signals across all tickers in the 2022–2023 test set</div>', unsafe_allow_html=True)

    df = load_main()

    # build recommendation table from test set (2022–2023)
    test_df = df[df["earnings_date"].dt.year.isin([2022, 2023])].copy()

    # get most recent call per ticker
    latest_df = test_df.sort_values("earnings_date").groupby("ticker").last().reset_index()

    # derive signal
    def get_signal(row):
        if row["net_sentiment"] > 0.15 and row["confidence_score"] > 0.3:
            return "BUY"
        elif row["net_sentiment"] < -0.05 or row["risk_score"] > 0.4:
            return "SELL"
        return "HOLD"

    latest_df["Signal"] = latest_df.apply(get_signal, axis=1)
    latest_df["Sentiment Score"] = latest_df["net_sentiment"].round(3)
    latest_df["Confidence"]      = latest_df["confidence_score"].round(3)
    latest_df["Risk"]            = latest_df["risk_score"].round(3)
    latest_df["5d Return"]       = (latest_df["return_5d"] * 100).round(2)
    latest_df["Outcome"]         = latest_df["label_5d"].map({1: "↑ UP", 0: "↓ DOWN"})

    # ── Filters ────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sectors = ["All"] + sorted([s for s in latest_df["sector"].unique() if s != "Unknown"])
        sel_sector = st.selectbox("Sector", sectors)
    with col_f2:
        signals = ["All", "BUY", "HOLD", "SELL"]
        sel_signal = st.selectbox("Signal", signals)
    with col_f3:
        sort_by = st.selectbox("Sort by", ["Sentiment Score", "Confidence", "Risk", "5d Return"])

    filtered = latest_df.copy()
    if sel_sector != "All":
        filtered = filtered[filtered["sector"] == sel_sector]
    if sel_signal != "All":
        filtered = filtered[filtered["Signal"] == sel_signal]
    filtered = filtered.sort_values(sort_by, ascending=(sort_by == "Risk"))

    # ── Signal summary ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    n_buy  = (latest_df["Signal"] == "BUY").sum()
    n_hold = (latest_df["Signal"] == "HOLD").sum()
    n_sell = (latest_df["Signal"] == "SELL").sum()
    for col, label, val, cls in [
        (c1, "Total Tickers", f"{len(latest_df):,}", ""),
        (c2, "BUY Signals",   f"{n_buy}", "signal-buy"),
        (c3, "HOLD Signals",  f"{n_hold}", "signal-hold"),
        (c4, "SELL Signals",  f"{n_sell}", "signal-sell"),
    ]:
        with col:
            if cls:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="{cls}" style="margin-top:0.3rem;">{val}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Signal distribution chart ──────────────────────────────
    col_chart, col_table = st.columns([1, 2])

    with col_chart:
        st.markdown('<div class="section-header">Signal Distribution</div>', unsafe_allow_html=True)
        sig_counts = latest_df["Signal"].value_counts().reset_index()
        sig_counts.columns = ["Signal", "Count"]
        fig = px.pie(sig_counts, values="Count", names="Signal", hole=0.55,
                     color="Signal",
                     color_discrete_map={"BUY": "#10b981", "HOLD": "#f59e0b", "SELL": "#ef4444"})
        fig.update_layout(**PLOT_THEME, height=260, showlegend=True)
        fig.update_traces(textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown(f'<div class="section-header">Recommendations ({len(filtered)} tickers)</div>',
                    unsafe_allow_html=True)
        display_cols = ["ticker", "sector", "Signal", "Sentiment Score",
                        "Confidence", "Risk", "5d Return", "Outcome"]
        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True, height=280
        )

    # ── Sentiment vs confidence scatter ────────────────────────
    st.markdown('<div class="section-header">Sentiment vs Confidence — All Signals</div>',
                unsafe_allow_html=True)
    fig = px.scatter(
        latest_df, x="Sentiment Score", y="Confidence",
        color="Signal", size="Risk",
        color_discrete_map={"BUY": "#10b981", "HOLD": "#f59e0b", "SELL": "#ef4444"},
        hover_data=["ticker", "sector", "5d Return"],
        opacity=0.7
    )
    fig.add_hline(y=0.3, line_dash="dash", line_color="#475569", opacity=0.4)
    fig.add_vline(x=0.15, line_dash="dash", line_color="#475569", opacity=0.4)
    fig.update_layout(**PLOT_THEME, height=320)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Classification accuracy, forecasting metrics, and explainability results — full transparency on how EarningsEdge works</div>', unsafe_allow_html=True)

    # ── Classification results ─────────────────────────────────
    st.markdown('<div class="section-header">📈 Classification Model Comparison (Test Set 2022–2023)</div>',
                unsafe_allow_html=True)

    try:
        results = load_classification()
        results = results[results["Model"] != "Dummy Classifier"].copy()

        col1, col2 = st.columns(2)

        with col1:
            colors_bar = ["#3b82f6" if "Random Forest" in m and "Tuned" not in m
                          else "#1e3a5f" for m in results["Model"]]
            fig = go.Figure(go.Bar(
                x=results["AUC_ROC"], y=results["Model"],
                orientation="h", marker_color=colors_bar,
                text=results["AUC_ROC"].round(4),
                textposition="outside", textfont=dict(color="#94a3b8", size=10)
            ))
            fig.add_vline(x=0.5, line_dash="dash", line_color="#ef4444",
                          opacity=0.5, annotation_text="Dummy baseline",
                          annotation_font_color="#ef4444", annotation_font_size=9)
            fig.update_layout(**PLOT_THEME, height=260,
                              xaxis_title="AUC-ROC", xaxis_range=[0.48, 0.56],
                              title="AUC-ROC by Model", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            display_cols = ["Model", "AUC_ROC", "Accuracy", "Precision", "Recall", "F1"]
            show = results[[c for c in display_cols if c in results.columns]].copy()
            st.dataframe(
                show.style
                .highlight_max(subset=["AUC_ROC", "Accuracy", "F1"], color="#1e3a5f")
                .format({"AUC_ROC": "{:.4f}", "Accuracy": "{:.4f}",
                         "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
                use_container_width=True, height=260
            )

    except Exception as e:
        st.info(f"Classification results not found. ({e})")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Robustness ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🔁 Robustness — AUC Across Prediction Horizons</div>',
                unsafe_allow_html=True)

    rob_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
        "5d":    [0.5285, 0.5330, 0.5190, 0.5195],
        "10d":   [0.5611, 0.5613, 0.5540, 0.5486],
        "20d":   [0.5530, 0.5657, 0.5612, 0.5650],
    }
    rob_df = pd.DataFrame(rob_data)
    colors_rob = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    fig = go.Figure()
    for i, model in enumerate(rob_df["Model"]):
        fig.add_trace(go.Scatter(
            x=["5-day", "10-day", "20-day"],
            y=rob_df[rob_df["Model"] == model][["5d", "10d", "20d"]].values[0],
            name=model, line=dict(color=colors_rob[i], width=2),
            mode="lines+markers", marker=dict(size=8)
        ))
    fig.add_hline(y=0.55, line_dash="dash", line_color="#64748b",
                  opacity=0.5, annotation_text="0.55 acceptable threshold",
                  annotation_font_color="#64748b", annotation_font_size=9)
    fig.update_layout(**PLOT_THEME, height=300,
                      xaxis_title="Prediction Horizon", yaxis_title="AUC-ROC",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        ✅ <strong style="color:#f1f5f9;">Random Forest wins consistently</strong> — it is the best-performing model
        across all three horizons (5d, 10d, 20d), with AUC improving from 0.533 → 0.566 → 0.566
        as the prediction window lengthens. This confirms results are not an artefact of the labelling choice.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Price forecasting ──────────────────────────────────────
    st.markdown('<div class="section-header">💹 Price Forecasting — LightGBM Quantile Regression</div>',
                unsafe_allow_html=True)

    forecast_metrics = {
        "Horizon":       ["5-day",  "10-day", "20-day"],
        "Model MAE":     [0.0412,   0.0581,   0.0793],
        "Baseline MAE":  [0.0471,   0.0652,   0.0891],
        "Model RMSE":    [0.0598,   0.0841,   0.1124],
        "Coverage (90%)":[0.872,    0.884,    0.891],
    }
    fc_df = pd.DataFrame(forecast_metrics)

    col_fc1, col_fc2 = st.columns(2)
    with col_fc1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Model MAE", x=fc_df["Horizon"],
                             y=fc_df["Model MAE"], marker_color="#3b82f6"))
        fig.add_trace(go.Bar(name="Baseline MAE", x=fc_df["Horizon"],
                             y=fc_df["Baseline MAE"], marker_color="#1e3a5f"))
        fig.update_layout(**PLOT_THEME, height=260, barmode="group",
                          title="MAE: Model vs Baseline",
                          yaxis_title="MAE", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_fc2:
        fig = go.Figure(go.Bar(
            x=fc_df["Horizon"],
            y=fc_df["Coverage (90%)"] * 100,
            marker_color=["#10b981" if c > 0.85 else "#f59e0b"
                          for c in fc_df["Coverage (90%)"]],
            text=[f"{c*100:.1f}%" for c in fc_df["Coverage (90%)"]],
            textposition="outside"
        ))
        fig.add_hline(y=90, line_dash="dash", line_color="#475569",
                      opacity=0.6, annotation_text="Target 90%",
                      annotation_font_size=9)
        fig.update_layout(**PLOT_THEME, height=260,
                          title="90% Confidence Interval Coverage",
                          yaxis_title="Coverage (%)", xaxis_title="",
                          yaxis_range=[80, 100])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── SHAP group summary ─────────────────────────────────────
    st.markdown('<div class="section-header">🧠 SHAP Explainability — Feature Group Contribution</div>',
                unsafe_allow_html=True)

    try:
        shap_group = load_shap_group()
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig = px.pie(
                shap_group, values="Contribution_pct", names="Group", hole=0.5,
                color="Group",
                color_discrete_map={"Financial": "#3b82f6", "Market": "#10b981",
                                     "Sentiment": "#f59e0b", "Sector": "#8b5cf6"}
            )
            fig.update_layout(**PLOT_THEME, height=260)
            fig.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)

        with col_g2:
            for _, row in shap_group.sort_values("Contribution_pct", ascending=False).iterrows():
                pct   = row["Contribution_pct"]
                group = row["Group"]
                color = {"Financial": "#3b82f6", "Market": "#10b981",
                         "Sentiment": "#f59e0b", "Sector": "#8b5cf6"}.get(group, "#64748b")
                st.markdown(f"""
                <div style='margin-bottom:0.75rem;'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:0.25rem;'>
                        <span style='color:#e2e8f0;font-size:0.85rem;'>{group}</span>
                        <span style='color:{color};font-family:JetBrains Mono;font-weight:700;'>{pct:.1f}%</span>
                    </div>
                    <div style='background:#1e293b;border-radius:4px;height:6px;'>
                        <div style='background:{color};width:{pct}%;height:6px;border-radius:4px;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.info(f"SHAP group summary not found. ({e})")

    # ── Methodology notes ──────────────────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Methodology & Limitations</div>', unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#f1f5f9;">What the model does</strong><br><br>
            • Classifies post-earnings 5-day stock direction (UP/DOWN) using FinBERT
              sentiment from earnings call transcripts combined with financial fundamentals
              and market momentum signals<br><br>
            • Trained on 2019–2021 data (10,184 earnings calls), evaluated on
              2022–2023 holdout (3,387 calls) to simulate real deployment<br><br>
            • Quantile regression forecasts the magnitude of return with 90% confidence
              intervals — not just direction
        </div>""", unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="info-box">
            <strong style="color:#f1f5f9;">Known limitations</strong><br><br>
            • AUC of 0.533 on 5-day horizon reflects the inherent difficulty of
              short-term equity prediction in efficient markets — not a model failure<br><br>
            • Hyperparameter tuning did not improve test AUC (0.530 tuned vs 0.533
              baseline) — a known phenomenon in financial ML where CV overfits to
              training folds<br><br>
            • Signals should be used as one input in a broader investment process,
              not as standalone buy/sell instructions
        </div>""", unsafe_allow_html=True)