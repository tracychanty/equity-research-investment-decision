import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Investor Copilot",
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

    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label {
        padding: 0.75rem 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 10px;
        font-size: 1.05rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: #111827;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .metric-card {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 12px; padding: 1.1rem 1.25rem;
    min-height: 100px; display: flex; flex-direction: column;
    justify-content: center;
    }
    .metric-card:hover { border-color: #3b82f6; }
    .metric-card { position: relative; overflow: visible; }
    .tip-toggle { display: none; }
    .tip-icon {
        position: absolute; top: 0.6rem; right: 0.7rem;
        width: 16px; height: 16px; border-radius: 50%;
        background: #1e293b; color: #94a3b8; font-size: 0.65rem;
        display: flex; align-items: center; justify-content: center;
        cursor: pointer; border: 1px solid #334155; user-select: none;
        z-index: 10;
    }
    .tip-icon:hover { background: #334155; color: #f1f5f9; }
    .tip-desc {
        display: none; position: absolute; top: 2.1rem; right: 0.6rem;
        width: 210px; background: #1e293b; border: 1px solid #334155;
        border-radius: 8px; padding: 0.65rem 0.8rem; color: #cbd5e1;
        font-size: 0.72rem; line-height: 1.4; z-index: 30;
        box-shadow: 0 6px 16px rgba(0,0,0,0.5);
    }
    .tip-toggle:checked ~ .tip-desc { display: block; }
    .metric-label {
        color: #aab4c4; font-size: 0.78rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.35rem;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .metric-value {
        color: #f1f5f9; font-size: 1.6rem; font-weight: 700;
        font-family: 'JetBrains Mono', monospace; line-height: 1;
    }
    .metric-sub { color: #8b96a8; font-size: 0.8rem; margin-top: 0.3rem; line-height: 1.4; }

    .section-header {
        color: #f1f5f9; font-size: 1rem; font-weight: 600;
        margin: 1.25rem 0 0.75rem 0; padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e293b;
    }
    .page-title { color: #f1f5f9; font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .page-subtitle { color: #64748b; font-size: 0.875rem; margin-bottom: 1.25rem; }
    [data-testid="stWidgetLabel"] p {
        color: #64748b !important; font-size: 0.85rem !important;
    }
            
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

# ── Back-to-top anchor (all pages) ──────────────────────────────
# Note: no JS/iframe here on purpose — the components.html approach
# depends on reaching into the parent document from inside a sandboxed
# iframe, which is unreliable across Streamlit versions/browsers. A
# plain anchor link is simpler and always works, at the cost of no
# fade-in-on-scroll and an instant (not animated) jump.
st.markdown("<div id='page-top'></div>", unsafe_allow_html=True)
st.markdown("""
<a href="#page-top" title="Back to top" style="
    position:fixed; bottom:2rem; right:2rem; width:44px; height:44px;
    border-radius:50%; background:#111827; border:1px solid #3b82f6;
    color:#3b82f6; display:flex; align-items:center; justify-content:center;
    font-size:1.3rem; text-decoration:none;
    box-shadow:0 4px 12px rgba(0,0,0,0.45); z-index:999999;">↑</a>
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
def load_classification():
    return pd.read_csv("Outputs/classification/final_model_comparison.csv")

# Feature → group mapping for SHAP drill-down. Reverse-engineered from
# shap_values_test.csv column names and validated against
# Outputs/shap_explainability/shap_group_summary.csv — grouping every
# feature this way reproduces the exact Mean_SHAP figures in that file
# (sum of per-feature mean|SHAP| within each group), so this is confirmed
# correct, not a guess. Notably 'beta' is grouped under Financial, not
# Market — that's what the notebook's original grouping actually used.
SHAP_SENTIMENT_FEATURES = [
    "risk_score", "qa_neutral_ratio", "uncertainty_score", "qa_sentiment",
    "mgmt_neutral_ratio", "mgmt_sentiment_volatility", "qa_negative_ratio",
    "qa_sentiment_volatility", "mgmt_sentiment", "confidence_score",
    "sentiment_surprise", "negative_ratio", "net_sentiment", "cost_cutting_score",
]
SHAP_MARKET_FEATURES = [
    "volatility_10d", "price_position_52w", "momentum_5d", "volatility_spread",
    "momentum_30d", "volatility_30d", "momentum_60d", "momentum_10d",
    "risk_adjusted_momentum_30d",
]
SHAP_FINANCIAL_FEATURES = [
    "debt_to_equity", "eps_revision", "revenue", "market_cap", "eps_forward",
    "profit_margin", "operating_margin", "forward_pe", "return_on_equity",
    "log_market_cap", "return_on_assets", "gross_margin", "pe_ratio", "beta",
]
def shap_feature_group_map(all_columns):
    """Build {feature: group} for every column in shap_values_test.csv."""
    mapping = {}
    for c in all_columns:
        if c.startswith("sector_"):
            mapping[c] = "Sector"
        elif c in SHAP_SENTIMENT_FEATURES:
            mapping[c] = "Sentiment"
        elif c in SHAP_MARKET_FEATURES:
            mapping[c] = "Market"
        elif c in SHAP_FINANCIAL_FEATURES:
            mapping[c] = "Financial"
    return mapping

@st.cache_data
def load_shap_values_test():
    return pd.read_csv("Outputs/shap_explainability/shap_values_test.csv")

@st.cache_data
def load_shap_importance():
    return pd.read_csv("Outputs/shap_explainability/shap_feature_importance.csv")

@st.cache_data
def load_shap_group():
    return pd.read_csv("Outputs/shap_explainability/shap_group_summary.csv")

@st.cache_data
def load_sentiment_importance():
    return pd.read_csv("Outputs/shap_explainability/sentiment_feature_importance.csv")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/logo.png", width=170)
    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-size:1.3rem;font-weight:700;color:#f1f5f9;'>📈 Investor Copilot</div>
        <div style='font-size:0.72rem;color:#475569;margin-top:0.2rem;'>Equity Research Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏢  Company Overview",
        "🔍  Explainability",
        "🏭  Peer Benchmarking",
        "💡  Investment Recommendations",
        "📊  Model Validation",
    ], label_visibility="collapsed")

# ── Scroll to top whenever the selected page changes ─────────────
if "prev_page" not in st.session_state:
    st.session_state.prev_page = page
    st.session_state.scroll_nonce = 0
elif st.session_state.prev_page != page:
    st.session_state.prev_page = page
    st.session_state.scroll_nonce = st.session_state.get("scroll_nonce", 0) + 1
    scroll_script = """
    <script>
    try {
        console.log('[Investor Copilot] page-switch scroll-to-top firing (nonce=%d)');
        const doc = window.parent.document;
        function scrollTop() {
            doc.querySelectorAll('*').forEach(function(el) {
                if (el.scrollTop > 0) { el.scrollTop = 0; }
            });
            doc.documentElement.scrollTop = 0;
            doc.body.scrollTop = 0;
            window.parent.scrollTo(0, 0);
        }
        scrollTop();
        setTimeout(scrollTop, 50);
        setTimeout(scrollTop, 200);
    } catch (e) {
        console.error('[Investor Copilot] scroll-to-top failed:', e);
    }
    </script>
    <!-- nonce:%d -->
    """ % (st.session_state.scroll_nonce, st.session_state.scroll_nonce)
    components.html(scroll_script, height=0)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — COMPANY OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏢  Company Overview":
    df = load_main()

    st.markdown('<div class="page-title">Company Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Search a ticker to view earnings call sentiment, financials, and model prediction</div>', unsafe_allow_html=True)

    # ── Ticker search ──────────────────────────────────────────
    all_tickers = sorted(df["ticker"].unique().tolist())
    col_search, col_stats = st.columns([1, 2])
    with col_search:
        ticker = st.selectbox("Select / Type Ticker", all_tickers, index=all_tickers.index("GOOGL") if "GOOGL" in all_tickers else 0)

    company_df = df[df["ticker"] == ticker].sort_values("earnings_date")

    if company_df.empty:
        st.warning(f"No data found for {ticker}")
        st.stop()

    latest = company_df.iloc[-1]
    prev = company_df.iloc[-2] if len(company_df) >= 2 else None

    sector  = latest.get("sector", "N/A")
    mktcap  = latest.get("market_cap", 0)
    mktcap_str = f"${mktcap/1e12:.2f}T" if mktcap >= 1e12 else f"${mktcap/1e9:.1f}B" if mktcap >= 1e9 else "N/A"
    def _safe_val(row, key, fallback):
        val = row.get(key, None)
        return val if (val is not None and not pd.isna(val)) else fallback

    company_name = _safe_val(latest, "company_name", None) or _safe_val(latest, "name", None) or ticker
    industry = _safe_val(latest, "industry", None) or sector
    quarter = latest.get("q", "N/A")
    latest_date_str = latest["earnings_date"].strftime("%b %d, %Y")

    # ── One-line takeaway ────────────────────────────────────────
    if prev is not None:
        sent_diff = latest.get("net_sentiment", 0) - prev.get("net_sentiment", 0)
        if sent_diff > 0.02:
            tone_phrase = "Tone improved"
        elif sent_diff < -0.02:
            tone_phrase = "Tone weakened"
        else:
            tone_phrase = "Tone was steady"
    else:
        tone_phrase = "First call on record"
    reaction_phrase = "positive" if latest.get("return_5d", 0) > 0 else "negative"
    takeaway = f"{tone_phrase} vs. the prior call, while the post-earnings market reaction was {reaction_phrase}."

    with col_stats:
        st.markdown(f"""
        <div style='display:flex;justify-content:flex-end;align-items:center;
                    gap:2.5rem;height:100%;margin-top:1.75rem;'>
            <div style='text-align:right;'>
                <div style='color:#475569;font-size:0.65rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.06em;'>Earnings Calls</div>
                <div style='color:#e2e8f0;font-size:1rem;font-family:JetBrains Mono;
                            font-weight:600;margin-top:0.35rem;'>{len(company_df)}</div>
            </div>
            <div style='width:1px;align-self:stretch;background:#1e293b;'></div>
            <div style='text-align:right;'>
                <div style='color:#475569;font-size:0.65rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.06em;'>Coverage</div>
                <div style='color:#e2e8f0;font-size:1rem;font-family:JetBrains Mono;
                            font-weight:600;margin-top:0.35rem;'>
                    {company_df["earnings_date"].min().strftime("%b %Y")} –
                    {company_df["earnings_date"].max().strftime("%b %Y")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Company header ─────────────────────────────────────────
    st.markdown(f"""
    <div style='background:#111827;border:1px solid #1e293b;border-radius:12px;
                padding:1.25rem 1.5rem;margin-bottom:1rem;'>
        <div style='display:flex;justify-content:space-between;align-items:flex-start;
                    flex-wrap:wrap;gap:1rem;'>
            <div>
                <div style='color:#3b82f6;font-size:1.8rem;font-weight:700;
                            font-family:JetBrains Mono;'>{ticker}</div>
                <div style='color:#94a3b8;font-size:0.95rem;margin-top:0.3rem;'>{industry} · {mktcap_str}</div>
            </div>
            <div style='text-align:right;'>
                <div style='color:#64748b;font-size:0.72rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.06em;'>Latest Call</div>
                <div style='color:#f1f5f9;font-size:1.1rem;font-family:JetBrains Mono;
                            font-weight:600;margin-top:0.35rem;'>{quarter} · {latest_date_str}</div>
            </div>
        </div>
        <div style='margin-top:1rem;padding-top:1rem;border-top:1px solid #1e293b;
                    color:#e2e8f0;font-size:1rem;line-height:1.6;'>
            💬 {takeaway}
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

    reaction_word = "positive" if latest.get("return_5d", 0) > 0 else "negative"
    signal_note = (f"Model leans {signal.lower()} on tone, despite a {reaction_word} last reaction."
                    if signal != "HOLD" else
                    f"Mixed signals — no strong tilt despite a {reaction_word} last reaction.")

    def score_delta(curr, prev_val, decimals=3):
        if prev is None or pd.isna(curr) or pd.isna(prev_val):
            return "<span style='color:#8b96a8;font-size:0.75rem;'>no prior call</span>"
        diff = curr - prev_val
        if abs(diff) < 1e-9:
            return ""
        arrow = "▲" if diff > 0 else "▼"
        color = "#4ade80" if diff > 0 else "#fb7185"
        return f"<span style='color:{color};font-size:0.75rem;font-weight:700;'>{arrow} {diff:+.{decimals}f} vs prior</span>"

    st.markdown("<div style='color:#94a3b8;font-size:0.78rem;font-weight:700;text-transform:uppercase;"
                "letter-spacing:0.06em;margin-bottom:0.6rem;'>🔮 Predictive Signals</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    predictive_cards = [
        (c1, "Signal", f'<div class="{signal_class}">{signal}</div>', signal_note,
         "Rule-based call combining latest net sentiment, confidence score, and risk score.", ""),
        (c2, "Net Sentiment", f"{sent_score:.3f}", "latest earnings call",
         "FinBERT sentiment (-1 to 1) averaged across the full transcript: management remarks + Q&A.",
         score_delta(latest.get('net_sentiment', np.nan), prev.get('net_sentiment', np.nan) if prev is not None else np.nan)),
        (c3, "Confidence Score", f"{latest.get('confidence_score', 0):.3f}", "management tone",
         "Share of assertive, forward-looking language in management's prepared remarks.",
         score_delta(latest.get('confidence_score', np.nan), prev.get('confidence_score', np.nan) if prev is not None else np.nan)),
        (c4, "Risk Score", f"{latest.get('risk_score', 0):.3f}", "downside language",
         "Share of hedging, uncertainty, and downside-risk language detected in the transcript.",
         score_delta(latest.get('risk_score', np.nan), prev.get('risk_score', np.nan) if prev is not None else np.nan)),
    ]
    for col, label, val, sub, tooltip, delta in predictive_cards:
        with col:
            tip_id = "tip-" + label.lower().replace(" ", "-").replace("(", "").replace(")", "")
            if tooltip:
                tip_icon = f'<input type="checkbox" id="{tip_id}" class="tip-toggle"><label for="{tip_id}" class="tip-icon">?</label>'
                tip_desc = f'<div class="tip-desc">{tooltip}</div>'
            else:
                tip_icon, tip_desc = "", ""
            delta_html = f'<div style="margin-top:0.2rem;">{delta}</div>' if delta else ""
            if label == "Signal":
                st.markdown(f'<div class="metric-card">{tip_icon}<div class="metric-label">{label}</div>{val}<div class="metric-sub">{sub}</div>{tip_desc}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card">{tip_icon}<div class="metric-label">{label}</div><div class="metric-value">{val}</div><div class="metric-sub">{sub}</div>{delta_html}{tip_desc}</div>', unsafe_allow_html=True)

    st.markdown("<div style='color:#94a3b8;font-size:0.78rem;font-weight:700;text-transform:uppercase;"
                "letter-spacing:0.06em;margin:1rem 0 0.6rem 0;'>📜 Historical Validation</div>", unsafe_allow_html=True)
    hist_cols = st.columns(4)
    with hist_cols[0]:
        st.markdown(f"""<div class="metric-card" style="border-style:dashed;">
            <div class="metric-label">Last Actual 5d Return</div>
            <div class="metric-value">{latest.get('return_5d', 0)*100:.2f}%</div>
            <div class="metric-sub">post-earnings actual (not a forecast)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Time window toggle ──────────────────────────────────────
    time_window = st.radio(
        "Time window", ["All", "3Y", "2Y", "1Y", "Last 8 calls"],
        horizontal=True, label_visibility="collapsed", key="time_window_toggle"
    )
    max_call_date = company_df["earnings_date"].max()
    if time_window == "3Y":
        chart_df = company_df[company_df["earnings_date"] >= max_call_date - pd.DateOffset(years=3)]
    elif time_window == "2Y":
        chart_df = company_df[company_df["earnings_date"] >= max_call_date - pd.DateOffset(years=2)]
    elif time_window == "1Y":
        chart_df = company_df[company_df["earnings_date"] >= max_call_date - pd.DateOffset(years=1)]
    elif time_window == "Last 8 calls":
        chart_df = company_df.tail(8)
    else:
        chart_df = company_df

    if chart_df.empty:
        st.caption(f"No calls in the selected window ({time_window}) — showing full history instead.")
        chart_df = company_df

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
                x=chart_df["earnings_date"], y=chart_df[col_name],
                name=name, line=dict(color=color, width=2.5),
                mode="lines+markers", marker=dict(size=6, color=color)
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.5)
        fig.update_layout(**PLOT_THEME, height=260,
                        legend=dict(orientation="h", y=1.15, x=0,
                                    font=dict(color="#e2e8f0", size=11)),
                        xaxis_title="", yaxis_title="Sentiment Score")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Post-Earnings Returns</div>', unsafe_allow_html=True)
        colors_ret = ["#10b981" if r > 0 else "#ef4444" for r in chart_df["return_5d"]]
        fig = go.Figure(go.Bar(
            x=chart_df["earnings_date"],
            y=chart_df["return_5d"] * 100,
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

    def fin_delta(curr_val, prev_val, is_pct=False, decimals=1):
        if prev is None or pd.isna(curr_val) or pd.isna(prev_val):
            return "<span style='color:#8b96a8;font-size:0.75rem;'>no prior call</span>"
        diff = curr_val - prev_val
        if abs(diff) < 1e-9:
            return ""
        arrow = "▲" if diff > 0 else "▼"
        color = "#4ade80" if diff > 0 else "#fb7185"
        diff_str = f"{diff*100:+.{decimals}f} pts" if is_pct else f"{diff:+.{decimals}f}"
        return f"<span style='color:{color};font-size:0.75rem;font-weight:700;'>{arrow} {diff_str} vs prior</span>"

    f1, f2, f3, f4, f5, f6 = st.columns(6)
    fin_cards = [
        (f1, "PE Ratio",      f"{latest.get('pe_ratio', 0):.1f}",
            fin_delta(latest.get('pe_ratio', np.nan), prev.get('pe_ratio', np.nan) if prev is not None else np.nan)),
        (f2, "Forward PE",    f"{latest.get('forward_pe', 0):.1f}",
            fin_delta(latest.get('forward_pe', np.nan), prev.get('forward_pe', np.nan) if prev is not None else np.nan)),
        (f3, "EPS (Forward)", f"${latest.get('eps_forward', 0):.2f}",
            fin_delta(latest.get('eps_forward', np.nan), prev.get('eps_forward', np.nan) if prev is not None else np.nan, decimals=2)),
        (f4, "Gross Margin",  f"{latest.get('gross_margin', 0)*100:.1f}%",
            fin_delta(latest.get('gross_margin', np.nan), prev.get('gross_margin', np.nan) if prev is not None else np.nan, is_pct=True)),
        (f5, "ROE",           f"{latest.get('return_on_equity', 0)*100:.1f}%",
            fin_delta(latest.get('return_on_equity', np.nan), prev.get('return_on_equity', np.nan) if prev is not None else np.nan, is_pct=True)),
        (f6, "Debt/Equity",   f"{latest.get('debt_to_equity', 0):.2f}",
            fin_delta(latest.get('debt_to_equity', np.nan), prev.get('debt_to_equity', np.nan) if prev is not None else np.nan, decimals=2)),
    ]
    for col, label, val, delta in fin_cards:
        with col:
            delta_html = f'<div style="margin-top:0.2rem;">{delta}</div>' if delta else ""
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.2rem;">{val}</div>
                {delta_html}
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
    st.markdown('<div class="page-subtitle">SHAP-based analysis of what drives model decisions globally and locally</div>', unsafe_allow_html=True)

    try:
        shap_imp   = load_shap_importance()
        shap_group = load_shap_group()
        sent_imp   = load_sentiment_importance()
        shap_vals  = load_shap_values_test()

        # ── Group contribution bars ────────────────────────────
        st.markdown('<div class="section-header">What Drives Predictions — Feature Group Contribution</div>', unsafe_allow_html=True)

        color_map = {"Financial": "#3b82f6", "Market": "#10b981",
                     "Sentiment": "#f59e0b", "Sector": "#8b5cf6"}
        group_order = shap_group.sort_values("Contribution_pct", ascending=False)["Group"].tolist()
        group_interpretations = {
            "Financial": "Profitability, revenue, and forward EPS dominate",
            "Market": "Price momentum and 52-week positioning drive the market-based signal",
            "Sentiment": "Q&A tone and sentiment surprise carry most of the signal",
            "Sector": "Sector membership plays a minor but non-zero role",
        }

        col_bars, col_finding = st.columns([1.4, 1])

        with col_bars:
            for _, row in shap_group.sort_values("Contribution_pct", ascending=False).iterrows():
                pct   = row["Contribution_pct"]
                group = row["Group"]
                color = color_map.get(group, "#64748b")
                tip_id = f"tip-group-{group.lower()}"
                interp = group_interpretations.get(group, "")
                st.markdown(f"""
                <div style='margin-bottom:0.9rem;position:relative;'>
                    <input type="checkbox" id="{tip_id}" class="tip-toggle">
                    <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>
                        <span style='color:#e2e8f0;font-size:0.875rem;font-weight:500;display:inline-flex;
                                    align-items:center;gap:0.4rem;'>
                            {group}
                            <label for="{tip_id}" class="tip-icon" style="position:static;">?</label>
                        </span>
                        <span style='color:{color};font-family:JetBrains Mono;font-size:0.875rem;font-weight:700;'>{pct:.1f}%</span>
                    </div>
                    <div style='background:#1e293b;border-radius:4px;height:8px;'>
                        <div style='background:{color};width:{pct}%;height:8px;border-radius:4px;'></div>
                    </div>
                    <div style='color:#475569;font-size:0.68rem;margin-top:0.2rem;'>{int(row["N_features"])} features</div>
                    <div class="tip-desc" style="position:static;width:auto;margin-top:0.4rem;
                                box-shadow:none;background:transparent;border:none;padding:0;
                                color:#94a3b8;font-size:0.76rem;">{group}: {interp}</div>
                </div>""", unsafe_allow_html=True)

        with col_finding:
            st.markdown("""
            <div class="finding-card">
                <div class="finding-title">🎯 Central Finding</div>
                <div class="finding-desc"><strong style="color:#f59e0b;font-size:0.95rem;">
                Sentiment contributes nearly one-quarter of total predictive power, directly validating that earnings call language contains incremental investment signal.
                </strong></div>
            </div>""", unsafe_allow_html=True)

        # ── Drill-down: click a group to see its individual features ──
        # (Full width, not confined to col_bars, so the chart has room to breathe.)
        st.markdown("<div style='color:#94a3b8;font-size:0.8rem;margin:0.9rem 0 0.5rem 0;'>"
                    "Click a group to see which features drive it:</div>", unsafe_allow_html=True)

        if "shap_selected_group" not in st.session_state:
            st.session_state.shap_selected_group = None

        btn_cols = st.columns(len(group_order))
        for i, group in enumerate(group_order):
            with btn_cols[i]:
                is_active = st.session_state.shap_selected_group == group
                if st.button(group, key=f"shap_group_btn_{group}",
                            type="primary" if is_active else "secondary",
                            use_container_width=True):
                    # Toggle off if clicking the already-selected group, else select it.
                    st.session_state.shap_selected_group = None if is_active else group
                    st.rerun()  # re-render immediately so button colors reflect the new state

        # Recolor the active button to match its group's chart color (only one
        # primary-type button exists on this page at a time, so this is safe
        # to target globally without fragile position-based CSS selectors).
        if st.session_state.shap_selected_group is not None:
            active_color = color_map.get(st.session_state.shap_selected_group, "#3b82f6")
            st.markdown(f"""
            <style>
            [data-testid="stButton"] button[kind="primary"] {{
                background-color: {active_color} !important;
                border-color: {active_color} !important;
                color: #0a0e1a !important;
            }}
            </style>
            """, unsafe_allow_html=True)

        selected_group = st.session_state.shap_selected_group

        if selected_group is not None:
            feature_group_map = shap_feature_group_map(shap_vals.columns)
            group_cols = [c for c, g in feature_group_map.items() if g == selected_group]
            mean_abs_shap = shap_vals[group_cols].abs().mean().sort_values(ascending=False)
            group_features = pd.DataFrame({
                "feature": mean_abs_shap.index,
                "mean_shap": mean_abs_shap.values,
            })
            group_features["pct_of_group"] = group_features["mean_shap"] / group_features["mean_shap"].sum() * 100

            # Show top 15 within the group to keep the chart readable for
            # larger groups (Financial/Sentiment have 14, Sector has 11).
            display_features = group_features.head(15)
            fig = px.bar(
                display_features.sort_values("pct_of_group"),
                x="pct_of_group", y="feature", orientation="h",
                color_discrete_sequence=[color_map.get(selected_group, "#64748b")],
                text=display_features.sort_values("pct_of_group")["pct_of_group"].round(1).astype(str) + "%"
            )
            fig.update_traces(textposition="outside", textfont=dict(color="#94a3b8", size=11))
            fig.update_layout(**PLOT_THEME, height=max(220, 28 * len(display_features)),
                              xaxis_title=f"% share within {selected_group} ({len(group_cols)} features)",
                              yaxis_title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Top features + sentiment breakdown ─────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-header">Top 10 Overall Features</div>', unsafe_allow_html=True)
            total_all_features_shap = (
                shap_vals[[c for c in shap_vals.columns if c not in ("ticker", "earnings_date")]]
                .abs().mean().sum()
            )  # matches the 100% base used by group %; excludes ID columns if present
            top10 = shap_imp.sort_values("mean_shap", ascending=False).head(10).copy()
            top10["pct_of_total"] = top10["mean_shap"] / total_all_features_shap * 100
            top10_sorted = top10.sort_values("mean_shap")
            fig = px.bar(
                top10_sorted,
                x="mean_shap", y="feature", orientation="h",
                color_discrete_sequence=["#3b82f6"],
                text=top10_sorted["pct_of_total"].round(1).astype(str) + "%"
            )
            fig.update_traces(textposition="outside", textfont=dict(color="#94a3b8", size=11),
                              hovertemplate="Mean |SHAP|: %{x:.5f}<extra></extra>")
            fig.update_layout(**PLOT_THEME, height=280,
                              xaxis_title="Mean |SHAP Value|", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                "<div style='text-align:left;margin:0.4rem 0 0.8rem 0;'>"
                "<span style='background:#1e293b;color:#93c5fd;padding:0.15rem 0.55rem;border-radius:6px;"
                "font-family:JetBrains Mono;font-size:0.75rem;'>Mean |SHAP|</span>"
                "<span style='color:#94a3b8;font-size:0.8rem;'> = importance magnitude</span></div>",
                unsafe_allow_html=True
            )

            st.markdown('<div class="section-header">Key Feature Insights</div>', unsafe_allow_html=True)
            top_insights = [
                ("price_position_52w dominates", "By a wide margin the single strongest predictor — more than double the next-ranked feature"),
                ("Fundamentals cluster near the top", "eps_forward, revenue, return_on_equity, market_cap, log_market_cap, and gross_margin fill 6 of the top 10 slots"),
                ("qa_sentiment cracks the top 5", "The only sentiment feature in the top 10 — ranks above every momentum/technical feature except price_position_52w"),
            ]
            for title, desc in top_insights:
                st.markdown(f"""
                <div style='padding:0.5rem 0;border-bottom:1px solid #0f172a;'>
                    <div style='color:#3b82f6;font-size:0.78rem;font-weight:600;'>{title}</div>
                    <div style='color:#94a3b8;font-size:0.76rem;margin-top:0.15rem;line-height:1.4;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="section-header">Top 10 Sentiment Drivers</div>', unsafe_allow_html=True)
            sent10 = sent_imp.copy()
            sent10["pct_of_total"] = sent10["mean_shap"] / total_all_features_shap * 100
            sent10_sorted = sent10.sort_values("mean_shap")
            fig = px.bar(
                sent10_sorted,
                x="mean_shap", y="feature", orientation="h",
                color_discrete_sequence=["#f59e0b"],
                text=sent10_sorted["pct_of_total"].round(1).astype(str) + "%"
            )
            fig.update_traces(textposition="outside", textfont=dict(color="#94a3b8", size=11),
                              hovertemplate="Mean |SHAP|: %{x:.5f}<extra></extra>")
            fig.update_layout(**PLOT_THEME, height=280,
                              xaxis_title="Mean |SHAP Value|", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                "<div style='text-align:left;margin:0.4rem 0 0.8rem 0;'>"
                "<span style='background:#1e293b;color:#93c5fd;padding:0.15rem 0.55rem;border-radius:6px;"
                "font-family:JetBrains Mono;font-size:0.75rem;'>Mean |SHAP|</span>"
                "<span style='color:#94a3b8;font-size:0.8rem;'> = importance magnitude</span></div>",
                unsafe_allow_html=True
            )

            st.markdown('<div class="section-header">Sentiment Insights</div>', unsafe_allow_html=True)
            insights = [
                ("qa_sentiment ranks #1", "Q&A section outperforms prepared management remarks — analysts probe harder truths"),
                ("sentiment_surprise ranks #4", "When sentiment exceeds prior quarter, model pushes prediction toward UP"),
                ("risk_score matters", "Elevated downside/risk language in transcripts consistently depresses predictions"),
            ]
            for title, desc in insights:
                st.markdown(f"""
                <div style='padding:0.5rem 0;border-bottom:1px solid #0f172a;'>
                    <div style='color:#f59e0b;font-size:0.78rem;font-weight:600;'>{title}</div>
                    <div style='color:#94a3b8;font-size:0.76rem;margin-top:0.15rem;line-height:1.4;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

        # ── Company-specific SHAP explainability ─────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Why This Prediction — Company-Specific SHAP Drivers</div>', unsafe_allow_html=True)

        if "ticker" not in shap_vals.columns:
            st.info("SHAP sample doesn't have ticker/date attached yet — "
                    "regenerate shap_values_test.csv with IDs to enable this section.")
        else:
            sample_tickers = sorted(shap_vals["ticker"].unique().tolist())
            SELECT_PROMPT = "— Select a ticker —"
            shap_ticker = st.selectbox(
                "Select a ticker to explain",
                [SELECT_PROMPT] + sample_tickers, index=0,
                help=f"Limited to the {len(sample_tickers)} tickers with at least one call "
                     f"in the 1,000-row SHAP sample (~30% of the full 3,387-call test set)."
            )

            if shap_ticker != SELECT_PROMPT:
                ticker_shap = shap_vals[shap_vals["ticker"] == shap_ticker].copy()
                ticker_shap["earnings_date"] = pd.to_datetime(ticker_shap["earnings_date"])
                latest_shap_row = ticker_shap.sort_values("earnings_date").iloc[-1]
                shap_call_date = latest_shap_row["earnings_date"]

                if len(ticker_shap) > 1:
                    st.caption(f"{len(ticker_shap)} of {shap_ticker}'s calls appear in the SHAP sample — "
                              f"showing the most recent one included: {shap_call_date.strftime('%b %d, %Y')}.")
                else:
                    st.caption(f"Showing {shap_ticker}'s only call included in the SHAP sample "
                              f"({shap_call_date.strftime('%b %d, %Y')}) — not necessarily their latest call overall.")

                feature_cols = [c for c in shap_vals.columns if c not in ("ticker", "earnings_date")]
                row_shap = latest_shap_row[feature_cols].astype(float)

                top_pos = row_shap.sort_values(ascending=False).head(5)
                top_neg = row_shap.sort_values(ascending=True).head(5)
                combined = pd.concat([top_pos, top_neg]).sort_values()
                bar_colors = ["#10b981" if v > 0 else "#ef4444" for v in combined.values]

                fig = go.Figure(go.Bar(
                    x=combined.values, y=combined.index, orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:+.5f}" for v in combined.values],
                ))
                fig.update_traces(
                    textposition="outside", textfont=dict(color="#94a3b8", size=11),
                    hovertemplate="SHAP: %{x:+.5f}<extra></extra>"
                )
                fig.add_vline(x=0, line_color="#475569", line_width=1)
                fig.update_layout(**PLOT_THEME, height=340,
                                  xaxis_title="SHAP value  (→ pushes prediction UP  |  ← pushes DOWN)",
                                  yaxis_title="", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                sentence_parts = []
                if len(top_pos):
                    sentence_parts.append(f"<strong style='color:#10b981;'>{top_pos.index[0]}</strong> pushed the prediction up the most")
                if len(top_neg):
                    sentence_parts.append(f"<strong style='color:#ef4444;'>{top_neg.index[0]}</strong> pulled it down the most")
                if sentence_parts:
                    st.markdown(f"""
                    <div class="finding-card">
                        <div class="finding-desc">{' — while '.join(sentence_parts)}.</div>
                    </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.info(f"SHAP outputs not found. Place files in Outputs/shap_explainability/. ({e})")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — PEER BENCHMARKING
# ══════════════════════════════════════════════════════════════
elif page == "🏭  Peer Benchmarking":
    st.markdown('<div class="page-title">Peer Benchmarking</div>', unsafe_allow_html=True)
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
# PAGE 4 — INVESTMENT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
elif page == "💡  Investment Recommendations":
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
# PAGE 5 — MODEL VALIDATION
# ══════════════════════════════════════════════════════════════
elif page == "📊  Model Validation":
    st.markdown('<div class="page-title">Model Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Classification accuracy, forecasting metrics, and explainability results — full transparency on how Investor Copilot works</div>', unsafe_allow_html=True)

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