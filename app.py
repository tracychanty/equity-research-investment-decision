import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Investor Copilot",
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
        white-space: normal; overflow-wrap: break-word; line-height: 1.3;
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
    a.nav-chip, a.nav-chip:visited {
        display: inline-block !important; background: #1e293b !important;
        border: 1px solid #334155 !important; color: #cbd5e1 !important;
        padding: 0.45rem 1.1rem !important; border-radius: 8px !important;
        font-size: 0.8rem !important; font-weight: 600 !important; text-decoration: none !important;
        margin-right: 0.5rem; margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4);
        transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease;
    }
    a.nav-chip:hover {
        background: #ffffff !important; border-color: #1e3a8a !important;
        color: #1e3a8a !important; text-decoration: none !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.5); transform: translateY(-1px);
    }
    a.nav-chip:active {
        transform: translateY(0px); box-shadow: 0 1px 2px rgba(0,0,0,0.4);
    }
    .badge-pill {
        display: inline-block; background: #0f2942; border: 1px solid #1e3a5f;
        color: #93c5fd; padding: 0.2rem 0.65rem; border-radius: 999px; font-size: 0.85rem;
    }
    .badge-pill-neutral {
        display: inline-block; background: #1e293b; border: 1px solid #334155;
        color: #94a3b8; padding: 0.2rem 0.65rem; border-radius: 999px; font-size: 0.78rem; font-style: italic;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 2.5rem;
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
@st.cache_data(show_spinner="Loading data...")
def load_main():
    df = pd.read_csv("Outputs/feature_engineering/ml_dataset_enhanced.csv")
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df

@st.cache_data(show_spinner="Loading data...")
def load_classification():
    return pd.read_csv("Outputs/classification/final_model_comparison.csv")

@st.cache_data(show_spinner="Loading data...")
def load_robustness():
    return pd.read_csv("Outputs/classification/robustness_summary.csv")

@st.cache_data(show_spinner="Loading data...")
def load_forecast_summary():
    return pd.read_csv("Outputs/price_forecasting/quantile_forecast_summary_tuned.csv")

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

@st.cache_data(show_spinner="Loading data...")
def load_shap_values_test():
    return pd.read_csv("Outputs/shap_explainability/shap_values_test.csv")

@st.cache_data(show_spinner="Loading data...")
def load_shap_importance():
    return pd.read_csv("Outputs/shap_explainability/shap_feature_importance.csv")

@st.cache_data(show_spinner="Loading data...")
def load_benchmark():
    df = pd.read_csv("Outputs/industry_benchmarking/industry_benchmarking_output.csv")
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df

@st.cache_data(show_spinner="Loading data...")
def load_shap_group():
    return pd.read_csv("Outputs/shap_explainability/shap_group_summary.csv")

@st.cache_data(show_spinner="Loading data...")
def load_sentiment_importance():
    return pd.read_csv("Outputs/shap_explainability/sentiment_feature_importance.csv")

@st.cache_data(show_spinner="Loading data...")
def load_event_abnormal_returns():
    return pd.read_csv("Outputs/event_study/abnormal_returns.csv")

@st.cache_data(show_spinner="Loading data...")
def load_event_significance():
    return pd.read_csv("Outputs/event_study/significance_summary.csv")

@st.cache_data(show_spinner="Loading data...")
def load_event_monotonic():
    return pd.read_csv("Outputs/event_study/monotonic_relationship.csv")

@st.cache_data(show_spinner="Loading data...")
def load_event_sector_car():
    return pd.read_csv("Outputs/event_study/sector_car_summary.csv")

@st.cache_data(show_spinner="Loading data...")
def load_portfolio_weights():
    df = pd.read_csv("Outputs/portfolio_construction/v3_portfolio_weights.csv")
    return df

@st.cache_data(show_spinner="Loading data...")
def load_portfolio_net_returns():
    return pd.read_csv("Outputs/portfolio_construction/v3_portfolio_net_returns_wide.csv")

@st.cache_data(show_spinner="Loading data...")
def load_portfolio_cumulative_returns():
    return pd.read_csv("Outputs/portfolio_construction/v3_portfolio_cumulative_returns.csv")

@st.cache_data(show_spinner="Loading data...")
def load_portfolio_drawdowns():
    return pd.read_csv("Outputs/portfolio_construction/v3_portfolio_drawdowns.csv")

@st.cache_data(show_spinner="Loading data...")
def load_portfolio_performance():
    return pd.read_csv("Outputs/portfolio_construction/portfolio_performance_metrics_report.csv")

@st.cache_data(show_spinner="Loading data...")
def load_main_strategy_summary():
    return pd.read_csv("Outputs/portfolio_construction/main_strategy_summary.csv")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem !important;
    }
    section[data-testid="stSidebar"] .stRadio label p {
        font-size: 1.05rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.image("assets/logo.png", width=110)
    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-size:1.5rem;font-weight:700;color:#f1f5f9;'>Investor Copilot</div>
        <div style='font-size:1.05rem;font-weight:600;color:#94a3b8;margin-top:0.3rem;'>EarningsEdge</div>
        <div style='font-size:0.78rem;color:#475569;margin-top:0.1rem;'>Equity Research Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
    "🏢  Company Overview",
    "🏭  Peer Benchmarking",
    "💡  Investment Recommendations",
    "🔍  Explainability",
    "📊  Model Validation",
    "🧠  AI Research Assistant",
    ], label_visibility="collapsed")

    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-size:0.8rem;color:#64748b;margin-top:0.5rem;'>By Tracy Chan &amp; Yanxin Li</div>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown('<div class="page-title">Explainability &amp; Economic Evidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">What drives the model, and whether that signal shows up in actual returns</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🧠 Model Explainability (SHAP)", "📈 Economic Evidence (Event Study)"])

    with tab1:
        st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;">What features drive the model\'s predictions.</div>', unsafe_allow_html=True)

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

    with tab2:
        st.markdown('<div style="color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;">Whether the sentiment signal actually shows up in abnormal stock returns around earnings.</div>', unsafe_allow_html=True)

        WINDOW_LABELS = {
            "AR_0": "Day 0 (announcement)",
            "CAR_0_1": "CAR [0, +1]",
            "CAR_1_3": "CAR [+1, +3]",
            "CAR_1_5": "CAR [+1, +5]",
            "CAR_1_10": "CAR [+1, +10]",
            "CAR_minus1_1": "CAR [-1, +1]",
        }

        try:
            abn_df = load_event_abnormal_returns()
            sig_df = load_event_significance()
            mono_df = load_event_monotonic()
            sector_car_df = load_event_sector_car()

            # ── SHAP → Event Study bridge ──────────────────────────────
            sentiment_pct = None
            try:
                sentiment_pct = shap_group.loc[shap_group["Group"] == "Sentiment", "Contribution_pct"].values[0]
            except Exception:
                pass

            car01_row = sig_df[sig_df["Window"] == "CAR_0_1"].iloc[0]
            is_significant = bool(car01_row["t_sig"]) or bool(car01_row["mw_sig"])
            sig_label = "Statistically significant" if is_significant else "Not statistically significant"

            badges = []
            if sentiment_pct is not None:
                badges.append(f"{sentiment_pct:.1f}% SHAP contribution")
            badges.append(f"{car01_row['Difference']*100:+.2f}pp spread")
            badges.append(sig_label)
            badges_html = "".join(f'<span class="badge-pill" style="margin-right:0.4rem;">{b}</span>' for b in badges)

            st.markdown(f"""
            <div class="finding-card">
                <div class="finding-title">🔗 Connecting the two lenses</div>
                <div style="margin:0.5rem 0 0.6rem 0;">{badges_html}</div>
                <div class="finding-desc">
                    The most positive sentiment quartile earned {car01_row['Q4_mean_CAR']*100:+.2f}% over the
                    [0,+1] day window versus {car01_row['Q1_mean_CAR']*100:+.2f}% for the most negative,
                    confirming the SHAP signal shows up in real market reactions.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            # ── Significance summary table ─────────────────────────────
            st.markdown('<div class="section-header">Sentiment Quartile Spread — Is It Statistically Significant?</div>', unsafe_allow_html=True)
            sig_fmt = pd.DataFrame({
                "Window": sig_df["Window"].map(lambda w: WINDOW_LABELS.get(w, w)),
                "Q4 Mean CAR (most positive)": (sig_df["Q4_mean_CAR"] * 100).map(lambda v: f"{v:+.2f}%"),
                "Q1 Mean CAR (most negative)": (sig_df["Q1_mean_CAR"] * 100).map(lambda v: f"{v:+.2f}%"),
                "Spread": (sig_df["Difference"] * 100).map(lambda v: f"{v:+.2f}pp"),
                "t-test": sig_df["t_sig"].fillna("n.s."),
                "Mann-Whitney": sig_df["mw_sig"].fillna("n.s."),
            })
            st.dataframe(sig_fmt, use_container_width=True, height=38 * (len(sig_fmt) + 1) + 3, hide_index=True)
            n_per_quartile = int(sig_df["n_Q4"].iloc[0])
            st.caption(
                f"*** p<0.01, ** p<0.05, n.s. = not significant. Q4/Q1 refer to the top and bottom sentiment "
                f"quartiles by net sentiment score, {n_per_quartile:,} events per quartile."
            )

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            # ── Monotonic relationship chart ───────────────────────────
            col_mono, col_car = st.columns(2)
            with col_mono:
                st.markdown('<div class="section-header">Return by Sentiment Quartile</div>', unsafe_allow_html=True)
                window_choice = st.selectbox(
                    "Window", options=mono_df["Window"].tolist(),
                    format_func=lambda w: WINDOW_LABELS.get(w, w), index=1,
                )
                mono_row = mono_df[mono_df["Window"] == window_choice].iloc[0]
                quartiles = ["Q1", "Q2", "Q3", "Q4"]
                values = [mono_row[q] * 100 for q in quartiles]
                colors = ["#ef4444" if v < 0 else "#10b981" for v in values]
                fig = go.Figure(go.Bar(
                    x=["Q1 (most neg.)", "Q2", "Q3", "Q4 (most pos.)"], y=values,
                    marker_color=colors, text=[f"{v:+.2f}%" for v in values], textposition="outside",
                ))
                fig.add_hline(y=0, line_color="#475569", line_width=1)
                fig.update_layout(**PLOT_THEME, height=320, yaxis_title="Mean return (%)", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                monotonic_label = {
                    "✓ Yes": "fully monotonic",
                    "~ Partial": "partially monotonic",
                    "✗ No": "not monotonic",
                }.get(mono_row["Monotonic"], mono_row["Monotonic"])
                st.caption(f"Quartile ordering: {monotonic_label}.")

            # ── CAR event-time chart ────────────────────────────────────
            with col_car:
                st.markdown('<div class="section-header">Cumulative Return Around Earnings</div>', unsafe_allow_html=True)
                days = list(range(-1, 11))
                day_cols = [f"AR_day_{d}" for d in days]
                mean_by_group = abn_df.groupby("sentiment_group")[day_cols].mean()
                car_traj = mean_by_group.cumsum(axis=1)
                fig = go.Figure()
                for grp, color in [("Positive", "#10b981"), ("Negative", "#ef4444")]:
                    if grp in car_traj.index:
                        fig.add_trace(go.Scatter(
                            x=days, y=car_traj.loc[grp, day_cols].values * 100,
                            mode="lines+markers", name=f"{grp} sentiment",
                            line=dict(color=color, width=2.5),
                        ))
                fig.add_vline(x=0, line_dash="dash", line_color="#475569", opacity=0.5)
                fig.update_layout(**PLOT_THEME, height=320, xaxis_title="Trading days relative to earnings",
                                  yaxis_title="Cumulative abnormal return (%)", legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Day 0 is the earnings call date. Abnormal-return differences are strongest in the "
                    "immediate post-earnings window (CAR [0,+1]) and become weaker, and sometimes partially "
                    "reverse, over the following days."
                )

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            # ── Additional evidence (optional deep dive) ────────────────
            with st.expander("📊 Additional Evidence — Sector Breakdown & Distributions", expanded=False):
                st.markdown('<div class="section-header" style="margin-top:0;">Sentiment Return Spread by Sector</div>', unsafe_allow_html=True)
                diff_cols = {"diff_AR_0": "Day 0", "diff_CAR_0_1": "CAR [0,+1]",
                            "diff_CAR_1_3": "CAR [+1,+3]", "diff_CAR_minus1_1": "CAR [-1,+1]"}
                present_cols = [c for c in diff_cols if c in sector_car_df.columns]
                sector_sorted = sector_car_df.sort_values("diff_CAR_0_1", ascending=True)
                z = (sector_sorted[present_cols] * 100).values
                fig = go.Figure(data=go.Heatmap(
                    z=z, x=[diff_cols[c] for c in present_cols], y=sector_sorted["sector"],
                    colorscale="RdYlGn", zmid=0, text=[[f"{v:+.1f}%" for v in row] for row in z],
                    texttemplate="%{text}", textfont={"size": 10}, colorbar=dict(title="Spread (pp)", thickness=12),
                    hoverinfo="skip",
                ))
                fig.update_layout(**PLOT_THEME, height=380, xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
                strongest_sector = sector_car_df.loc[sector_car_df["diff_CAR_0_1"].idxmax()]
                weakest_sector = sector_car_df.loc[sector_car_df["diff_CAR_0_1"].idxmin()]
                st.markdown(f"""
                <div style="color:#8b96a8;font-size:0.8rem;line-height:1.6;margin-top:0.3rem;">
                    Spread = mean CAR for the most positive sentiment quartile minus the most negative, by sector.
                    <ul style="margin:0.3rem 0 0 1.1rem;padding:0;">
                        <li>Strongest on CAR [0,+1]: {strongest_sector['sector']} ({strongest_sector['diff_CAR_0_1']*100:+.1f}pp)</li>
                        <li>Weakest: {weakest_sector['sector']} ({weakest_sector['diff_CAR_0_1']*100:+.1f}pp)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
                col_scatter, col_dist = st.columns(2)

                with col_scatter:
                    st.markdown('<div class="section-header" style="margin-top:0;">Sentiment Surprise vs Return</div>', unsafe_allow_html=True)
                    sample_df = abn_df.sample(n=min(3000, len(abn_df)), random_state=42)
                    fig = px.scatter(
                        sample_df, x="sentiment_surprise", y="CAR_0_1", color="sentiment_group",
                        color_discrete_map={"Positive": "#10b981", "Negative": "#ef4444"}, opacity=0.25,
                    )
                    fig.update_traces(marker=dict(size=5))
                    slope, intercept = np.polyfit(abn_df["sentiment_surprise"], abn_df["CAR_0_1"], 1)
                    x_fit = np.linspace(abn_df["sentiment_surprise"].min(), abn_df["sentiment_surprise"].max(), 50)
                    y_fit = slope * x_fit + intercept
                    fig.add_trace(go.Scatter(
                        x=x_fit, y=y_fit, mode="lines", name="Fitted trend",
                        line=dict(color="#f1f5f9", width=2, dash="dash"),
                    ))
                    corr = abn_df["sentiment_surprise"].corr(abn_df["CAR_0_1"])
                    fig.update_layout(**PLOT_THEME, height=320, xaxis_title="Sentiment surprise (QoQ change)",
                                      yaxis_title="CAR [0,+1]", legend=dict(orientation="h", y=1.12))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"Correlation (all {len(abn_df):,} events): r = {corr:.3f}. Random sample of "
                        f"{len(sample_df):,} points shown for rendering performance; trend line fit on full data."
                    )

                with col_dist:
                    st.markdown('<div class="section-header" style="margin-top:0;">Return Distribution by Sentiment</div>', unsafe_allow_html=True)
                    lo, hi = abn_df["CAR_0_1"].quantile([0.01, 0.99])
                    fig = go.Figure()
                    for grp, color in [("Positive", "#10b981"), ("Negative", "#ef4444")]:
                        subset = abn_df[abn_df["sentiment_group"] == grp]["CAR_0_1"].clip(lo, hi) * 100
                        fig.add_trace(go.Histogram(x=subset, name=f"{grp} sentiment", marker_color=color, opacity=0.6, nbinsx=40))
                    fig.update_layout(**PLOT_THEME, height=320, barmode="overlay",
                                      xaxis_title="CAR [0,+1] (%)", yaxis_title="Count",
                                      legend=dict(orientation="h", y=1.12))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Winsorized at the 1st/99th percentile to keep the central comparison readable; extreme tail events are compressed to the axis edges rather than excluded.")

        except FileNotFoundError as e:
            st.info(f"Event study outputs not found. Place files in Outputs/event_study/. ({e})")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — PEER BENCHMARKING
# ══════════════════════════════════════════════════════════════
elif page == "🏭  Peer Benchmarking":
    st.markdown('<div class="page-title">Peer Benchmarking</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Sector-relative scoring, peer rankings, and buy/hold/avoid recommendations</div>', unsafe_allow_html=True)

    try:
        bench = load_benchmark()

        rec_colors = {"Buy": "#10b981", "Hold": "#f59e0b", "Avoid": "#ef4444"}

        # ── Sector + Industry + Focus Ticker selectors ──────────
        sectors = sorted([s for s in bench["sector"].unique() if s != "Unknown"])
        col_s, col_i, col_t = st.columns([1, 1, 1.5])
        with col_s:
            selected_sector = st.selectbox("Sector", sectors)

        sector_df_full = bench[bench["sector"] == selected_sector]

        with col_i:
            industries = sorted(sector_df_full["industry"].dropna().unique().tolist())
            selected_industry = st.selectbox("Industry (optional)", ["All Industries"] + industries)

        sector_df = (sector_df_full if selected_industry == "All Industries"
                    else sector_df_full[sector_df_full["industry"] == selected_industry])

        with col_t:
            sector_tickers = sorted(sector_df["ticker"].unique().tolist())
            focus_tickers = st.multiselect("Focus Ticker(s)", sector_tickers, default=[])

        # ── Sector overview KPIs ────────────────────────────────
        st.markdown('<div class="section-header">Sector Overview</div>', unsafe_allow_html=True)
        buy_rate = (sector_df["recommendation"] == "Buy").mean() * 100
        top_row = sector_df.loc[sector_df["overall_score"].idxmax()]
        c1, c2, c3, c4 = st.columns(4)
        metrics = [
            (c1, "Companies", f"{sector_df['ticker'].nunique():,}", selected_sector),
            (c2, "Top Score", f"{top_row['overall_score']:.3f}", f"{top_row['ticker']} — best in sector"),
            (c3, "Avg 20d Predicted Return", f"{sector_df['predicted_return'].mean()*100:.2f}%", "model forecast"),
            (c4, "Buy Rate", f"{buy_rate:.1f}%", "of sector rated Buy"),
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

        # ── Recommendation mix + Top 10 leaders ─────────────────
        scope_label = selected_sector if selected_industry == "All Industries" else f"{selected_industry} ({selected_sector})"
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">Recommendation Mix</div>', unsafe_allow_html=True)
            rec_counts = sector_df["recommendation"].value_counts().reindex(
                ["Buy", "Hold", "Avoid"], fill_value=0).reset_index()
            rec_counts.columns = ["recommendation", "count"]
            total_n = rec_counts["count"].sum()
            rec_counts["pct"] = (rec_counts["count"] / total_n * 100) if total_n > 0 else 0
            fig = go.Figure(go.Bar(
                x=rec_counts["recommendation"], y=rec_counts["count"],
                marker_color=[rec_colors.get(r, "#64748b") for r in rec_counts["recommendation"]],
                text=[f"{c} ({p:.0f}%)" for c, p in zip(rec_counts["count"], rec_counts["pct"])],
                textposition="outside", textfont=dict(color="#94a3b8")
            ))
            fig.update_layout(**PLOT_THEME, height=300, xaxis_title="", yaxis_title="Companies")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Scope: {scope_label}")

        with col2:
            st.markdown('<div class="section-header">Top 10 Leaders</div>', unsafe_allow_html=True)
            st.markdown('<div style="color:#64748b;font-size:0.72rem;margin-top:-0.6rem;margin-bottom:0.5rem;">ranked within selected sector/industry</div>', unsafe_allow_html=True)
            top10_sector = sector_df.nlargest(10, "overall_score")
            colors_top10 = [rec_colors.get(r, "#64748b") for r in top10_sector["recommendation"]]
            y_labels = [f"#{int(r)} {t}" for r, t in
                       zip(top10_sector["overall_rank_in_sector"], top10_sector["ticker"])]
            fig = go.Figure(go.Bar(
                x=top10_sector["overall_score"], y=y_labels,
                orientation="h", marker_color=colors_top10,
                text=top10_sector["overall_score"].round(3),
                textposition="outside", textfont=dict(color="#94a3b8", size=10),
                customdata=top10_sector[["industry", "predicted_return"]],
                hovertemplate="<b>%{y}</b><br>Industry: %{customdata[0]}<br>"
                             "20d Predicted Return: %{customdata[1]:.2%}<br>Score: %{x:.3f}<extra></extra>"
            ))
            fig.update_layout(**PLOT_THEME, height=300,
                              xaxis_title="Overall Score (sector-wide rank)", yaxis_title="")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            legend_html = "".join([
                f"<span style='display:inline-flex;align-items:center;gap:0.3rem;margin-right:1rem;'>"
                f"<span style='width:9px;height:9px;border-radius:2px;background:{c};display:inline-block;'></span>"
                f"<span style='color:#94a3b8;font-size:0.75rem;'>{r}</span></span>"
                for r, c in rec_colors.items()
            ])
            st.markdown(f"<div>{legend_html}</div>", unsafe_allow_html=True)
            st.caption(f"Scope: {scope_label}. Overall Score is still computed against the full sector "
                      f"percentile, not just this industry filter — see Methodology below.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Focus ticker(s) vs sector average, or vs each other ──
        if len(focus_tickers) >= 1:
            radar_dims = ["Sentiment", "Financial Quality", "Forecast", "Valuation", "Leverage", "Safety"]
            radar_cols = ["sentiment_pct", "financial_pct", "forecast_pct", "valuation_pct", "leverage_pct"]
            palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#06b6d4", "#ec4899", "#84cc16"]

            def _radar_vals(row):
                vals = [row[c] * 100 for c in radar_cols]
                vals.append((1 - row["risk_pct"]) * 100)  # Safety = inverted risk_pct
                return vals

            if len(focus_tickers) == 1:
                title_text = f"{focus_tickers[0]} vs. {selected_sector} Average"
            elif len(focus_tickers) <= 5:
                title_text = f"Comparing {' vs '.join(focus_tickers)} — {selected_sector}"
            else:
                shown = ', '.join(focus_tickers[:5])
                title_text = f"Comparing {shown} +{len(focus_tickers)-5} more — {selected_sector}"
            st.markdown(f'<div class="section-header">{title_text}</div>', unsafe_allow_html=True)

            # ── Data freshness ───────────────────────────────────
            as_of = bench["earnings_date"].max()
            fresh_rows = sector_df[sector_df["ticker"].isin(focus_tickers)].set_index("ticker").loc[focus_tickers]
            fresh_parts = [
                f"{t}: {d.strftime('%b %d, %Y')} ({(as_of - d).days}d before most recent call in dataset)"
                for t, d in fresh_rows["earnings_date"].items()
            ]
            st.caption(f"Latest call used per ticker (dataset's most recent call overall: "
                      f"{as_of.strftime('%b %d, %Y')}) — " + " · ".join(fresh_parts))

            col_r1, col_r2 = st.columns([1.3, 1])
            with col_r1:
                fig = go.Figure()

                if len(focus_tickers) == 1:
                    # Single ticker: compare against the sector median reference line.
                    ticker_row = sector_df[sector_df["ticker"] == focus_tickers[0]].iloc[0]
                    sector_vals = [50] * len(radar_dims)
                    fig.add_trace(go.Scatterpolar(
                        r=sector_vals + [sector_vals[0]], theta=radar_dims + [radar_dims[0]],
                        name="Sector Median (50th pct)", line=dict(color="#475569", dash="dash"),
                        fill="none"
                    ))
                    ticker_vals = _radar_vals(ticker_row)
                    fig.add_trace(go.Scatterpolar(
                        r=ticker_vals + [ticker_vals[0]], theta=radar_dims + [radar_dims[0]],
                        name=focus_tickers[0], line=dict(color=palette[0], width=2),
                        fill="toself", fillcolor="rgba(59,130,246,0.15)"
                    ))
                else:
                    # Multiple tickers: compare against each other, no sector median line.
                    for i, t in enumerate(focus_tickers):
                        row = sector_df[sector_df["ticker"] == t].iloc[0]
                        vals = _radar_vals(row)
                        color = palette[i % len(palette)]
                        fig.add_trace(go.Scatterpolar(
                            r=vals + [vals[0]], theta=radar_dims + [radar_dims[0]],
                            name=t, line=dict(color=color, width=2)
                        ))

                fig.update_layout(
                    polar=dict(
                        bgcolor="#111827",
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1e293b",
                                        linecolor="#1e293b", tickfont=dict(color="#64748b", size=9)),
                        angularaxis=dict(gridcolor="#1e293b", linecolor="#1e293b",
                                         tickfont=dict(color="#94a3b8", size=10))
                    ),
                    paper_bgcolor="#111827", font=dict(color="#94a3b8"),
                    height=320, showlegend=True,
                    legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
                    margin=dict(l=30, r=30, t=30, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Percentile rank within sector (0-100) on each dimension, higher = stronger on all "
                          "six. Safety = 100 − Risk Percentile, matching the (1 − risk_pct) inversion used "
                          "in the overall_score formula (see Methodology below).")

            with col_r2:
                if len(focus_tickers) == 1:
                    ticker_row = sector_df[sector_df["ticker"] == focus_tickers[0]].iloc[0]
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:0.6rem;">
                        <div class="metric-label">Overall Rank in Sector</div>
                        <div class="metric-value" style="font-size:1.3rem;">#{int(ticker_row['overall_rank_in_sector'])} of {int(ticker_row['peer_count'])}</div>
                        <div class="metric-sub">{ticker_row['rank_percentile']*100:.0f}th percentile</div>
                    </div>
                    <div class="metric-card" style="margin-bottom:0.6rem;">
                        <div class="metric-label">Recommendation</div>
                        <div class="metric-value" style="font-size:1.3rem;color:{rec_colors.get(ticker_row['recommendation'], '#94a3b8')};">{ticker_row['recommendation']}</div>
                        <div class="metric-sub">20d predicted return: {ticker_row['predicted_return']*100:+.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Safety Percentile</div>
                        <div class="metric-value" style="font-size:1.3rem;">{(1 - ticker_row['risk_pct'])*100:.0f}</div>
                        <div class="metric-sub">higher = safer vs. peers (raw risk score: {ticker_row['risk_score_combined']:.3f})</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Side-by-side comparison table for multiple selected tickers.
                    comp_df = sector_df[sector_df["ticker"].isin(focus_tickers)][
                        ["ticker", "overall_rank_in_sector", "rank_percentile", "overall_score",
                         "recommendation", "predicted_return", "risk_pct", "earnings_date"]
                    ].copy()
                    comp_df = comp_df.set_index("ticker").loc[focus_tickers].reset_index()
                    comp_df["overall_rank_in_sector"] = comp_df["overall_rank_in_sector"].astype(int)
                    comp_df["rank_percentile"] = (comp_df["rank_percentile"] * 100).round(0).astype(int)
                    comp_df["predicted_return"] = (comp_df["predicted_return"] * 100).round(2)
                    comp_df["risk_pct"] = (comp_df["risk_pct"] * 100).round(0).astype(int)
                    comp_df["earnings_date"] = comp_df["earnings_date"].dt.strftime("%b %d, %Y")
                    comp_df.columns = ["Ticker", "Rank", "Rank Pctl", "Score", "Rec.",
                                       "20d Pred. Return %", "Risk Pct", "Latest Call"]
                    st.dataframe(
                        comp_df.style.format({"Score": "{:.3f}", "20d Pred. Return %": "{:+.2f}"}),
                        use_container_width=True, height=min(300, 38 * (len(comp_df) + 1))
                    )

            # ── Weighted score contribution (radar shows all 6 as equal- ──
            # sized, but the formula weights them 0.25/0.25/0.20/0.10/0.10/0.10)
            st.markdown('<div class="section-header">Weighted Score Contribution</div>', unsafe_allow_html=True)
            weights = {"Sentiment": 0.25, "Financial Quality": 0.25, "Forecast": 0.20,
                      "Valuation": 0.10, "Leverage": 0.10, "Safety": 0.10}
            weight_cols = {"Sentiment": "sentiment_pct", "Financial Quality": "financial_pct",
                          "Forecast": "forecast_pct", "Valuation": "valuation_pct",
                          "Leverage": "leverage_pct"}
            contrib_rows = sector_df[sector_df["ticker"].isin(focus_tickers)].set_index("ticker").loc[focus_tickers]
            fig_contrib = go.Figure()
            for dim, w in weights.items():
                if dim == "Safety":
                    vals = (1 - contrib_rows["risk_pct"]) * w
                else:
                    vals = contrib_rows[weight_cols[dim]] * w
                fig_contrib.add_trace(go.Bar(
                    name=f"{dim} ({w:.0%})", y=contrib_rows.index, x=vals,
                    orientation="h",
                    hovertemplate=f"{dim}: " + "%{x:.3f}<extra></extra>"
                ))
            fig_contrib.update_layout(**PLOT_THEME, height=max(220, 55 * len(focus_tickers) + 90),
                                      barmode="stack", xaxis_title="Contribution to Overall Score",
                                      yaxis_title="",
                                      legend=dict(orientation="h", y=1.25, x=0, xanchor="left", yanchor="bottom"))
            fig_contrib.update_layout(margin=dict(t=90, b=60))
            st.plotly_chart(fig_contrib, use_container_width=True)
            st.caption("Segment width = each dimension's actual weighted contribution (bar total = "
                      "Overall Score), unlike the radar above which shows all six as visually equal.")

            # ── Sector-relative deltas ────────────────────────────
            st.markdown('<div class="section-header">Sector-Relative Deltas</div>', unsafe_allow_html=True)
            delta_specs = [
                ("sentiment_vs_sector", "Sentiment", True),
                ("financial_vs_sector", "Financial Quality", True),
                ("forecast_vs_sector", "20d Forecast Return", True),
                ("risk_vs_sector", "Risk", False),  # higher risk_vs_sector = worse, invert color logic
            ]
            delta_cols = st.columns(len(focus_tickers))
            for col, t in zip(delta_cols, focus_tickers):
                row = sector_df[sector_df["ticker"] == t].iloc[0]
                with col:
                    badges = ""
                    for field, label, higher_is_better in delta_specs:
                        v = row[field]
                        is_good = (v > 0) if higher_is_better else (v < 0)
                        color = "#10b981" if is_good else ("#64748b" if abs(v) < 1e-9 else "#ef4444")
                        arrow = "▲" if v > 0 else ("—" if abs(v) < 1e-9 else "▼")
                        badges += (f"<div style='display:flex;justify-content:space-between;padding:0.2rem 0;'>"
                                  f"<span style='color:#94a3b8;font-size:0.72rem;'>{label}</span>"
                                  f"<span style='color:{color};font-size:0.72rem;font-weight:600;'>{arrow} {v:+.3f}</span></div>")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{t}</div>
                        {badges}
                    </div>""", unsafe_allow_html=True)
            st.caption("Raw value minus sector average for each metric (not percentile-based). "
                      "For Risk, lower is better, so a red ▲ here means more risk than sector peers.")

            # ── Forecast uncertainty band for selected tickers ───
            st.markdown('<div class="section-header">20d Forecast Band — Selected Tickers</div>', unsafe_allow_html=True)
            fband = sector_df[sector_df["ticker"].isin(focus_tickers)].set_index("ticker").loc[focus_tickers].reset_index()
            fig_band = go.Figure()
            fig_band.add_trace(go.Scatter(
                x=fband["predicted_return"] * 100, y=fband["ticker"],
                mode="markers",
                marker=dict(color=[rec_colors.get(r, "#94a3b8") for r in fband["recommendation"]], size=10),
                error_x=dict(
                    type="data", symmetric=False,
                    array=(fband["upper_bound"] - fband["predicted_return"]) * 100,
                    arrayminus=(fband["predicted_return"] - fband["lower_bound"]) * 100,
                    color="#475569", thickness=1.5, width=4
                ),
                text=fband["ticker"],
                hovertemplate="%{text}<br>Predicted: %{x:.2f}%<extra></extra>"
            ))
            fig_band.add_vline(x=0, line_dash="dash", line_color="#475569", opacity=0.5)
            fig_band.update_layout(**PLOT_THEME, height=max(180, 60 * len(fband)),
                                   xaxis_title="20d Predicted Return (%) with 90% Confidence Interval",
                                   yaxis_title="")
            st.plotly_chart(fig_band, use_container_width=True)
            avg_width = fband["interval_width"].mean() * 100
            st.caption(f"Dots show the point forecast; bars show the 90% confidence interval "
                      f"(lower_bound to upper_bound) — note these are not always symmetric around the "
                      f"point forecast. Average interval width for these tickers: {avg_width:.1f} pts — "
                      f"wide bands mean the model is highly uncertain about the magnitude, even when "
                      f"confident about direction.")

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Predicted return vs overall score scatter ───────────
        st.markdown('<div class="section-header">20d Predicted Return vs. Overall Score — All Companies in Sector</div>',
                   unsafe_allow_html=True)
        scatter_df = sector_df.copy()
        fig = go.Figure()
        for rec_val, color in rec_colors.items():
            sub = scatter_df[scatter_df["recommendation"] == rec_val]
            fig.add_trace(go.Scatter(
                x=sub["overall_score"], y=sub["predicted_return"] * 100,
                mode="markers", name=rec_val,
                marker=dict(color=color, size=8, opacity=0.65),
                text=sub["ticker"], hovertemplate="%{text}<br>Score: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>"
            ))
        if len(focus_tickers) >= 1:
            fr = sector_df[sector_df["ticker"].isin(focus_tickers)]
            fig.add_trace(go.Scatter(
                x=fr["overall_score"], y=fr["predicted_return"] * 100,
                mode="markers+text", text=fr["ticker"], textposition="top center",
                textfont=dict(color="#f1f5f9", size=11),
                marker=dict(color="#f1f5f9", size=14, symbol="star", line=dict(width=1, color="#0a0e1a")),
                showlegend=False, hoverinfo="skip"
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="#475569", opacity=0.5)
        # Recommendation thresholds from the scoring methodology:
        # Buy needs score >= 0.70 (and predicted_return > 0); Hold needs score >= 0.45.
        fig.add_vline(x=0.45, line_dash="dot", line_color="#f59e0b", opacity=0.5,
                      annotation_text="Hold threshold (0.45)", annotation_position="top",
                      annotation_font_color="#f59e0b", annotation_font_size=9)
        fig.add_vline(x=0.70, line_dash="dot", line_color="#10b981", opacity=0.5,
                      annotation_text="Buy threshold (0.70)", annotation_position="top",
                      annotation_font_color="#10b981", annotation_font_size=9)
        fig.update_layout(**PLOT_THEME, height=350,
                          xaxis_title="Overall Score", yaxis_title="20d Predicted Return (%)",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Buy also requires 20d predicted return > 0, so names right of 0.70 but below "
                  "the 0% line are not Buy.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        with st.expander("📋 Methodology — How Overall Score & Recommendations Are Computed", expanded=False):
            st.markdown('''<div class="info-box">
                <div style="color:#cbd5e1;font-size:0.85rem;margin-bottom:0.9rem;">Overall score combines six sector-relative dimensions, with sentiment and financial quality weighted most heavily.</div>
                <strong style="color:#f1f5f9;">Overall Score Formula</strong><br><br>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#e2e8f0;line-height:1.9;background:#0d1420;border:1px solid #1e293b;border-radius:6px;padding:0.75rem 1rem;margin-bottom:0.9rem;">
                    Overall&nbsp;Score&nbsp;=&nbsp;0.25·Sentiment&nbsp;+&nbsp;0.25·Financial&nbsp;Quality&nbsp;+&nbsp;0.20·Forecast<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+&nbsp;0.10·Valuation&nbsp;+&nbsp;0.10·Leverage&nbsp;+&nbsp;0.10·Safety
                </div>
                <table style="width:100%;border-collapse:collapse;font-size:0.82rem;margin-bottom:0.9rem;">
                    <tr style="border-bottom:1px solid #1e293b;">
                        <td style="color:#94a3b8;padding:0.3rem 0.5rem 0.3rem 0;font-weight:600;">Tier</td>
                        <td style="color:#94a3b8;padding:0.3rem 0.5rem;font-weight:600;">Component</td>
                        <td style="color:#94a3b8;padding:0.3rem 0 0.3rem 0.5rem;font-weight:600;text-align:right;">Weight</td>
                    </tr>
                    <tr style="border-bottom:1px solid #1e293b;"><td style="padding:0.3rem 0.5rem 0.3rem 0;color:#e2e8f0;">Core</td><td style="padding:0.3rem 0.5rem;color:#e2e8f0;">Sentiment</td><td style="padding:0.3rem 0 0.3rem 0.5rem;text-align:right;color:#f1f5f9;">25%</td></tr>
                    <tr style="border-bottom:1px solid #1e293b;"><td style="padding:0.3rem 0.5rem 0.3rem 0;color:#e2e8f0;">Core</td><td style="padding:0.3rem 0.5rem;color:#e2e8f0;">Financial Quality</td><td style="padding:0.3rem 0 0.3rem 0.5rem;text-align:right;color:#f1f5f9;">25%</td></tr>
                    <tr style="border-bottom:1px solid #1e293b;"><td style="padding:0.3rem 0.5rem 0.3rem 0;color:#e2e8f0;">Core</td><td style="padding:0.3rem 0.5rem;color:#e2e8f0;">Forecast</td><td style="padding:0.3rem 0 0.3rem 0.5rem;text-align:right;color:#f1f5f9;">20%</td></tr>
                    <tr style="border-bottom:1px solid #1e293b;"><td style="padding:0.3rem 0.5rem 0.3rem 0;color:#e2e8f0;">Supporting</td><td style="padding:0.3rem 0.5rem;color:#e2e8f0;">Valuation</td><td style="padding:0.3rem 0 0.3rem 0.5rem;text-align:right;color:#f1f5f9;">10%</td></tr>
                    <tr style="border-bottom:1px solid #1e293b;"><td style="padding:0.3rem 0.5rem 0.3rem 0;color:#e2e8f0;">Supporting</td><td style="padding:0.3rem 0.5rem;color:#e2e8f0;">Leverage</td><td style="padding:0.3rem 0 0.3rem 0.5rem;text-align:right;color:#f1f5f9;">10%</td></tr>
                    <tr><td style="padding:0.3rem 0.5rem 0.3rem 0;color:#e2e8f0;">Supporting</td><td style="padding:0.3rem 0.5rem;color:#e2e8f0;">Safety</td><td style="padding:0.3rem 0 0.3rem 0.5rem;text-align:right;color:#f1f5f9;">10%</td></tr>
                </table>
                Risk is inverted via Safety = 1 − Risk to preserve directional consistency across components, ensuring that a higher contribution uniformly corresponds to a stronger overall score.
                </div>''', unsafe_allow_html=True)

            st.markdown('<div class="info-box" style="margin-top:0.75rem;"><strong style="color:#f1f5f9;">Recommendation Thresholds</strong><br><br>• <span style="color:#10b981;">Buy</span>: Overall Score ≥ 0.70 <em>and</em> predicted return > 0<br>• <span style="color:#f59e0b;">Hold</span>: Overall Score ≥ 0.45<br>• <span style="color:#ef4444;">Avoid</span>: Overall Score below 0.45</div>', unsafe_allow_html=True)

            st.markdown('<div class="info-box" style="margin-top:0.75rem;border-color:#1e3a5f;">⚠️ <strong style="color:#f1f5f9;">Important Note</strong><br><br>Every score is a <em>sector-relative</em> percentile, so a company is only ever compared against its direct peers, not the whole market — a "0.70 Overall Score" in one sector is <strong>not directly comparable across sectors</strong>.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.info(f"Industry benchmarking outputs not found. ({e})")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — INVESTMENT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
elif page == "💡  Investment Recommendations":
    st.markdown('<div class="page-title">Investment Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">ML-optimized portfolio construction and backtested performance — 2022–2023 test set</div>', unsafe_allow_html=True)

    st.markdown("""
    <div>
        <a href="#rec-summary" class="nav-chip">What to Do</a>
        <a href="#rec-why" class="nav-chip">Why</a>
        <a href="#rec-holdings" class="nav-chip">Holdings</a>
        <a href="#rec-risks" class="nav-chip">Risks</a>
        <a href="#rec-methodology" class="nav-chip">Methodology</a>
    </div>
    """, unsafe_allow_html=True)

    MAIN_STRATEGY = "ML_Optimized"
    BENCHMARK = "Full_Universe"
    MAX_POSITION_WEIGHT = 0.20
    MAX_SECTOR_WEIGHT = 0.40
    PORTFOLIO_COLORS = {
        "ML_Optimized": "#10b981",
        "Full_Universe": "#64748b",
        "Equal_Weight": "#3b82f6",
        "Market_Cap_Weight": "#f59e0b",
        "ML_Score_Weight": "#a78bfa",
        "Long_Short": "#ef4444",
    }

    try:
        weights_df = load_portfolio_weights()
        cum_wide = load_portfolio_cumulative_returns()
        net_wide = load_portfolio_net_returns()
        perf_df = load_portfolio_performance()
        main_summary = load_main_strategy_summary()

        main_row = main_summary.iloc[0]
        bench_row = perf_df[perf_df["Portfolio"] == BENCHMARK].iloc[0]
        ls_row = perf_df[perf_df["Portfolio"] == "Long_Short"].iloc[0] if "Long_Short" in perf_df["Portfolio"].values else None

        latest_quarter = weights_df["quarter"].max()
        latest_port = weights_df[
            (weights_df["quarter"] == latest_quarter) &
            (weights_df["portfolio_type"] == MAIN_STRATEGY)
        ].sort_values("weight", ascending=False).copy()

        expected_return = (latest_port["weight"] * latest_port["predicted_return"]).sum() * 100
        max_weight = latest_port["weight"].max() * 100
        top_sector = latest_port.groupby("sector")["weight"].sum().idxmax()
        top_sector_wt = latest_port.groupby("sector")["weight"].sum().max() * 100

        # ── Hero recommendation block ────────────────────────────
        st.markdown('<div id="rec-summary"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="finding-card" style="border-left-color:#10b981; padding:1.1rem 1.4rem;">
            <div style="color:#64748b;font-size:0.72rem;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:0.35rem;">
                Recommended Action — {latest_quarter}
            </div>
            <div style="color:#f1f5f9;font-size:1.3rem;font-weight:700;margin-bottom:0.2rem;">
                Hold ML-Optimized Portfolio
            </div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-bottom:0.7rem;">
                {len(latest_port)} long-only positions selected for the next quarter
            </div>
            <div style="margin-bottom:1.4rem;">
                <span class="badge-pill" style="margin-right:0.4rem;">Long-only</span>
                <span class="badge-pill" style="margin-right:0.4rem;">Backtested</span>
                <span class="badge-pill" style="margin-right:0.4rem;">{int(main_row['Number of Quarters'])} quarters</span>
                <span class="badge-pill">Transaction costs included</span>
            </div>
            <div style="display:flex;gap:1.8rem;align-items:flex-end;
                        margin-bottom:0.9rem;padding-bottom:0.9rem;border-bottom:1px solid #1e293b;">
                <div style="padding-right:1.8rem;border-right:1px solid #1e293b;">
                    <div style="color:#94a3b8;font-size:0.85rem;">Expected Return (20d)</div>
                    <div style="color:#34d399;font-size:1.3rem;font-weight:700;line-height:1.3;">{expected_return:+.1f}%</div>
                </div>
                <div style="display:flex;gap:2rem;flex-wrap:wrap;">
                    <div><div style="color:#94a3b8;font-size:0.85rem;">Max Position</div><div style="color:#e2e8f0;font-size:1.3rem;font-weight:700;line-height:1.3;">{max_weight:.1f}% <span style="color:#64748b;font-weight:400;font-size:0.95rem;">(cap {MAX_POSITION_WEIGHT*100:.0f}%)</span></div></div>
                    <div><div style="color:#94a3b8;font-size:0.85rem;">Largest Sector</div><div style="color:#e2e8f0;font-size:1.3rem;font-weight:700;line-height:1.3;">{top_sector} <span style="color:#64748b;font-weight:400;font-size:0.95rem;">({top_sector_wt:.0f}%)</span></div></div>
                    <div><div style="color:#94a3b8;font-size:0.85rem;">Sharpe Ratio</div><div style="color:#e2e8f0;font-size:1.3rem;font-weight:700;line-height:1.3;">{main_row['Sharpe Ratio']:.2f}</div></div>
                </div>
            </div>
            <div style="color:#94a3b8;font-size:0.92rem;line-height:1.55;margin-bottom:0.4rem;">
                <strong style="color:#cbd5e1;">Why:</strong> best risk-adjusted performance among long-only
                strategies tested, with {main_row['CAGR']:.1f}% CAGR and {main_row['Maximum Drawdown']:.1f}% max
                drawdown vs {bench_row['CAGR']:.1f}% CAGR for the benchmark.
            </div>
            <div style="color:#a1885f;font-size:0.85rem;opacity:0.9;line-height:1.5;">
                ⚠ <strong>Caveat:</strong> backtest covers only {int(main_row['Number of Quarters'])} quarters;
                Long/Short scored a higher headline return but is excluded here for implementation realism.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        quarter_min = weights_df["quarter"].min()
        quarter_max = weights_df["quarter"].max()

        # ── KPI cards: main strategy vs benchmark ─────────────────
        st.markdown('<div id="rec-why"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Why ML-Optimized Won</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        kpis = [
            (c1, "CAGR", f"{main_row['CAGR']:.1f}%", f"vs {bench_row['CAGR']:.1f}% benchmark"),
            (c2, "Sharpe Ratio", f"{main_row['Sharpe Ratio']:.2f}", f"vs {bench_row['Sharpe Ratio']:.2f} benchmark"),
            (c3, "Active Return vs Benchmark", f"{main_row['Annualized Active Return vs Benchmark']:+.1f}%", f"Information Ratio {main_row['Information Ratio vs Benchmark']:.2f}"),
            (c4, "Max Drawdown", f"{main_row['Maximum Drawdown']:.1f}%", f"vs {bench_row['Maximum Drawdown']:.1f}% benchmark"),
            (c5, "Quarters Tested", f"{int(main_row['Number of Quarters'])}", f"{quarter_min} – {quarter_max}"),
        ]
        for col, label, val, sub in kpis:
            with col:
                st.markdown(f"""
                <div class="metric-card" style="min-height:150px;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

        # ── Simplified cumulative growth chart (ML_Optimized + Full_Universe always shown) ──
        other_strategies = [c for c in cum_wide.columns if c not in ("quarter", MAIN_STRATEGY, BENCHMARK)]
        extra_selected = st.multiselect(
            "Add strategies to compare",
            options=other_strategies,
            default=[],
        )
        with st.expander("How each strategy differs", expanded=False):
            st.markdown("""
            <div class="info-box">
            <strong style="color:#cbd5e1;">ML Optimized</strong><br>
            Top-ranked names, with weights optimized for return under risk and concentration constraints.
            <span class="badge-pill-neutral">Featured recommendation</span>
            <div style="margin-top:0.7rem;"></div>
            <strong style="color:#cbd5e1;">Full Universe</strong><br>
            Benchmark using the average return of all eligible tickers, with no selection.
            <div style="margin-top:0.7rem;"></div>
            <strong style="color:#cbd5e1;">Equal Weight</strong><br>
            Same selected names as ML Optimized, but each holding receives the same weight.
            <div style="margin-top:0.7rem;"></div>
            <strong style="color:#cbd5e1;">Market Cap Weight</strong><br>
            Same selected names, weighted by market capitalization.
            <div style="margin-top:0.7rem;"></div>
            <strong style="color:#cbd5e1;">Score Weight</strong><br>
            Same selected names, weighted more heavily toward higher-conviction scores.
            <div style="margin-top:0.7rem;"></div>
            <strong style="color:#cbd5e1;">Long/Short</strong><br>
            Long top-ranked names and short bottom-ranked names.
            <span class="badge-pill-neutral">Not long-only, so excluded from the core recommendation</span>
            </div>
            """, unsafe_allow_html=True)
        lines_to_plot = [MAIN_STRATEGY, BENCHMARK] + extra_selected

        fig = go.Figure()
        for col in lines_to_plot:
            fig.add_trace(go.Scatter(
                x=cum_wide["quarter"], y=cum_wide[col],
                mode="lines+markers", name=col.replace("_", " "),
                line=dict(
                    width=3 if col == MAIN_STRATEGY else 1.5,
                    color=PORTFOLIO_COLORS.get(col, "#94a3b8"),
                    dash="solid" if col in (MAIN_STRATEGY, BENCHMARK) else "dot",
                ),
            ))
        fig.update_layout(**PLOT_THEME, height=300, xaxis_title="Quarter", yaxis_title="Growth of $1",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        # ── Compact performance table (3 strategies only) ───────────
        show_cols = ["Portfolio", "CAGR", "Sharpe Ratio", "Maximum Drawdown"]
        perf_small = perf_df[perf_df["Portfolio"].isin([MAIN_STRATEGY, BENCHMARK, "Long_Short"])][show_cols]
        perf_small = perf_small.set_index("Portfolio").reindex([MAIN_STRATEGY, "Long_Short", BENCHMARK]).reset_index()
        perf_small_fmt = pd.DataFrame({
            "Portfolio": perf_small["Portfolio"].str.replace("_", " "),
            "CAGR": perf_small["CAGR"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"),
            "Sharpe": perf_small["Sharpe Ratio"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
            "Max Drawdown": perf_small["Maximum Drawdown"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—"),
        })
        st.dataframe(perf_small_fmt, use_container_width=True, height=145, hide_index=True)

        if ls_row is not None:
            st.markdown(f"""
            <div class="info-box" style="border-color:#7c2d12;font-size:0.88rem;padding:0.75rem 1rem;">
                <strong style="color:#fbbf24;">Why not Long/Short?</strong> Although it had the strongest
                headline performance, we do not make it the core recommendation because it requires shorting
                and its results are based on only {int(main_row['Number of Quarters'])} test quarters.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Latest quarter recommended portfolio ─────────────────────
        st.markdown('<div id="rec-holdings"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-header">Recommended Portfolio — {latest_quarter} '
            f'(ML Optimized, {len(latest_port)} holdings)</div>', unsafe_allow_html=True)

        top3_pct = latest_port.nlargest(3, "weight")["weight"].sum() * 100
        st.caption(
            f"Top 3 positions = {top3_pct:.1f}% of portfolio  ·  "
            f"Largest sector concentration = {top_sector} at {top_sector_wt:.1f}%"
        )

        col_hold, col_sector = st.columns([2, 1])
        with col_hold:
            display = latest_port[["ticker", "sector", "weight", "predicted_return", "final_score"]].copy()
            display["weight"] = (display["weight"] * 100).round(2)
            display["predicted_return"] = (display["predicted_return"] * 100).round(2)
            display["final_score"] = display["final_score"].round(3)
            display.columns = ["Ticker", "Sector", "Weight (%)", "Predicted Return (%)", "Score"]
            table_height = 38 * (len(display) + 1) + 3
            st.dataframe(display.reset_index(drop=True), use_container_width=True, height=table_height)
            st.caption(
                "Ticker-level probability-of-up is not shown in this portfolio table because the current "
                "backtest export does not include classification probability outputs."
            )

        with col_sector:
            sector_alloc = latest_port.groupby("sector")["weight"].sum().reset_index()
            sector_alloc["weight_pct"] = sector_alloc["weight"] * 100
            sector_alloc = sector_alloc.sort_values("weight_pct", ascending=True)
            fig = px.bar(
                sector_alloc, x="weight_pct", y="sector", orientation="h",
                text=sector_alloc["weight_pct"].round(1).astype(str) + "%",
            )
            fig.update_traces(marker_color="#10b981", textposition="outside")
            fig.update_layout(**PLOT_THEME, height=340, xaxis_title="Weight (%)", yaxis_title="",
                              showlegend=False)
            fig.update_layout(margin=dict(l=10, r=30, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Risks & constraints ────────────────────────────────────
        st.markdown('<div id="rec-risks"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Risks &amp; Constraints</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="info-box">
        <ul style="margin:0;padding-left:1.1rem;line-height:1.7;">
            <li><strong style="color:#cbd5e1;">20% single-name cap</strong> — no position can exceed one-fifth
            of the portfolio.</li>
            <li><strong style="color:#cbd5e1;">40% sector cap</strong> — no single sector can exceed two-fifths
            of the portfolio.</li>
            <li><strong style="color:#cbd5e1;">Long-only</strong> — no short positions; investable through a
            standard brokerage account.</li>
            <li><strong style="color:#cbd5e1;">Transaction costs included</strong> — all returns are net of a
            10bps cost assumption on portfolio turnover.</li>
            <li><strong style="color:#cbd5e1;">Backtest length: only {int(main_row['Number of Quarters'])} quarters</strong>
            ({quarter_min}–{quarter_max}) — CAGR and Sortino can look artificially extreme over so few periods;
            a single strong or weak quarter moves them disproportionately. Treat this section as directional
            evidence the strategy works, not a precise, statistically robust estimate.</li>
        </ul>
        </div>''', unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        st.markdown('<div id="rec-methodology"></div>', unsafe_allow_html=True)
        with st.expander("📋 Methodology — How It Was Built", expanded=False):
            st.markdown('''<div class="info-box">
            <ul style="margin:0;padding-left:1.1rem;line-height:1.7;">
                <li><strong style="color:#cbd5e1;">Ranking logic:</strong> tickers are scored each quarter on a
                composite <em>final score</em> — expected alpha, probability-adjusted return, forecast confidence,
                overall benchmarking score, and return-to-uncertainty, net of downside risk.</li>
                <li><strong style="color:#cbd5e1;">Selection:</strong> the top 8–15 names (10% of the eligible
                universe, capped) are kept each quarter.</li>
                <li><strong style="color:#cbd5e1;">Weighting:</strong> weights are chosen to maximize expected
                return subject to a risk penalty.</li>
                <li><strong style="color:#cbd5e1;">Constraints:</strong> 20% max single-position weight, 40% max
                sector weight.</li>
                <li><strong style="color:#cbd5e1;">Alternatives tested:</strong> equal-weight, market-cap-weight,
                score-weight, and a long-short overlay are backtested alongside the optimized portfolio.</li>
                <li><strong style="color:#cbd5e1;">Transaction costs:</strong> all returns are net of a 10bps
                cost assumption on portfolio turnover.</li>
                <li><strong style="color:#cbd5e1;">Benchmark definition:</strong> <code>Full_Universe</code> is
                the unweighted average return of every eligible ticker each quarter — no selection or weighting
                applied, used as the passive baseline.</li>
                <li><strong style="color:#cbd5e1;">Why Long/Short is excluded:</strong> it posts the highest
                backtested CAGR, but a 100% win rate and 0% drawdown over only 5 quarters is more consistent
                with a small-sample artifact than a durable edge — and it requires a shorting-capable account,
                which most readers of this dashboard don't have.</li>
            </ul>
            </div>''', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)

        with st.expander("📊 Full Strategy Comparison", expanded=False):
            all_strategies = [c for c in cum_wide.columns if c != "quarter"]
            selected_strategies = st.multiselect(
                "Select strategies to compare",
                options=all_strategies,
                default=[],
            )
            with st.popover("How each strategy differs"):
                st.markdown("""
                <div style="font-size:0.85rem;line-height:1.6;">
                <strong style="color:#cbd5e1;">ML Optimized</strong><br>
                Top-ranked names, with weights optimized for return under risk and concentration constraints.
                <span class="badge-pill-neutral">Featured recommendation</span>
                <div style="margin-top:0.7rem;"></div>
                <strong style="color:#cbd5e1;">Full Universe</strong><br>
                Benchmark using the average return of all eligible tickers, with no selection.
                <div style="margin-top:0.7rem;"></div>
                <strong style="color:#cbd5e1;">Equal Weight</strong><br>
                Same selected names as ML Optimized, but each holding receives the same weight.
                <div style="margin-top:0.7rem;"></div>
                <strong style="color:#cbd5e1;">Market Cap Weight</strong><br>
                Same selected names, weighted by market capitalization.
                <div style="margin-top:0.7rem;"></div>
                <strong style="color:#cbd5e1;">Score Weight</strong><br>
                Same selected names, weighted more heavily toward higher-conviction scores.
                <div style="margin-top:0.7rem;"></div>
                <strong style="color:#cbd5e1;">Long/Short</strong><br>
                Long top-ranked names and short bottom-ranked names.
                <span class="badge-pill-neutral">Not long-only, so excluded from the core recommendation</span>
                </div>
                """, unsafe_allow_html=True)
            if selected_strategies:
                fig = go.Figure()
                for col in selected_strategies:
                    fig.add_trace(go.Scatter(
                        x=cum_wide["quarter"], y=cum_wide[col],
                        mode="lines+markers", name=col.replace("_", " "),
                        line=dict(
                            width=3 if col == MAIN_STRATEGY else 1.5,
                            color=PORTFOLIO_COLORS.get(col, "#94a3b8"),
                            dash="solid" if col in (MAIN_STRATEGY, BENCHMARK) else "dot",
                        ),
                    ))
                fig.update_layout(**PLOT_THEME, height=340, xaxis_title="Quarter", yaxis_title="Growth of $1",
                                  legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, use_container_width=True)

                row_order = ["ML_Optimized", "Equal_Weight", "Market_Cap_Weight", "ML_Score_Weight", "Long_Short", "Full_Universe"]
                rows_present = [r for r in row_order if r in net_wide.columns]
                quarters_list = net_wide["quarter"].tolist()
                z = (net_wide.set_index("quarter")[rows_present] * 100).T.values
                text = [[f"{v:+.1f}%" for v in row] for row in z]
                fig_hm = go.Figure(data=go.Heatmap(
                    z=z, x=quarters_list, y=[r.replace("_", " ") for r in rows_present],
                    colorscale="RdYlGn", zmid=0, text=text, texttemplate="%{text}",
                    textfont={"size": 11}, colorbar=dict(title="Return (%)", thickness=12),
                    hoverinfo="skip",
                ))
                fig_hm.update_layout(**PLOT_THEME, height=300, xaxis_title="Quarter", yaxis_title="")
                st.plotly_chart(fig_hm, use_container_width=True)

                show_cols = ["Portfolio", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown", "Win Rate"]
                perf_display = perf_df[show_cols].sort_values("CAGR", ascending=False).reset_index(drop=True)
                perf_fmt = pd.DataFrame({
                    "Portfolio": perf_display["Portfolio"].str.replace("_", " "),
                    "CAGR": perf_display["CAGR"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"),
                    "Sharpe": perf_display["Sharpe Ratio"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
                    "Sortino": perf_display["Sortino Ratio"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a*"),
                    "Max Drawdown": perf_display["Maximum Drawdown"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—"),
                    "Win Rate": perf_display["Win Rate"].map(lambda v: f"{v:.0f}%" if pd.notna(v) else "—"),
                })
                st.dataframe(perf_fmt, use_container_width=True, height=195, hide_index=True)
                st.caption("*n/a — Sortino is undefined when a strategy has zero losing quarters in the backtest window.")
            else:
                st.caption("Select one or more strategies above to view the full comparison chart, quarterly return heatmap, and metrics table.")

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.info(f"Portfolio construction outputs not found. ({e})")

# ══════════════════════════════════════════════════════════════
# PAGE 5 — MODEL VALIDATION
# ══════════════════════════════════════════════════════════════
elif page == "📊  Model Validation":
    st.markdown('<div class="page-title">Model Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Held-out test performance, robustness checks, and forecast reliability</div>', unsafe_allow_html=True)

    # ── Key Takeaways (KPI cards) ────────────────────────────────
    try:
        _results_kt = load_classification()
        _results_kt = _results_kt[_results_kt["Model"] != "Dummy Classifier"].copy()
        _best_row = _results_kt.loc[_results_kt["AUC_ROC"].idxmax()]

        _rob_kt = load_robustness()
        _rob_kt = _rob_kt[_rob_kt["Model"] != "Dummy Classifier"].copy()
        _rf_rob = _rob_kt[_rob_kt["Model"] == "Random Forest"].iloc[0]
        _horizon_trend = f"{_rf_rob['AUC_5d']:.3f} → {_rf_rob['AUC_20d']:.3f}"

        _fc_kt = load_forecast_summary()
        _r2_range = f"{_fc_kt['R2'].min():.3f}–{_fc_kt['R2'].max():.3f}"

        kpi_cols = st.columns(4)
        kpis = [
            ("Best Classifier", _best_row["Model"], f"AUC-ROC {_best_row['AUC_ROC']:.3f}"),
            ("5-Day AUC-ROC", f"{_best_row['AUC_ROC']:.3f}", "vs. 0.500 dummy baseline"),
            ("Forecasting Power", f"R² {_r2_range}", "limited magnitude accuracy"),
            ("Random Forest AUC Trend", _horizon_trend, "5-day AUC → 20-day AUC"),
        ]
        for col, (label, value, sub) in zip(kpi_cols, kpis):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.15rem;">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        st.caption("Evaluated on a held-out 2022–2023 test set of 3,387 earnings calls (never seen during training).")
    except Exception as e:
        st.caption(f"Key takeaways unavailable ({e})")

    # ── Classification results ─────────────────────────────────
    st.markdown('<div class="section-header">Classification Model Comparison (Test Set 2022–2023)</div>',
                unsafe_allow_html=True)

    try:
        results = load_classification()
        results = results[results["Model"] != "Dummy Classifier"].copy()

        final_model_row = results[results["Notes"].fillna("").str.contains("FINAL MODEL")] \
            if "Notes" in results.columns else pd.DataFrame()
        final_model_name = final_model_row.iloc[0]["Model"] if not final_model_row.empty else "Random Forest"
        st.markdown(f"""
        <div style='display:inline-block;background:#111827;border:1px solid #3b82f6;border-radius:20px;
                    padding:0.35rem 0.9rem;margin-bottom:0.75rem;'>
            <span style='color:#64748b;font-size:0.75rem;'>Selected final model:</span>
            <span style='color:#3b82f6;font-weight:700;font-size:0.85rem;margin-left:0.4rem;'>{final_model_name} (baseline)</span>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            colors_bar = ["#3b82f6" if "Random Forest" in m and "Tuned" not in m
                          else "#1e3a5f" for m in results["Model"]]
            fig = go.Figure(go.Bar(
                x=results["AUC_ROC"], y=results["Model"],
                orientation="h", marker_color=colors_bar,
                text=[f"{v:.4f}" for v in results["AUC_ROC"]],
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
            # Precision vs Recall - more intuitive for investment use than F1 alone:
            # Precision = when the model signals BUY, how often is it right;
            # Recall = of the actual UP moves, how many did it catch.
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Bar(name="Precision", y=results["Model"], x=results["Precision"],
                                    orientation="h", marker_color="#3b82f6"))
            fig_pr.add_trace(go.Bar(name="Recall", y=results["Model"], x=results["Recall"],
                                    orientation="h", marker_color="#10b981"))
            fig_pr.update_layout(**PLOT_THEME, height=280, barmode="group",
                              title="Precision vs Recall by Model",
                              xaxis_title="Score", yaxis_title="")
            fig_pr.update_layout(legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"))
            fig_pr.update_layout(margin=dict(l=20, r=20, t=50, b=90))
            st.plotly_chart(fig_pr, use_container_width=True)

        _auc_min, _auc_max = results["AUC_ROC"].min(), results["AUC_ROC"].max()
        _best_model_name = results.loc[results["AUC_ROC"].idxmax(), "Model"]
        st.caption(f"{_best_model_name} leads, but all models cluster in a tight "
                  f"{_auc_min:.3f}–{_auc_max:.3f} AUC range.")

        if "show_classification_table" not in st.session_state:
            st.session_state.show_classification_table = False

        if st.button("Show Detailed Metrics Table" if not st.session_state.show_classification_table
                     else "Hide Detailed Metrics Table",
                     use_container_width=True):
            st.session_state.show_classification_table = not st.session_state.show_classification_table
            st.rerun()

        if st.session_state.show_classification_table:
            display_cols = ["Model", "AUC_ROC", "Accuracy", "Precision", "Recall", "F1"]
            show = results[[c for c in display_cols if c in results.columns]].copy()
            st.dataframe(
                show.style
                .highlight_max(subset=["AUC_ROC", "Accuracy", "F1"], color="#1e3a5f")
                .format({"AUC_ROC": "{:.4f}", "Accuracy": "{:.4f}",
                         "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}),
                use_container_width=True, height=210
            )

    except Exception as e:
        st.info(f"Classification results not found. ({e})")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Robustness ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Robustness — AUC Across Prediction Horizons</div>',
                unsafe_allow_html=True)

    try:
        rob_df = load_robustness()
        rob_df_models = rob_df[rob_df["Model"] != "Dummy Classifier"].copy()
        colors_rob = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

        fig = go.Figure()
        for i, model in enumerate(rob_df_models["Model"]):
            row = rob_df_models[rob_df_models["Model"] == model].iloc[0]
            y_vals = [row["AUC_5d"], row["AUC_10d"], row["AUC_20d"]]
            is_winner = (model == "Random Forest")
            color = colors_rob[i % len(colors_rob)]
            fig.add_trace(go.Scatter(
                x=["5-day", "10-day", "20-day"], y=y_vals,
                name=model, line=dict(color=color, width=4 if is_winner else 2),
                mode="lines+markers",
                marker=dict(size=11 if is_winner else 8),
            ))

        fig.add_hline(y=0.5, line_dash="dash", line_color="#ef4444",
                      opacity=0.5, annotation_text="Dummy baseline",
                      annotation_font_color="#ef4444", annotation_font_size=9)
        fig.add_hline(y=0.55, line_dash="dash", line_color="#64748b",
                      opacity=0.5, annotation_text="0.55 acceptable threshold",
                      annotation_font_color="#64748b", annotation_font_size=9)
        fig.update_layout(**PLOT_THEME, height=300,
                          xaxis_title="Prediction Horizon", yaxis_title="AUC-ROC")
        # Separate vertical legend to the right of the plot, rather than labels
        # clustered at the line endpoints (which collide when models finish
        # nearly identical - e.g. Random Forest and LightGBM are only 0.0007
        # AUC apart at 20-day).
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"))
        fig.update_layout(margin=dict(l=20, r=160, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        best_row = rob_df_models.set_index("Model")
        rf_row = best_row.loc["Random Forest"] if "Random Forest" in best_row.index else None
        if rf_row is not None:
            st.markdown(f"""
            <div class="info-box">
                ✅ <strong style="color:#f1f5f9;">Random Forest wins consistently</strong> — it is the best-performing model
                across all three horizons (5d, 10d, 20d), with AUC improving from {rf_row['AUC_5d']:.3f} → {rf_row['AUC_10d']:.3f} → {rf_row['AUC_20d']:.3f}
                as the prediction window lengthens. This confirms results are not an artefact of the labelling choice.
            </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.info(f"Robustness results not found. ({e})")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Price forecasting ──────────────────────────────────────
    st.markdown('<div class="section-header">Price Forecasting — LightGBM Quantile Regression</div>',
                unsafe_allow_html=True)

    try:
        fc_df = load_forecast_summary()
        fc_df = fc_df.copy()
        fc_df["Horizon"] = fc_df["Target"].map({
            "return_5d": "5-day", "return_10d": "10-day", "return_20d": "20-day"
        }).fillna(fc_df["Target"])

        col_fc1, col_fc2, col_fc3 = st.columns(3)
        with col_fc1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Model MAE", x=fc_df["Horizon"],
                                 y=fc_df["Model_MAE"], marker_color="#3b82f6"))
            fig.add_trace(go.Bar(name="Baseline MAE", x=fc_df["Horizon"],
                                 y=fc_df["Baseline_MAE"], marker_color="#1e3a5f"))
            fig.update_layout(**PLOT_THEME, height=280, barmode="group",
                              title="MAE: Model vs Baseline",
                              yaxis_title="MAE", xaxis_title="")
            fig.update_layout(legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"))
            fig.update_layout(margin=dict(l=20, r=20, t=50, b=80))
            st.plotly_chart(fig, use_container_width=True)

        with col_fc2:
            fig = go.Figure(go.Bar(
                x=fc_df["Horizon"],
                y=fc_df["Coverage"] * 100,
                marker_color=["#10b981" if c > 0.85 else "#f59e0b"
                              for c in fc_df["Coverage"]],
                text=[f"{c*100:.1f}%" for c in fc_df["Coverage"]],
                textposition="outside"
            ))
            fig.add_hline(y=90, line_dash="dash", line_color="#475569",
                          opacity=0.6, annotation_text="Target 90%",
                          annotation_font_size=9)
            fig.update_layout(**PLOT_THEME, height=260,
                              title="90% CI Coverage",
                              yaxis_title="Coverage (%)", xaxis_title="",
                              yaxis_range=[75, 100])
            st.plotly_chart(fig, use_container_width=True)

        with col_fc3:
            fig = go.Figure(go.Bar(
                x=fc_df["Horizon"], y=fc_df["R2"],
                marker_color="#8b5cf6",
                text=[f"{r:.3f}" for r in fc_df["R2"]],
                textposition="outside"
            ))
            fig.update_layout(**PLOT_THEME, height=260,
                              title="R² by Horizon",
                              yaxis_title="R²", xaxis_title="",
                              yaxis_range=[0, max(0.05, fc_df["R2"].max() * 1.5)])
            st.plotly_chart(fig, use_container_width=True)

        r2_range = f"{fc_df['R2'].min():.3f}–{fc_df['R2'].max():.3f}"
        st.markdown(f"""
        <div class="info-box">
            ⚠️ <strong style="color:#f1f5f9;">Model barely beats baseline on magnitude</strong> — Model MAE is
            only marginally better than (and for 5-day, essentially tied with) a naive baseline across all
            horizons, and R² stays near zero ({r2_range}). Consistent with the classification results, this
            confirms that predicting the <em>magnitude</em> of 5–20 day returns is extremely difficult beyond
            a naive baseline in efficient markets. 90% CI coverage ({fc_df['Coverage'].min()*100:.1f}%–{fc_df['Coverage'].max()*100:.1f}%)
            also falls short of the 90% target, suggesting the quantile intervals are somewhat too narrow.
        </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.info(f"Price forecasting results not found. ({e})")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Methodology notes (collapsible) ─────────────────────────
    with st.expander("📋 Methodology & Limitations", expanded=False):
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

    # ── What this means in practice ──────────────────────────────
    st.markdown("""
    <div class="finding-card" style="margin-top:0.75rem;">
        <div class="finding-title">💡 What This Means in Practice</div>
        <div class="finding-desc">Use the model as a ranking and signal tool, not a precise 
                return forecaster. It performs modestly better than a random baseline at 
                identifying which earnings calls lean UP vs DOWN, but it does not predict 
                the magnitude of stock moves with high precision. Treat its output as one 
                input within a broader investment process.</div>
    </div>""", unsafe_allow_html=True)
