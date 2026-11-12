"""Streamlit front-end for the multi-agent investment & market analyst."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analyst import DEFAULT_WEIGHTS, Orchestrator
from analyst.backtest import sma_crossover_backtest
from analyst.indicators import rsi, sma

st.set_page_config(page_title="AI Investment & Market Analyst", page_icon="📈", layout="wide")

RATING_COLORS = {
    "STRONG BUY": "#0a7f3f",
    "BUY": "#3fa66b",
    "HOLD": "#8a8f98",
    "SELL": "#d9772b",
    "STRONG SELL": "#c0392b",
    "N/A": "#8a8f98",
}


# --------------------------------------------------------------------- helpers
def _get_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.getenv(name, "")
    except Exception:  # no secrets.toml present
        return os.getenv(name, "")


@st.cache_data(ttl=900, show_spinner=False)
def run_analysis(tickers: tuple, period: str, demo: bool, benchmark: str,
                 weights: tuple, use_llm: bool, api_key: str):
    orch = Orchestrator(
        weights=dict(weights),
        use_llm=use_llm,
        api_key=api_key or None,
        demo=demo,
        benchmark=benchmark or None,
    )
    return orch.analyze_many(list(tickers), period)


def price_figure(prices: pd.DataFrame, ticker: str) -> go.Figure:
    close = prices["Close"]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=(f"{ticker} price", "Volume", "RSI (14)"),
    )
    fig.add_trace(
        go.Candlestick(x=prices.index, open=prices["Open"], high=prices["High"],
                       low=prices["Low"], close=close, name="Price"), row=1, col=1)
    for w, color in ((20, "#f5a623"), (50, "#4a90e2"), (200, "#9b59b6")):
        ma = sma(close, w)
        if ma.notna().any():
            fig.add_trace(go.Scatter(x=prices.index, y=ma, name=f"SMA {w}",
                                     line=dict(width=1.4, color=color)), row=1, col=1)
    fig.add_trace(go.Bar(x=prices.index, y=prices["Volume"], name="Volume",
                         marker_color="#8a8f98"), row=2, col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=rsi(close), name="RSI",
                             line=dict(color="#e74c3c", width=1.4)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=3, col=1)
    fig.update_layout(height=760, xaxis_rangeslider_visible=False, legend_orientation="h",
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def score_bar(report) -> go.Figure:
    names = [a.agent.title() for a in report.agents] + ["Composite"]
    scores = [a.score for a in report.agents] + [report.score]
    colors = ["#4a90e2"] * len(report.agents) + [RATING_COLORS.get(report.rating, "#8a8f98")]
    fig = go.Figure(go.Bar(x=names, y=scores, marker_color=colors, text=[f"{s:+.2f}" for s in scores],
                           textposition="outside"))
    fig.update_yaxes(range=[-1.1, 1.1], zeroline=True, title="Score (-1 bearish / +1 bullish)")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
    return fig


# --------------------------------------------------------------------- sidebar
st.title("📈 Multi-Agent Investment & Market Analyst")
st.warning(
    "Educational tool only — not financial advice. Signals are rule-based and "
    "automated; they can be wrong. Do your own research before investing.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Configuration")
    tickers_in = st.text_input("Tickers (comma-separated)", "AAPL, MSFT, NVDA",
                               help="Yahoo Finance symbols, e.g. AAPL, TSLA, RELIANCE.NS, TCS.NS, ^NSEI")
    period = st.selectbox("History", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    benchmark = st.text_input("Benchmark (for beta)", "SPY", help="e.g. SPY, QQQ, ^NSEI")
    demo = st.checkbox("Demo mode (synthetic offline data)", value=False,
                       help="Use this if Yahoo Finance is rate-limiting you or you want a quick test.")

    with st.expander("Agent weights"):
        w_tech = st.slider("Technical", 0.0, 1.0, DEFAULT_WEIGHTS["technical"], 0.05)
        w_fund = st.slider("Fundamental", 0.0, 1.0, DEFAULT_WEIGHTS["fundamental"], 0.05)
        w_sent = st.slider("Sentiment", 0.0, 1.0, DEFAULT_WEIGHTS["sentiment"], 0.05)
        w_risk = st.slider("Risk", 0.0, 1.0, DEFAULT_WEIGHTS["risk"], 0.05)

    with st.expander("LLM narrative (optional)"):
        default_key = _get_secret("ANTHROPIC_API_KEY")
        use_llm = st.checkbox("Write the final note with Claude", value=bool(default_key))
        api_key = st.text_input("Anthropic API key", value=default_key, type="password",
                                help="Stored only in this session. Or set ANTHROPIC_API_KEY in secrets.")

    run = st.button("Run analysis", type="primary")

if run:
    tickers = tuple(t.strip().upper() for t in tickers_in.split(",") if t.strip())
    if not tickers:
        st.error("Enter at least one ticker.")
    elif len(tickers) > 15:
        st.error("Please analyse 15 tickers or fewer at a time.")
    elif w_tech + w_fund + w_sent + w_risk <= 0:
        st.error("At least one agent weight must be above zero.")
    else:
        weights = (("technical", w_tech), ("fundamental", w_fund), ("sentiment", w_sent), ("risk", w_risk))
        with st.spinner("Agents are analysing…"):
            st.session_state["reports"] = run_analysis(
                tickers, period, demo, benchmark.strip(), weights, use_llm, api_key)

reports = st.session_state.get("reports")
if not reports:
    st.info("Set your tickers in the sidebar and click **Run analysis**. "
            "Tip: tick *Demo mode* to try the app without any network data.")
    st.stop()

ok = [r for r in reports if not r.error]
for r in reports:
    if r.error:
        st.error(f"{r.ticker}: {r.error}")
if not ok:
    st.stop()

tab_dash, tab_chart, tab_agents, tab_bt, tab_about = st.tabs(
    ["Dashboard", "Charts", "Agent details", "Backtest", "How it works"])

# ------------------------------------------------------------------- dashboard
with tab_dash:
    st.subheader("Watchlist ranking")
    table = pd.DataFrame([r.summary_row() for r in ok]).sort_values("Score", ascending=False)
    st.dataframe(table, hide_index=True)
    st.download_button("Download table (CSV)", table.to_csv(index=False), "analysis_summary.csv", "text/csv")

    st.divider()
    sel = st.selectbox("Focus ticker", [r.ticker for r in ok], key="focus")
    rep = next(r for r in ok if r.ticker == sel)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Rating**<br><span style='font-size:1.8rem;font-weight:700;"
                f"color:{RATING_COLORS.get(rep.rating, '#888')}'>{rep.rating}</span>",
                unsafe_allow_html=True)
    c2.metric("Composite score", f"{rep.score:+.2f}")
    c3.metric("Confidence", f"{rep.confidence:.0%}")
    c4.metric("Suggested position", f"{rep.position_pct:.1%}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"#### Analyst note ({'Claude' if rep.narrative_source == 'llm' else 'rule-based'})")
        st.write(rep.narrative)
        st.download_button("Download report (Markdown)", rep.to_markdown(), f"{rep.ticker}_report.md")
    with right:
        st.plotly_chart(score_bar(rep))

# ---------------------------------------------------------------------- charts
with tab_chart:
    sel_c = st.selectbox("Ticker", [r.ticker for r in ok], key="chart_t")
    rc = next(r for r in ok if r.ticker == sel_c)
    st.plotly_chart(price_figure(rc.context.prices, rc.ticker))

# ---------------------------------------------------------------------- agents
with tab_agents:
    sel_a = st.selectbox("Ticker", [r.ticker for r in ok], key="agent_t")
    ra = next(r for r in ok if r.ticker == sel_a)
    for a in ra.agents:
        label = f"{a.agent.title()} — {a.signal} (score {a.score:+.2f}, confidence {a.confidence:.0%})"
        with st.expander(label, expanded=False):
            if a.error:
                st.error(a.error)
            st.write(a.summary)
            if a.agent == "sentiment" and a.details.get("items"):
                st.dataframe(pd.DataFrame(a.details["items"]), hide_index=True)
            else:
                st.json({k: v for k, v in a.details.items() if k != "items"})
            st.caption(f"Ran in {a.runtime_s}s")

# -------------------------------------------------------------------- backtest
with tab_bt:
    st.caption("Long-only SMA crossover vs buy & hold on the selected history. "
               "Past performance does not predict future results.")
    sel_b = st.selectbox("Ticker", [r.ticker for r in ok], key="bt_t")
    rb = next(r for r in ok if r.ticker == sel_b)
    b1, b2, b3 = st.columns(3)
    fast = b1.number_input("Fast SMA", 5, 100, 20)
    slow = b2.number_input("Slow SMA", 10, 250, 50)
    cost = b3.number_input("Cost (bps per trade)", 0.0, 100.0, 5.0)
    try:
        bt = sma_crossover_backtest(rb.context.prices, int(fast), int(slow), float(cost))
        st.line_chart(bt.equity)
        fmt = bt.metrics.copy().astype(object)
        for col in fmt.columns:
            for idx in fmt.index:
                v = bt.metrics.loc[idx, col]
                fmt.loc[idx, col] = (f"{v:.2f}" if idx.startswith("Sharpe")
                                     else f"{int(v)}" if idx == "Position changes"
                                     else f"{v:.1%}")
        st.dataframe(fmt)
    except ValueError as exc:
        st.warning(str(exc))

# ----------------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
**Pipeline**

1. **Data layer** pulls prices, fundamentals and headlines (Yahoo Finance via `yfinance`).
2. **Four specialist agents** run in parallel, each returning a score from -1 to +1 and a confidence:
   - *Technical*: trend (50/200-day), RSI, MACD, Bollinger bands
   - *Fundamental*: P/E, PEG, margins, ROE, leverage, revenue growth
   - *Sentiment*: transparent lexicon scoring of recent headlines
   - *Risk*: volatility, max drawdown, VaR, beta, volatility-targeted position cap
3. **Strategist** combines scores using your weights **multiplied by each agent's confidence**
   (an agent with no data gets no say), applies a risk override, then writes the note
   (rule-based, or with Claude if you add an API key).

**Limitations**: free data can be delayed or incomplete, sentiment is keyword-based,
and none of this accounts for your personal situation. Educational use only.
"""
    )
