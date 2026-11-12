import numpy as np
import pandas as pd
import pytest

from analyst import Orchestrator
from analyst.agents import (
    AnalysisContext,
    FundamentalAgent,
    RiskAgent,
    SentimentAgent,
    TechnicalAgent,
)
from analyst.agents.base import BaseAgent, score_to_rating
from analyst.agents.sentiment import score_headline
from analyst.backtest import sma_crossover_backtest
from analyst.data import synthetic_fundamentals, synthetic_headlines, synthetic_prices
from analyst.indicators import bollinger, macd, max_drawdown, rsi, sma


def ctx_for(ticker="TEST"):
    return AnalysisContext(
        ticker=ticker,
        prices=synthetic_prices(ticker, "2y"),
        fundamentals=synthetic_fundamentals(ticker),
        headlines=synthetic_headlines(ticker),
        benchmark=synthetic_prices("SPY", "2y"),
        demo=True,
    )


# ---------------------------------------------------------------- indicators
def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 100, dtype=float))
    assert rsi(up).iloc[-1] == 100.0
    down = pd.Series(np.arange(100, 1, -1, dtype=float))
    assert rsi(down).iloc[-1] < 1.0
    r = rsi(synthetic_prices("X")["Close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_sma_and_bollinger():
    s = pd.Series(np.arange(1, 31, dtype=float))
    assert sma(s, 5).iloc[-1] == pytest.approx(28.0)
    bb = bollinger(s)
    assert (bb["upper"].dropna() >= bb["lower"].dropna()).all()


def test_macd_columns():
    m = macd(synthetic_prices("X")["Close"])
    assert set(m.columns) == {"macd", "signal", "hist"}


def test_max_drawdown():
    assert max_drawdown(pd.Series([100.0, 120.0, 60.0, 90.0])) == pytest.approx(-0.5)


# -------------------------------------------------------------------- agents
def test_score_to_rating_bounds():
    assert score_to_rating(0.9) == "STRONG BUY"
    assert score_to_rating(0.3) == "BUY"
    assert score_to_rating(0.0) == "HOLD"
    assert score_to_rating(-0.3) == "SELL"
    assert score_to_rating(-0.9) == "STRONG SELL"


@pytest.mark.parametrize("agent", [TechnicalAgent(), FundamentalAgent(), SentimentAgent(), RiskAgent()])
def test_agents_return_valid_results(agent):
    res = agent.run(ctx_for())
    assert res.error is None
    assert -1.0 <= res.score <= 1.0
    assert 0.0 <= res.confidence <= 1.0
    assert res.summary


def test_agent_failure_is_contained():
    ctx = ctx_for()
    ctx.prices = ctx.prices.iloc[:10]  # too short
    res = TechnicalAgent().run(ctx)
    assert res.error and res.confidence == 0.0 and res.signal == "N/A"


def test_fundamental_handles_missing_data():
    ctx = ctx_for()
    ctx.fundamentals = {}
    res = FundamentalAgent().run(ctx)
    assert res.confidence == 0.0 and res.score == 0.0


def test_sentiment_lexicon():
    assert score_headline("Company beats estimates, record growth") > 0
    assert score_headline("Shares plunge on fraud probe and weak outlook") < 0
    assert score_headline("Company holds annual meeting") == 0


def test_risk_has_position_cap():
    res = RiskAgent().run(ctx_for())
    cap = res.details["suggested_max_position_pct"]
    assert 0.01 <= cap <= 0.15


def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent()


# ---------------------------------------------------------------- orchestrator
def test_orchestrator_demo_end_to_end():
    orch = Orchestrator(demo=True)
    rep = orch.analyze("aapl")
    assert rep.error is None
    assert rep.ticker == "AAPL"
    assert rep.rating in {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}
    assert len(rep.agents) == 4
    assert "not financial advice" in rep.narrative.lower()
    assert "AAPL" in rep.to_markdown()


def test_analyze_many_dedupes_and_orders():
    reps = Orchestrator(demo=True).analyze_many(["aapl", "MSFT", "AAPL", " "], "6mo")
    assert [r.ticker for r in reps] == ["AAPL", "MSFT"]


def test_zero_confidence_agents_get_no_vote():
    orch = Orchestrator(demo=True)
    ctx = ctx_for()
    ctx.fundamentals = {}
    ctx.headlines = []
    results = [a.run(ctx) for a in orch.agents]
    decision = orch.strategist.decide(ctx, results)
    assert decision.weights_used["fundamental"] == 0.0
    assert decision.weights_used["sentiment"] == 0.0


def test_llm_failure_falls_back_to_rules(monkeypatch):
    import analyst.agents.strategist as strat

    def boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(strat, "generate_narrative", boom)
    rep = Orchestrator(demo=True, use_llm=True, api_key="fake").analyze("AAPL")
    assert rep.narrative_source == "rules"
    assert "LLM narrative unavailable" in rep.narrative


# ------------------------------------------------------------------ backtest
def test_backtest_shapes_and_no_lookahead():
    prices = synthetic_prices("BT", "2y")
    bt = sma_crossover_backtest(prices, 20, 50)
    assert len(bt.equity) == len(prices)
    assert bt.equity.iloc[0, 0] == pytest.approx(10_000.0, rel=0.05)
    assert {"Total return", "CAGR", "Max drawdown"} <= set(bt.metrics.index)


def test_backtest_validates_windows():
    with pytest.raises(ValueError):
        sma_crossover_backtest(synthetic_prices("BT"), 50, 20)
    with pytest.raises(ValueError):
        sma_crossover_backtest(synthetic_prices("BT", "1mo"), 20, 50)
