"""Orchestrator: fetches data, runs specialist agents in parallel, then the strategist."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .agents import (
    AgentResult,
    AnalysisContext,
    FundamentalAgent,
    RiskAgent,
    SentimentAgent,
    StrategistAgent,
    TechnicalAgent,
)
from .data import (
    DataUnavailable,
    get_fundamentals,
    get_headlines,
    get_price_history,
)

DEFAULT_WEIGHTS = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.15, "risk": 0.20}


@dataclass
class AnalysisReport:
    ticker: str
    as_of: str
    rating: str = "N/A"
    score: float = 0.0
    confidence: float = 0.0
    position_pct: float = 0.0
    narrative: str = ""
    narrative_source: str = "rules"
    agents: list[AgentResult] = field(default_factory=list)
    context: Optional[AnalysisContext] = None
    error: Optional[str] = None

    def summary_row(self) -> dict:
        row = {
            "Ticker": self.ticker,
            "Rating": self.rating,
            "Score": self.score,
            "Confidence": self.confidence,
            "Position %": round(self.position_pct * 100, 1),
        }
        for a in self.agents:
            row[a.agent.title()] = a.score
        if self.error:
            row["Error"] = self.error
        return row

    def to_markdown(self) -> str:
        out = [f"# {self.ticker} — {self.rating}", f"_As of {self.as_of}_", ""]
        if self.error:
            out.append(f"**Error:** {self.error}")
            return "\n".join(out)
        out += [
            f"- Composite score: **{self.score:+.2f}**",
            f"- Confidence: **{self.confidence:.0%}**",
            f"- Suggested position: **{self.position_pct:.1%}**",
            "",
            "## Narrative",
            self.narrative,
            "",
            "## Agent findings",
        ]
        for a in self.agents:
            out.append(f"### {a.agent.title()} — {a.signal} ({a.score:+.2f}, conf {a.confidence:.0%})")
            out.append(a.summary)
            out.append("")
        out.append("_Educational analysis only. Not financial advice._")
        return "\n".join(out)


class Orchestrator:
    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        use_llm: bool = False,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        demo: bool = False,
        benchmark: Optional[str] = "SPY",
        max_workers: int = 4,
    ):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.demo = demo
        self.benchmark = benchmark
        self.max_workers = max_workers
        self.agents = [TechnicalAgent(), FundamentalAgent(), SentimentAgent(), RiskAgent()]
        self.strategist = StrategistAgent(self.weights, api_key=api_key, use_llm=use_llm, model=model)

    # ------------------------------------------------------------------ data
    def build_context(self, ticker: str, period: str = "1y") -> AnalysisContext:
        ticker = ticker.strip().upper()
        prices = get_price_history(ticker, period, demo=self.demo)
        bench = None
        if self.benchmark and self.benchmark.upper() != ticker:
            try:
                bench = get_price_history(self.benchmark.upper(), period, demo=self.demo)
            except DataUnavailable:
                bench = None  # beta is optional
        return AnalysisContext(
            ticker=ticker,
            prices=prices,
            fundamentals=get_fundamentals(ticker, demo=self.demo),
            headlines=get_headlines(ticker, demo=self.demo),
            benchmark=bench,
            demo=self.demo,
        )

    # --------------------------------------------------------------- analysis
    def analyze(self, ticker: str, period: str = "1y") -> AnalysisReport:
        as_of = datetime.now().strftime("%Y-%m-%d %H:%M")
        ticker = ticker.strip().upper()
        try:
            ctx = self.build_context(ticker, period)
        except DataUnavailable as exc:
            return AnalysisReport(ticker=ticker, as_of=as_of, error=str(exc), narrative=str(exc))

        # Specialist agents run concurrently (I/O-light here, but extensible to API-heavy agents)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(lambda a: a.run(ctx), self.agents))

        decision = self.strategist.decide(ctx, results)
        return AnalysisReport(
            ticker=ticker,
            as_of=as_of,
            rating=decision.rating,
            score=decision.score,
            confidence=decision.confidence,
            position_pct=decision.position_pct,
            narrative=decision.narrative,
            narrative_source=decision.narrative_source,
            agents=results,
            context=ctx,
        )

    def analyze_many(self, tickers: list[str], period: str = "1y") -> list[AnalysisReport]:
        clean: list[str] = []
        for t in tickers:
            t = t.strip().upper()
            if t and t not in clean:
                clean.append(t)
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(clean)))) as pool:
            return list(pool.map(lambda t: self.analyze(t, period), clean))
