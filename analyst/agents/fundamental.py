"""Fundamental analysis agent: valuation, profitability, leverage, growth."""

from __future__ import annotations

from .base import AgentResult, AnalysisContext, BaseAgent, clamp, fnum, score_to_rating


def _num(info: dict, key: str):
    v = info.get(key)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _score_pe(pe: float) -> float:
    if pe <= 0:
        return -0.8
    if pe < 15:
        return 1.0
    if pe < 25:
        return 0.4
    if pe < 35:
        return -0.2
    return -0.8


def _score_peg(peg: float) -> float:
    if peg <= 0:
        return -0.3
    if peg < 1:
        return 1.0
    if peg < 2:
        return 0.2
    return -0.5


def _score_margin(m: float) -> float:
    if m > 0.20:
        return 1.0
    if m > 0.10:
        return 0.5
    if m > 0:
        return 0.0
    return -1.0


def _score_roe(r: float) -> float:
    if r > 0.20:
        return 1.0
    if r > 0.10:
        return 0.4
    if r > 0:
        return 0.0
    return -1.0


def _score_de(d: float) -> float:
    # Yahoo reports debt/equity in percent (e.g. 150 == 1.5x)
    if d < 50:
        return 0.6
    if d < 150:
        return 0.0
    return -0.7


def _score_growth(g: float) -> float:
    if g > 0.15:
        return 1.0
    if g > 0.05:
        return 0.4
    if g > 0:
        return 0.0
    return -0.8


class FundamentalAgent(BaseAgent):
    name = "fundamental"
    title = "Fundamental Analyst"

    N_METRICS = 6

    def analyze(self, ctx: AnalysisContext) -> AgentResult:
        info = ctx.fundamentals or {}
        pe = _num(info, "forwardPE") if _num(info, "forwardPE") is not None else _num(info, "trailingPE")
        peg = _num(info, "pegRatio")
        margin = _num(info, "profitMargins")
        roe = _num(info, "returnOnEquity")
        de = _num(info, "debtToEquity")
        growth = _num(info, "revenueGrowth")

        parts: list[tuple[str, float, str]] = []
        if pe is not None:
            parts.append(("valuation (P/E)", _score_pe(pe), f"P/E {pe:.1f}"))
        if peg is not None:
            parts.append(("growth-adjusted value (PEG)", _score_peg(peg), f"PEG {peg:.2f}"))
        if margin is not None:
            parts.append(("profitability", _score_margin(margin), f"net margin {margin:.1%}"))
        if roe is not None:
            parts.append(("return on equity", _score_roe(roe), f"ROE {roe:.1%}"))
        if de is not None:
            parts.append(("leverage", _score_de(de), f"debt/equity {de:.0f}%"))
        if growth is not None:
            parts.append(("revenue growth", _score_growth(growth), f"revenue growth {growth:.1%}"))

        details = {
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "market_cap": fnum(info.get("marketCap"), 0),
            "pe": fnum(pe, 2),
            "peg": fnum(peg, 2),
            "profit_margin": fnum(margin),
            "roe": fnum(roe),
            "debt_to_equity": fnum(de, 1),
            "revenue_growth": fnum(growth),
            "metrics_available": len(parts),
        }

        if not parts:
            return AgentResult(
                agent=self.name,
                signal="HOLD",
                score=0.0,
                confidence=0.0,
                summary="No fundamental data available for this ticker (common for ETFs, indices and crypto).",
                details=details,
            )

        score = clamp(sum(p[1] for p in parts) / len(parts))
        coverage = len(parts) / self.N_METRICS
        confidence = clamp(0.2 + 0.7 * coverage, 0, 1)
        best = max(parts, key=lambda p: p[1])
        worst = min(parts, key=lambda p: p[1])
        summary = (
            f"Based on {len(parts)}/{self.N_METRICS} fundamentals. "
            f"Strongest: {best[0]} ({best[2]}). Weakest: {worst[0]} ({worst[2]})."
        )
        return AgentResult(
            agent=self.name,
            signal=score_to_rating(score),
            score=round(score, 3),
            confidence=round(confidence, 3),
            summary=summary,
            details=details,
        )
