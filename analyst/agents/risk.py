"""Risk agent: volatility, drawdown, VaR, beta and volatility-targeted sizing."""

from __future__ import annotations

import numpy as np

from ..indicators import annualized_vol, beta, daily_returns, max_drawdown, value_at_risk
from .base import AgentResult, AnalysisContext, BaseAgent, clamp, fnum

TARGET_VOL = 0.20
BASE_POSITION = 0.10
MAX_POSITION = 0.15
MIN_POSITION = 0.01


def risk_level(vol: float, mdd: float) -> str:
    if vol > 0.50 or mdd < -0.50:
        return "VERY HIGH RISK"
    if vol > 0.35 or mdd < -0.30:
        return "HIGH RISK"
    if vol > 0.20 or mdd < -0.15:
        return "MODERATE RISK"
    return "LOW RISK"


class RiskAgent(BaseAgent):
    name = "risk"
    title = "Risk Manager"

    def analyze(self, ctx: AnalysisContext) -> AgentResult:
        df = ctx.prices
        if df is None or len(df) < 30:
            raise ValueError("need at least 30 daily bars for risk analysis")

        close = df["Close"].astype(float)
        rets = daily_returns(close)
        vol = annualized_vol(rets)
        mdd = max_drawdown(close)
        var95 = value_at_risk(rets, 0.95)

        b = None
        if ctx.benchmark is not None and len(ctx.benchmark) > 30:
            b = beta(rets, daily_returns(ctx.benchmark["Close"].astype(float)))

        # Higher risk -> more negative score
        score = 0.0
        score += 0.3 if vol < 0.20 else 0.0 if vol < 0.35 else -0.3 if vol < 0.50 else -0.6
        score += 0.3 if mdd > -0.15 else 0.0 if mdd > -0.30 else -0.3 if mdd > -0.50 else -0.6
        score = clamp(score)

        level = risk_level(vol, mdd)
        position = float(np.clip(BASE_POSITION * (TARGET_VOL / max(vol, 0.05)), MIN_POSITION, MAX_POSITION))

        beta_txt = f", beta {b:.2f}" if b is not None else ""
        return AgentResult(
            agent=self.name,
            signal=level,
            score=round(score, 3),
            confidence=0.85,
            summary=(
                f"Annualised volatility {vol:.1%}, max drawdown {mdd:.1%}, "
                f"1-day 95% VaR {var95:.1%}{beta_txt}. Volatility-targeted position cap: "
                f"{position:.1%} of portfolio."
            ),
            details={
                "annualized_volatility": fnum(vol),
                "max_drawdown": fnum(mdd),
                "var_95_1d": fnum(var95),
                "beta": fnum(b, 2),
                "risk_level": level,
                "suggested_max_position_pct": fnum(position),
            },
        )
