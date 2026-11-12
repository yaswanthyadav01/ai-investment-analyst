"""Technical analysis agent: trend, momentum, mean-reversion."""

from __future__ import annotations

from ..indicators import bollinger, macd, rsi, sma
from .base import AgentResult, AnalysisContext, BaseAgent, clamp, fnum, score_to_rating


class TechnicalAgent(BaseAgent):
    name = "technical"
    title = "Technical Analyst"

    def analyze(self, ctx: AnalysisContext) -> AgentResult:
        df = ctx.prices
        if df is None or len(df) < 60:
            raise ValueError("need at least 60 daily bars for technical analysis")

        close = df["Close"].astype(float)
        last = float(close.iloc[-1])
        sma20, sma50, sma200 = sma(close, 20), sma(close, 50), sma(close, 200)
        rsi14 = float(rsi(close).iloc[-1])
        m = macd(close).iloc[-1]
        bb = bollinger(close).iloc[-1]

        score = 0.0
        reasons: list[str] = []

        # Trend
        if last > sma50.iloc[-1]:
            score += 0.30
            reasons.append("price above 50-day average")
        else:
            score -= 0.30
            reasons.append("price below 50-day average")

        has_200 = not sma200.isna().iloc[-1]
        if has_200:
            if sma50.iloc[-1] > sma200.iloc[-1]:
                score += 0.20
                reasons.append("50-day above 200-day (bullish regime)")
            else:
                score -= 0.20
                reasons.append("50-day below 200-day (bearish regime)")

        # Momentum / mean reversion
        if rsi14 < 30:
            score += 0.20
            reasons.append(f"RSI {rsi14:.0f} oversold")
        elif rsi14 > 70:
            score -= 0.20
            reasons.append(f"RSI {rsi14:.0f} overbought")
        else:
            reasons.append(f"RSI {rsi14:.0f} neutral")

        if m["hist"] > 0:
            score += 0.20
            reasons.append("MACD histogram positive")
        else:
            score -= 0.20
            reasons.append("MACD histogram negative")

        pct_b = float(bb["pct_b"])
        if pct_b > 1:
            score -= 0.10
            reasons.append("price above upper Bollinger band")
        elif pct_b < 0:
            score += 0.10
            reasons.append("price below lower Bollinger band")

        score = clamp(score)
        history_factor = 1.0 if has_200 else 0.75
        confidence = clamp(0.4 + 0.6 * abs(score), 0, 1) * history_factor

        return AgentResult(
            agent=self.name,
            signal=score_to_rating(score),
            score=round(score, 3),
            confidence=round(confidence, 3),
            summary="Technical picture: " + "; ".join(reasons) + ".",
            details={
                "last_close": fnum(last),
                "sma20": fnum(sma20.iloc[-1]),
                "sma50": fnum(sma50.iloc[-1]),
                "sma200": fnum(sma200.iloc[-1]),
                "rsi14": fnum(rsi14, 2),
                "macd": fnum(m["macd"]),
                "macd_signal": fnum(m["signal"]),
                "macd_hist": fnum(m["hist"]),
                "bollinger_pct_b": fnum(pct_b),
            },
        )
