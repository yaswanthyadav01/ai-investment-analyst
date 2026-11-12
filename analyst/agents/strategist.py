"""Strategist agent: fuses the specialist agents into one rating + narrative."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from ..llm import generate_narrative
from .base import RATING_ORDER, AgentResult, AnalysisContext, clamp, fnum, score_to_rating

log = logging.getLogger(__name__)


@dataclass
class Decision:
    rating: str
    score: float
    confidence: float
    position_pct: float
    narrative: str
    narrative_source: str  # "llm" | "rules"
    weights_used: dict


class StrategistAgent:
    name = "strategist"
    title = "Chief Strategist"

    def __init__(self, weights: dict[str, float], api_key: Optional[str] = None,
                 use_llm: bool = False, model: Optional[str] = None):
        self.weights = weights
        self.api_key = api_key
        self.use_llm = use_llm and bool(api_key)
        self.model = model

    # ------------------------------------------------------------------ fusion
    def decide(self, ctx: AnalysisContext, results: list[AgentResult]) -> Decision:
        eff: dict[str, float] = {}
        for r in results:
            w = float(self.weights.get(r.agent, 0.0))
            eff[r.agent] = w * clamp(r.confidence, 0, 1)

        total_eff = sum(eff.values())
        total_w = sum(float(self.weights.get(r.agent, 0.0)) for r in results) or 1.0

        if total_eff <= 0:
            score, confidence = 0.0, 0.0
        else:
            score = sum(eff[r.agent] * r.score for r in results) / total_eff
            confidence = total_eff / total_w
        score = clamp(score)
        rating = score_to_rating(score)

        by_name = {r.agent: r for r in results}
        risk = by_name.get("risk")

        # Risk override: never recommend maximum conviction on very risky assets
        if risk and risk.details.get("risk_level") == "VERY HIGH RISK" and rating == "STRONG BUY":
            rating = "BUY"

        position = 0.0
        if rating in ("BUY", "STRONG BUY") and risk:
            cap = float(risk.details.get("suggested_max_position_pct") or 0.0)
            position = cap if rating == "STRONG BUY" else cap * 0.6
        position = round(position, 4)

        narrative, source = self._narrative(ctx, results, rating, score, confidence, position)
        return Decision(
            rating=rating,
            score=round(score, 3),
            confidence=round(clamp(confidence, 0, 1), 3),
            position_pct=position,
            narrative=narrative,
            narrative_source=source,
            weights_used={k: round(v, 4) for k, v in eff.items()},
        )

    # --------------------------------------------------------------- narrative
    def _rule_narrative(self, ctx, results, rating, score, confidence, position) -> str:
        lines = [
            f"{ctx.ticker}: overall rating {rating} (composite score {score:+.2f}, "
            f"confidence {confidence:.0%})."
        ]
        for r in results:
            if r.error:
                lines.append(f"- {r.agent.title()}: unavailable ({r.error}).")
            else:
                lines.append(f"- {r.agent.title()} [{r.signal}, {r.score:+.2f}]: {r.summary}")
        if position > 0:
            lines.append(f"Suggested starting position: up to {position:.1%} of portfolio.")
        else:
            lines.append("No new position suggested at this time.")
        lines.append("This is an automated, educational analysis and not financial advice.")
        return "\n".join(lines)

    def _narrative(self, ctx, results, rating, score, confidence, position) -> tuple[str, str]:
        rules = self._rule_narrative(ctx, results, rating, score, confidence, position)
        if not self.use_llm:
            return rules, "rules"
        payload = {
            "ticker": ctx.ticker,
            "rating": rating,
            "composite_score": round(score, 3),
            "confidence": round(confidence, 3),
            "suggested_position_pct": position,
            "agents": [
                {
                    "agent": r.agent,
                    "signal": r.signal,
                    "score": r.score,
                    "confidence": r.confidence,
                    "summary": r.summary,
                    "details": {k: v for k, v in r.details.items() if k != "items"},
                }
                for r in results
            ],
        }
        prompt = "Write the investment note from these agent findings:\n" + json.dumps(payload, indent=2)
        try:
            return generate_narrative(prompt, self.api_key, self.model), "llm"
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM narrative failed, using rule-based text: %s", exc)
            return rules + f"\n(LLM narrative unavailable: {type(exc).__name__})", "rules"


__all__ = ["StrategistAgent", "Decision", "RATING_ORDER", "fnum"]
