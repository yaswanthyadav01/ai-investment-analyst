"""News sentiment agent (transparent lexicon scoring; no heavy NLP deps)."""

from __future__ import annotations

import re

from .base import AgentResult, AnalysisContext, BaseAgent, clamp, score_to_rating

POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "rallies", "gain", "gains",
    "growth", "record", "upgrade", "upgrades", "upgraded", "strong", "stronger", "profit",
    "profits", "profitable", "outperform", "bullish", "boost", "boosts", "boosting", "win",
    "wins", "partnership", "launch", "launches", "innovation", "expansion", "improving",
    "improves", "optimistic", "breakthrough", "opportunity", "accelerates", "demand",
    "dividend", "buyback", "approval", "approved", "secures",
}
NEGATIVE = {
    "miss", "misses", "missed", "fall", "falls", "plunge", "plunges", "slide", "slides",
    "drop", "drops", "loss", "losses", "downgrade", "downgrades", "downgraded", "weak",
    "weaker", "weakness", "probe", "lawsuit", "sues", "fraud", "recall", "bearish", "cut",
    "cuts", "layoffs", "decline", "declines", "concern", "concerns", "risk", "risks",
    "warning", "warns", "investigation", "slowing", "slump", "bankruptcy", "default",
    "fine", "fined", "scandal", "delay", "delays", "regulatory",
}

_WORD = re.compile(r"[a-z']+")


def score_headline(text: str) -> float:
    words = _WORD.findall(text.lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


class SentimentAgent(BaseAgent):
    name = "sentiment"
    title = "News Sentiment Analyst"

    def analyze(self, ctx: AnalysisContext) -> AgentResult:
        heads = [h for h in (ctx.headlines or []) if h]
        if not heads:
            return AgentResult(
                agent=self.name,
                signal="HOLD",
                score=0.0,
                confidence=0.0,
                summary="No recent headlines were available.",
                details={"headlines_scored": 0, "items": []},
            )

        scored = [(h, score_headline(h)) for h in heads]
        avg = sum(s for _, s in scored) / len(scored)
        score = clamp(avg)
        # More headlines -> more trust, but lexicon scoring is crude so cap at 0.7
        confidence = clamp(0.15 + 0.55 * min(1.0, len(heads) / 10), 0, 0.7)
        pos = sum(s > 0 for _, s in scored)
        neg = sum(s < 0 for _, s in scored)
        return AgentResult(
            agent=self.name,
            signal=score_to_rating(score),
            score=round(score, 3),
            confidence=round(confidence, 3),
            summary=(
                f"Scored {len(heads)} headlines: {pos} positive, {neg} negative, "
                f"{len(heads) - pos - neg} neutral (average sentiment {avg:+.2f})."
            ),
            details={
                "headlines_scored": len(heads),
                "positive": pos,
                "negative": neg,
                "items": [{"headline": h, "score": round(s, 2)} for h, s in scored],
            },
        )
