"""Shared types and the BaseAgent contract."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

RATING_ORDER = ["STRONG SELL", "SELL", "HOLD", "BUY", "STRONG BUY"]


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def fnum(x: Any, nd: int = 4) -> Optional[float]:
    """JSON-safe float: None for missing/NaN/inf."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd)


def score_to_rating(score: float) -> str:
    if score >= 0.5:
        return "STRONG BUY"
    if score >= 0.2:
        return "BUY"
    if score <= -0.5:
        return "STRONG SELL"
    if score <= -0.2:
        return "SELL"
    return "HOLD"


@dataclass
class AnalysisContext:
    """Everything agents may look at for one ticker."""

    ticker: str
    prices: pd.DataFrame
    fundamentals: dict = field(default_factory=dict)
    headlines: list[str] = field(default_factory=list)
    benchmark: Optional[pd.DataFrame] = None
    demo: bool = False


@dataclass
class AgentResult:
    agent: str
    signal: str
    score: float  # -1 (very bearish / risky) .. +1 (very bullish / safe)
    confidence: float  # 0 .. 1
    summary: str
    details: dict = field(default_factory=dict)
    error: Optional[str] = None
    runtime_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class BaseAgent(ABC):
    name = "base"
    title = "Base agent"

    @abstractmethod
    def analyze(self, ctx: AnalysisContext) -> AgentResult:
        ...

    def run(self, ctx: AnalysisContext) -> AgentResult:
        """Run safely: one failing agent never takes down the whole report."""
        t0 = time.perf_counter()
        try:
            result = self.analyze(ctx)
        except Exception as exc:  # noqa: BLE001
            result = AgentResult(
                agent=self.name,
                signal="N/A",
                score=0.0,
                confidence=0.0,
                summary=f"{self.title} could not complete: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )
        result.runtime_s = round(time.perf_counter() - t0, 3)
        return result
