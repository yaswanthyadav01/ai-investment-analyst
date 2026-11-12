"""Simple vectorised backtest: SMA-crossover strategy vs buy & hold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import TRADING_DAYS, max_drawdown, sma


@dataclass
class BacktestResult:
    equity: pd.DataFrame  # columns: Strategy, Buy & Hold
    metrics: pd.DataFrame  # index: metric name, columns: Strategy, Buy & Hold


def _metrics(equity: pd.Series, returns: pd.Series, trades: int) -> dict:
    n = max(len(returns), 1)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    years = n / TRADING_DAYS
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    std = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std and std > 0 else float("nan")
    return {
        "Total return": total,
        "CAGR": cagr,
        "Sharpe (rf=0)": sharpe,
        "Max drawdown": max_drawdown(equity),
        "Position changes": trades,
    }


def sma_crossover_backtest(
    prices: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    cost_bps: float = 5.0,
    initial: float = 10_000.0,
) -> BacktestResult:
    """Long when fast SMA > slow SMA, otherwise cash.

    The signal is shifted one bar so a trade can only use information
    available *before* the bar it trades on (no look-ahead bias).
    """
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    close = prices["Close"].astype(float)
    if len(close) <= slow + 5:
        raise ValueError(f"need more than {slow + 5} bars for this backtest")

    ret = close.pct_change().fillna(0.0)
    fast_ma, slow_ma = sma(close, fast), sma(close, slow)
    raw = (fast_ma > slow_ma).astype(float)
    raw[slow_ma.isna()] = 0.0
    position = raw.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strat_ret = position * ret - turnover * cost_bps / 1e4

    equity = pd.DataFrame(
        {
            "Strategy": (1 + strat_ret).cumprod() * initial,
            "Buy & Hold": (1 + ret).cumprod() * initial,
        }
    )
    metrics = pd.DataFrame(
        {
            "Strategy": _metrics(equity["Strategy"], strat_ret, int((turnover > 0).sum())),
            "Buy & Hold": _metrics(equity["Buy & Hold"], ret, 1),
        }
    )
    return BacktestResult(equity=equity, metrics=metrics)
