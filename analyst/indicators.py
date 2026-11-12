"""Pure-pandas technical indicators and risk statistics (no TA-Lib needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses at all in the window -> RSI is 100
    out = out.where(~((avg_loss == 0.0) & avg_gain.notna()), 100.0)
    return out


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (series - lower) / (upper - lower)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(close: pd.Series) -> float:
    """Most negative peak-to-trough decline, e.g. -0.35 for -35%."""
    curve = close / close.cummax() - 1.0
    return float(curve.min())


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical 1-day VaR as a positive loss fraction."""
    return float(-np.percentile(returns, (1 - level) * 100))


def beta(asset_returns: pd.Series, bench_returns: pd.Series) -> float | None:
    joined = pd.concat([asset_returns, bench_returns], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return None
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
    if cov[1, 1] == 0:
        return None
    return float(cov[0, 1] / cov[1, 1])
