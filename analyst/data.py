"""Data layer: live data via yfinance, with deterministic synthetic demo data.

Demo mode lets the whole app run offline (CI, tests, or when Yahoo rate-limits).
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


class DataUnavailable(RuntimeError):
    """Raised when market data can't be fetched for a ticker."""


PERIOD_DAYS = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "2y": 504, "5y": 1260}

_DEMO_HEADLINES = [
    "{t} beats quarterly earnings expectations as revenue growth accelerates",
    "Analysts upgrade {t} citing strong demand and improving margins",
    "{t} faces regulatory probe, shares slide on investor concern",
    "{t} announces new product launch and record customer growth",
    "Weak guidance raises concern about {t} outlook amid slowing demand",
    "{t} secures major partnership, boosting long-term opportunity",
]


def _seed(ticker: str, salt: str = "") -> int:
    return zlib.crc32((ticker.upper() + salt).encode())


# --------------------------------------------------------------------------- demo
def synthetic_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    n = PERIOD_DAYS.get(period, 252)
    rng = np.random.default_rng(_seed(ticker))
    mu = rng.uniform(0.0, 0.25) / 252
    sigma = rng.uniform(0.18, 0.45) / np.sqrt(252)
    rets = rng.normal(mu, sigma, n)
    start = rng.uniform(40, 400)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.r_[start, close[:-1]] * (1 + rng.normal(0, 0.002, n))
    spread = np.abs(rng.normal(0, sigma, n)) * 0.6
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_000, 50_000_000, n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def synthetic_fundamentals(ticker: str) -> dict:
    rng = np.random.default_rng(_seed(ticker, "fund"))
    return {
        "shortName": f"{ticker.upper()} (demo)",
        "sector": "Demo Sector",
        "forwardPE": float(rng.uniform(10, 40)),
        "trailingPE": float(rng.uniform(12, 45)),
        "pegRatio": float(rng.uniform(0.5, 3.0)),
        "profitMargins": float(rng.uniform(-0.02, 0.30)),
        "returnOnEquity": float(rng.uniform(-0.02, 0.35)),
        "debtToEquity": float(rng.uniform(10, 220)),
        "revenueGrowth": float(rng.uniform(-0.05, 0.30)),
        "marketCap": float(rng.uniform(5e9, 3e12)),
    }


def synthetic_headlines(ticker: str) -> list[str]:
    rng = np.random.default_rng(_seed(ticker, "news"))
    picks = rng.choice(len(_DEMO_HEADLINES), size=5, replace=False)
    return [_DEMO_HEADLINES[i].format(t=ticker.upper()) for i in picks]


# --------------------------------------------------------------------------- live
def get_price_history(ticker: str, period: str = "1y", demo: bool = False) -> pd.DataFrame:
    if demo:
        return synthetic_prices(ticker, period)
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataUnavailable("yfinance is not installed") from exc
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception as exc:
        raise DataUnavailable(f"Could not fetch prices for '{ticker}': {exc}") from exc
    if df is None or df.empty:
        raise DataUnavailable(f"No price data returned for '{ticker}'. Check the symbol.")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def get_fundamentals(ticker: str, demo: bool = False) -> dict:
    if demo:
        return synthetic_fundamentals(ticker)
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        return dict(info)
    except Exception:
        return {}


def get_headlines(ticker: str, demo: bool = False, limit: int = 15) -> list[str]:
    if demo:
        return synthetic_headlines(ticker)
    try:
        import yfinance as yf

        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    titles: list[str] = []
    for item in items[:limit]:
        # yfinance has returned both flat and nested ("content") shapes
        title = item.get("title") or (item.get("content") or {}).get("title")
        if title:
            titles.append(str(title))
    return titles
