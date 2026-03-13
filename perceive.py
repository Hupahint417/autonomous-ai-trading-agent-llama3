"""
Perceive Module — Fetches market data, computes indicators, and collects news.
Feeds the Reason step with EMA, RSI, MACD, Bollinger Bands, and news sentiment.
"""

import logging
from typing import Any, Optional

import ccxt
import pandas as pd

from sentiment import get_news_sentiment
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

logger = logging.getLogger(__name__)


def _get_exchange(exchange_id: str, api_key: Optional[str] = None, secret: Optional[str] = None):
    """Create CCXT exchange instance."""
    ex_class = getattr(ccxt, exchange_id, None)
    if ex_class is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")
    opts = {"enableRateLimit": True}
    if api_key and secret:
        return ex_class({"apiKey": api_key, "secret": secret, "options": opts})
    return ex_class({"options": opts})


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "15m",
    limit: int = 100,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV from exchange via CCXT."""
    exchange = _get_exchange(exchange_id, api_key, secret)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    ema_fast: int = 9,
    ema_slow: int = 21,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> dict[str, Any]:
    """
    Compute EMA, RSI, MACD, Bollinger Bands from OHLCV.
    Returns dict with latest values and signals for the LLM.
    """
    if df.empty or len(df) < max(ema_slow, rsi_period, macd_slow + macd_signal):
        return {}

    close = df["close"]

    # RSI
    rsi_ind = RSIIndicator(close=close, window=rsi_period)
    rsi = rsi_ind.rsi()

    # EMA
    ema_f = EMAIndicator(close=close, window=ema_fast)
    ema_s = EMAIndicator(close=close, window=ema_slow)
    ema_fast_vals = ema_f.ema_indicator()
    ema_slow_vals = ema_s.ema_indicator()

    # EMA crossover: BULLISH = fast > slow, BEARISH = fast < slow
    ema_cross = "NEUTRAL"
    if len(ema_fast_vals) >= 1 and len(ema_slow_vals) >= 1:
        f, s = ema_fast_vals.iloc[-1], ema_slow_vals.iloc[-1]
        ema_cross = "BULLISH" if f > s else "BEARISH"

    # MACD
    macd_ind = MACD(
        close=close,
        window_fast=macd_fast,
        window_slow=macd_slow,
        window_sign=macd_signal,
    )
    macd_line = macd_ind.macd()
    macd_sig = macd_ind.macd_signal()
    macd_hist = macd_ind.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()

    # BB position: UPPER / MIDDLE / LOWER
    last_close = close.iloc[-1]
    last_upper = bb_upper.iloc[-1]
    last_lower = bb_lower.iloc[-1]
    last_mid = bb_mid.iloc[-1]
    if last_close >= last_upper:
        bb_pos = "UPPER"
    elif last_close <= last_lower:
        bb_pos = "LOWER"
    else:
        bb_pos = "MIDDLE"

    return {
        "price": float(last_close),
        "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
        "ema_fast": float(ema_fast_vals.iloc[-1]),
        "ema_slow": float(ema_slow_vals.iloc[-1]),
        "ema_cross": ema_cross,
        "macd_histogram": float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else None,
        "bollinger_position": bb_pos,
        "timestamp": df["timestamp"].iloc[-1].isoformat(),
    }


def perceive(
    symbol: str,
    exchange_id: str,
    timeframe: str = "15m",
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    rsi_period: int = 14,
    ema_fast: int = 9,
    ema_slow: int = 21,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    news_provider: str = "cryptopanic",
    news_lookback_hours: int = 2,
    news_max_headlines: int = 5,
) -> dict[str, Any]:
    """
    Full perception step: fetch OHLCV, compute indicators, fetch news.
    Returns a dict with technical_signals, news_headlines, news_sentiment.
    """
    df = fetch_ohlcv(exchange_id, symbol, timeframe, api_key=api_key, secret=secret)
    indicators = compute_indicators(
        df,
        rsi_period=rsi_period,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
    )

    headlines, sentiment = get_news_sentiment(
        provider=news_provider,
        symbol=symbol,
        lookback_hours=news_lookback_hours,
        max_headlines=news_max_headlines,
    )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "technical_signals": indicators,
        "news_headlines": [h["title"] for h in headlines],
        "news_sentiment": sentiment,
        "headlines_count": len(headlines),
    }
