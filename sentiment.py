"""
News Sentiment Module — Scores crypto news headlines for the trading agent.
Supports CryptoPanic and Finnhub APIs. Returns aggregate sentiment (-1 to 1) and headlines.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def fetch_cryptopanic_news(
    api_key: str,
    symbol: str = "BTC",
    lookback_hours: int = 2,
    max_headlines: int = 5,
) -> tuple[list[dict], float]:
    """
    Fetch news from CryptoPanic API.
    Returns (headlines, aggregate_sentiment) where sentiment is -1 (bearish) to 1 (bullish).
    CryptoPanic uses votes: positive, negative, important, liked, disliked, etc.
    """
    if not api_key:
        return [], 0.0

    base_symbol = symbol.split("/")[0] if "/" in symbol else symbol
    # Map common symbols: BTC/USDT -> BTC, ETH/USDT -> ETH
    currency = base_symbol[:3] if len(base_symbol) >= 3 else base_symbol

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": api_key,
        "currencies": currency,
        "filter": "hot",
        "public": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("CryptoPanic API error: %s", e)
        return [], 0.0
    except ValueError as e:
        logger.warning("CryptoPanic JSON error: %s", e)
        return [], 0.0

    results = data.get("results", [])
    headlines = []
    sentiments = []

    # Filter by lookback
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

    for post in results[:max_headlines * 2]:  # Fetch extra to filter
        pub = post.get("published_at")
        if pub:
            try:
                pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                if pub_dt.tzinfo:
                    pub_dt = pub_dt.replace(tzinfo=None)
                if pub_dt < cutoff:
                    continue
            except (ValueError, TypeError):
                pass

        title = post.get("title", "")
        votes = post.get("votes", {})

        # CryptoPanic votes: positive, negative, important, liked, disliked
        pos = votes.get("positive", 0)
        neg = votes.get("negative", 0)
        imp = votes.get("important", 0)

        headlines.append({"title": title, "url": post.get("url", "")})

        if pos + neg > 0:
            score = (pos - neg) / (pos + neg)
        else:
            # No votes: use neutral or slight bearish for "important" macro news
            score = -0.1 if imp > 0 else 0.0
        sentiments.append(score)

        if len(headlines) >= max_headlines:
            break

    agg = sum(sentiments) / len(sentiments) if sentiments else 0.0
    agg = max(-1.0, min(1.0, agg))
    return headlines, agg


def fetch_finnhub_news(
    api_key: str,
    symbol: str = "BTC",
    lookback_hours: int = 2,
    max_headlines: int = 5,
) -> tuple[list[dict], float]:
    """
    Fetch crypto news from Finnhub.
    Returns (headlines, aggregate_sentiment). Finnhub doesn't provide votes;
    we use a simple keyword-based heuristic for demo.
    """
    if not api_key:
        return [], 0.0

    url = "https://finnhub.io/api/v1/news"
    params = {"category": "crypto", "token": api_key}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.warning("Finnhub API error: %s", e)
        return [], 0.0
    except ValueError:
        return [], 0.0

    headlines = []
    sentiments = []
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

    # Finnhub returns list of articles (or {"data": [...]} in some endpoints)
    articles = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []

    bearish_words = {"crash", "fall", "drop", "decline", "bear", "sell", "risk", "fed", "rate hike"}
    bullish_words = {"surge", "rally", "gain", "rise", "bull", "buy", "adoption", "breakthrough"}

    for art in articles:
        if len(headlines) >= max_headlines:
            break
        ts = art.get("datetime", 0)
        if ts:
            pub_dt = datetime.utcfromtimestamp(ts)
            if pub_dt < cutoff:
                continue

        title = art.get("headline", art.get("title", ""))
        headlines.append({"title": title, "url": art.get("url", "")})

        # Simple keyword sentiment when no votes available
        t_lower = title.lower()
        b = sum(1 for w in bullish_words if w in t_lower)
        r = sum(1 for w in bearish_words if w in t_lower)
        if b + r > 0:
            score = (b - r) / (b + r)
        else:
            score = 0.0
        sentiments.append(score)

    agg = sum(sentiments) / len(sentiments) if sentiments else 0.0
    agg = max(-1.0, min(1.0, agg))
    return headlines, agg


def get_news_sentiment(
    provider: str = "cryptopanic",
    symbol: str = "BTC/USDT",
    lookback_hours: int = 2,
    max_headlines: int = 5,
) -> tuple[list[dict], float]:
    """
    Main entry: fetch news and compute sentiment.
    Returns (headlines, sentiment_score) where sentiment is -1 to 1.
    """
    if provider == "cryptopanic":
        key = os.getenv("CRYPTOPANIC_API_KEY", "")
        return fetch_cryptopanic_news(key, symbol, lookback_hours, max_headlines)
    if provider == "finnhub":
        key = os.getenv("FINNHUB_API_KEY", "")
        return fetch_finnhub_news(key, symbol, lookback_hours, max_headlines)
    return [], 0.0
