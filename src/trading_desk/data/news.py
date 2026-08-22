"""News/sentiment feeds: Finnhub as the primary near-real-time headline source
(generous free tier, fits a multi-times-daily cycle) — both per-symbol company
news and general/macro market news — plus Alpha Vantage sentiment as a slower
secondary signal (thin free tier — call at most once/symbol/day and cache the
result, never per-cycle)."""

from datetime import date, timedelta
from typing import Any, Dict, List

import httpx

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"


def fetch_finnhub_headlines(
    symbol: str,
    api_key: str,
    http_client: httpx.Client,
    lookback_days: int = 2,
) -> List[Dict[str, str]]:
    """Returns `[{"headline": ..., "url": ...}, ...]` — the URL is carried
    through so the dashboard can cite and link the actual source behind a
    view's rationale, not just its text."""
    today = date.today()
    params = {
        "symbol": symbol,
        "from": (today - timedelta(days=lookback_days)).isoformat(),
        "to": today.isoformat(),
        "token": api_key,
    }
    response = http_client.get(f"{FINNHUB_BASE_URL}/company-news", params=params)
    response.raise_for_status()
    articles = response.json()
    return [
        {"headline": article["headline"], "url": article.get("url", "")}
        for article in articles
        if article.get("headline")
    ]


def fetch_finnhub_general_news(
    api_key: str,
    http_client: httpx.Client,
    category: str = "general",
) -> List[Dict[str, str]]:
    """Market-wide headlines (not tied to one symbol) — gives the per-asset
    view call broader macro context (rates, indices, geopolitics) alongside
    the company-specific headlines from `fetch_finnhub_headlines`. Same
    `{"headline", "url"}` shape."""
    params = {"category": category, "token": api_key}
    response = http_client.get(f"{FINNHUB_BASE_URL}/news", params=params)
    response.raise_for_status()
    articles = response.json()
    return [
        {"headline": article["headline"], "url": article.get("url", "")}
        for article in articles
        if article.get("headline")
    ]


def fetch_alphavantage_sentiment(
    symbol: str,
    api_key: str,
    http_client: httpx.Client,
) -> Dict[str, Any]:
    """Returns Alpha Vantage's raw per-ticker sentiment payload for `symbol`.
    Callers are responsible for caching this (free tier: 25 requests/day)."""
    params = {"function": "NEWS_SENTIMENT", "tickers": symbol, "apikey": api_key}
    response = http_client.get(ALPHAVANTAGE_BASE_URL, params=params)
    response.raise_for_status()
    payload = response.json()
    for item in payload.get("feed", []):
        for ticker_sentiment in item.get("ticker_sentiment", []):
            if ticker_sentiment.get("ticker") == symbol:
                return {
                    "relevance_score": float(ticker_sentiment.get("relevance_score", 0)),
                    "sentiment_score": float(ticker_sentiment.get("ticker_sentiment_score", 0)),
                    "sentiment_label": ticker_sentiment.get("ticker_sentiment_label", "Neutral"),
                }
    return {"relevance_score": 0.0, "sentiment_score": 0.0, "sentiment_label": "Neutral"}
