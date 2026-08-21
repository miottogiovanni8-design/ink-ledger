"""Market capitalization feed for the Black-Litterman equilibrium prior.

Equities get their real market cap from Finnhub. ETFs don't have a
market cap in the same sense (they're not companies) — the equilibrium
weight for an ETF is instead proxied by its trailing average dollar
trading volume, a standard practical substitute when true AUM isn't
available. This is a documented simplification, not hidden: it means the
ETF sleeve's prior is liquidity-weighted rather than AUM-weighted.
"""

from typing import Dict, List

import httpx
import pandas as pd

FINNHUB_PROFILE_URL = "https://finnhub.io/api/v1/stock/profile2"


def fetch_market_cap(symbol: str, api_key: str, http_client: httpx.Client) -> float:
    """Returns market cap in USD (Finnhub reports it in millions)."""
    response = http_client.get(FINNHUB_PROFILE_URL, params={"symbol": symbol, "token": api_key})
    response.raise_for_status()
    payload = response.json()
    market_cap_millions = payload.get("marketCapitalization", 0.0)
    return float(market_cap_millions) * 1_000_000


def fetch_market_caps(symbols: List[str], api_key: str, http_client: httpx.Client) -> Dict[str, float]:
    return {symbol: fetch_market_cap(symbol, api_key, http_client) for symbol in symbols}


def dollar_volume_proxy_weights(price_panel: pd.DataFrame, volume_panel: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """Average dollar volume over the trailing `lookback` days, per ticker —
    used as the equilibrium-weight proxy for ETFs (no market cap available)."""
    recent_price = price_panel.tail(lookback)
    recent_volume = volume_panel.tail(lookback)
    avg_dollar_volume = (recent_price * recent_volume).mean()
    return avg_dollar_volume.to_dict()
