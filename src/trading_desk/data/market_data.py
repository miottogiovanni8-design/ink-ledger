"""Fetches OHLCV bars from Alpaca. Equities and ETFs both trade as regular
stock tickers on Alpaca, so a single StockHistoricalDataClient covers the
whole v2 investable universe — no separate crypto data path."""

from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Alpaca's bars endpoint returns an empty result when no `start` is given —
# `limit` alone does not imply "most recent N bars." lookback_bars counts
# trading days; 1.6x calendar days plus a small pad comfortably covers
# weekends/holidays without over- or under-fetching.
def _lookback_start(lookback_bars: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=int(lookback_bars * 1.6) + 5)


def fetch_bars(
    symbol: str,
    stock_client: StockHistoricalDataClient,
    lookback_bars: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """Returns a DataFrame indexed by timestamp with open/high/low/close/volume columns."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=timeframe, limit=lookback_bars, start=_lookback_start(lookback_bars)
    )
    bar_set = stock_client.get_stock_bars(request)
    df = bar_set.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    return df


def fetch_price_panel(
    symbols: List[str],
    stock_client: StockHistoricalDataClient,
    lookback_bars: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """Wide panel — rows are dates, columns are tickers, values are close
    price — the direct input to Black-Litterman's covariance and prior
    computation (`engine/black_litterman.py`)."""
    # For a multi-symbol request, `limit` caps the *combined* bar count
    # across every symbol, not bars per symbol — even a generous multiple
    # of lookback_bars can silently starve the last few symbols once the
    # combined cap is hit. `start` alone bounds the query correctly here;
    # leaving `limit` unset lets the client paginate until that date floor
    # is satisfied for every symbol, not until an arbitrary count is hit.
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=_lookback_start(lookback_bars),
    )
    bar_set = stock_client.get_stock_bars(request)
    df = bar_set.df
    if not isinstance(df.index, pd.MultiIndex):
        return df[["close"]].rename(columns={"close": symbols[0]})
    panel = df["close"].unstack(level="symbol")
    return panel.dropna()


def fetch_volume_panel(
    symbols: List[str],
    stock_client: StockHistoricalDataClient,
    lookback_bars: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """Wide panel of trading volume — used for the ETF market-cap proxy
    (`data/fundamentals.py::dollar_volume_proxy_weights`)."""
    # See fetch_price_panel: no `limit` here either, for the same reason.
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=_lookback_start(lookback_bars),
    )
    bar_set = stock_client.get_stock_bars(request)
    df = bar_set.df
    if not isinstance(df.index, pd.MultiIndex):
        return df[["volume"]].rename(columns={"volume": symbols[0]})
    panel = df["volume"].unstack(level="symbol")
    return panel.dropna()
