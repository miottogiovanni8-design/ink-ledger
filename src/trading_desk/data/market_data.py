"""Fetches OHLCV bars from Alpaca. Equities and ETFs both trade as regular
stock tickers on Alpaca, so a single StockHistoricalDataClient covers the
whole v2 investable universe — no separate crypto data path."""

from typing import List

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def fetch_bars(
    symbol: str,
    stock_client: StockHistoricalDataClient,
    lookback_bars: int = 252,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame:
    """Returns a DataFrame indexed by timestamp with open/high/low/close/volume columns."""
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, limit=lookback_bars)
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
    request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=timeframe, limit=lookback_bars)
    bar_set = stock_client.get_stock_bars(request)
    df = bar_set.df
    if not isinstance(df.index, pd.MultiIndex):
        return df[["close"]].rename(columns={"close": symbols[0]})
    panel = df["close"].unstack(level="symbol")
    return panel.dropna()
