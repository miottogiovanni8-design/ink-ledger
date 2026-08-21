"""Fetches OHLCV bars from Alpaca and turns them into IndicatorSnapshot objects
the prefilter/decision engine consume."""

from typing import List, Optional

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trading_desk.data import indicators
from trading_desk.engine.schemas import AssetClass, IndicatorSnapshot


def fetch_bars(
    symbol: str,
    asset_class: AssetClass,
    stock_client: StockHistoricalDataClient,
    crypto_client: CryptoHistoricalDataClient,
    lookback_bars: int = 60,
    timeframe: TimeFrame = TimeFrame.Hour,
) -> pd.DataFrame:
    """Returns a DataFrame indexed by timestamp with open/high/low/close/volume columns."""
    if asset_class == "equity":
        request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, limit=lookback_bars)
        bar_set = stock_client.get_stock_bars(request)
    else:
        request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, limit=lookback_bars)
        bar_set = crypto_client.get_crypto_bars(request)

    df = bar_set.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    return df


def build_indicator_snapshot(
    symbol: str,
    asset_class: AssetClass,
    bars: pd.DataFrame,
    headlines: Optional[List[str]] = None,
    has_fresh_headline: bool = False,
) -> IndicatorSnapshot:
    close = bars["close"]
    rsi_series = indicators.rsi(close)
    macd_df = indicators.macd(close)
    bb_df = indicators.bollinger_bands(close)
    atr_series = indicators.atr(bars["high"], bars["low"], close)

    return IndicatorSnapshot(
        symbol=symbol,
        asset_class=asset_class,
        price=float(close.iloc[-1]),
        rsi_14=float(rsi_series.iloc[-1]),
        macd=float(macd_df["macd"].iloc[-1]),
        macd_signal=float(macd_df["signal"].iloc[-1]),
        macd_prev=float(macd_df["macd"].iloc[-2]),
        macd_signal_prev=float(macd_df["signal"].iloc[-2]),
        bollinger_upper=float(bb_df["upper"].iloc[-1]),
        bollinger_lower=float(bb_df["lower"].iloc[-1]),
        atr_14=float(atr_series.iloc[-1]),
        has_fresh_headline=has_fresh_headline,
        headlines=headlines or [],
    )
