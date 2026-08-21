import pandas as pd
import pytest

from trading_desk.data import indicators


def test_rsi_strictly_increasing_series_is_100_after_warmup():
    close = pd.Series(range(1, 31), dtype=float)
    result = indicators.rsi(close, period=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_strictly_decreasing_series_is_0_after_warmup():
    close = pd.Series(range(30, 0, -1), dtype=float)
    result = indicators.rsi(close, period=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_rsi_flat_series_is_neutral_or_bounded():
    close = pd.Series([100.0] * 20)
    result = indicators.rsi(close, period=14)
    assert 0 <= result.iloc[-1] <= 100


def test_macd_flat_series_is_zero():
    close = pd.Series([100.0] * 40)
    result = indicators.macd(close)
    assert result["macd"].iloc[-1] == pytest.approx(0.0)
    assert result["signal"].iloc[-1] == pytest.approx(0.0)


def test_macd_uptrend_is_positive():
    close = pd.Series(range(1, 41), dtype=float)
    result = indicators.macd(close)
    assert result["macd"].iloc[-1] > 0


def test_bollinger_bands_flat_series_collapses_to_price():
    close = pd.Series([100.0] * 25)
    result = indicators.bollinger_bands(close, period=20)
    last = result.iloc[-1]
    assert last["mid"] == pytest.approx(100.0)
    assert last["upper"] == pytest.approx(100.0)
    assert last["lower"] == pytest.approx(100.0)


def test_bollinger_bands_upper_above_lower_for_volatile_series():
    close = pd.Series([100, 105, 95, 110, 90, 108, 92, 106, 94, 102, 98, 104, 96, 107, 93, 109, 91, 103, 97, 105, 99])
    result = indicators.bollinger_bands(close, period=20)
    last = result.iloc[-1]
    assert last["upper"] > last["mid"] > last["lower"]


def test_atr_flat_ohlc_is_zero():
    high = pd.Series([100.0] * 20)
    low = pd.Series([100.0] * 20)
    close = pd.Series([100.0] * 20)
    result = indicators.atr(high, low, close, period=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_atr_positive_when_there_is_range():
    high = pd.Series([102.0] * 20)
    low = pd.Series([98.0] * 20)
    close = pd.Series([100.0] * 20)
    result = indicators.atr(high, low, close, period=14)
    assert result.iloc[-1] == pytest.approx(4.0)
