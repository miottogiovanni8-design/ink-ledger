import pandas as pd

from trading_desk.data.market_data import build_indicator_snapshot


def make_bars(n: int = 40) -> pd.DataFrame:
    closes = [100 + i * 0.5 for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * n,
        }
    )


def test_build_indicator_snapshot_from_uptrend_bars():
    bars = make_bars()
    snapshot = build_indicator_snapshot("AAPL", "equity", bars, headlines=["headline 1"], has_fresh_headline=True)

    assert snapshot.symbol == "AAPL"
    assert snapshot.asset_class == "equity"
    assert snapshot.price == bars["close"].iloc[-1]
    assert 0 <= snapshot.rsi_14 <= 100
    assert snapshot.atr_14 >= 0
    assert snapshot.has_fresh_headline is True
    assert snapshot.headlines == ["headline 1"]


def test_build_indicator_snapshot_defaults_headlines_to_empty():
    bars = make_bars()
    snapshot = build_indicator_snapshot("BTC/USD", "crypto", bars)
    assert snapshot.headlines == []
    assert snapshot.has_fresh_headline is False
