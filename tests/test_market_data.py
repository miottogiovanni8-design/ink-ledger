from dataclasses import dataclass

import pandas as pd

from trading_desk.data.market_data import fetch_bars, fetch_price_panel, fetch_volume_panel, _lookback_start


@dataclass
class FakeBarSet:
    df: pd.DataFrame


class FakeStockClient:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.last_request = None

    def get_stock_bars(self, request):
        self.last_request = request
        return FakeBarSet(self._df)


def make_single_symbol_df() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {"open": [1, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5], "low": [1, 2, 3, 4, 5], "close": [1.1, 2.1, 3.1, 4.1, 5.1], "volume": [100] * 5},
        index=dates,
    )


def make_multi_symbol_df() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    rows = []
    index = []
    for symbol, base in [("AAPL", 100), ("MSFT", 300)]:
        for i, d in enumerate(dates):
            index.append((symbol, d))
            rows.append({"open": base + i, "high": base + i + 1, "low": base + i - 1, "close": base + i + 0.5, "volume": 1000})
    multi_index = pd.MultiIndex.from_tuples(index, names=["symbol", "timestamp"])
    return pd.DataFrame(rows, index=multi_index)


def test_fetch_bars_single_symbol_returns_plain_index():
    client = FakeStockClient(make_single_symbol_df())
    df = fetch_bars("AAPL", client)
    assert list(df["close"]) == [1.1, 2.1, 3.1, 4.1, 5.1]
    assert not isinstance(df.index, pd.MultiIndex)


def test_fetch_price_panel_pivots_to_wide_format():
    client = FakeStockClient(make_multi_symbol_df())
    panel = fetch_price_panel(["AAPL", "MSFT"], client)

    assert set(panel.columns) == {"AAPL", "MSFT"}
    assert len(panel) == 3
    assert panel["AAPL"].iloc[0] == 100.5
    assert panel["MSFT"].iloc[0] == 300.5


def test_fetch_price_panel_drops_rows_with_missing_data():
    df = make_multi_symbol_df()
    panel_with_gap = df.drop(index=[("MSFT", df.index.get_level_values("timestamp")[0])])
    client = FakeStockClient(panel_with_gap)
    panel = fetch_price_panel(["AAPL", "MSFT"], client)
    assert len(panel) == 2  # the row missing MSFT's price is dropped


def test_fetch_volume_panel_pivots_to_wide_format():
    client = FakeStockClient(make_multi_symbol_df())
    panel = fetch_volume_panel(["AAPL", "MSFT"], client)
    assert set(panel.columns) == {"AAPL", "MSFT"}
    assert panel["AAPL"].iloc[0] == 1000
    assert panel["MSFT"].iloc[0] == 1000


def test_fetch_bars_always_sets_an_explicit_start():
    # Alpaca's bars endpoint returns an empty result when no `start` is
    # given — `limit` alone does not mean "most recent N bars." Regression
    # guard for that: every request built by this module must set `start`.
    client = FakeStockClient(make_single_symbol_df())
    fetch_bars("AAPL", client, lookback_bars=30)
    assert client.last_request.start is not None


def test_fetch_price_panel_always_sets_an_explicit_start():
    client = FakeStockClient(make_multi_symbol_df())
    fetch_price_panel(["AAPL", "MSFT"], client, lookback_bars=30)
    assert client.last_request.start is not None


def test_fetch_volume_panel_always_sets_an_explicit_start():
    client = FakeStockClient(make_multi_symbol_df())
    fetch_volume_panel(["AAPL", "MSFT"], client, lookback_bars=30)
    assert client.last_request.start is not None


def test_lookback_start_pads_for_weekends_and_holidays():
    start = _lookback_start(lookback_bars=100)
    # ~1.6 calendar days per trading day plus a small pad, so it reaches
    # noticeably further back than a naive 1-calendar-day-per-bar guess.
    from datetime import datetime, timezone

    days_back = (datetime.now(timezone.utc) - start).days
    assert days_back >= 160
