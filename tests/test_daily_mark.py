from dataclasses import dataclass

import pandas as pd

from trading_desk.cli.daily_mark import fetch_benchmark_price


@dataclass
class FakeBarSet:
    df: pd.DataFrame


class FakeStockClient:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_stock_bars(self, request):
        return FakeBarSet(self._df)


def test_fetch_benchmark_price_returns_latest_close():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    df = pd.DataFrame({"close": [560.0, 562.5, 565.25]}, index=dates)
    client = FakeStockClient(df)

    price = fetch_benchmark_price(client)

    assert price == 565.25
