from dataclasses import dataclass

import pandas as pd
import pytest

from trading_desk.cli.daily_mark import compute_baseline_index, fetch_benchmark_price


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


def test_compute_baseline_index_sums_weight_times_price():
    weights = {"AAPL": 0.6, "MSFT": 0.4}
    prices = {"AAPL": 200.0, "MSFT": 300.0}

    index = compute_baseline_index(weights, prices)

    assert index == pytest.approx(0.6 * 200.0 + 0.4 * 300.0)


def test_compute_baseline_index_returns_none_for_empty_weights():
    assert compute_baseline_index({}, {"AAPL": 200.0}) is None


def test_compute_baseline_index_returns_none_when_a_price_is_missing():
    weights = {"AAPL": 0.6, "MSFT": 0.4}
    prices = {"AAPL": 200.0}  # MSFT missing

    assert compute_baseline_index(weights, prices) is None
