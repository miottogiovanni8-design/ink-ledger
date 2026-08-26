import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

import trading_desk.cli.daily_mark as daily_mark
from trading_desk.cli.daily_mark import compute_baseline_index, fetch_benchmark_price, run_daily_mark
from trading_desk.config import settings


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


def test_run_daily_mark_regenerates_the_dashboard_snapshot(monkeypatch):
    """Reproduces the bug behind a live complaint: two days of real daily
    marks sat in the database with the equity chart never showing them,
    because run_daily_mark only ever wrote to the db, never to
    dashboard_snapshot.json — the one file the live dashboard actually
    reads. A mark is worthless to the chart until this also runs."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        snapshot_path = str(Path(tmp) / "snapshot.json")
        monkeypatch.setattr(settings, "db_path", db_path)
        monkeypatch.setattr(settings, "snapshot_path", snapshot_path)
        monkeypatch.setattr(settings, "alpaca_api_key", "test-key")
        monkeypatch.setattr(settings, "alpaca_secret_key", "test-secret")
        monkeypatch.setattr(daily_mark, "get_account_equity", lambda client: 101234.56)
        monkeypatch.setattr(daily_mark, "get_account_cash", lambda client: 500.0)
        monkeypatch.setattr(daily_mark, "fetch_benchmark_price", lambda client: 561.78)

        run_daily_mark()

        with open(snapshot_path) as f:
            snapshot = json.load(f)

        assert len(snapshot["equity_curve"]) == 1
        assert snapshot["equity_curve"][0]["equity"] == pytest.approx(101234.56)
