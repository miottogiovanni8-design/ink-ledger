from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from trading_desk.execution.reconcile import compute_realized_pnl_eur, find_closing_order, reconcile_open_trades
from trading_desk.persistence.models import Trade


@dataclass
class FakeOrder:
    symbol: str
    filled_avg_price: float
    filled_at: Optional[datetime]


def make_trade(**overrides) -> Trade:
    defaults = dict(
        symbol="AAPL",
        asset_class="equity",
        direction="long",
        entry_price=190.0,
        stop_loss_price=185.0,
        take_profit_price=200.0,
        size_eur=15.0,
        status="open",
        opened_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    defaults.update(overrides)
    return Trade(**defaults)


def test_find_closing_order_picks_most_recent_matching_fill():
    opened_after = datetime.now(timezone.utc) - timedelta(hours=2)
    older = FakeOrder("AAPL", 198.0, opened_after + timedelta(hours=1))
    newer = FakeOrder("AAPL", 199.0, opened_after + timedelta(hours=1, minutes=30))
    other_symbol = FakeOrder("MSFT", 400.0, opened_after + timedelta(hours=1))
    before_open = FakeOrder("AAPL", 150.0, opened_after - timedelta(hours=5))

    result = find_closing_order([older, newer, other_symbol, before_open], "AAPL", opened_after)
    assert result is newer


def test_find_closing_order_returns_none_when_no_match():
    assert find_closing_order([], "AAPL", datetime.now(timezone.utc)) is None


def test_compute_realized_pnl_long_profit():
    trade = make_trade(direction="long", entry_price=100.0, size_eur=15.0)
    pnl = compute_realized_pnl_eur(trade, exit_price=110.0)
    assert pnl == pytest.approx(1.5)  # +10% return * 15 EUR notional


def test_compute_realized_pnl_short_profit():
    trade = make_trade(direction="short", entry_price=100.0, size_eur=15.0)
    pnl = compute_realized_pnl_eur(trade, exit_price=90.0)
    assert pnl == pytest.approx(1.5)  # price fell 10%, short profits


def test_reconcile_skips_trades_still_open_at_broker():
    trade = make_trade()
    closed = reconcile_open_trades([trade], broker_open_symbols={"AAPL"}, closed_orders=[])
    assert closed == 0
    assert trade.status == "open"


def test_reconcile_closes_trade_with_matching_fill():
    trade = make_trade(entry_price=190.0, direction="long", size_eur=15.0)
    filled_at = trade.opened_at + timedelta(hours=1)
    order = FakeOrder("AAPL", 198.0, filled_at)

    closed = reconcile_open_trades([trade], broker_open_symbols=set(), closed_orders=[order])

    assert closed == 1
    assert trade.status == "closed"
    assert trade.exit_price == pytest.approx(198.0)
    assert trade.closed_at == filled_at
    assert trade.realized_pnl_eur == pytest.approx(15.0 * (198.0 - 190.0) / 190.0)


def test_reconcile_leaves_trade_open_when_no_fill_found_yet():
    trade = make_trade()
    closed = reconcile_open_trades([trade], broker_open_symbols=set(), closed_orders=[])
    assert closed == 0
    assert trade.status == "open"
