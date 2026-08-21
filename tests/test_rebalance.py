import pytest
from alpaca.trading.enums import OrderSide

from trading_desk.execution.rebalance import compute_rebalance_trades, positions_market_value, submit_rebalance_trades


class FakePosition:
    def __init__(self, symbol, market_value):
        self.symbol = symbol
        self.market_value = market_value


class FakeTradingClient:
    def __init__(self):
        self.submitted_orders = []

    def submit_order(self, order_request):
        self.submitted_orders.append(order_request)
        return {"id": "fake-order-id"}


def test_positions_market_value_extracts_symbol_and_value():
    positions = [FakePosition("AAPL", "1500.25"), FakePosition("MSFT", "800.0")]
    result = positions_market_value(positions)
    assert result == {"AAPL": 1500.25, "MSFT": 800.0}


class TestComputeRebalanceTrades:
    def test_buys_new_target_position(self):
        trades = compute_rebalance_trades({"AAPL": 0.10}, {}, total_equity=10_000, min_trade_usd=5.0)
        assert trades == {"AAPL": pytest.approx(1000.0)}

    def test_sells_position_dropped_from_target(self):
        trades = compute_rebalance_trades({}, {"AAPL": 500.0}, total_equity=10_000, min_trade_usd=5.0)
        assert trades == {"AAPL": pytest.approx(-500.0)}

    def test_small_deltas_are_dropped(self):
        trades = compute_rebalance_trades({"AAPL": 0.10}, {"AAPL": 998.0}, total_equity=10_000, min_trade_usd=5.0)
        assert trades == {}

    def test_delta_above_threshold_is_kept(self):
        trades = compute_rebalance_trades({"AAPL": 0.10}, {"AAPL": 990.0}, total_equity=10_000, min_trade_usd=5.0)
        assert trades == {"AAPL": pytest.approx(10.0)}

    def test_multiple_symbols_computed_independently(self):
        trades = compute_rebalance_trades(
            {"AAPL": 0.10, "MSFT": 0.05},
            {"AAPL": 500.0, "XLK": 300.0},
            total_equity=10_000,
            min_trade_usd=5.0,
        )
        assert trades == {"AAPL": pytest.approx(500.0), "MSFT": pytest.approx(500.0), "XLK": pytest.approx(-300.0)}


class TestSubmitRebalanceTrades:
    def test_positive_delta_submits_buy(self):
        client = FakeTradingClient()
        submit_rebalance_trades(client, {"AAPL": 250.0})
        order = client.submitted_orders[0]
        assert order.side == OrderSide.BUY
        assert order.notional == pytest.approx(250.0)

    def test_negative_delta_submits_sell_with_absolute_notional(self):
        client = FakeTradingClient()
        submit_rebalance_trades(client, {"AAPL": -250.0})
        order = client.submitted_orders[0]
        assert order.side == OrderSide.SELL
        assert order.notional == pytest.approx(250.0)

    def test_submits_one_order_per_symbol(self):
        client = FakeTradingClient()
        submit_rebalance_trades(client, {"AAPL": 100.0, "MSFT": -50.0})
        assert len(client.submitted_orders) == 2
