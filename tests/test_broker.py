import pytest
from alpaca.trading.enums import OrderClass, OrderSide

from trading_desk.execution.broker import submit_bracket_order


class FakeTradingClient:
    def __init__(self):
        self.submitted_orders = []

    def submit_order(self, order_request):
        self.submitted_orders.append(order_request)
        return {"id": "fake-order-id", "status": "accepted"}


def test_submit_bracket_order_long_uses_buy_side():
    client = FakeTradingClient()
    submit_bracket_order(client, "AAPL", "long", notional_usd=15.0, stop_loss_price=185.0, take_profit_price=200.0)

    order = client.submitted_orders[0]
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.order_class == OrderClass.BRACKET
    assert order.notional == pytest.approx(15.0)
    assert order.take_profit.limit_price == pytest.approx(200.0)
    assert order.stop_loss.stop_price == pytest.approx(185.0)


def test_submit_bracket_order_short_uses_sell_side():
    client = FakeTradingClient()
    submit_bracket_order(client, "AAPL", "short", notional_usd=15.0, stop_loss_price=200.0, take_profit_price=185.0)

    order = client.submitted_orders[0]
    assert order.side == OrderSide.SELL


def test_submit_bracket_order_rejects_hold_direction():
    client = FakeTradingClient()
    with pytest.raises(ValueError):
        submit_bracket_order(client, "AAPL", "hold", notional_usd=15.0, stop_loss_price=185.0, take_profit_price=200.0)
    assert client.submitted_orders == []
