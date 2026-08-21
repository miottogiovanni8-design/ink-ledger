"""Alpaca paper-trading execution: bracket orders (entry + stop-loss +
take-profit in one call) so protective exits are enforced by the broker
itself between cron cycles, not by our own polling cadence."""

from typing import Any, List

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

from trading_desk.engine.schemas import Direction


def submit_bracket_order(
    client: TradingClient,
    symbol: str,
    direction: Direction,
    notional_usd: float,
    stop_loss_price: float,
    take_profit_price: float,
) -> Any:
    if direction not in ("long", "short"):
        raise ValueError(f"cannot submit a bracket order for direction={direction!r}")

    side = OrderSide.BUY if direction == "long" else OrderSide.SELL

    order_request = MarketOrderRequest(
        symbol=symbol,
        notional=round(notional_usd, 2),
        side=side,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
        stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
    )
    return client.submit_order(order_request)


def get_account_equity(client: TradingClient) -> float:
    account = client.get_account()
    return float(account.equity)


def get_account_cash(client: TradingClient) -> float:
    account = client.get_account()
    return float(account.cash)


def get_daily_pnl(client: TradingClient) -> float:
    """Equity change since the prior trading day's close (Alpaca tracks this
    as `last_equity`) — this is what the daily loss circuit breaker watches."""
    account = client.get_account()
    return float(account.equity) - float(account.last_equity)


def get_open_positions(client: TradingClient) -> List[Any]:
    return client.get_all_positions()
