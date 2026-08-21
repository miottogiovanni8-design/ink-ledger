"""Moves the live portfolio toward target weights: diff current holdings
against target weights x equity, submit plain notional market orders for
whatever's left. No per-position stop-loss/take-profit — risk here is
managed at the portfolio level (the optimizer's volatility target and the
drawdown circuit breaker), the way an asset manager actually rebalances a
book, not the way a trader manages a single position.
"""

from typing import Any, Dict, List

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def positions_market_value(positions: List[Any]) -> Dict[str, float]:
    return {p.symbol: float(p.market_value) for p in positions}


def compute_rebalance_trades(
    target_weights: Dict[str, float],
    current_positions_usd: Dict[str, float],
    total_equity: float,
    min_trade_usd: float = 5.0,
) -> Dict[str, float]:
    """Returns {symbol: notional_delta_usd} — positive means buy, negative
    means sell. Deltas smaller than `min_trade_usd` are dropped so the
    system doesn't churn tiny positions to chase rounding noise."""
    trades: Dict[str, float] = {}
    all_symbols = set(target_weights) | set(current_positions_usd)
    for symbol in all_symbols:
        target_value = target_weights.get(symbol, 0.0) * total_equity
        current_value = current_positions_usd.get(symbol, 0.0)
        delta = target_value - current_value
        if abs(delta) >= min_trade_usd:
            trades[symbol] = delta
    return trades


def submit_rebalance_trades(client: TradingClient, trades: Dict[str, float]) -> List[Any]:
    orders = []
    for symbol, delta_usd in trades.items():
        side = OrderSide.BUY if delta_usd > 0 else OrderSide.SELL
        order_request = MarketOrderRequest(
            symbol=symbol,
            notional=round(abs(delta_usd), 2),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        orders.append(client.submit_order(order_request))
    return orders
