"""Reconciles our own trade ledger with what actually happened at the broker.

Bracket orders' stop-loss/take-profit legs fill asynchronously, between our
cron cycles — so every cycle starts by checking which DB-open trades are no
longer open positions at Alpaca, and closing them out with the fill that
caused it. Pure/testable: callers fetch the broker data, this module only
reasons about already-fetched objects.
"""

from datetime import datetime
from typing import Any, List, Optional, Set

from trading_desk.persistence.models import Trade


def find_closing_order(closed_orders: List[Any], symbol: str, opened_after: datetime) -> Optional[Any]:
    candidates = [
        order
        for order in closed_orders
        if order.symbol == symbol and getattr(order, "filled_at", None) and order.filled_at > opened_after
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda order: order.filled_at)


def compute_realized_pnl_eur(trade: Trade, exit_price: float) -> float:
    direction_sign = 1 if trade.direction == "long" else -1
    price_return = (exit_price - trade.entry_price) / trade.entry_price
    return trade.size_eur * price_return * direction_sign


def reconcile_open_trades(
    open_trades: List[Trade],
    broker_open_symbols: Set[str],
    closed_orders: List[Any],
) -> int:
    closed_count = 0
    for trade in open_trades:
        if trade.symbol in broker_open_symbols:
            continue
        closing_order = find_closing_order(closed_orders, trade.symbol, trade.opened_at)
        if closing_order is None:
            continue
        exit_price = float(closing_order.filled_avg_price)
        trade.status = "closed"
        trade.closed_at = closing_order.filled_at
        trade.exit_price = exit_price
        trade.realized_pnl_eur = compute_realized_pnl_eur(trade, exit_price)
        closed_count += 1
    return closed_count
