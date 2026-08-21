"""Alpaca account/position reads shared by the daily mark and weekly
rebalance jobs. Order submission itself lives in execution/rebalance.py —
v2 moves target weights to the broker via plain notional market orders, not
per-position bracket orders (risk is managed at the portfolio level now,
not with a stop-loss on every single name)."""

from typing import Any, List

from alpaca.trading.client import TradingClient


def get_account_equity(client: TradingClient) -> float:
    account = client.get_account()
    return float(account.equity)


def get_account_cash(client: TradingClient) -> float:
    account = client.get_account()
    return float(account.cash)


def get_daily_pnl(client: TradingClient) -> float:
    """Equity change since the prior trading day's close (Alpaca tracks this as `last_equity`)."""
    account = client.get_account()
    return float(account.equity) - float(account.last_equity)


def get_open_positions(client: TradingClient) -> List[Any]:
    return client.get_all_positions()
