"""Turns the immutable transaction log into a weighted-average cost basis
per symbol — the standard institutional method (not FIFO/LIFO lot
selection, which would need per-lot tracking this notional-based ledger
doesn't have). Positions are derived at read time from transactions rather
than stored as mutable state, so the transaction log stays the single
source of truth."""

from typing import Any, Dict, List, Optional


def compute_cost_basis(transactions: List[Dict[str, Any]]) -> Optional[float]:
    """transactions: chronological list of {side, notional_usd, price} for
    one symbol. Returns the weighted-average cost per share of the
    currently open position, or None if nothing is currently held."""
    shares = 0.0
    cost = 0.0
    for tx in transactions:
        if tx["price"] <= 0:
            continue
        tx_shares = tx["notional_usd"] / tx["price"]
        if tx["side"] == "buy":
            shares += tx_shares
            cost += tx["notional_usd"]
        else:
            if shares <= 0:
                continue
            avg_cost = cost / shares
            sold_shares = min(tx_shares, shares)
            shares -= sold_shares
            cost -= avg_cost * sold_shares

    if shares <= 1e-9:
        return None
    return cost / shares


def build_holdings_detail(
    weights: Dict[str, float],
    total_equity: float,
    latest_prices: Dict[str, float],
    name_map: Dict[str, str],
    transactions_by_symbol: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    holdings = []
    for symbol, weight in weights.items():
        price = latest_prices.get(symbol)
        cost_basis = compute_cost_basis(transactions_by_symbol.get(symbol, []))
        pct_since_purchase = ((price - cost_basis) / cost_basis) if (price and cost_basis) else None
        holdings.append(
            {
                "symbol": symbol,
                "name": name_map.get(symbol, symbol),
                "weight": weight,
                "market_value_usd": weight * total_equity,
                "current_price": price,
                "cost_basis": cost_basis,
                "pct_since_purchase": pct_since_purchase,
            }
        )
    return holdings
