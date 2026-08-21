"""Assembles the real inputs for skill_analysis and attribution from stored
state — pairing each view with what actually happened next, and turning
holding-level P&L into sector-level returns. The pure math lives in
engine/skill_analysis.py and engine/attribution.py; this module only reads
and joins."""

import json
from typing import Any, Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_desk.engine.skill_analysis import directional_hit
from trading_desk.persistence.models import RebalanceEvent, ViewRecord


def views_with_realized_outcomes(session: Session) -> List[Dict[str, Any]]:
    """Pairs each view against the price move of its symbol between the
    week it was made and the following week's rebalance — the closest
    thing to "what actually happened after this call" the stored data
    supports. Views from the most recent rebalance have no "next" prices
    yet and are excluded."""
    events = session.execute(select(RebalanceEvent).order_by(RebalanceEvent.created_at)).scalars().all()
    if len(events) < 2:
        return []

    prices_by_event: Dict[int, Dict[str, float]] = {
        e.id: (json.loads(e.latest_prices_json) if e.latest_prices_json else {}) for e in events
    }
    next_event_id: Dict[int, int] = {events[i].id: events[i + 1].id for i in range(len(events) - 1)}

    views = session.execute(
        select(ViewRecord).where(ViewRecord.rebalance_event_id.isnot(None))
    ).scalars().all()

    results = []
    for v in views:
        next_id = next_event_id.get(v.rebalance_event_id)
        if next_id is None:
            continue
        entry_price = prices_by_event.get(v.rebalance_event_id, {}).get(v.symbol)
        exit_price = prices_by_event.get(next_id, {}).get(v.symbol)
        if not entry_price or not exit_price:
            continue
        realized_return = (exit_price - entry_price) / entry_price
        results.append(
            {
                "symbol": v.symbol,
                "expected_return_annualized": v.expected_return_annualized,
                "confidence": v.confidence,
                "realized_return": realized_return,
                "correct": directional_hit(v.expected_return_annualized, realized_return),
            }
        )
    return results


def sector_benchmark_returns(session: Session, sector_etf_map: Dict[str, str]) -> Dict[str, float]:
    """Sector index return proxy, from the earliest to the most recent
    rebalance's tracked closing prices for that sector's SPDR ETF."""
    events = session.execute(select(RebalanceEvent).order_by(RebalanceEvent.created_at)).scalars().all()
    if len(events) < 2:
        return {}
    first_prices = json.loads(events[0].latest_prices_json) if events[0].latest_prices_json else {}
    last_prices = json.loads(events[-1].latest_prices_json) if events[-1].latest_prices_json else {}

    returns = {}
    for sector, etf in sector_etf_map.items():
        p0, p1 = first_prices.get(etf), last_prices.get(etf)
        if p0 and p1:
            returns[sector] = (p1 - p0) / p0
    return returns


def portfolio_sector_returns(holdings_detail: List[Dict[str, Any]], sector_map: Dict[str, str]) -> Dict[str, float]:
    """Weighted-average realized return (since each position's own entry)
    per sector, weighted by each holding's portfolio weight."""
    weighted_sum: Dict[str, float] = {}
    weight_total: Dict[str, float] = {}
    for h in holdings_detail:
        if h.get("pct_since_purchase") is None:
            continue
        sector = sector_map.get(h["symbol"], "Other")
        weighted_sum[sector] = weighted_sum.get(sector, 0.0) + h["weight"] * h["pct_since_purchase"]
        weight_total[sector] = weight_total.get(sector, 0.0) + h["weight"]
    return {s: weighted_sum[s] / weight_total[s] for s in weighted_sum if weight_total[s] > 0}
