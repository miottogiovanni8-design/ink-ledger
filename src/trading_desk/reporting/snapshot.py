"""Builds the single JSON snapshot that is the join point between the
engine's SQLite state and everything that renders it (dashboard, email,
on-demand recap). Risk metrics per scenario (return/vol/Sharpe/VaR/CVaR/
exposure) are computed once at rebalance time and stored in
RebalanceEvent.scenarios_json — this module just reads and assembles,
it does not recompute anything from price data."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_desk.metrics import stats
from trading_desk.persistence.models import EquitySnapshot, RebalanceEvent, ViewRecord

SCHEMA_VERSION = 2


def _equity_curve(session: Session) -> List[Dict[str, Any]]:
    rows = session.execute(select(EquitySnapshot).order_by(EquitySnapshot.taken_at)).scalars().all()
    return [{"t": row.taken_at.isoformat(), "equity": row.equity_eur} for row in rows]


def _latest_rebalance(session: Session) -> Optional[RebalanceEvent]:
    return session.execute(
        select(RebalanceEvent).order_by(RebalanceEvent.created_at.desc()).limit(1)
    ).scalars().first()


def _investment_committee_notes(session: Session, since: Optional[datetime], limit: int = 50) -> List[Dict[str, Any]]:
    query = select(ViewRecord).order_by(ViewRecord.created_at.desc()).limit(limit)
    if since is not None:
        query = select(ViewRecord).where(ViewRecord.created_at >= since).order_by(ViewRecord.created_at.desc()).limit(limit)
    rows = session.execute(query).scalars().all()
    return [
        {
            "created_at": v.created_at.isoformat(),
            "symbol": v.symbol,
            "asset_class": v.asset_class,
            "expected_return_annualized": v.expected_return_annualized,
            "confidence": v.confidence,
            "rationale": v.rationale,
            "key_signals": json.loads(v.key_signals) if v.key_signals else [],
        }
        for v in rows
    ]


def build_snapshot(session: Session) -> Dict[str, Any]:
    equity_curve = _equity_curve(session)
    equity_values = [point["equity"] for point in equity_curve]
    returns = stats.returns_from_equity_curve(equity_values)

    rebalance = _latest_rebalance(session)
    scenarios = json.loads(rebalance.scenarios_json) if rebalance else {}
    notes = _investment_committee_notes(session, since=rebalance.created_at if rebalance else None)

    performance_stats: Dict[str, Any] = {
        "sharpe_ratio": stats.sharpe_ratio(returns),
        "sortino_ratio": stats.sortino_ratio(returns),
        "max_drawdown": stats.max_drawdown(equity_values),
        "total_return": (equity_values[-1] - equity_values[0]) / equity_values[0] if len(equity_values) >= 2 else 0.0,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "equity_curve": equity_curve,
        "performance_stats": performance_stats,
        "active_risk_profile": rebalance.active_risk_profile if rebalance else "balanced",
        "rebalance_generated_at": rebalance.created_at.isoformat() if rebalance else None,
        "scenarios": scenarios,
        "investment_committee_notes": notes,
    }


def write_snapshot(snapshot: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)


def read_snapshot(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)
