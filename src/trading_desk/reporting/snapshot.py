"""Builds the single JSON snapshot that is the join point between the
engine's SQLite state and everything that renders it (dashboard, email,
on-demand recap). Risk metrics per scenario (return/vol/Sharpe/VaR/CVaR/
exposure) are computed once at rebalance time and stored in
RebalanceEvent.scenarios_json — this module just reads and assembles,
it does not recompute anything from price data. The one exception is the
equity-vs-benchmark comparison, which only needs the daily marks already
stored (equity + raw SPY close) — no price-panel refetch required."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_desk.metrics import stats
from trading_desk.persistence.models import EquitySnapshot, RebalanceEvent, ViewRecord

SCHEMA_VERSION = 2


def _daily_marks(session: Session) -> List[EquitySnapshot]:
    return session.execute(select(EquitySnapshot).order_by(EquitySnapshot.taken_at)).scalars().all()


def _benchmark_curve(marks: List[EquitySnapshot]) -> List[Dict[str, Any]]:
    """Indexes SPY to the same starting equity as the portfolio, so the two
    lines are directly comparable in euro terms on the same chart — not a
    dual-axis trick, both series share one scale."""
    priced = [m for m in marks if m.benchmark_price is not None]
    if not priced:
        return []
    base_equity = marks[0].equity_eur
    base_price = priced[0].benchmark_price
    return [
        {"t": m.taken_at.isoformat(), "equity": base_equity * (m.benchmark_price / base_price)}
        for m in priced
    ]


def _latest_rebalance(session: Session) -> Optional[RebalanceEvent]:
    return session.execute(
        select(RebalanceEvent).order_by(RebalanceEvent.created_at.desc()).limit(1)
    ).scalars().first()


def _risk_profile_history(session: Session, limit: int = 26) -> List[Dict[str, Any]]:
    rows = session.execute(
        select(RebalanceEvent).order_by(RebalanceEvent.created_at.desc()).limit(limit)
    ).scalars().all()
    return [
        {
            "date": r.created_at.isoformat(),
            "active_risk_profile": r.active_risk_profile,
            "executed": r.executed,
            "skip_reason": r.skip_reason,
        }
        for r in reversed(rows)
    ]


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
    marks = _daily_marks(session)
    equity_curve = [{"t": m.taken_at.isoformat(), "equity": m.equity_eur} for m in marks]
    benchmark_curve = _benchmark_curve(marks)

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

    if len(benchmark_curve) >= 3:
        benchmark_values = [point["equity"] for point in benchmark_curve]
        benchmark_returns = stats.returns_from_equity_curve(benchmark_values)
        alpha, beta = stats.alpha_beta(returns, benchmark_returns)
        performance_stats["alpha_annualized"] = alpha
        performance_stats["beta"] = beta
        performance_stats["benchmark_total_return"] = (
            (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "performance_stats": performance_stats,
        "active_risk_profile": rebalance.active_risk_profile if rebalance else "balanced",
        "rebalance_generated_at": rebalance.created_at.isoformat() if rebalance else None,
        "risk_profile_history": _risk_profile_history(session),
        "scenarios": scenarios,
        "investment_committee_notes": notes,
    }


def write_snapshot(snapshot: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)


def read_snapshot(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)
