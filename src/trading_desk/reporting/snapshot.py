"""Builds the single JSON snapshot that is the join point between the trading
engine and the dashboard: written after every cycle, read by whoever renders
the dashboard (on-demand or weekly recap) or sends the email."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_desk.metrics import stats
from trading_desk.persistence.models import Decision, EquitySnapshot, Trade

SCHEMA_VERSION = 1


def _equity_curve(session: Session) -> List[Dict[str, Any]]:
    rows = session.execute(select(EquitySnapshot).order_by(EquitySnapshot.taken_at)).scalars().all()
    return [{"t": row.taken_at.isoformat(), "equity": row.equity_eur} for row in rows]


def _closed_trade_pnls(session: Session) -> List[float]:
    rows = (
        session.execute(select(Trade.realized_pnl_eur).where(Trade.status == "closed"))
        .scalars()
        .all()
    )
    return [pnl for pnl in rows if pnl is not None]


def _open_positions(session: Session) -> List[Dict[str, Any]]:
    rows = session.execute(select(Trade).where(Trade.status == "open")).scalars().all()
    return [
        {
            "symbol": t.symbol,
            "asset_class": t.asset_class,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "stop_loss_price": t.stop_loss_price,
            "take_profit_price": t.take_profit_price,
            "size_eur": t.size_eur,
            "opened_at": t.opened_at.isoformat(),
        }
        for t in rows
    ]


def _trade_journal(session: Session, limit: int = 50) -> List[Dict[str, Any]]:
    rows = (
        session.execute(select(Decision).order_by(Decision.created_at.desc()).limit(limit))
        .scalars()
        .all()
    )
    return [
        {
            "created_at": d.created_at.isoformat(),
            "symbol": d.symbol,
            "asset_class": d.asset_class,
            "direction": d.direction,
            "confidence": d.confidence,
            "rationale": d.rationale,
            "key_signals": json.loads(d.key_signals) if d.key_signals else [],
            "risk_flags": json.loads(d.risk_flags) if d.risk_flags else [],
            "skipped_by_risk_layer": d.skipped_by_risk_layer,
            "skip_reason": d.skip_reason,
        }
        for d in rows
    ]


def build_snapshot(session: Session, benchmark_returns: Optional[List[float]] = None) -> Dict[str, Any]:
    equity_curve = _equity_curve(session)
    equity_values = [point["equity"] for point in equity_curve]
    returns = stats.returns_from_equity_curve(equity_values)
    closed_pnls = _closed_trade_pnls(session)

    computed_stats: Dict[str, Any] = {
        "sharpe_ratio": stats.sharpe_ratio(returns),
        "sortino_ratio": stats.sortino_ratio(returns),
        "max_drawdown": stats.max_drawdown(equity_values),
        "win_rate": stats.win_rate(closed_pnls),
        "profit_factor": stats.profit_factor(closed_pnls),
        "closed_trades_count": len(closed_pnls),
        "total_realized_pnl_eur": sum(closed_pnls),
    }
    if benchmark_returns:
        alpha, beta = stats.alpha_beta(returns, benchmark_returns)
        computed_stats["alpha_annualized"] = alpha
        computed_stats["beta"] = beta

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "equity_curve": equity_curve,
        "positions": _open_positions(session),
        "trade_journal": _trade_journal(session),
        "stats": computed_stats,
    }


def write_snapshot(snapshot: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)


def read_snapshot(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)
