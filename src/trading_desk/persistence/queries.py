"""Small read helpers that turn raw DB rows into the portfolio-state numbers
the risk layer and decision engine need. Kept separate from models.py so the
query shape can evolve without touching the schema."""

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trading_desk.persistence.models import EquitySnapshot, Trade


def open_positions_count(session: Session) -> int:
    return session.execute(
        select(func.count()).select_from(Trade).where(Trade.status == "open")
    ).scalar_one()


def peak_equity(session: Session, fallback: float) -> float:
    result = session.execute(select(func.max(EquitySnapshot.equity_eur))).scalar_one_or_none()
    return result if result is not None else fallback


def todays_opened_notional_eur(session: Session) -> float:
    """Sum of size_eur for trades opened today (UTC) — this is what 'spent so
    far today' means for the daily budget, independent of P&L."""
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    result = session.execute(
        select(func.coalesce(func.sum(Trade.size_eur), 0.0)).where(Trade.opened_at >= today_start)
    ).scalar_one()
    return float(result)
