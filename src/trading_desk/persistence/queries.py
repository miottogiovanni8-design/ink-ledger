"""Small read helpers: the equity peak for the drawdown circuit breaker, and
month-to-date estimated Claude API spend for the spend-alert threshold."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trading_desk.persistence.models import ApiSpendLog, EquitySnapshot


def peak_equity(session: Session, fallback: float) -> float:
    result = session.execute(select(func.max(EquitySnapshot.equity_eur))).scalar_one_or_none()
    return result if result is not None else fallback


def month_to_date_spend_usd(session: Session, as_of: Optional[datetime] = None) -> float:
    as_of = as_of or datetime.now(timezone.utc)
    month_start = as_of.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    result = session.execute(
        select(func.sum(ApiSpendLog.estimated_cost_usd)).where(ApiSpendLog.created_at >= month_start)
    ).scalar_one_or_none()
    return result if result is not None else 0.0
