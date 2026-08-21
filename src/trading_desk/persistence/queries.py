"""Small read helper the risk layer needs: the equity peak, for the drawdown
circuit breaker."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trading_desk.persistence.models import EquitySnapshot


def peak_equity(session: Session, fallback: float) -> float:
    result = session.execute(select(func.max(EquitySnapshot.equity_eur))).scalar_one_or_none()
    return result if result is not None else fallback
