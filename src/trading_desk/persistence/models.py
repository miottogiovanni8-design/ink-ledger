from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Decision(Base):
    """Every decision the LLM (or the risk layer, if it skipped the LLM) produced."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    symbol: Mapped[str] = mapped_column(String(16))
    asset_class: Mapped[str] = mapped_column(String(16))  # "equity" | "crypto"
    direction: Mapped[str] = mapped_column(String(8))  # "long" | "short" | "hold"
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text)
    key_signals: Mapped[str] = mapped_column(Text)  # JSON-encoded list[str]
    risk_flags: Mapped[str] = mapped_column(Text)  # JSON-encoded list[str]
    skipped_by_risk_layer: Mapped[bool] = mapped_column(default=False)
    skip_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    trade: Mapped[Optional["Trade"]] = relationship(back_populates="decision", uselist=False)


class Trade(Base):
    """A single executed (paper) position, from entry to exit."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_id: Mapped[Optional[int]] = mapped_column(ForeignKey("decisions.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(16))
    asset_class: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss_price: Mapped[float] = mapped_column(Float)
    take_profit_price: Mapped[float] = mapped_column(Float)
    size_eur: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_pnl_eur: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open")  # "open" | "closed"
    alpaca_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    decision: Mapped[Optional["Decision"]] = relationship(back_populates="trade")


class EquitySnapshot(Base):
    """Point-in-time total account equity, sampled once per cycle — feeds the equity curve."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    equity_eur: Mapped[float] = mapped_column(Float)
    cash_eur: Mapped[float] = mapped_column(Float)
    open_positions_count: Mapped[int] = mapped_column(Integer, default=0)
