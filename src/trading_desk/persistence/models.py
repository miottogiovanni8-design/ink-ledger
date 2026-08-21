from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ViewRecord(Base):
    """One asset's weekly analyst view, persisted for the investment
    committee notes — the LLM's only output in this pipeline.
    rebalance_event_id links it to the week it was made in, which is what
    lets skill_analysis pair a view against the asset's subsequently
    realized return (read from the *next* RebalanceEvent's prices)."""

    __tablename__ = "view_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    rebalance_event_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(16))
    asset_class: Mapped[str] = mapped_column(String(8))  # "equity" | "etf"
    expected_return_annualized: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text)
    key_signals: Mapped[str] = mapped_column(Text)  # JSON-encoded list[str]


class RebalanceEvent(Base):
    """One weekly rebalance run: the resulting weights (and expected
    return/volatility/Sharpe) for all three risk profiles, so the dashboard
    can offer a real client-side risk-profile toggle without a backend, plus
    which profile was actually executed and the prior/posterior return
    vectors for auditability."""

    __tablename__ = "rebalance_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    active_risk_profile: Mapped[str] = mapped_column(String(16))
    scenarios_json: Mapped[str] = mapped_column(Text)  # {"conservative": {...}, "balanced": {...}, "aggressive": {...}}
    prior_returns_json: Mapped[str] = mapped_column(Text)
    posterior_returns_json: Mapped[str] = mapped_column(Text)
    latest_prices_json: Mapped[str] = mapped_column(Text, default="{}")  # {"AAPL": 231.42, ...} — the closes used this run
    executed: Mapped[bool] = mapped_column(default=False)
    skip_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Transaction(Base):
    """One executed buy/sell leg of a rebalance — the source of truth for
    both the transaction history and each holding's cost basis (computed
    at read time from these rows, not stored redundantly). Price is the
    last known close at decision time, not a confirmed broker fill price —
    see execution/rebalance.py for why that's the documented approximation."""

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    rebalance_event_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(16))
    asset_class: Mapped[str] = mapped_column(String(8))
    side: Mapped[str] = mapped_column(String(4))  # "buy" | "sell"
    notional_usd: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text)


class EquitySnapshot(Base):
    """Daily mark-to-market equity point — feeds the equity curve and the
    dashboard's day/week history scrubber. benchmark_price is the S&P 500
    (SPY) close on the same day, stored raw (not indexed) so the snapshot
    builder can index both series to the same starting point at read time."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    equity_eur: Mapped[float] = mapped_column(Float)
    cash_eur: Mapped[float] = mapped_column(Float)
    benchmark_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
