import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_desk.persistence.db import get_session
from trading_desk.persistence.models import EquitySnapshot, Trade
from trading_desk.persistence.queries import open_positions_count, peak_equity, todays_opened_notional_eur


def test_open_positions_count():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            session.add(Trade(symbol="AAPL", asset_class="equity", direction="long", entry_price=100, stop_loss_price=95, take_profit_price=110, size_eur=10, status="open"))
            session.add(Trade(symbol="MSFT", asset_class="equity", direction="short", entry_price=100, stop_loss_price=105, take_profit_price=90, size_eur=10, status="closed"))

        with get_session(db_path) as session:
            assert open_positions_count(session) == 1


def test_peak_equity_uses_max_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            for equity in [1000.0, 1100.0, 950.0]:
                session.add(EquitySnapshot(equity_eur=equity, cash_eur=equity, open_positions_count=0))

        with get_session(db_path) as session:
            assert peak_equity(session, fallback=0.0) == 1100.0


def test_peak_equity_falls_back_when_empty():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            assert peak_equity(session, fallback=500.0) == 500.0


def test_todays_opened_notional_only_counts_todays_trades():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            session.add(
                Trade(
                    symbol="AAPL",
                    asset_class="equity",
                    direction="long",
                    entry_price=100,
                    stop_loss_price=95,
                    take_profit_price=110,
                    size_eur=15.0,
                    status="open",
                    opened_at=datetime.now(timezone.utc),
                )
            )
            session.add(
                Trade(
                    symbol="MSFT",
                    asset_class="equity",
                    direction="short",
                    entry_price=100,
                    stop_loss_price=105,
                    take_profit_price=90,
                    size_eur=20.0,
                    status="closed",
                    opened_at=datetime.now(timezone.utc) - timedelta(days=2),
                )
            )

        with get_session(db_path) as session:
            assert todays_opened_notional_eur(session) == 15.0
