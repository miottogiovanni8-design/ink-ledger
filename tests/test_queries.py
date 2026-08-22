import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_desk.persistence.db import get_session
from trading_desk.persistence.models import ApiSpendLog, EquitySnapshot
from trading_desk.persistence.queries import month_to_date_spend_usd, peak_equity


def test_peak_equity_uses_max_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            for equity in [1000.0, 1100.0, 950.0]:
                session.add(EquitySnapshot(equity_eur=equity, cash_eur=equity))

        with get_session(db_path) as session:
            assert peak_equity(session, fallback=0.0) == 1100.0


def test_peak_equity_falls_back_when_empty():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            assert peak_equity(session, fallback=500.0) == 500.0


def test_month_to_date_spend_sums_only_current_month():
    as_of = datetime(2026, 8, 22, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            session.add(ApiSpendLog(estimated_cost_usd=0.10, created_at=as_of - timedelta(days=1)))
            session.add(ApiSpendLog(estimated_cost_usd=0.05, created_at=as_of))
            session.add(ApiSpendLog(estimated_cost_usd=0.30, created_at=as_of - timedelta(days=25)))  # last month

        with get_session(db_path) as session:
            total = month_to_date_spend_usd(session, as_of=as_of)

    assert total == pytest.approx(0.15)  # excludes the last-month row


def test_month_to_date_spend_zero_when_empty():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            assert month_to_date_spend_usd(session) == 0.0
