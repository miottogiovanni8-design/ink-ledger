import tempfile
from pathlib import Path

from trading_desk.persistence.db import get_session
from trading_desk.persistence.models import EquitySnapshot
from trading_desk.persistence.queries import peak_equity


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
