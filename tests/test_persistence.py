import json
import tempfile
from pathlib import Path

from trading_desk.persistence.db import get_session
from trading_desk.persistence.models import Decision, EquitySnapshot, Trade


def test_roundtrip_decision_and_trade():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")

        with get_session(db_path) as session:
            decision = Decision(
                symbol="AAPL",
                asset_class="equity",
                direction="long",
                confidence=0.72,
                rationale="RSI oversold + positive earnings headline",
                key_signals=json.dumps(["RSI 28", "positive headline sentiment"]),
                risk_flags=json.dumps([]),
            )
            session.add(decision)
            session.flush()

            trade = Trade(
                decision_id=decision.id,
                symbol="AAPL",
                asset_class="equity",
                direction="long",
                entry_price=190.0,
                stop_loss_price=185.0,
                take_profit_price=198.75,
                size_eur=15.0,
            )
            session.add(trade)

        with get_session(db_path) as session:
            stored = session.query(Trade).filter_by(symbol="AAPL").one()
            assert stored.decision.rationale.startswith("RSI oversold")
            assert stored.status == "open"


def test_equity_snapshot_persists():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            session.add(EquitySnapshot(equity_eur=1000.0, cash_eur=1000.0, open_positions_count=0))

        with get_session(db_path) as session:
            rows = session.query(EquitySnapshot).all()
            assert len(rows) == 1
            assert rows[0].equity_eur == 1000.0
