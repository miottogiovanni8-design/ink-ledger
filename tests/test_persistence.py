import json
import tempfile
from pathlib import Path

from trading_desk.persistence.db import get_session
from trading_desk.persistence.models import EquitySnapshot, RebalanceEvent, Transaction, ViewRecord


def test_roundtrip_view_record():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")

        with get_session(db_path) as session:
            session.add(
                ViewRecord(
                    symbol="AAPL",
                    asset_class="equity",
                    expected_return_annualized=0.09,
                    confidence=0.55,
                    rationale="Strong iPhone cycle plus services growth.",
                    key_signals=json.dumps(["services revenue +15% YoY"]),
                )
            )

        with get_session(db_path) as session:
            stored = session.query(ViewRecord).filter_by(symbol="AAPL").one()
            assert stored.rationale.startswith("Strong iPhone cycle")
            assert json.loads(stored.key_signals) == ["services revenue +15% YoY"]


def test_roundtrip_rebalance_event():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        scenarios = {
            "conservative": {"weights": {"AAPL": 0.1}, "expected_return": 0.06, "volatility": 0.08, "sharpe": 0.75},
            "balanced": {"weights": {"AAPL": 0.15}, "expected_return": 0.09, "volatility": 0.14, "sharpe": 0.64},
            "aggressive": {"weights": {"AAPL": 0.2}, "expected_return": 0.12, "volatility": 0.22, "sharpe": 0.55},
        }
        with get_session(db_path) as session:
            session.add(
                RebalanceEvent(
                    active_risk_profile="balanced",
                    scenarios_json=json.dumps(scenarios),
                    prior_returns_json=json.dumps({"AAPL": 0.07}),
                    posterior_returns_json=json.dumps({"AAPL": 0.09}),
                    executed=True,
                )
            )

        with get_session(db_path) as session:
            stored = session.query(RebalanceEvent).one()
            assert stored.active_risk_profile == "balanced"
            assert json.loads(stored.scenarios_json)["balanced"]["expected_return"] == 0.09


def test_roundtrip_transaction():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            session.add(
                Transaction(
                    symbol="AAPL",
                    asset_class="equity",
                    side="buy",
                    notional_usd=150.0,
                    price=200.0,
                    rationale="Strong iPhone cycle plus services growth.",
                )
            )

        with get_session(db_path) as session:
            stored = session.query(Transaction).one()
            assert stored.symbol == "AAPL"
            assert stored.side == "buy"
            assert stored.notional_usd == 150.0
            assert stored.rationale.startswith("Strong iPhone cycle")


def test_equity_snapshot_persists():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            session.add(EquitySnapshot(equity_eur=1000.0, cash_eur=1000.0))

        with get_session(db_path) as session:
            rows = session.query(EquitySnapshot).all()
            assert len(rows) == 1
            assert rows[0].equity_eur == 1000.0
