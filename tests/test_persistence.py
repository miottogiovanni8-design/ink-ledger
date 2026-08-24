import json
import sqlite3
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


def test_get_session_backfills_columns_added_to_the_model_after_the_db_was_created():
    """Reproduces the exact production incident: a live db file created
    before rationale_it/sources_json existed on the model, still missing
    them, with a pre-existing row that must survive the migration."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE view_records (
                id INTEGER PRIMARY KEY,
                created_at DATETIME,
                rebalance_event_id INTEGER,
                symbol VARCHAR(16),
                asset_class VARCHAR(8),
                expected_return_annualized FLOAT,
                confidence FLOAT,
                rationale TEXT,
                key_signals TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO view_records (symbol, asset_class, expected_return_annualized, confidence, rationale, key_signals)"
            " VALUES ('AAPL', 'equity', 0.09, 0.55, 'pre-migration row', '[]')"
        )
        conn.commit()
        conn.close()

        with get_session(db_path) as session:
            stored = session.query(ViewRecord).filter_by(symbol="AAPL").one()
            assert stored.rationale == "pre-migration row"
            assert stored.rationale_it == ""
            assert stored.sources_json == "[]"

            session.add(
                ViewRecord(
                    symbol="NVDA",
                    asset_class="equity",
                    expected_return_annualized=0.12,
                    confidence=0.6,
                    rationale="AI capex still expanding.",
                    rationale_it="Il capex AI è ancora in espansione.",
                    key_signals=json.dumps([]),
                    sources_json=json.dumps([{"headline": "h", "url": "https://example.com"}]),
                )
            )

        with get_session(db_path) as session:
            rows = {r.symbol: r for r in session.query(ViewRecord).all()}
            assert rows["AAPL"].rationale_it == ""
            assert rows["NVDA"].rationale_it == "Il capex AI è ancora in espansione."
            assert json.loads(rows["NVDA"].sources_json)[0]["headline"] == "h"
