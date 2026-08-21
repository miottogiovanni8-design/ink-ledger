import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_desk.persistence.db import get_session
from trading_desk.persistence.models import EquitySnapshot, RebalanceEvent, ViewRecord
from trading_desk.reporting.snapshot import build_snapshot, read_snapshot, write_snapshot

REBALANCE_TIME = datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc)

SCENARIOS = {
    "conservative": {"weights": {"AAPL": 0.10}, "expected_return": 0.06, "volatility": 0.08, "sharpe": 0.75, "var_95": 0.01, "cvar_95": 0.015, "sector_exposure": {"Technology": 0.10}, "factor_exposure": {}},
    "balanced": {"weights": {"AAPL": 0.15}, "expected_return": 0.09, "volatility": 0.14, "sharpe": 0.64, "var_95": 0.02, "cvar_95": 0.03, "sector_exposure": {"Technology": 0.15}, "factor_exposure": {}},
    "aggressive": {"weights": {"AAPL": 0.20}, "expected_return": 0.12, "volatility": 0.22, "sharpe": 0.55, "var_95": 0.035, "cvar_95": 0.05, "sector_exposure": {"Technology": 0.20}, "factor_exposure": {}},
}


def _seed(session):
    for i, equity in enumerate([1000.0, 1010.0, 990.0, 1025.0]):
        session.add(EquitySnapshot(equity_eur=equity, cash_eur=equity * 0.5, taken_at=REBALANCE_TIME - timedelta(days=3 - i)))

    session.add(
        RebalanceEvent(
            created_at=REBALANCE_TIME,
            active_risk_profile="balanced",
            scenarios_json=json.dumps(SCENARIOS),
            prior_returns_json=json.dumps({"AAPL": 0.07}),
            posterior_returns_json=json.dumps({"AAPL": 0.09}),
            executed=True,
        )
    )

    session.add(
        ViewRecord(
            created_at=REBALANCE_TIME + timedelta(minutes=5),
            symbol="AAPL",
            asset_class="equity",
            expected_return_annualized=0.09,
            confidence=0.55,
            rationale="Strong iPhone cycle plus services growth.",
            key_signals=json.dumps(["services revenue +15% YoY"]),
        )
    )
    session.add(
        ViewRecord(
            created_at=REBALANCE_TIME - timedelta(days=10),
            symbol="AAPL",
            asset_class="equity",
            expected_return_annualized=0.03,
            confidence=0.4,
            rationale="Old view from a prior rebalance, should not appear.",
            key_signals=json.dumps([]),
        )
    )


def test_build_snapshot_shape_and_content():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            _seed(session)

        with get_session(db_path) as session:
            snapshot = build_snapshot(session)

    assert snapshot["schema_version"] == 2
    assert len(snapshot["equity_curve"]) == 4
    assert snapshot["active_risk_profile"] == "balanced"
    assert snapshot["scenarios"]["aggressive"]["volatility"] == 0.22
    assert snapshot["performance_stats"]["total_return"] > 0

    notes = snapshot["investment_committee_notes"]
    assert len(notes) == 1
    assert notes[0]["rationale"].startswith("Strong iPhone cycle")


def test_build_snapshot_with_no_rebalance_yet():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            session.add(EquitySnapshot(equity_eur=1000.0, cash_eur=1000.0))

        with get_session(db_path) as session:
            snapshot = build_snapshot(session)

    assert snapshot["scenarios"] == {}
    assert snapshot["active_risk_profile"] == "balanced"
    assert snapshot["investment_committee_notes"] == []


def test_write_and_read_snapshot_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "snapshot.json")
        snapshot = {"schema_version": 2, "performance_stats": {"sharpe_ratio": 1.23}}
        write_snapshot(snapshot, path)
        assert read_snapshot(path) == snapshot
