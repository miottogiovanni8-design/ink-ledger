import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_desk.persistence.db import get_session
from trading_desk.persistence.models import EquitySnapshot, RebalanceEvent, Transaction, ViewRecord
from trading_desk.reporting.snapshot import build_snapshot, read_snapshot, write_snapshot

REBALANCE_TIME = datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc)

SCENARIOS = {
    "conservative": {"weights": {"AAPL": 0.10}, "expected_return": 0.06, "volatility": 0.08, "sharpe": 0.75, "var_95": 0.01, "cvar_95": 0.015, "sector_exposure": {"Technology": 0.10}, "factor_exposure": {}},
    "balanced": {"weights": {"AAPL": 0.15}, "expected_return": 0.09, "volatility": 0.14, "sharpe": 0.64, "var_95": 0.02, "cvar_95": 0.03, "sector_exposure": {"Technology": 0.15}, "factor_exposure": {}},
    "aggressive": {"weights": {"AAPL": 0.20}, "expected_return": 0.12, "volatility": 0.22, "sharpe": 0.55, "var_95": 0.035, "cvar_95": 0.05, "sector_exposure": {"Technology": 0.20}, "factor_exposure": {}},
}


def _seed(session, include_older_rebalance=False):
    equities = [1000.0, 1010.0, 990.0, 1025.0]
    benchmarks = [500.0, 502.0, 498.0, 505.0]
    for i, (equity, bench) in enumerate(zip(equities, benchmarks)):
        session.add(
            EquitySnapshot(
                equity_eur=equity,
                cash_eur=equity * 0.5,
                benchmark_price=bench,
                taken_at=REBALANCE_TIME - timedelta(days=3 - i),
            )
        )

    if include_older_rebalance:
        session.add(
            RebalanceEvent(
                created_at=REBALANCE_TIME - timedelta(days=7),
                active_risk_profile="conservative",
                scenarios_json=json.dumps(SCENARIOS),
                prior_returns_json=json.dumps({"AAPL": 0.05}),
                posterior_returns_json=json.dumps({"AAPL": 0.06}),
                executed=True,
            )
        )

    session.add(
        RebalanceEvent(
            created_at=REBALANCE_TIME,
            active_risk_profile="balanced",
            scenarios_json=json.dumps(SCENARIOS),
            prior_returns_json=json.dumps({"AAPL": 0.07}),
            posterior_returns_json=json.dumps({"AAPL": 0.09}),
            latest_prices_json=json.dumps({"AAPL": 220.0}),
            executed=True,
        )
    )

    session.add(
        Transaction(
            executed_at=REBALANCE_TIME,
            symbol="AAPL",
            asset_class="equity",
            side="buy",
            notional_usd=150.0,
            price=200.0,
            rationale="Strong iPhone cycle plus services growth.",
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

    assert snapshot["schema_version"] == 6
    assert snapshot["is_sample_data"] is False
    assert len(snapshot["equity_curve"]) == 4
    assert snapshot["active_risk_profile"] == "balanced"
    assert snapshot["scenarios"]["aggressive"]["volatility"] == 0.22
    assert snapshot["performance_stats"]["total_return"] > 0

    notes = snapshot["investment_committee_notes"]
    assert len(notes) == 1
    assert notes[0]["rationale"].startswith("Strong iPhone cycle")


def test_build_snapshot_includes_benchmark_curve_and_alpha_beta():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            _seed(session)

        with get_session(db_path) as session:
            snapshot = build_snapshot(session)

    benchmark_curve = snapshot["benchmark_curve"]
    assert len(benchmark_curve) == 4
    # indexed to the portfolio's starting equity (1000.0), not the raw SPY price (500.0)
    assert benchmark_curve[0]["equity"] == 1000.0

    assert "alpha_annualized" in snapshot["performance_stats"]
    assert "beta" in snapshot["performance_stats"]
    assert "benchmark_total_return" in snapshot["performance_stats"]


def test_build_snapshot_baseline_curve_empty_without_frozen_allocation():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            _seed(session)

        with get_session(db_path) as session:
            snapshot = build_snapshot(session)

    assert snapshot["baseline_curve"] == []
    assert "baseline_total_return" not in snapshot["performance_stats"]
    assert "active_management_effect" not in snapshot["performance_stats"]


def test_build_snapshot_includes_baseline_curve_and_active_management_effect():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            equities = [1000.0, 1010.0, 990.0, 1025.0]
            baselines = [200.0, 202.0, 198.0, 201.0]  # a flat buy-and-hold basket, +0.5% total
            for i, (equity, base) in enumerate(zip(equities, baselines)):
                session.add(
                    EquitySnapshot(
                        equity_eur=equity,
                        cash_eur=equity * 0.5,
                        baseline_index_raw=base,
                        taken_at=REBALANCE_TIME - timedelta(days=3 - i),
                    )
                )

        with get_session(db_path) as session:
            snapshot = build_snapshot(session)

    baseline_curve = snapshot["baseline_curve"]
    assert len(baseline_curve) == 4
    # indexed to the portfolio's starting equity (1000.0), not the raw basket index (200.0)
    assert baseline_curve[0]["equity"] == 1000.0
    assert baseline_curve[-1]["equity"] == pytest.approx(1000.0 * (201.0 / 200.0))

    stats = snapshot["performance_stats"]
    assert stats["baseline_total_return"] == pytest.approx((201.0 - 200.0) / 200.0)
    assert stats["active_management_effect"] == pytest.approx(stats["total_return"] - stats["baseline_total_return"])


def test_build_snapshot_risk_profile_history_oldest_first():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            _seed(session, include_older_rebalance=True)

        with get_session(db_path) as session:
            snapshot = build_snapshot(session)

    history = snapshot["risk_profile_history"]
    assert len(history) == 2
    assert history[0]["active_risk_profile"] == "conservative"
    assert history[1]["active_risk_profile"] == "balanced"


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
    assert snapshot["benchmark_curve"] == []
    assert "alpha_annualized" not in snapshot["performance_stats"]


def test_build_snapshot_holdings_detail_and_transaction_history():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            _seed(session)

        with get_session(db_path) as session:
            snapshot = build_snapshot(session)

    holdings = snapshot["holdings_detail"]
    assert len(holdings) == 1
    h = holdings[0]
    assert h["symbol"] == "AAPL"
    assert h["name"] == "Apple Inc."
    assert h["current_price"] == 220.0
    assert h["cost_basis"] == pytest.approx(200.0)
    assert h["pct_since_purchase"] == pytest.approx(0.10)

    assert snapshot["latest_prices"] == {"AAPL": 220.0}

    history = snapshot["transaction_history"]
    assert len(history) == 1
    assert history[0]["symbol"] == "AAPL"
    assert history[0]["side"] == "buy"
    assert history[0]["rationale"].startswith("Strong iPhone cycle")


def test_build_snapshot_skill_analysis_and_attribution_with_two_rebalances():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        week1 = REBALANCE_TIME - timedelta(weeks=1)

        with get_session(db_path) as session:
            session.add(EquitySnapshot(equity_eur=1000.0, cash_eur=500.0, taken_at=week1))
            session.add(EquitySnapshot(equity_eur=1020.0, cash_eur=500.0, taken_at=REBALANCE_TIME))

            event1 = RebalanceEvent(
                created_at=week1, active_risk_profile="balanced",
                scenarios_json=json.dumps(SCENARIOS), prior_returns_json="{}", posterior_returns_json="{}",
                latest_prices_json=json.dumps({"AAPL": 200.0, "XLK": 100.0}), executed=True,
            )
            session.add(event1)
            session.flush()
            session.add(
                ViewRecord(
                    created_at=week1, rebalance_event_id=event1.id, symbol="AAPL", asset_class="equity",
                    expected_return_annualized=0.08, confidence=0.6, rationale="Bullish.", key_signals="[]",
                )
            )
            session.add(
                Transaction(
                    executed_at=week1, symbol="AAPL", asset_class="equity", side="buy",
                    notional_usd=100.0, price=200.0, rationale="Initial build.",
                )
            )

            event2 = RebalanceEvent(
                created_at=REBALANCE_TIME, active_risk_profile="balanced",
                scenarios_json=json.dumps(SCENARIOS), prior_returns_json="{}", posterior_returns_json="{}",
                latest_prices_json=json.dumps({"AAPL": 220.0, "XLK": 105.0}), executed=True,
            )
            session.add(event2)

        with get_session(db_path) as session:
            snapshot = build_snapshot(session)

    skill = snapshot["skill_analysis"]
    assert skill["sample_size"] == 1
    assert skill["information_coefficient"] is None  # a single point has no variance to correlate
    total_bucketed = sum(b["count"] for b in skill["calibration_buckets"])
    assert total_bucketed == 1  # the 0.6-confidence view lands in exactly one bucket

    attribution = snapshot["attribution"]
    assert attribution["by_sector"]  # populated since XLK gives us a Technology benchmark return
    tech = next(s for s in attribution["by_sector"] if s["sector"] == "Technology")
    assert tech["portfolio_weight"] == pytest.approx(0.15)


def test_write_and_read_snapshot_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "snapshot.json")
        snapshot = {"schema_version": 2, "performance_stats": {"sharpe_ratio": 1.23}}
        write_snapshot(snapshot, path)
        assert read_snapshot(path) == snapshot
