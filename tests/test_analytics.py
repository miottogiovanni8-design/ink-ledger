import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_desk.persistence.db import get_session
from trading_desk.persistence.models import RebalanceEvent, ViewRecord
from trading_desk.reporting.analytics import (
    portfolio_sector_returns,
    sector_benchmark_returns,
    views_with_realized_outcomes,
)

T0 = datetime(2026, 4, 6, 9, 0, tzinfo=timezone.utc)


def _seed_two_weeks(session):
    event1 = RebalanceEvent(
        created_at=T0,
        active_risk_profile="balanced",
        scenarios_json="{}",
        prior_returns_json="{}",
        posterior_returns_json="{}",
        latest_prices_json=json.dumps({"AAPL": 100.0, "XLK": 200.0}),
        executed=True,
    )
    event2 = RebalanceEvent(
        created_at=T0 + timedelta(weeks=1),
        active_risk_profile="balanced",
        scenarios_json="{}",
        prior_returns_json="{}",
        posterior_returns_json="{}",
        latest_prices_json=json.dumps({"AAPL": 110.0, "XLK": 190.0}),
        executed=True,
    )
    session.add(event1)
    session.add(event2)
    session.flush()

    view1 = ViewRecord(
        created_at=T0,
        rebalance_event_id=event1.id,
        symbol="AAPL",
        asset_class="equity",
        expected_return_annualized=0.09,
        confidence=0.6,
        rationale="Bullish view.",
        key_signals="[]",
    )
    view2 = ViewRecord(
        created_at=T0 + timedelta(weeks=1),
        rebalance_event_id=event2.id,
        symbol="AAPL",
        asset_class="equity",
        expected_return_annualized=0.05,
        confidence=0.5,
        rationale="No next data yet.",
        key_signals="[]",
    )
    session.add(view1)
    session.add(view2)
    return event1, event2


def test_views_with_realized_outcomes_computes_return_and_hit():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            _seed_two_weeks(session)

        with get_session(db_path) as session:
            results = views_with_realized_outcomes(session)

    # only view1 has a "next" week to compute a realized return from
    assert len(results) == 1
    r = results[0]
    assert r["symbol"] == "AAPL"
    assert r["realized_return"] == pytest.approx(0.10)  # (110-100)/100
    assert r["correct"] is True  # positive view, positive realized return


def test_views_with_realized_outcomes_empty_with_single_rebalance():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            event = RebalanceEvent(
                active_risk_profile="balanced", scenarios_json="{}", prior_returns_json="{}",
                posterior_returns_json="{}", latest_prices_json=json.dumps({"AAPL": 100.0}), executed=True,
            )
            session.add(event)
            session.flush()
            session.add(
                ViewRecord(
                    rebalance_event_id=event.id, symbol="AAPL", asset_class="equity",
                    expected_return_annualized=0.05, confidence=0.5, rationale="x", key_signals="[]",
                )
            )

        with get_session(db_path) as session:
            assert views_with_realized_outcomes(session) == []


def test_sector_benchmark_returns_from_first_and_last_event():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            _seed_two_weeks(session)

        with get_session(db_path) as session:
            returns = sector_benchmark_returns(session, {"Technology": "XLK"})

    assert returns["Technology"] == pytest.approx((190.0 - 200.0) / 200.0)


class TestPortfolioSectorReturns:
    def test_weighted_average_within_sector(self):
        holdings = [
            {"symbol": "AAPL", "weight": 0.10, "pct_since_purchase": 0.10},
            {"symbol": "MSFT", "weight": 0.05, "pct_since_purchase": -0.02},
        ]
        sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
        result = portfolio_sector_returns(holdings, sector_map)
        # weighted avg: (0.10*0.10 + 0.05*-0.02) / (0.10+0.05) = (0.01 - 0.001)/0.15
        assert result["Technology"] == pytest.approx((0.01 - 0.001) / 0.15)

    def test_skips_holdings_without_cost_basis(self):
        holdings = [{"symbol": "XLK", "weight": 0.10, "pct_since_purchase": None}]
        result = portfolio_sector_returns(holdings, {"XLK": "Technology"})
        assert result == {}

    def test_unmapped_symbol_falls_into_other(self):
        holdings = [{"symbol": "ZZZ", "weight": 0.10, "pct_since_purchase": 0.05}]
        result = portfolio_sector_returns(holdings, {})
        assert result["Other"] == pytest.approx(0.05)
