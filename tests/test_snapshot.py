import json
import tempfile
from pathlib import Path

from trading_desk.persistence.db import get_session
from trading_desk.persistence.models import Decision, EquitySnapshot, Trade
from trading_desk.reporting.snapshot import build_snapshot, read_snapshot, write_snapshot


def _seed(session):
    for equity in [1000.0, 1010.0, 990.0, 1025.0]:
        session.add(EquitySnapshot(equity_eur=equity, cash_eur=equity * 0.5, open_positions_count=1))

    decision = Decision(
        symbol="AAPL",
        asset_class="equity",
        direction="long",
        confidence=0.7,
        rationale="RSI oversold with a positive headline.",
        key_signals=json.dumps(["RSI 28 oversold"]),
        risk_flags=json.dumps([]),
    )
    session.add(decision)
    session.flush()

    session.add(
        Trade(
            decision_id=decision.id,
            symbol="AAPL",
            asset_class="equity",
            direction="long",
            entry_price=190.0,
            stop_loss_price=185.0,
            take_profit_price=200.0,
            size_eur=15.0,
            status="closed",
            exit_price=198.0,
            realized_pnl_eur=8.0,
        )
    )
    session.add(
        Trade(
            symbol="MSFT",
            asset_class="equity",
            direction="short",
            entry_price=400.0,
            stop_loss_price=410.0,
            take_profit_price=380.0,
            size_eur=15.0,
            status="open",
        )
    )


def test_build_snapshot_shape_and_stats():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.sqlite")
        with get_session(db_path) as session:
            _seed(session)

        with get_session(db_path) as session:
            snapshot = build_snapshot(session)

    assert snapshot["schema_version"] == 1
    assert len(snapshot["equity_curve"]) == 4
    assert snapshot["stats"]["closed_trades_count"] == 1
    assert snapshot["stats"]["total_realized_pnl_eur"] == 8.0
    assert snapshot["stats"]["win_rate"] == 1.0
    assert len(snapshot["positions"]) == 1
    assert snapshot["positions"][0]["symbol"] == "MSFT"
    assert len(snapshot["trade_journal"]) == 1
    assert snapshot["trade_journal"][0]["rationale"].startswith("RSI oversold")
    assert snapshot["trade_journal"][0]["key_signals"] == ["RSI 28 oversold"]


def test_write_and_read_snapshot_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "snapshot.json")
        snapshot = {"schema_version": 1, "stats": {"sharpe_ratio": 1.23}}
        write_snapshot(snapshot, path)
        loaded = read_snapshot(path)
        assert loaded == snapshot
