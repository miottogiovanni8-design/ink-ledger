import pytest

from trading_desk.engine.schemas import TradeDecision
from trading_desk.risk.sizing import clamp_decision_to_risk_limits


def make_decision(**overrides) -> TradeDecision:
    defaults = dict(
        symbol="AAPL",
        asset_class="equity",
        direction="long",
        confidence=0.7,
        position_size_usd=1000.0,  # deliberately huge, should get clamped
        stop_loss_price=None,
        take_profit_price=None,
        rationale="RSI oversold.",
    )
    defaults.update(overrides)
    return TradeDecision(**defaults)


def test_hold_decision_is_zeroed_out():
    decision = make_decision(direction="hold", position_size_usd=50.0)
    clamped = clamp_decision_to_risk_limits(
        decision,
        entry_price=100.0,
        atr=2.0,
        daily_budget_eur=100.0,
        remaining_budget_eur=80.0,
        risk_pct_per_trade=0.05,
        max_position_pct_of_budget=0.15,
        atr_multiplier=1.5,
        rr_ratio=1.75,
    )
    assert clamped.position_size_usd == 0.0
    assert clamped.stop_loss_price is None
    assert clamped.take_profit_price is None


def test_missing_stop_loss_is_derived_from_atr():
    decision = make_decision(stop_loss_price=None, take_profit_price=None)
    clamped = clamp_decision_to_risk_limits(
        decision,
        entry_price=100.0,
        atr=2.0,
        daily_budget_eur=100.0,
        remaining_budget_eur=80.0,
        risk_pct_per_trade=0.05,
        max_position_pct_of_budget=0.15,
        atr_multiplier=1.5,
        rr_ratio=1.75,
    )
    assert clamped.stop_loss_price == pytest.approx(97.0)  # 100 - 1.5*2
    assert clamped.take_profit_price == pytest.approx(105.25)  # 100 + 1.75*3


def test_oversized_llm_proposal_is_capped_at_risk_based_size():
    decision = make_decision(position_size_usd=1000.0, stop_loss_price=97.0)
    clamped = clamp_decision_to_risk_limits(
        decision,
        entry_price=100.0,
        atr=2.0,
        daily_budget_eur=100.0,
        remaining_budget_eur=80.0,
        risk_pct_per_trade=0.05,
        max_position_pct_of_budget=0.15,
        atr_multiplier=1.5,
        rr_ratio=1.75,
    )
    # risk-based size = (100 * 0.05) / 0.03 = 166.67, capped at 15% of budget = 15.0
    assert clamped.position_size_usd == pytest.approx(15.0)


def test_size_never_exceeds_remaining_daily_budget():
    decision = make_decision(position_size_usd=10.0, stop_loss_price=97.0)
    clamped = clamp_decision_to_risk_limits(
        decision,
        entry_price=100.0,
        atr=2.0,
        daily_budget_eur=100.0,
        remaining_budget_eur=5.0,
        risk_pct_per_trade=0.05,
        max_position_pct_of_budget=0.15,
        atr_multiplier=1.5,
        rr_ratio=1.75,
    )
    assert clamped.position_size_usd == pytest.approx(5.0)


def test_llm_provided_stop_loss_is_respected_not_overridden():
    decision = make_decision(stop_loss_price=95.0, take_profit_price=110.0)
    clamped = clamp_decision_to_risk_limits(
        decision,
        entry_price=100.0,
        atr=2.0,
        daily_budget_eur=100.0,
        remaining_budget_eur=80.0,
        risk_pct_per_trade=0.05,
        max_position_pct_of_budget=0.15,
        atr_multiplier=1.5,
        rr_ratio=1.75,
    )
    assert clamped.stop_loss_price == pytest.approx(95.0)
    assert clamped.take_profit_price == pytest.approx(110.0)
