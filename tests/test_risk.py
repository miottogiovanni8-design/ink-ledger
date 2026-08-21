import pytest

from trading_desk.risk import circuit_breakers as cb
from trading_desk.risk import sizing


class TestSizing:
    def test_position_size_scales_with_budget(self):
        size = sizing.position_size_eur(
            daily_budget_eur=100,
            risk_pct_per_trade=0.05,
            stop_loss_distance_pct=0.5,
            max_position_pct_of_budget=0.5,
        )
        assert size == pytest.approx(10.0)

    def test_position_size_capped_at_max_pct_of_budget(self):
        size = sizing.position_size_eur(
            daily_budget_eur=100,
            risk_pct_per_trade=0.5,
            stop_loss_distance_pct=0.01,
            max_position_pct_of_budget=0.15,
        )
        assert size == pytest.approx(15.0)

    def test_position_size_rejects_non_positive_stop_distance(self):
        with pytest.raises(ValueError):
            sizing.position_size_eur(100, 0.05, 0.0, 0.15)

    def test_stop_loss_long_is_below_entry(self):
        stop = sizing.stop_loss_price(entry_price=100, atr=2, direction="long", atr_multiplier=1.5)
        assert stop == pytest.approx(97.0)

    def test_stop_loss_short_is_above_entry(self):
        stop = sizing.stop_loss_price(entry_price=100, atr=2, direction="short", atr_multiplier=1.5)
        assert stop == pytest.approx(103.0)

    def test_take_profit_long_respects_rr_ratio(self):
        tp = sizing.take_profit_price(entry_price=100, stop_loss=97, direction="long", rr_ratio=2.0)
        assert tp == pytest.approx(106.0)

    def test_take_profit_short_respects_rr_ratio(self):
        tp = sizing.take_profit_price(entry_price=100, stop_loss=103, direction="short", rr_ratio=2.0)
        assert tp == pytest.approx(94.0)

    def test_stop_loss_distance_pct(self):
        assert sizing.stop_loss_distance_pct(100, 97) == pytest.approx(0.03)

    def test_stop_loss_distance_pct_rejects_non_positive_entry(self):
        with pytest.raises(ValueError):
            sizing.stop_loss_distance_pct(0, 97)


class TestCircuitBreakers:
    def test_daily_loss_breaker_not_tripped_within_budget(self):
        result = cb.daily_loss_breaker(daily_pnl_eur=-50, daily_budget_eur=100, threshold_pct=1.0)
        assert result.entries_allowed is True

    def test_daily_loss_breaker_tripped_at_full_budget_loss(self):
        result = cb.daily_loss_breaker(daily_pnl_eur=-100, daily_budget_eur=100, threshold_pct=1.0)
        assert result.entries_allowed is False
        assert "daily loss" in result.reason

    def test_daily_loss_breaker_ignores_gains(self):
        result = cb.daily_loss_breaker(daily_pnl_eur=250, daily_budget_eur=100, threshold_pct=1.0)
        assert result.entries_allowed is True

    def test_drawdown_breaker_not_tripped_below_threshold(self):
        result = cb.drawdown_breaker(current_equity=900, peak_equity=1000, threshold_pct=0.18)
        assert result.entries_allowed is True

    def test_drawdown_breaker_tripped_at_threshold(self):
        result = cb.drawdown_breaker(current_equity=820, peak_equity=1000, threshold_pct=0.18)
        assert result.entries_allowed is False
        assert "drawdown" in result.reason

    def test_drawdown_breaker_handles_zero_peak_equity(self):
        result = cb.drawdown_breaker(current_equity=0, peak_equity=0, threshold_pct=0.18)
        assert result.entries_allowed is True

    def test_max_concurrent_positions_breaker(self):
        assert cb.max_concurrent_positions_breaker(4, 5).entries_allowed is True
        result = cb.max_concurrent_positions_breaker(5, 5)
        assert result.entries_allowed is False
        assert "max concurrent" in result.reason

    def test_evaluate_all_gates_returns_first_tripped(self):
        result = cb.evaluate_all_gates(
            daily_pnl_eur=-100,
            daily_budget_eur=100,
            daily_loss_threshold_pct=1.0,
            current_equity=820,
            peak_equity=1000,
            drawdown_threshold_pct=0.18,
            open_positions=5,
            max_positions=5,
        )
        assert result.entries_allowed is False
        assert "daily loss" in result.reason

    def test_evaluate_all_gates_all_clear(self):
        result = cb.evaluate_all_gates(
            daily_pnl_eur=10,
            daily_budget_eur=100,
            daily_loss_threshold_pct=1.0,
            current_equity=1000,
            peak_equity=1000,
            drawdown_threshold_pct=0.18,
            open_positions=2,
            max_positions=5,
        )
        assert result.entries_allowed is True
        assert result.reason is None
