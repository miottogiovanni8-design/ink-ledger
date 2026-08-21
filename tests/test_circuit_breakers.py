from trading_desk.risk import circuit_breakers as cb


def test_drawdown_breaker_not_tripped_below_threshold():
    result = cb.drawdown_breaker(current_equity=900, peak_equity=1000, threshold_pct=0.18)
    assert result.rebalancing_allowed is True
    assert result.reason is None


def test_drawdown_breaker_tripped_at_threshold():
    result = cb.drawdown_breaker(current_equity=820, peak_equity=1000, threshold_pct=0.18)
    assert result.rebalancing_allowed is False
    assert "drawdown" in result.reason


def test_drawdown_breaker_handles_zero_peak_equity():
    result = cb.drawdown_breaker(current_equity=0, peak_equity=0, threshold_pct=0.18)
    assert result.rebalancing_allowed is True


def test_drawdown_breaker_new_peak_is_never_tripped():
    result = cb.drawdown_breaker(current_equity=1200, peak_equity=1200, threshold_pct=0.18)
    assert result.rebalancing_allowed is True
