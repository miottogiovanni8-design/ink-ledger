import math

import pytest

from trading_desk.metrics import stats


def test_returns_from_equity_curve():
    returns = stats.returns_from_equity_curve([100, 110, 99])
    assert returns[0] == pytest.approx(0.1)
    assert returns[1] == pytest.approx(-0.1)


def test_returns_from_equity_curve_needs_at_least_two_points():
    assert stats.returns_from_equity_curve([100]) == []
    assert stats.returns_from_equity_curve([]) == []


def test_sharpe_ratio_known_value():
    result = stats.sharpe_ratio([0.04, 0.02], periods_per_year=1)
    assert result == pytest.approx(2.121320343559643, rel=1e-9)


def test_sharpe_ratio_zero_variance_returns_zero():
    assert stats.sharpe_ratio([0.02, 0.02, 0.02], periods_per_year=1) == 0.0


def test_sharpe_ratio_needs_at_least_two_points():
    assert stats.sharpe_ratio([0.02]) == 0.0


def test_sortino_ratio_known_value():
    result = stats.sortino_ratio([0.04, -0.02], periods_per_year=1)
    assert result == pytest.approx(1 / math.sqrt(2), rel=1e-9)


def test_sortino_ratio_no_downside_returns_zero():
    assert stats.sortino_ratio([0.04, 0.02], periods_per_year=1) == 0.0


def test_max_drawdown_known_value():
    assert stats.max_drawdown([100, 120, 90, 110]) == pytest.approx(0.25)


def test_max_drawdown_monotonic_up_is_zero():
    assert stats.max_drawdown([100, 110, 120, 130]) == pytest.approx(0.0)


def test_max_drawdown_empty_is_zero():
    assert stats.max_drawdown([]) == 0.0


def test_win_rate_known_value():
    assert stats.win_rate([10, -5, 20, -2, 0]) == pytest.approx(0.4)


def test_win_rate_no_trades_is_zero():
    assert stats.win_rate([]) == 0.0


def test_profit_factor_known_value():
    assert stats.profit_factor([10, -5, 20, -2]) == pytest.approx(30 / 7)


def test_profit_factor_no_losses_is_none():
    assert stats.profit_factor([10, 20]) is None


def test_alpha_beta_known_linear_relationship():
    benchmark = [0.01, 0.02, 0.03, 0.04]
    strategy = [0.021, 0.041, 0.061, 0.081]  # = 2 * benchmark + 0.001
    alpha, beta = stats.alpha_beta(strategy, benchmark, periods_per_year=1)
    assert beta == pytest.approx(2.0, rel=1e-9)
    assert alpha == pytest.approx(0.001, rel=1e-9)


def test_alpha_beta_flat_benchmark_is_undefined():
    alpha, beta = stats.alpha_beta([0.01, 0.02], [0.0, 0.0])
    assert (alpha, beta) == (0.0, 0.0)


def test_alpha_beta_too_short_is_zero():
    alpha, beta = stats.alpha_beta([0.01], [0.02])
    assert (alpha, beta) == (0.0, 0.0)
