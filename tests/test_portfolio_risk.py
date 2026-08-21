import numpy as np
import pandas as pd
import pytest

from trading_desk.engine.portfolio_risk import evaluate_scenario, exposure_by_tag, historical_var_cvar


class TestEvaluateScenario:
    def test_two_asset_known_values(self):
        expected_returns = pd.Series({"A": 0.10, "B": 0.06})
        cov = pd.DataFrame({"A": [0.04, 0.0], "B": [0.0, 0.01]}, index=["A", "B"])  # uncorrelated
        weights = {"A": 0.5, "B": 0.5}

        ret, vol, sharpe = evaluate_scenario(weights, expected_returns, cov, risk_free_rate=0.02)

        assert ret == pytest.approx(0.08)  # 0.5*0.10 + 0.5*0.06
        expected_var = 0.5**2 * 0.04 + 0.5**2 * 0.01  # no covariance term (uncorrelated)
        assert vol == pytest.approx(expected_var**0.5)
        assert sharpe == pytest.approx((0.08 - 0.02) / (expected_var**0.5))


class TestHistoricalVarCvar:
    def test_exact_percentile_no_interpolation(self):
        values = np.arange(-50, 51) / 1000.0  # -0.05 .. 0.05, 101 points
        returns_panel = pd.DataFrame({"A": values})
        var, cvar = historical_var_cvar({"A": 1.0}, returns_panel, confidence_level=0.90)

        assert var == pytest.approx(0.04, abs=1e-9)
        assert cvar == pytest.approx(0.045, abs=1e-9)

    def test_cvar_is_at_least_var(self):
        rng = np.random.default_rng(3)
        returns_panel = pd.DataFrame({"A": rng.normal(0.0005, 0.02, 500), "B": rng.normal(0.0003, 0.015, 500)})
        var, cvar = historical_var_cvar({"A": 0.6, "B": 0.4}, returns_panel, confidence_level=0.95)
        assert var >= 0
        assert cvar >= var

    def test_higher_volatility_gives_higher_var(self):
        rng = np.random.default_rng(11)
        calm = pd.DataFrame({"A": rng.normal(0.0, 0.005, 500)})
        volatile = pd.DataFrame({"A": rng.normal(0.0, 0.05, 500)})
        var_calm, _ = historical_var_cvar({"A": 1.0}, calm, confidence_level=0.95)
        var_volatile, _ = historical_var_cvar({"A": 1.0}, volatile, confidence_level=0.95)
        assert var_volatile > var_calm


class TestExposureByTag:
    def test_sums_weights_by_tag(self):
        weights = {"AAPL": 0.10, "MSFT": 0.08, "XLF": 0.15}
        tag_map = {"AAPL": "Technology", "MSFT": "Technology", "XLF": "Financials"}
        exposure = exposure_by_tag(weights, tag_map)
        assert exposure == {"Technology": pytest.approx(0.18), "Financials": pytest.approx(0.15)}

    def test_unmapped_symbol_falls_back_to_other(self):
        exposure = exposure_by_tag({"ZZZ": 0.05}, {})
        assert exposure == {"Other": pytest.approx(0.05)}
