import numpy as np
import pandas as pd
import pytest

from trading_desk.engine.black_litterman import blend_views, compute_covariance, compute_market_prior, optimize_portfolio
from trading_desk.engine.schemas import PortfolioView

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def make_price_panel(seed: int = 7, n_days: int = 252) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-08-01", periods=n_days, freq="B")
    panel = {}
    for i, ticker in enumerate(TICKERS):
        drift = 0.0003 + i * 0.0001
        vol = 0.01 + i * 0.003
        returns = rng.normal(drift, vol, n_days)
        prices = 100 * np.cumprod(1 + returns)
        panel[ticker] = prices
    return pd.DataFrame(panel, index=dates)


def make_view(symbol, expected_return, confidence) -> PortfolioView:
    return PortfolioView(
        symbol=symbol,
        asset_class="equity",
        expected_return_annualized=expected_return,
        confidence=confidence,
        rationale="test view",
        rationale_it="view di prova",
    )


class TestCovariance:
    def test_covariance_matrix_shape_and_symmetry(self):
        cov = compute_covariance(make_price_panel())
        assert cov.shape == (4, 4)
        assert list(cov.columns) == TICKERS
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)

    def test_covariance_matrix_is_positive_semidefinite(self):
        cov = compute_covariance(make_price_panel())
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert (eigenvalues >= -1e-8).all()


class TestMarketPrior:
    def test_prior_returns_one_value_per_ticker(self):
        panel = make_price_panel()
        cov = compute_covariance(panel)
        market_caps = {t: 1e9 * (i + 1) for i, t in enumerate(TICKERS)}
        prior = compute_market_prior(panel["AAA"], market_caps, cov)
        assert set(prior.index) == set(TICKERS)
        assert prior.notna().all()


class TestBlendViews:
    def test_no_views_returns_prior_unchanged(self):
        panel = make_price_panel()
        cov = compute_covariance(panel)
        market_caps = {t: 1e9 for t in TICKERS}
        prior = compute_market_prior(panel["AAA"], market_caps, cov)

        posterior_returns, posterior_cov = blend_views(cov, prior, [])

        pd.testing.assert_series_equal(posterior_returns, prior)
        pd.testing.assert_frame_equal(posterior_cov, cov)

    def test_higher_confidence_pulls_posterior_closer_to_view(self):
        panel = make_price_panel()
        cov = compute_covariance(panel)
        market_caps = {t: 1e9 for t in TICKERS}
        prior = compute_market_prior(panel["AAA"], market_caps, cov)

        extreme_view_value = prior["AAA"] + 0.50  # far from equilibrium

        low_conf_returns, _ = blend_views(cov, prior, [make_view("AAA", extreme_view_value, 0.05)])
        high_conf_returns, _ = blend_views(cov, prior, [make_view("AAA", extreme_view_value, 0.95)])

        dist_low = abs(low_conf_returns["AAA"] - extreme_view_value)
        dist_high = abs(high_conf_returns["AAA"] - extreme_view_value)
        assert dist_high < dist_low

    def test_view_on_one_asset_barely_moves_unrelated_assets(self):
        panel = make_price_panel()
        cov = compute_covariance(panel)
        market_caps = {t: 1e9 for t in TICKERS}
        prior = compute_market_prior(panel["AAA"], market_caps, cov)

        posterior_returns, _ = blend_views(cov, prior, [make_view("AAA", prior["AAA"] + 0.50, 0.95)])

        # DDD has low correlation-driven spillover; posterior should stay close to its prior
        assert abs(posterior_returns["DDD"] - prior["DDD"]) < abs(posterior_returns["AAA"] - prior["AAA"])


class TestOptimizePortfolio:
    def _posterior(self):
        panel = make_price_panel()
        cov = compute_covariance(panel)
        market_caps = {t: 1e9 for t in TICKERS}
        prior = compute_market_prior(panel["AAA"], market_caps, cov)
        views = [make_view(t, prior[t] + 0.03, 0.6) for t in TICKERS]
        return blend_views(cov, prior, views)

    def test_weights_sum_to_one_and_respect_bounds(self):
        posterior_returns, posterior_cov = self._posterior()
        for profile in ["conservative", "balanced", "aggressive"]:
            weights = optimize_portfolio(posterior_returns, posterior_cov, profile, max_weight_pct=0.4)
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)
            for w in weights.values():
                assert -1e-6 <= w <= 0.4 + 1e-6

    def test_conservative_has_lower_realized_volatility_than_aggressive(self):
        from pypfopt.efficient_frontier import EfficientFrontier

        posterior_returns, posterior_cov = self._posterior()
        conservative_weights = optimize_portfolio(posterior_returns, posterior_cov, "conservative", max_weight_pct=0.4)
        aggressive_weights = optimize_portfolio(posterior_returns, posterior_cov, "aggressive", max_weight_pct=0.4)

        def realized_vol(weights):
            w = pd.Series(weights)
            return float(np.sqrt(w @ posterior_cov @ w))

        assert realized_vol(conservative_weights) < realized_vol(aggressive_weights)
