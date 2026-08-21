"""Black-Litterman portfolio construction: blends CAPM market equilibrium
with Claude-generated views, then optimizes for a target risk profile.

Academic basis: Black & Litterman (1992); using LLM output as the view
vector follows "Integrating LLM-Generated Views into Mean-Variance
Optimization Using the Black-Litterman Model" (ICLR 2025 workshop,
arXiv:2504.14345). Math is delegated to PyPortfolioOpt rather than
reimplemented — Ledoit-Wolf shrinkage for covariance (the library's own docs
advise against raw sample covariance), Idzorek's method to turn 0-1
confidences directly into the view uncertainty matrix Omega.
"""

from typing import Dict, List, Tuple

import pandas as pd
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns, market_implied_risk_aversion
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage

from trading_desk.engine.schemas import PortfolioView, RiskProfile

TARGET_VOLATILITY = {
    "conservative": 0.08,
    "aggressive": 0.22,
}


def compute_covariance(price_panel: pd.DataFrame) -> pd.DataFrame:
    """price_panel: rows are dates, columns are tickers, values are close prices."""
    return CovarianceShrinkage(price_panel).ledoit_wolf()


def compute_market_prior(
    market_prices: pd.Series,
    market_caps: Dict[str, float],
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Reverse-optimizes the market's implied risk aversion from a benchmark
    price series, then derives equilibrium expected returns (Pi) from
    market-cap weights and the covariance matrix — the Black-Litterman
    prior, before any views are applied."""
    risk_aversion = market_implied_risk_aversion(market_prices, risk_free_rate=risk_free_rate)
    return market_implied_prior_returns(market_caps, risk_aversion, cov_matrix, risk_free_rate=risk_free_rate)


def blend_views(
    cov_matrix: pd.DataFrame,
    prior: pd.Series,
    views: List[PortfolioView],
) -> Tuple[pd.Series, pd.DataFrame]:
    """Combines the equilibrium prior with Claude's views via Black-Litterman.
    No views means no information beyond the market — returns the prior
    unchanged rather than degrading it through an empty-view BL pass."""
    if not views:
        return prior, cov_matrix

    absolute_views = {v.symbol: v.expected_return_annualized for v in views}
    confidences = [v.confidence for v in views]

    bl = BlackLittermanModel(
        cov_matrix,
        pi=prior,
        absolute_views=absolute_views,
        omega="idzorek",
        view_confidences=confidences,
    )
    return bl.bl_returns(), bl.bl_cov()


def optimize_portfolio(
    posterior_returns: pd.Series,
    posterior_cov: pd.DataFrame,
    risk_profile: RiskProfile,
    max_weight_pct: float = 0.15,
) -> Dict[str, float]:
    """Maps the risk profile to an EfficientFrontier objective: conservative
    and aggressive target an explicit annualized volatility band; balanced
    maximizes the Sharpe ratio (the tangency portfolio) within the same
    long-only, single-name weight cap."""
    ef = EfficientFrontier(posterior_returns, posterior_cov, weight_bounds=(0, max_weight_pct))
    try:
        if risk_profile == "conservative":
            ef.efficient_risk(TARGET_VOLATILITY["conservative"])
        elif risk_profile == "aggressive":
            ef.efficient_risk(TARGET_VOLATILITY["aggressive"])
        else:
            ef.max_sharpe(risk_free_rate=0.0)
    except ValueError:
        # the requested objective is infeasible for this universe/posterior
        # (target vol below the achievable minimum, or no asset beats the
        # risk-free rate) — minimum volatility is always feasible and is the
        # sensible degenerate fallback for every profile.
        ef.min_volatility()
    return ef.clean_weights()
