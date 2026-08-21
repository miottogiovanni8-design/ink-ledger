"""Ex-ante portfolio risk metrics shown on the risk exposure panel:
expected return/volatility/Sharpe for a weight vector, historical VaR/CVaR,
and exposure grouped by an arbitrary tag (sector or factor)."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pypfopt.base_optimizer import portfolio_performance


def evaluate_scenario(
    weights: Dict[str, float],
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> Tuple[float, float, float]:
    """Returns (expected_return, volatility, sharpe) for an arbitrary weight
    vector — used to evaluate all three risk-profile scenarios against the
    same posterior, independent of which one was actually optimized for."""
    return portfolio_performance(weights, expected_returns, cov_matrix, risk_free_rate=risk_free_rate)


def historical_var_cvar(
    weights: Dict[str, float],
    returns_panel: pd.DataFrame,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Historical-simulation VaR/CVaR from a daily-returns panel. Both are
    reported as positive loss fractions (0.03 means a 3% one-day loss)."""
    symbols = list(weights.keys())
    aligned = returns_panel[symbols]
    weight_vector = pd.Series(weights)[symbols]
    portfolio_returns = aligned @ weight_vector

    var_cutoff = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    var = -float(var_cutoff)

    tail_losses = portfolio_returns[portfolio_returns <= var_cutoff]
    cvar = -float(tail_losses.mean()) if len(tail_losses) > 0 else var
    return var, cvar


def exposure_by_tag(weights: Dict[str, float], tag_map: Dict[str, str]) -> Dict[str, float]:
    """Sums portfolio weight by an arbitrary tag — feeds both the sector and
    the factor exposure bars on the dashboard from the same function."""
    exposure: Dict[str, float] = {}
    for symbol, weight in weights.items():
        tag = tag_map.get(symbol, "Other")
        exposure[tag] = exposure.get(tag, 0.0) + weight
    return exposure
