"""Performance statistics computed off the equity curve and trade journal.

Pure functions over plain lists/floats — no DB or network dependency, so this
is fully unit-testable against known fixtures.
"""

import math
from typing import List, Optional, Tuple

TRADING_PERIODS_PER_YEAR = 252


def returns_from_equity_curve(equity: List[float]) -> List[float]:
    """Simple period-over-period returns from a sequence of equity values."""
    if len(equity) < 2:
        return []
    return [(equity[i] - equity[i - 1]) / equity[i - 1] for i in range(1, len(equity)) if equity[i - 1] != 0]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def sharpe_ratio(
    returns: List[float],
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sharpe ratio. Risk-free rate is a per-period rate (not annual)."""
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free_rate for r in returns]
    std = _stdev(excess)
    if std == 0:
        return 0.0
    return (_mean(excess) / std) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: List[float],
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualized Sortino ratio — like Sharpe, but only penalizes downside deviation."""
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free_rate for r in returns]
    downside = [min(r, 0.0) for r in excess]
    downside_std = math.sqrt(sum(d**2 for d in downside) / len(downside)) if downside else 0.0
    if downside_std == 0:
        return 0.0
    return (_mean(excess) / downside_std) * math.sqrt(periods_per_year)


def max_drawdown(equity: List[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.15 = -15%)."""
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            drawdown = (peak - value) / peak
            worst = max(worst, drawdown)
    return worst


def win_rate(trade_pnls: List[float]) -> float:
    if not trade_pnls:
        return 0.0
    wins = sum(1 for pnl in trade_pnls if pnl > 0)
    return wins / len(trade_pnls)


def profit_factor(trade_pnls: List[float]) -> Optional[float]:
    """Gross profit / gross loss. Returns None when there are no losing trades
    to divide by (undefined, not infinite — surface it as unavailable)."""
    gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def alpha_beta(
    strategy_returns: List[float],
    benchmark_returns: List[float],
    periods_per_year: int = TRADING_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> Tuple[float, float]:
    """Annualized alpha and beta of strategy returns vs. a benchmark, via OLS
    on excess returns. Series are aligned by truncating to the shorter length."""
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 2:
        return 0.0, 0.0

    strat = [r - risk_free_rate for r in strategy_returns[:n]]
    bench = [r - risk_free_rate for r in benchmark_returns[:n]]

    bench_mean = _mean(bench)
    bench_var = sum((b - bench_mean) ** 2 for b in bench) / (n - 1)
    if bench_var == 0:
        return 0.0, 0.0

    strat_mean = _mean(strat)
    covariance = sum((strat[i] - strat_mean) * (bench[i] - bench_mean) for i in range(n)) / (n - 1)

    beta = covariance / bench_var
    alpha_per_period = strat_mean - beta * bench_mean
    alpha_annualized = alpha_per_period * periods_per_year
    return alpha_annualized, beta
