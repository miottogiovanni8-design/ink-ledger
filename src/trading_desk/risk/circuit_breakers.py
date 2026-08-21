"""Deterministic risk gates evaluated before every decision-engine call.

These run in front of the LLM: if a breaker is tripped, the LLM is never
invoked for new entries — the risk layer disposes, the model only proposes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RiskGateResult:
    entries_allowed: bool
    reason: Optional[str] = None


def daily_loss_breaker(
    daily_pnl_eur: float,
    daily_budget_eur: float,
    threshold_pct: float,
) -> RiskGateResult:
    """Halt new entries once today's realized+unrealized loss reaches
    `threshold_pct` of the daily budget (default: 100%, i.e. lose at most
    one day's budget in one day)."""
    loss_limit = -daily_budget_eur * threshold_pct
    if daily_pnl_eur <= loss_limit:
        return RiskGateResult(
            False,
            f"daily loss circuit breaker tripped: P&L {daily_pnl_eur:.2f} EUR <= limit {loss_limit:.2f} EUR",
        )
    return RiskGateResult(True)


def drawdown_breaker(
    current_equity: float,
    peak_equity: float,
    threshold_pct: float,
) -> RiskGateResult:
    """Pause the entire system if drawdown from the equity peak exceeds
    `threshold_pct`. Requires manual review/reset to resume."""
    if peak_equity <= 0:
        return RiskGateResult(True)
    drawdown = (peak_equity - current_equity) / peak_equity
    if drawdown >= threshold_pct:
        return RiskGateResult(
            False,
            f"max drawdown circuit breaker tripped: {drawdown:.1%} >= {threshold_pct:.1%} — system paused, needs manual reset",
        )
    return RiskGateResult(True)


def max_concurrent_positions_breaker(open_positions: int, max_positions: int) -> RiskGateResult:
    if open_positions >= max_positions:
        return RiskGateResult(
            False,
            f"max concurrent positions reached: {open_positions}/{max_positions}",
        )
    return RiskGateResult(True)


def evaluate_all_gates(
    daily_pnl_eur: float,
    daily_budget_eur: float,
    daily_loss_threshold_pct: float,
    current_equity: float,
    peak_equity: float,
    drawdown_threshold_pct: float,
    open_positions: int,
    max_positions: int,
) -> RiskGateResult:
    """Evaluate every gate in order; return the first tripped gate, or an
    all-clear result if none trip."""
    gates = [
        daily_loss_breaker(daily_pnl_eur, daily_budget_eur, daily_loss_threshold_pct),
        drawdown_breaker(current_equity, peak_equity, drawdown_threshold_pct),
        max_concurrent_positions_breaker(open_positions, max_positions),
    ]
    for gate in gates:
        if not gate.entries_allowed:
            return gate
    return RiskGateResult(True)
