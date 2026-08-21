"""The one standing circuit breaker for a long-only, weekly-rebalanced
portfolio: if drawdown from the equity peak exceeds the threshold, pause
rebalancing entirely until a human resets it. Position-level and daily-loss
breakers from the v1 intraday design don't have a clean analogue in a weekly
allocation framework, so they were removed rather than kept as dead weight.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RiskGateResult:
    rebalancing_allowed: bool
    reason: Optional[str] = None


def drawdown_breaker(
    current_equity: float,
    peak_equity: float,
    threshold_pct: float,
) -> RiskGateResult:
    """Pause rebalancing if drawdown from the equity peak exceeds
    `threshold_pct`. Requires manual review/reset to resume."""
    if peak_equity <= 0:
        return RiskGateResult(True)
    drawdown = (peak_equity - current_equity) / peak_equity
    if drawdown >= threshold_pct:
        return RiskGateResult(
            False,
            f"max drawdown circuit breaker tripped: {drawdown:.1%} >= {threshold_pct:.1%} — rebalancing paused, needs manual reset",
        )
    return RiskGateResult(True)
