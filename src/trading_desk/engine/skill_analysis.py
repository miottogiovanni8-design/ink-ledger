"""Answers the question a real risk manager asks first: does the AI's view
actually have predictive skill, separate from just being long a rising
market? Two standard quant-research tools, both textbook (Grinold & Kahn's
"Active Portfolio Management" is the reference for the Information
Coefficient and calibration analysis of a forecasting signal).
"""

from typing import Dict, List, Optional

import numpy as np


def information_coefficient(predicted: List[float], realized: List[float]) -> Optional[float]:
    """Pearson correlation between the view's predicted expected return and
    the asset's subsequently realized return, across every view with a
    known outcome. This is the standard metric (IC) for whether a
    forecasting signal has any skill at all — an IC of 0 means the views
    are noise; consistently above ~0.05-0.10 is considered a genuine edge
    in most quant equity contexts."""
    if len(predicted) < 2 or len(predicted) != len(realized):
        return None
    if np.std(predicted) < 1e-12 or np.std(realized) < 1e-12:
        return None
    return float(np.corrcoef(predicted, realized)[0, 1])


def directional_hit(expected_return: float, realized_return: float) -> bool:
    """Did the view call the right direction? Ties (either side exactly
    zero) count as a miss — a real view should commit to a direction."""
    if expected_return == 0 or realized_return == 0:
        return False
    return (expected_return > 0) == (realized_return > 0)


def calibration_buckets(
    records: List[Dict[str, float]],
    edges: List[float] = [0.0, 0.4, 0.6, 0.8, 1.0],
) -> List[Dict]:
    """records: [{"confidence": float, "correct": bool}, ...]. Buckets views
    by stated confidence and reports the actual hit rate per bucket — a
    well-calibrated model's hit rate should climb with its stated
    confidence; a flat or inverted curve means the confidence score isn't
    meaningful."""
    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        is_last = i == len(edges) - 2
        in_bucket = [
            r for r in records
            if lo <= r["confidence"] < hi or (is_last and r["confidence"] == hi)
        ]
        hit_rate = (sum(1 for r in in_bucket if r["correct"]) / len(in_bucket)) if in_bucket else None
        buckets.append({"range_low": lo, "range_high": hi, "count": len(in_bucket), "hit_rate": hit_rate})
    return buckets


def decompose_return(total_return: float, benchmark_return: float, beta: float) -> Dict[str, float]:
    """Splits the portfolio's total return into what beta exposure to the
    benchmark alone would explain, and what's left over (the "skill"
    component — the part attributable to the AI's views and the
    optimizer's tilts, not just being long the market)."""
    beta_contribution = beta * benchmark_return
    skill_contribution = total_return - beta_contribution
    return {"beta_contribution": beta_contribution, "skill_contribution": skill_contribution}
