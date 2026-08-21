"""Brinson-Fachler performance attribution: decomposes the portfolio's
active return (portfolio return minus benchmark return) into allocation
(over/underweighting sectors that did well or poorly), selection (picking
better/worse performers within a sector), and interaction. Textbook
three-effect formula — the standard a fund's performance team reports to
an investment committee, not an invented metric.
"""

from typing import Dict, List


def brinson_attribution(
    sectors: List[str],
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    portfolio_returns: Dict[str, float],
    benchmark_returns: Dict[str, float],
) -> Dict:
    benchmark_total_return = sum(
        benchmark_weights.get(s, 0.0) * benchmark_returns.get(s, 0.0) for s in sectors
    )

    by_sector = []
    total_allocation = total_selection = total_interaction = 0.0

    for sector in sectors:
        wp = portfolio_weights.get(sector, 0.0)
        wb = benchmark_weights.get(sector, 0.0)
        rp = portfolio_returns.get(sector, 0.0)
        rb = benchmark_returns.get(sector, 0.0)

        allocation = (wp - wb) * (rb - benchmark_total_return)
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)

        total_allocation += allocation
        total_selection += selection
        total_interaction += interaction

        by_sector.append(
            {
                "sector": sector,
                "portfolio_weight": wp,
                "benchmark_weight": wb,
                "allocation_effect": allocation,
                "selection_effect": selection,
                "interaction_effect": interaction,
            }
        )

    return {
        "by_sector": by_sector,
        "benchmark_total_return": benchmark_total_return,
        "total_allocation_effect": total_allocation,
        "total_selection_effect": total_selection,
        "total_interaction_effect": total_interaction,
        "total_active_return": total_allocation + total_selection + total_interaction,
    }
