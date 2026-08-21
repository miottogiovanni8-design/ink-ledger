import pytest

from trading_desk.engine.attribution import brinson_attribution


def test_brinson_attribution_known_two_sector_example():
    # Hand-verified: benchmark_total_return = 0.5*0.08 + 0.5*0.03 = 0.055
    # portfolio_total_return = 0.6*0.10 + 0.4*0.02 = 0.068
    # active return = 0.068 - 0.055 = 0.013, which must equal
    # allocation + selection + interaction.
    result = brinson_attribution(
        sectors=["A", "B"],
        portfolio_weights={"A": 0.6, "B": 0.4},
        benchmark_weights={"A": 0.5, "B": 0.5},
        portfolio_returns={"A": 0.10, "B": 0.02},
        benchmark_returns={"A": 0.08, "B": 0.03},
    )

    assert result["benchmark_total_return"] == pytest.approx(0.055)
    assert result["total_allocation_effect"] == pytest.approx(0.005)
    assert result["total_selection_effect"] == pytest.approx(0.005)
    assert result["total_interaction_effect"] == pytest.approx(0.003)
    assert result["total_active_return"] == pytest.approx(0.013)

    sector_a = next(s for s in result["by_sector"] if s["sector"] == "A")
    assert sector_a["allocation_effect"] == pytest.approx(0.0025)
    assert sector_a["selection_effect"] == pytest.approx(0.01)
    assert sector_a["interaction_effect"] == pytest.approx(0.002)


def test_identical_portfolio_and_benchmark_has_zero_active_return():
    result = brinson_attribution(
        sectors=["A", "B"],
        portfolio_weights={"A": 0.5, "B": 0.5},
        benchmark_weights={"A": 0.5, "B": 0.5},
        portfolio_returns={"A": 0.05, "B": 0.05},
        benchmark_returns={"A": 0.05, "B": 0.05},
    )
    assert result["total_active_return"] == pytest.approx(0.0)
    assert result["total_allocation_effect"] == pytest.approx(0.0)
    assert result["total_selection_effect"] == pytest.approx(0.0)


def test_missing_sector_data_defaults_to_zero():
    result = brinson_attribution(
        sectors=["A", "C"],  # "C" has no data in either weights/returns dicts
        portfolio_weights={"A": 1.0},
        benchmark_weights={"A": 1.0},
        portfolio_returns={"A": 0.10},
        benchmark_returns={"A": 0.08},
    )
    sector_c = next(s for s in result["by_sector"] if s["sector"] == "C")
    assert sector_c["portfolio_weight"] == 0.0
    assert sector_c["allocation_effect"] == pytest.approx(0.0)
