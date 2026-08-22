from dataclasses import dataclass

import pytest

from trading_desk.metrics.cost_tracking import estimate_view_call_cost_usd


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def test_estimate_view_call_cost_uncached():
    usage = FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = estimate_view_call_cost_usd(usage)
    assert cost == pytest.approx(3.00 + 15.00)


def test_estimate_view_call_cost_includes_cache_write_and_read():
    usage = FakeUsage(cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
    cost = estimate_view_call_cost_usd(usage)
    assert cost == pytest.approx(3.00 * 1.25 + 3.00 * 0.10)


def test_estimate_view_call_cost_zero_usage():
    assert estimate_view_call_cost_usd(FakeUsage()) == 0.0


def test_estimate_view_call_cost_handles_missing_cache_fields():
    class BareUsage:
        input_tokens = 500_000
        output_tokens = 200_000

    cost = estimate_view_call_cost_usd(BareUsage())
    assert cost == pytest.approx(0.5 * 3.00 + 0.2 * 15.00)
