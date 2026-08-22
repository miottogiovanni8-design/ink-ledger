"""Estimates the USD cost of Claude API calls from response token usage,
using public per-token pricing — not Anthropic's own billing figure (that
requires an Admin API key, unavailable for individual accounts). Feeds a
self-reported, month-to-date spend estimate that the weekly pipeline
compares against an alert threshold, as a second layer alongside the hard
spend limit set directly in the Anthropic console."""

from typing import Any

# Standard (non-promotional) per-million-token pricing, USD. Kept stable
# rather than tracking temporary intro discounts, since a promo expiring
# mid-month would otherwise make the estimate quietly wrong.
SONNET_5_INPUT_PER_MTOK = 3.00
SONNET_5_OUTPUT_PER_MTOK = 15.00

# Anthropic's standard prompt-caching multipliers on the base input price.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def estimate_view_call_cost_usd(usage: Any) -> float:
    """`usage` is an Anthropic Messages API Usage object (or anything
    exposing the same attributes) from a claude-sonnet-5 call."""
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_creation_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    cost = (
        input_tokens * SONNET_5_INPUT_PER_MTOK
        + cache_creation_tokens * SONNET_5_INPUT_PER_MTOK * CACHE_WRITE_MULTIPLIER
        + cache_read_tokens * SONNET_5_INPUT_PER_MTOK * CACHE_READ_MULTIPLIER
        + output_tokens * SONNET_5_OUTPUT_PER_MTOK
    ) / 1_000_000
    return cost
