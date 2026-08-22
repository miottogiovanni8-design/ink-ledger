from typing import List, Literal

from pydantic import BaseModel, Field

AssetClass = Literal["equity", "etf"]
RiskProfile = Literal["conservative", "balanced", "aggressive"]


class PortfolioView(BaseModel):
    """One asset's expected-return view, as produced by the Claude analyst
    call. Mirrors the `record_portfolio_view` tool schema. Feeds directly
    into Black-Litterman as an absolute view (expected_return_annualized)
    with an Idzorek confidence (confidence) — the LLM never proposes weights
    or trades, only a return expectation and how sure it is of it."""

    symbol: str
    asset_class: AssetClass
    expected_return_annualized: float = Field(ge=-1.0, le=2.0)
    confidence: float = Field(ge=0.01, le=0.99)
    rationale: str
    key_signals: List[str] = Field(default_factory=list)


PORTFOLIO_VIEW_TOOL = {
    "name": "record_portfolio_view",
    "description": (
        "Record an annualized expected-return view for one asset, with the "
        "confidence and reasoning that will be persisted in the investment "
        "committee notes. This is a research view, not a trade instruction — "
        "portfolio weights are computed separately via Black-Litterman."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "asset_class": {"type": "string", "enum": ["equity", "etf"]},
            "expected_return_annualized": {
                "type": "number",
                "description": "Expected annualized total return, e.g. 0.08 for +8%/year, -0.05 for -5%/year.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.01,
                "maximum": 0.99,
                "description": "Idzorek-style confidence in this view, from 0.01 (almost no conviction) to 0.99 (very high conviction). Never exactly 0 or 1.",
            },
            "rationale": {"type": "string"},
            "key_signals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific evidence cited in the rationale (a headline, a metric, a macro factor). Use an empty array if none.",
            },
        },
        "required": ["symbol", "asset_class", "expected_return_annualized", "confidence", "rationale", "key_signals"],
        "additionalProperties": False,
    },
}
