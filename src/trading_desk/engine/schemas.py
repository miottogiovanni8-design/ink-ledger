from typing import List, Literal, Optional

from pydantic import BaseModel, Field

AssetClass = Literal["equity", "crypto"]
Direction = Literal["long", "short", "hold"]


class PortfolioState(BaseModel):
    """Deterministic account state handed to the LLM as context — it never
    computes this itself, so it can't misjudge remaining risk budget."""

    equity_eur: float
    daily_budget_eur: float
    daily_budget_spent_eur: float
    open_positions: int
    max_positions: int
    daily_pnl_eur: float


class IndicatorSnapshot(BaseModel):
    symbol: str
    asset_class: AssetClass
    price: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_prev: float
    macd_signal_prev: float
    bollinger_upper: float
    bollinger_lower: float
    atr_14: float
    has_fresh_headline: bool = False
    headlines: List[str] = Field(default_factory=list)


class TradeDecision(BaseModel):
    """Mirrors the `record_trade_decision` tool schema the Claude decision
    engine returns. `rationale` is what lands in the trade journal."""

    symbol: str
    asset_class: AssetClass
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    position_size_usd: float = Field(default=0.0, ge=0.0)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    rationale: str
    key_signals: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    skipped_by_risk_layer: bool = False
    skip_reason: Optional[str] = None


TRADE_DECISION_TOOL = {
    "name": "record_trade_decision",
    "description": (
        "Record a trading decision for one candidate symbol, with the reasoning "
        "that will be persisted in the trade journal."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "asset_class": {"type": "string", "enum": ["equity", "crypto"]},
            "direction": {"type": "string", "enum": ["long", "short", "hold"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "position_size_usd": {"type": "number", "minimum": 0},
            "stop_loss_price": {"type": "number"},
            "take_profit_price": {"type": "number"},
            "rationale": {"type": "string"},
            "key_signals": {"type": "array", "items": {"type": "string"}},
            "risk_flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["symbol", "asset_class", "direction", "confidence", "rationale"],
    },
}
