"""Claude decision engine: turns a candidate's technical + news context into a
structured, journal-ready trade decision via forced tool use.

This module never talks to the risk layer or the broker — callers are
expected to have already cleared `risk.circuit_breakers.evaluate_all_gates`
before invoking `request_trade_decision`. The LLM proposes; the deterministic
risk/execution code disposes.
"""

from typing import Any, List

from trading_desk.engine.schemas import TRADE_DECISION_TOOL, IndicatorSnapshot, PortfolioState, TradeDecision

DECISION_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are the decision engine of a paper-trading desk. For the \
candidate symbol described in the user message, decide whether to go long, \
short, or hold, sized within the stated remaining daily budget.

Rules you must respect:
- Never propose a position larger than the remaining daily budget allows.
- Always propose a stop_loss_price and take_profit_price for long/short \
  decisions; hold decisions are exempt from stop/take-profit.
- `rationale` must be a specific, plain-English explanation referencing the \
  actual technical/news signals given — it is stored verbatim in the trade \
  journal and shown to a human reviewing the strategy.
- If the signals are mixed or weak, prefer "hold" over a low-confidence trade.
- You must call the record_trade_decision tool exactly once with your decision."""


def build_user_content(
    portfolio: PortfolioState,
    snapshot: IndicatorSnapshot,
    trigger_reasons: List[str],
) -> str:
    remaining_budget = portfolio.daily_budget_eur - portfolio.daily_budget_spent_eur
    lines = [
        f"Symbol: {snapshot.symbol} ({snapshot.asset_class})",
        f"Price: {snapshot.price}",
        f"RSI(14): {snapshot.rsi_14}",
        f"MACD: {snapshot.macd} / signal {snapshot.macd_signal} (prev: {snapshot.macd_prev} / {snapshot.macd_signal_prev})",
        f"Bollinger bands: lower {snapshot.bollinger_lower}, upper {snapshot.bollinger_upper}",
        f"ATR(14): {snapshot.atr_14}",
        f"Screening triggers: {', '.join(trigger_reasons) if trigger_reasons else 'none'}",
        f"Recent headlines: {' | '.join(snapshot.headlines) if snapshot.headlines else 'none'}",
        "",
        f"Portfolio equity: {portfolio.equity_eur:.2f} EUR",
        f"Daily budget: {portfolio.daily_budget_eur:.2f} EUR, spent so far: {portfolio.daily_budget_spent_eur:.2f} EUR, remaining: {remaining_budget:.2f} EUR",
        f"Open positions: {portfolio.open_positions}/{portfolio.max_positions}",
        f"Today's P&L so far: {portfolio.daily_pnl_eur:.2f} EUR",
    ]
    return "\n".join(lines)


def parse_decision_response(response: Any) -> TradeDecision:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "record_trade_decision":
            return TradeDecision(**block.input)
    raise ValueError("decision engine response did not include a record_trade_decision tool call")


def request_trade_decision(
    client: Any,
    portfolio: PortfolioState,
    snapshot: IndicatorSnapshot,
    trigger_reasons: List[str],
    model: str = DECISION_MODEL,
) -> TradeDecision:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[TRADE_DECISION_TOOL],
        tool_choice={"type": "tool", "name": "record_trade_decision"},
        messages=[{"role": "user", "content": build_user_content(portfolio, snapshot, trigger_reasons)}],
    )
    return parse_decision_response(response)
