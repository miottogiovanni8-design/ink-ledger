from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from trading_desk.engine.decision import build_user_content, parse_decision_response, request_trade_decision
from trading_desk.engine.schemas import IndicatorSnapshot, PortfolioState, TradeDecision


def make_portfolio(**overrides) -> PortfolioState:
    defaults = dict(
        equity_eur=1000.0,
        daily_budget_eur=100.0,
        daily_budget_spent_eur=20.0,
        open_positions=1,
        max_positions=5,
        daily_pnl_eur=-5.0,
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


def make_snapshot(**overrides) -> IndicatorSnapshot:
    defaults = dict(
        symbol="AAPL",
        asset_class="equity",
        price=190.0,
        rsi_14=25.0,
        macd=0.1,
        macd_signal=0.0,
        macd_prev=-0.1,
        macd_signal_prev=0.0,
        bollinger_upper=195.0,
        bollinger_lower=185.0,
        atr_14=2.5,
        has_fresh_headline=True,
        headlines=["AAPL beats earnings estimates"],
    )
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


@dataclass
class FakeToolUseBlock:
    input: Dict[str, Any]
    type: str = "tool_use"
    name: str = "record_trade_decision"


@dataclass
class FakeMessage:
    content: List[Any] = field(default_factory=list)


class FakeMessagesAPI:
    def __init__(self, response: FakeMessage):
        self._response = response
        self.last_call_kwargs: Dict[str, Any] = {}

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response


class FakeAnthropicClient:
    def __init__(self, response: FakeMessage):
        self.messages = FakeMessagesAPI(response)


def test_build_user_content_includes_remaining_budget_and_signals():
    content = build_user_content(make_portfolio(), make_snapshot(), ["RSI 25.0 oversold"])
    assert "AAPL" in content
    assert "remaining: 80.00 EUR" in content
    assert "RSI 25.0 oversold" in content
    assert "AAPL beats earnings estimates" in content


def test_parse_decision_response_extracts_tool_call():
    decision_payload = {
        "symbol": "AAPL",
        "asset_class": "equity",
        "direction": "long",
        "confidence": 0.7,
        "position_size_usd": 15.0,
        "stop_loss_price": 185.0,
        "take_profit_price": 200.0,
        "rationale": "RSI oversold with a positive earnings headline.",
        "key_signals": ["RSI 25 oversold", "positive earnings headline"],
        "risk_flags": [],
    }
    response = FakeMessage(content=[FakeToolUseBlock(input=decision_payload)])
    decision = parse_decision_response(response)
    assert isinstance(decision, TradeDecision)
    assert decision.symbol == "AAPL"
    assert decision.direction == "long"
    assert decision.confidence == pytest.approx(0.7)


def test_parse_decision_response_raises_without_tool_call():
    response = FakeMessage(content=[])
    with pytest.raises(ValueError):
        parse_decision_response(response)


def test_request_trade_decision_wires_model_tools_and_forced_tool_choice():
    decision_payload = {
        "symbol": "AAPL",
        "asset_class": "equity",
        "direction": "hold",
        "confidence": 0.4,
        "rationale": "Mixed signals, staying flat.",
    }
    response = FakeMessage(content=[FakeToolUseBlock(input=decision_payload)])
    client = FakeAnthropicClient(response)

    decision = request_trade_decision(client, make_portfolio(), make_snapshot(), ["RSI 25.0 oversold"])

    assert decision.direction == "hold"
    call_kwargs = client.messages.last_call_kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "record_trade_decision"}
    assert call_kwargs["tools"][0]["name"] == "record_trade_decision"
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
