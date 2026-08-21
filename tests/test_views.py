from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from trading_desk.engine.schemas import PortfolioView
from trading_desk.engine.views import build_user_content, parse_view_response, request_portfolio_view


@dataclass
class FakeToolUseBlock:
    input: Dict[str, Any]
    type: str = "tool_use"
    name: str = "record_portfolio_view"


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


def test_build_user_content_includes_headlines_and_sector():
    content = build_user_content("AAPL", "equity", ["Apple beats estimates"], sector="Technology")
    assert "AAPL" in content
    assert "Technology" in content
    assert "Apple beats estimates" in content


def test_build_user_content_handles_no_headlines():
    content = build_user_content("XLK", "etf", [])
    assert "none" in content


def test_parse_view_response_extracts_tool_call():
    payload = {
        "symbol": "AAPL",
        "asset_class": "equity",
        "expected_return_annualized": 0.09,
        "confidence": 0.55,
        "rationale": "Strong iPhone cycle plus services growth.",
        "key_signals": ["services revenue +15% YoY"],
    }
    response = FakeMessage(content=[FakeToolUseBlock(input=payload)])
    view = parse_view_response(response)
    assert isinstance(view, PortfolioView)
    assert view.symbol == "AAPL"
    assert view.expected_return_annualized == pytest.approx(0.09)
    assert view.confidence == pytest.approx(0.55)


def test_parse_view_response_raises_without_tool_call():
    with pytest.raises(ValueError):
        parse_view_response(FakeMessage(content=[]))


def test_request_portfolio_view_wires_model_and_forced_tool_choice():
    payload = {
        "symbol": "XLF",
        "asset_class": "etf",
        "expected_return_annualized": 0.05,
        "confidence": 0.4,
        "rationale": "Rate cut cycle likely to support financials moderately.",
    }
    response = FakeMessage(content=[FakeToolUseBlock(input=payload)])
    client = FakeAnthropicClient(response)

    view = request_portfolio_view(client, "XLF", "etf", ["Fed signals rate cuts"], sector="Financials")

    assert view.symbol == "XLF"
    call_kwargs = client.messages.last_call_kwargs
    assert call_kwargs["model"] == "claude-sonnet-5"
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "record_portfolio_view"}
    assert call_kwargs["tools"][0]["name"] == "record_portfolio_view"
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
