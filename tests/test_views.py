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
    usage: Any = None


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


def test_build_user_content_includes_sentiment_when_provided():
    sentiment = {"sentiment_label": "Somewhat-Bullish", "sentiment_score": 0.32, "relevance_score": 0.85}
    content = build_user_content("AAPL", "equity", [], sentiment=sentiment)
    assert "Somewhat-Bullish" in content
    assert "0.32" in content
    assert "0.85" in content


def test_build_user_content_omits_sentiment_when_absent():
    content = build_user_content("AAPL", "equity", [])
    assert "Analyst sentiment" not in content


def test_build_user_content_includes_macro_headlines_when_provided():
    content = build_user_content("AAPL", "equity", [], macro_headlines=["Fed holds rates steady"])
    assert "Fed holds rates steady" in content
    assert "macro" in content.lower()


def test_parse_view_response_extracts_tool_call():
    payload = {
        "symbol": "AAPL",
        "asset_class": "equity",
        "expected_return_annualized": 0.09,
        "confidence": 0.55,
        "rationale": "Strong iPhone cycle plus services growth.",
        "rationale_it": "Forte ciclo iPhone e crescita dei servizi.",
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
        "rationale_it": "Il ciclo di taglio dei tassi dovrebbe sostenere moderatamente il settore finanziario.",
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


def test_request_portfolio_view_forwards_sentiment_and_macro_headlines():
    payload = {
        "symbol": "XLF",
        "asset_class": "etf",
        "expected_return_annualized": 0.05,
        "confidence": 0.4,
        "rationale": "Rate cut cycle likely to support financials moderately.",
        "rationale_it": "Il ciclo di taglio dei tassi dovrebbe sostenere moderatamente il settore finanziario.",
    }
    response = FakeMessage(content=[FakeToolUseBlock(input=payload)])
    client = FakeAnthropicClient(response)
    sentiment = {"sentiment_label": "Bullish", "sentiment_score": 0.5, "relevance_score": 0.9}

    request_portfolio_view(
        client, "XLF", "etf", [], sector="Financials",
        sentiment=sentiment, macro_headlines=["Fed holds rates steady"],
    )

    content = client.messages.last_call_kwargs["messages"][0]["content"]
    assert "Bullish" in content
    assert "Fed holds rates steady" in content


def test_request_portfolio_view_appends_usage_when_log_provided():
    payload = {
        "symbol": "AAPL",
        "asset_class": "equity",
        "expected_return_annualized": 0.08,
        "confidence": 0.5,
        "rationale": "Steady growth.",
        "rationale_it": "Crescita costante.",
    }
    fake_usage = {"input_tokens": 120, "output_tokens": 80}
    response = FakeMessage(content=[FakeToolUseBlock(input=payload)], usage=fake_usage)
    client = FakeAnthropicClient(response)
    usage_log: List[Any] = []

    request_portfolio_view(client, "AAPL", "equity", [], usage_log=usage_log)

    assert usage_log == [fake_usage]


def test_request_portfolio_view_skips_usage_log_when_not_provided():
    payload = {
        "symbol": "AAPL",
        "asset_class": "equity",
        "expected_return_annualized": 0.08,
        "confidence": 0.5,
        "rationale": "Steady growth.",
        "rationale_it": "Crescita costante.",
    }
    response = FakeMessage(content=[FakeToolUseBlock(input=payload)], usage={"input_tokens": 1})
    client = FakeAnthropicClient(response)

    view = request_portfolio_view(client, "AAPL", "equity", [])

    assert view.symbol == "AAPL"
