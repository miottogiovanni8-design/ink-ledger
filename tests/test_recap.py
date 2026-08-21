from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List

from trading_desk.reporting.recap import build_recap, generate_narrative, period_label


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


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


def make_snapshot():
    return {
        "performance_stats": {"sharpe_ratio": 1.2, "max_drawdown": 0.05, "total_return": 0.06},
        "active_risk_profile": "balanced",
        "scenarios": {
            "balanced": {
                "weights": {"AAPL": 0.15},
                "expected_return": 0.09,
                "volatility": 0.14,
                "sharpe": 0.64,
                "var_95": 0.02,
                "cvar_95": 0.03,
            }
        },
        "investment_committee_notes": [
            {"symbol": "AAPL", "expected_return_annualized": 0.09, "confidence": 0.55, "rationale": "Strong iPhone cycle."},
        ],
    }


def test_period_label_formats_range():
    label = period_label(date(2026, 8, 14), date(2026, 8, 21))
    assert label == "Aug 14 - Aug 21, 2026"


def test_generate_narrative_extracts_text_block():
    response = FakeMessage(content=[FakeTextBlock(text="Modest gains this week, Sharpe of 1.2.")])
    client = FakeAnthropicClient(response)

    narrative = generate_narrative(client, make_snapshot())

    assert client.messages.last_call_kwargs["model"] == "claude-opus-5"
    assert narrative == "Modest gains this week, Sharpe of 1.2."
    user_content = client.messages.last_call_kwargs["messages"][0]["content"]
    assert "balanced" in user_content
    assert "Strong iPhone cycle" in user_content
    assert '"weights"' not in user_content  # weights excluded from the narrative prompt


def test_build_recap_without_client_skips_narrative():
    recap = build_recap(make_snapshot(), date(2026, 8, 14), date(2026, 8, 21))
    assert recap["narrative"] is None
    assert recap["period_label"] == "Aug 14 - Aug 21, 2026"


def test_build_recap_with_client_includes_narrative():
    response = FakeMessage(content=[FakeTextBlock(text="Solid week.")])
    client = FakeAnthropicClient(response)

    recap = build_recap(make_snapshot(), date(2026, 8, 14), date(2026, 8, 21), client=client)

    assert recap["narrative"] == "Solid week."
