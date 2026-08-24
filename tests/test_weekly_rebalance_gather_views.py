from dataclasses import dataclass, field
from typing import Any, Dict, List

from trading_desk.cli.weekly_rebalance import gather_views
from trading_desk.config import settings


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

    def create(self, **kwargs):
        return self._response


class FakeAnthropicClient:
    def __init__(self, response: FakeMessage):
        self.messages = FakeMessagesAPI(response)


def test_gather_views_overrides_a_corrupted_symbol_with_the_one_actually_requested(monkeypatch):
    """Reproduces a live production crash: Claude echoed back a mangled
    symbol ('">V instead of "V", almost certainly a scraped headline's
    stray markup bleeding into the tool call) and it reached PyPortfolioOpt
    unmodified, which rejected it as "not in the universe". gather_views
    already knows which symbol it asked about, so it must never trust the
    echoed one over that."""
    monkeypatch.setattr(settings, "equity_universe", ["V"])
    monkeypatch.setattr(settings, "etf_universe", [])
    monkeypatch.setattr(settings, "finnhub_api_key", "")
    monkeypatch.setattr(settings, "alphavantage_api_key", "")

    payload = {
        "symbol": "'\">V",
        "asset_class": "equity",
        "expected_return_annualized": 0.05,
        "confidence": 0.4,
        "rationale": "Steady payments volume growth.",
        "rationale_it": "Crescita costante dei volumi di pagamento.",
    }
    response = FakeMessage(content=[FakeToolUseBlock(input=payload)])
    client = FakeAnthropicClient(response)

    views, sources_by_symbol = gather_views(client, http_client=None, universe=["V"])

    assert len(views) == 1
    assert views[0].symbol == "V"
    assert sources_by_symbol == {"V": []}
