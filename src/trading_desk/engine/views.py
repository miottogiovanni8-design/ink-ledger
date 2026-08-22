"""Claude analyst call: for one asset, produces an annualized expected-return
view with a confidence level — never a trade or a weight. Portfolio
construction is entirely deterministic (Black-Litterman + EfficientFrontier
in engine/black_litterman.py); this module is the only place an LLM's
judgment enters the pipeline, and it enters as a return expectation, not an
instruction.
"""

from typing import Any, Dict, List, Optional

from trading_desk.engine.schemas import PORTFOLIO_VIEW_TOOL, AssetClass, PortfolioView

VIEW_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a fundamental/macro research analyst at an \
investment desk. For the asset described in the user message, form a \
12-month annualized expected total return view, and how confident you are \
in it.

Rules you must respect:
- Base the view on fundamentals, recent news/headlines, and macro context — \
  never on short-term technical price patterns.
- expected_return_annualized is a total return expectation (e.g. 0.08 for \
  +8%/year), not a price target and not a probability.
- confidence reflects how much conviction you actually have — most views \
  should be moderate (0.3-0.6); reserve high confidence (>0.8) for cases \
  with clear, specific supporting evidence. A view with weak or generic \
  reasoning should carry low confidence, not a confident-sounding rationale.
- rationale must cite the specific evidence used (a headline, a fundamental \
  metric, a macro factor) — it is stored verbatim in the investment \
  committee notes and shown to a human reviewing the process.
- You must call the record_portfolio_view tool exactly once."""


def build_user_content(
    symbol: str,
    asset_class: AssetClass,
    headlines: List[str],
    sector: str = "",
    sentiment: Optional[Dict[str, Any]] = None,
    macro_headlines: Optional[List[str]] = None,
) -> str:
    lines = [
        f"Asset: {symbol} ({asset_class})",
    ]
    if sector:
        lines.append(f"Sector/factor: {sector}")
    lines.append(f"Recent headlines: {' | '.join(headlines) if headlines else 'none'}")
    if sentiment:
        lines.append(
            f"Analyst sentiment (Alpha Vantage): {sentiment.get('sentiment_label', 'Neutral')} "
            f"(score {sentiment.get('sentiment_score', 0.0):.2f}, "
            f"relevance {sentiment.get('relevance_score', 0.0):.2f})"
        )
    if macro_headlines:
        lines.append(f"Broader market/macro headlines: {' | '.join(macro_headlines)}")
    return "\n".join(lines)


def parse_view_response(response: Any) -> PortfolioView:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "record_portfolio_view":
            return PortfolioView(**block.input)
    raise ValueError("view response did not include a record_portfolio_view tool call")


def request_portfolio_view(
    client: Any,
    symbol: str,
    asset_class: AssetClass,
    headlines: List[str],
    sector: str = "",
    sentiment: Optional[Dict[str, Any]] = None,
    macro_headlines: Optional[List[str]] = None,
    model: str = VIEW_MODEL,
    usage_log: Optional[List[Any]] = None,
) -> PortfolioView:
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[PORTFOLIO_VIEW_TOOL],
        tool_choice={"type": "tool", "name": "record_portfolio_view"},
        messages=[{
            "role": "user",
            "content": build_user_content(symbol, asset_class, headlines, sector, sentiment, macro_headlines),
        }],
    )
    if usage_log is not None:
        usage_log.append(response.usage)
    return parse_view_response(response)
