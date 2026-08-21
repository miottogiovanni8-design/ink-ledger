"""Shared recap builder for both the weekly scheduled job and the on-demand
chat command — same snapshot, same narrative generator, different trigger."""

import json
from datetime import date
from typing import Any, Dict, Optional

RECAP_MODEL = "claude-opus-5"

NARRATIVE_SYSTEM_PROMPT = """You are writing a weekly recap narrative for an AI-augmented \
Black-Litterman investment desk, read by a technical recruiter evaluating the project's \
engineering and reasoning quality. Write 3-5 sentences, plain and honest — cite the \
concrete numbers given (expected return, volatility, Sharpe, VaR/CVaR for the active \
risk profile), do not editorialize or hype. If performance was poor, say so plainly and \
note what the drawdown circuit breaker did to contain it, if anything."""


def period_label(start: date, end: date) -> str:
    return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"


def generate_narrative(client: Any, snapshot: Dict[str, Any], model: str = RECAP_MODEL) -> str:
    active_profile = snapshot.get("active_risk_profile", "balanced")
    active_scenario = snapshot.get("scenarios", {}).get(active_profile, {})

    user_content = (
        "Performance stats:\n"
        + json.dumps(snapshot["performance_stats"], indent=2)
        + f"\n\nActive risk profile: {active_profile}\n"
        + "Active scenario metrics:\n"
        + json.dumps(
            {k: v for k, v in active_scenario.items() if k not in ("weights",)},
            indent=2,
        )
        + "\n\nRecent investment committee notes:\n"
        + "\n".join(
            f"- {n['symbol']} (expected return {n['expected_return_annualized']:.1%}, "
            f"confidence {n['confidence']:.2f}): {n['rationale']}"
            for n in snapshot["investment_committee_notes"][:10]
        )
    )
    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=NARRATIVE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("recap narrative response did not include a text block")


def build_recap(
    snapshot: Dict[str, Any],
    start: date,
    end: date,
    client: Optional[Any] = None,
    model: str = RECAP_MODEL,
) -> Dict[str, Any]:
    label = period_label(start, end)
    narrative = generate_narrative(client, snapshot, model) if client is not None else None
    return {"period_label": label, "narrative": narrative, "snapshot": snapshot}
