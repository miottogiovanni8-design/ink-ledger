"""Shared recap builder for both the weekly scheduled job and the on-demand
chat command — same snapshot, same narrative generator, different trigger."""

import json
from datetime import date
from typing import Any, Dict, Optional

RECAP_MODEL = "claude-opus-5"

NARRATIVE_SYSTEM_PROMPT = """You are writing a weekly recap narrative for an AI \
paper-trading desk, read by a technical recruiter evaluating the project's \
engineering and reasoning quality. Write 3-5 sentences, plain and honest — \
cite the concrete numbers given, do not editorialize or hype. If performance \
was poor, say so plainly and note what the risk layer did to contain it."""


def period_label(start: date, end: date) -> str:
    return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"


def generate_narrative(client: Any, snapshot: Dict[str, Any], model: str = RECAP_MODEL) -> str:
    user_content = (
        "Stats:\n"
        + json.dumps(snapshot["stats"], indent=2)
        + "\n\nOpen positions: "
        + str(len(snapshot["positions"]))
        + "\n\nRecent trade journal entries:\n"
        + "\n".join(
            f"- {e['symbol']} {e['direction']} (confidence {e['confidence']:.2f}): {e['rationale']}"
            for e in snapshot["trade_journal"][:10]
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
