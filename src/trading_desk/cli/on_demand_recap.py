"""Entrypoint: `python -m trading_desk.cli.on_demand_recap`

Regenerates the dashboard snapshot from current DB state and prints it to
stdout. Meant to be run when the user asks for a recap in chat: read this
output, then republish the dashboard Artifact with it — no LLM call happens
here, since the chat session is already Claude reasoning over the data live.
"""

import json
import logging

from trading_desk.config import settings
from trading_desk.persistence.db import get_session, init_db
from trading_desk.reporting.snapshot import build_snapshot, write_snapshot

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")


def run_on_demand_recap() -> dict:
    init_db(settings.db_path)
    with get_session(settings.db_path) as session:
        snapshot = build_snapshot(session)
    write_snapshot(snapshot, settings.snapshot_path)
    return snapshot


if __name__ == "__main__":
    print(json.dumps(run_on_demand_recap(), indent=2))
