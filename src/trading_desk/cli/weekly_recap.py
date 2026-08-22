"""Entrypoint: `python -m trading_desk.cli.weekly_recap`

Low-frequency, delay-tolerant job: reads the current DB state (already
populated by weekly_rebalance.py, including all three risk-profile
scenarios), generates the Claude Opus 5 narrative, sends the email, and
rewrites the dashboard snapshot JSON. Plain headless Python — the Anthropic
and Resend SDKs are regular API calls, so this runs as a GitHub Actions job
like weekly_rebalance.py and daily_mark.py, no Claude Code session needed.
"""

import logging
from datetime import date, timedelta

import anthropic
import resend

from trading_desk.config import settings
from trading_desk.persistence.db import get_session, init_db
from trading_desk.reporting.email_sender import render_recap_html, send_recap_email
from trading_desk.reporting.recap import build_recap
from trading_desk.reporting.snapshot import build_snapshot, write_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("weekly_recap")


def run_weekly_recap() -> dict:
    init_db(settings.db_path)

    end = date.today()
    start = end - timedelta(days=7)

    with get_session(settings.db_path) as session:
        snapshot = build_snapshot(session)

    write_snapshot(snapshot, settings.snapshot_path)

    anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    recap = build_recap(snapshot, start, end, client=anthropic_client)

    html = render_recap_html(recap["period_label"], snapshot, narrative=recap["narrative"] or "")

    if settings.resend_api_key and settings.email_from and settings.email_to:
        resend.api_key = settings.resend_api_key
        send_recap_email(
            resend.Emails,
            settings.email_from,
            settings.email_to,
            subject=f"Ink Ledger Recap — {recap['period_label']}",
            html=html,
        )
        logger.info("weekly recap email sent to %s", settings.email_to)
    else:
        logger.info("email not configured, skipping send")

    return recap


if __name__ == "__main__":
    run_weekly_recap()
