"""Entrypoint: `python -m trading_desk.cli.weekly_recap`

Low-frequency, delay-tolerant job (unlike run_cycle): reads the current DB
state, fetches SPY as an equity benchmark, generates the Claude Opus 5
narrative, renders + sends the email, and republishes the dashboard data.
Intended to run from a Claude Code scheduled routine, not GitHub Actions —
see the plan's scheduling section for why the two jobs are split.
"""

import logging
from datetime import date, timedelta

import anthropic
import resend
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trading_desk.config import settings
from trading_desk.metrics.stats import returns_from_equity_curve
from trading_desk.persistence.db import get_session, init_db
from trading_desk.reporting.email_sender import render_recap_html, send_recap_email
from trading_desk.reporting.recap import build_recap
from trading_desk.reporting.snapshot import build_snapshot, write_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("weekly_recap")

BENCHMARK_SYMBOL = "SPY"


def fetch_benchmark_returns(stock_client: StockHistoricalDataClient, start: date, end: date):
    request = StockBarsRequest(
        symbol_or_symbols=BENCHMARK_SYMBOL,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bar_set = stock_client.get_stock_bars(request)
    df = bar_set.df
    if hasattr(df.index, "get_level_values"):
        closes = df.xs(BENCHMARK_SYMBOL, level="symbol")["close"].tolist()
    else:
        closes = df["close"].tolist()
    return returns_from_equity_curve(closes)


def run_weekly_recap() -> dict:
    init_db(settings.db_path)

    end = date.today()
    start = end - timedelta(days=7)

    stock_client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    try:
        benchmark_returns = fetch_benchmark_returns(stock_client, start, end)
    except Exception as exc:  # benchmark is a nice-to-have, never block the recap on it
        logger.warning("could not fetch benchmark returns: %s", exc)
        benchmark_returns = None

    with get_session(settings.db_path) as session:
        snapshot = build_snapshot(session, benchmark_returns=benchmark_returns)

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
            subject=f"Trading Desk Recap — {recap['period_label']}",
            html=html,
        )
        logger.info("weekly recap email sent to %s", settings.email_to)
    else:
        logger.info("email not configured, skipping send")

    return recap


if __name__ == "__main__":
    run_weekly_recap()
