"""Entrypoint: `python -m trading_desk.cli.daily_mark`

Cheap, no-LLM daily job: reads account equity/cash from Alpaca and appends
one EquitySnapshot row. Runs every market day so the dashboard's history
scrubber has daily granularity even though rebalancing itself is weekly.
Called by GitHub Actions — see .github/workflows/daily-mark.yml.
"""

import logging

from alpaca.trading.client import TradingClient

from trading_desk.config import settings
from trading_desk.execution.broker import get_account_cash, get_account_equity
from trading_desk.persistence.db import get_session, init_db
from trading_desk.persistence.models import EquitySnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("daily_mark")


def run_daily_mark() -> None:
    init_db(settings.db_path)
    trading_client = TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key, paper=settings.alpaca_paper)

    equity = get_account_equity(trading_client)
    cash = get_account_cash(trading_client)

    with get_session(settings.db_path) as session:
        session.add(EquitySnapshot(equity_eur=equity, cash_eur=cash))

    logger.info("marked equity=%.2f cash=%.2f", equity, cash)


if __name__ == "__main__":
    run_daily_mark()
