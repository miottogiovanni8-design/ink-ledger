"""Entrypoint: `python -m trading_desk.cli.daily_mark`

Cheap, no-LLM daily job: reads account equity/cash from Alpaca plus the
S&P 500 (SPY) close, and appends one EquitySnapshot row. Runs every market
day so the dashboard's history scrubber has daily granularity even though
rebalancing itself is weekly, and so the equity-vs-benchmark chart has a
matching daily benchmark series. Called by GitHub Actions — see
.github/workflows/daily-mark.yml.
"""

import logging

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from trading_desk.config import settings
from trading_desk.data.market_data import fetch_bars
from trading_desk.execution.broker import get_account_cash, get_account_equity
from trading_desk.persistence.db import get_session, init_db
from trading_desk.persistence.models import EquitySnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("daily_mark")


def fetch_benchmark_price(stock_client: StockHistoricalDataClient) -> float:
    bars = fetch_bars(settings.benchmark_symbol, stock_client, lookback_bars=1)
    return float(bars["close"].iloc[-1])


def run_daily_mark() -> None:
    init_db(settings.db_path)
    trading_client = TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key, paper=settings.alpaca_paper)
    stock_client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)

    equity = get_account_equity(trading_client)
    cash = get_account_cash(trading_client)

    try:
        benchmark_price = fetch_benchmark_price(stock_client)
    except Exception as exc:  # benchmark is a nice-to-have, never block the mark on it
        logger.warning("could not fetch benchmark price: %s", exc)
        benchmark_price = None

    with get_session(settings.db_path) as session:
        session.add(EquitySnapshot(equity_eur=equity, cash_eur=cash, benchmark_price=benchmark_price))

    logger.info("marked equity=%.2f cash=%.2f benchmark=%s", equity, cash, benchmark_price)


if __name__ == "__main__":
    run_daily_mark()
