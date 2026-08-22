"""Entrypoint: `python -m trading_desk.cli.daily_mark`

Cheap, no-LLM daily job: reads account equity/cash from Alpaca plus the
S&P 500 (SPY) close, and appends one EquitySnapshot row. Runs every market
day so the dashboard's history scrubber has daily granularity even though
rebalancing itself is weekly, and so the equity-vs-benchmark chart has a
matching daily benchmark series. Also marks the frozen buy-and-hold baseline
(see BaselineAllocation) at that day's prices, if one has been set — a
zero-extra-account control arm computed purely from price data already
fetched here, never a separately executed portfolio. Called by GitHub
Actions — see .github/workflows/daily-mark.yml.
"""

import json
import logging
from typing import Dict, Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from sqlalchemy import select

from trading_desk.config import settings
from trading_desk.data.market_data import fetch_bars, fetch_price_panel
from trading_desk.execution.broker import get_account_cash, get_account_equity
from trading_desk.persistence.db import get_session, init_db
from trading_desk.persistence.models import BaselineAllocation, EquitySnapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("daily_mark")


def fetch_benchmark_price(stock_client: StockHistoricalDataClient) -> float:
    bars = fetch_bars(settings.benchmark_symbol, stock_client, lookback_bars=1)
    return float(bars["close"].iloc[-1])


def compute_baseline_index(weights: Dict[str, float], prices: Dict[str, float]) -> Optional[float]:
    """Sum(weight * price) for the frozen basket — a raw index level, not a
    currency amount. It gets indexed to the portfolio's starting equity at
    snapshot-build time, the same treatment as benchmark_price. Returns None
    if any frozen symbol is missing a price rather than silently
    understating the basket."""
    if not weights:
        return None
    total = 0.0
    for symbol, weight in weights.items():
        price = prices.get(symbol)
        if price is None:
            return None
        total += weight * price
    return total


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
        baseline = session.execute(
            select(BaselineAllocation).order_by(BaselineAllocation.frozen_at).limit(1)
        ).scalars().first()
        baseline_weights = json.loads(baseline.weights_json) if baseline else None

    baseline_index_raw = None
    if baseline_weights:
        try:
            price_panel = fetch_price_panel(list(baseline_weights.keys()), stock_client, lookback_bars=1)
            latest_prices = price_panel.iloc[-1].to_dict()
            baseline_index_raw = compute_baseline_index(baseline_weights, latest_prices)
        except Exception as exc:  # same nice-to-have treatment as the benchmark
            logger.warning("could not fetch baseline basket prices: %s", exc)

    with get_session(settings.db_path) as session:
        session.add(
            EquitySnapshot(
                equity_eur=equity,
                cash_eur=cash,
                benchmark_price=benchmark_price,
                baseline_index_raw=baseline_index_raw,
            )
        )

    logger.info(
        "marked equity=%.2f cash=%.2f benchmark=%s baseline_index=%s",
        equity, cash, benchmark_price, baseline_index_raw,
    )


if __name__ == "__main__":
    run_daily_mark()
