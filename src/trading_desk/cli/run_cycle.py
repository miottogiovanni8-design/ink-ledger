"""Entrypoint: `python -m trading_desk.cli.run_cycle --universe equity|crypto`

One full cycle: reconcile fills since last cycle, screen the watchlist,
run the risk-gated decision engine on candidates, execute clamped decisions
as bracket orders, snapshot equity, and write the dashboard JSON. Called by
the GitHub Actions cron — see .github/workflows/trading-cycle-*.yml.
"""

import argparse
import json
import logging
from datetime import datetime, timezone

import anthropic
import httpx
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from trading_desk.config import settings
from trading_desk.data.market_data import build_indicator_snapshot, fetch_bars
from trading_desk.data.news import fetch_finnhub_headlines
from trading_desk.engine.decision import request_trade_decision
from trading_desk.engine.prefilter import is_candidate
from trading_desk.engine.schemas import PortfolioState
from trading_desk.execution.broker import get_account_cash, get_account_equity, get_daily_pnl, get_open_positions, submit_bracket_order
from trading_desk.execution.reconcile import reconcile_open_trades
from trading_desk.persistence.db import get_session, init_db
from trading_desk.persistence.models import Decision, EquitySnapshot, Trade
from trading_desk.persistence.queries import open_positions_count, peak_equity, todays_opened_notional_eur
from trading_desk.reporting.snapshot import build_snapshot, write_snapshot
from trading_desk.risk.circuit_breakers import evaluate_all_gates
from trading_desk.risk.sizing import clamp_decision_to_risk_limits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_cycle")


def run_cycle(universe: str) -> None:
    init_db(settings.db_path)

    trading_client = TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key, paper=settings.alpaca_paper)
    stock_data_client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    crypto_data_client = CryptoHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    http_client = httpx.Client(timeout=10.0)

    watchlist = settings.equity_watchlist if universe == "equity" else settings.crypto_watchlist

    with get_session(settings.db_path) as session:
        _reconcile(session, trading_client)

        equity_eur = get_account_equity(trading_client)
        daily_pnl_eur = get_daily_pnl(trading_client)
        current_open_positions = open_positions_count(session)
        current_peak_equity = peak_equity(session, fallback=equity_eur)
        budget_spent = todays_opened_notional_eur(session)

        for symbol in watchlist:
            _process_symbol(
                session=session,
                symbol=symbol,
                universe=universe,
                trading_client=trading_client,
                stock_data_client=stock_data_client,
                crypto_data_client=crypto_data_client,
                anthropic_client=anthropic_client,
                http_client=http_client,
                equity_eur=equity_eur,
                daily_pnl_eur=daily_pnl_eur,
                open_positions=current_open_positions,
                peak_equity_eur=current_peak_equity,
                budget_spent=budget_spent,
            )
            # re-read after each symbol so budget/position gates reflect same-cycle trades
            current_open_positions = open_positions_count(session)
            budget_spent = todays_opened_notional_eur(session)

        session.add(
            EquitySnapshot(
                equity_eur=get_account_equity(trading_client),
                cash_eur=get_account_cash(trading_client),
                open_positions_count=open_positions_count(session),
            )
        )
        session.flush()

        snapshot = build_snapshot(session)

    write_snapshot(snapshot, settings.snapshot_path)
    logger.info("cycle complete, snapshot written to %s", settings.snapshot_path)


def _reconcile(session, trading_client: TradingClient) -> None:
    open_trades = session.query(Trade).filter_by(status="open").all()
    if not open_trades:
        return
    broker_open_symbols = {p.symbol for p in get_open_positions(trading_client)}
    earliest_open = min(t.opened_at for t in open_trades)
    closed_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=earliest_open)
    )
    closed_count = reconcile_open_trades(open_trades, broker_open_symbols, closed_orders)
    if closed_count:
        logger.info("reconciled %d closed trade(s)", closed_count)


def _process_symbol(
    session,
    symbol: str,
    universe: str,
    trading_client: TradingClient,
    stock_data_client: StockHistoricalDataClient,
    crypto_data_client: CryptoHistoricalDataClient,
    anthropic_client: anthropic.Anthropic,
    http_client: httpx.Client,
    equity_eur: float,
    daily_pnl_eur: float,
    open_positions: int,
    peak_equity_eur: float,
    budget_spent: float,
) -> None:
    bars = fetch_bars(symbol, universe, stock_data_client, crypto_data_client, timeframe=TimeFrame.Hour)
    if len(bars) < 20:
        logger.info("%s: not enough bars yet, skipping", symbol)
        return

    headlines = []
    if universe == "equity" and settings.finnhub_api_key:
        try:
            headlines = fetch_finnhub_headlines(symbol, settings.finnhub_api_key, http_client)
        except httpx.HTTPError as exc:
            logger.warning("%s: finnhub fetch failed: %s", symbol, exc)

    snapshot = build_indicator_snapshot(
        symbol, universe, bars, headlines=headlines[:3], has_fresh_headline=bool(headlines)
    )

    candidate, reasons = is_candidate(snapshot)
    if not candidate:
        return

    gate = evaluate_all_gates(
        daily_pnl_eur=daily_pnl_eur,
        daily_budget_eur=settings.daily_budget_eur,
        daily_loss_threshold_pct=settings.daily_loss_circuit_breaker_pct,
        current_equity=equity_eur,
        peak_equity=peak_equity_eur,
        drawdown_threshold_pct=settings.max_drawdown_circuit_breaker_pct,
        open_positions=open_positions,
        max_positions=settings.max_concurrent_positions,
    )
    if not gate.entries_allowed:
        session.add(
            Decision(
                symbol=symbol,
                asset_class=universe,
                direction="hold",
                confidence=0.0,
                rationale=f"Skipped by risk layer: {gate.reason}",
                key_signals=json.dumps(reasons),
                risk_flags=json.dumps([gate.reason]),
                skipped_by_risk_layer=True,
                skip_reason=gate.reason,
            )
        )
        logger.info("%s: risk gate blocked entries (%s)", symbol, gate.reason)
        return

    portfolio = PortfolioState(
        equity_eur=equity_eur,
        daily_budget_eur=settings.daily_budget_eur,
        daily_budget_spent_eur=budget_spent,
        open_positions=open_positions,
        max_positions=settings.max_concurrent_positions,
        daily_pnl_eur=daily_pnl_eur,
    )
    decision = request_trade_decision(anthropic_client, portfolio, snapshot, reasons)

    decision_row = Decision(
        symbol=decision.symbol,
        asset_class=decision.asset_class,
        direction=decision.direction,
        confidence=decision.confidence,
        rationale=decision.rationale,
        key_signals=json.dumps(decision.key_signals),
        risk_flags=json.dumps(decision.risk_flags),
        skipped_by_risk_layer=False,
    )
    session.add(decision_row)
    session.flush()

    if decision.direction == "hold":
        logger.info("%s: decision=hold (confidence %.2f)", symbol, decision.confidence)
        return

    remaining_budget = settings.daily_budget_eur - budget_spent
    clamped = clamp_decision_to_risk_limits(
        decision,
        entry_price=snapshot.price,
        atr=snapshot.atr_14,
        daily_budget_eur=settings.daily_budget_eur,
        remaining_budget_eur=remaining_budget,
        risk_pct_per_trade=settings.risk_pct_per_trade,
        max_position_pct_of_budget=settings.max_position_pct_of_budget,
        atr_multiplier=settings.stop_loss_atr_multiplier,
        rr_ratio=settings.take_profit_rr_ratio,
    )

    if clamped.position_size_usd <= 0:
        logger.info("%s: clamped size is zero, not executing", symbol)
        return

    order = submit_bracket_order(
        trading_client,
        symbol=symbol,
        direction=clamped.direction,
        notional_usd=clamped.position_size_usd,
        stop_loss_price=clamped.stop_loss_price,
        take_profit_price=clamped.take_profit_price,
    )

    session.add(
        Trade(
            decision_id=decision_row.id,
            symbol=symbol,
            asset_class=universe,
            direction=clamped.direction,
            entry_price=snapshot.price,
            stop_loss_price=clamped.stop_loss_price,
            take_profit_price=clamped.take_profit_price,
            size_eur=clamped.position_size_usd,
            status="open",
            alpaca_order_id=str(order.id),
        )
    )
    logger.info("%s: executed %s, size %.2f EUR", symbol, clamped.direction, clamped.position_size_usd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["equity", "crypto"], required=True)
    args = parser.parse_args()
    run_cycle(args.universe)


if __name__ == "__main__":
    main()
