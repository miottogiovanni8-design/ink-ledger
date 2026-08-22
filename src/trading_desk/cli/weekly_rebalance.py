"""Entrypoint: `python -m trading_desk.cli.weekly_rebalance [--profile conservative|balanced|aggressive]`

The full weekly pipeline: fetch prices + fundamentals, compute the CAPM
equilibrium prior, get a Claude view for every asset in the universe, blend
via Black-Litterman, optimize all three risk-profile scenarios, execute one
of them (unless the drawdown breaker is tripped), and write the dashboard
snapshot. Runs two ways:

- Scheduled (GitHub Actions, weekly): executes `settings.active_risk_profile`.
- On-demand (`--profile`, run from a chat request to switch risk profile
  immediately rather than waiting for next week's cron): executes the
  requested profile and makes it the new active one going forward.

Either way, all three scenarios are recomputed and persisted every run —
switching which one executes doesn't change what gets calculated, only
which weights reach the broker.
"""

import argparse
import json
import logging
from typing import Optional

import anthropic
import httpx
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from sqlalchemy import select

from trading_desk.config import DEFAULT_FACTOR_MAP, DEFAULT_SECTOR_MAP, settings
from trading_desk.data.fundamentals import dollar_volume_proxy_weights, fetch_market_caps
from trading_desk.data.market_data import fetch_price_panel, fetch_volume_panel
from trading_desk.data.news import fetch_alphavantage_sentiment, fetch_finnhub_general_news, fetch_finnhub_headlines
from trading_desk.engine.black_litterman import blend_views, compute_covariance, compute_market_prior, optimize_portfolio
from trading_desk.engine.portfolio_risk import evaluate_scenario, exposure_by_tag, historical_var_cvar
from trading_desk.engine.schemas import PortfolioView
from trading_desk.engine.views import request_portfolio_view
from trading_desk.execution.broker import get_account_equity, get_open_positions
from trading_desk.execution.rebalance import compute_rebalance_trades, positions_market_value, submit_rebalance_trades
from trading_desk.persistence.db import get_session, init_db
from trading_desk.persistence.models import BaselineAllocation, RebalanceEvent, Transaction, ViewRecord
from trading_desk.persistence.queries import peak_equity
from trading_desk.reporting.snapshot import build_snapshot, write_snapshot
from trading_desk.risk.circuit_breakers import drawdown_breaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("weekly_rebalance")

RISK_PROFILES = ["conservative", "balanced", "aggressive"]


def _asset_class_for(symbol: str) -> str:
    return "equity" if symbol in settings.equity_universe else "etf"


def _sector_or_factor(symbol: str) -> str:
    return DEFAULT_SECTOR_MAP.get(symbol) or DEFAULT_FACTOR_MAP.get(symbol, "")


def gather_views(anthropic_client, http_client, universe, macro_headlines: Optional[list] = None) -> list:
    views: list[PortfolioView] = []
    for symbol in universe:
        headlines = []
        sentiment = None
        is_equity = _asset_class_for(symbol) == "equity"
        if is_equity and settings.finnhub_api_key:
            try:
                headlines = fetch_finnhub_headlines(symbol, settings.finnhub_api_key, http_client)[:3]
            except httpx.HTTPError as exc:
                logger.warning("%s: finnhub fetch failed: %s", symbol, exc)
        if is_equity and settings.alphavantage_api_key:
            try:
                sentiment = fetch_alphavantage_sentiment(symbol, settings.alphavantage_api_key, http_client)
            except httpx.HTTPError as exc:
                logger.warning("%s: alphavantage fetch failed: %s", symbol, exc)
        view = request_portfolio_view(
            anthropic_client,
            symbol,
            _asset_class_for(symbol),
            headlines,
            sector=_sector_or_factor(symbol),
            sentiment=sentiment,
            macro_headlines=macro_headlines,
        )
        views.append(view)
    return views


def run_weekly_rebalance(risk_profile_override: Optional[str] = None) -> None:
    active_profile = risk_profile_override or settings.active_risk_profile
    init_db(settings.db_path)

    universe = settings.equity_universe + settings.etf_universe
    stock_client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    trading_client = TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key, paper=settings.alpaca_paper)
    anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    http_client = httpx.Client(timeout=10.0)

    full_symbols = universe + [settings.benchmark_symbol]
    price_panel = fetch_price_panel(full_symbols, stock_client, lookback_bars=settings.lookback_days)
    universe_prices = price_panel[universe]
    market_prices = price_panel[settings.benchmark_symbol]
    daily_returns = universe_prices.pct_change().dropna()

    cov_matrix = compute_covariance(universe_prices)

    equity_caps = fetch_market_caps(settings.equity_universe, settings.finnhub_api_key, http_client)
    volume_panel = fetch_volume_panel(settings.etf_universe, stock_client, lookback_bars=settings.lookback_days)
    etf_proxy_weights = dollar_volume_proxy_weights(price_panel[settings.etf_universe], volume_panel)
    market_caps = {**equity_caps, **etf_proxy_weights}

    prior = compute_market_prior(market_prices, market_caps, cov_matrix)

    macro_headlines = []
    if settings.finnhub_api_key:
        try:
            macro_headlines = fetch_finnhub_general_news(settings.finnhub_api_key, http_client)[:5]
        except httpx.HTTPError as exc:
            logger.warning("macro headlines fetch failed: %s", exc)

    views = gather_views(anthropic_client, http_client, universe, macro_headlines)

    with get_session(settings.db_path) as session:
        view_records = []
        for view in views:
            vr = ViewRecord(
                symbol=view.symbol,
                asset_class=view.asset_class,
                expected_return_annualized=view.expected_return_annualized,
                confidence=view.confidence,
                rationale=view.rationale,
                key_signals=json.dumps(view.key_signals),
            )
            session.add(vr)
            view_records.append(vr)

        views_by_symbol = {v.symbol: v for v in views}
        latest_prices = price_panel.iloc[-1].to_dict()

        posterior_returns, posterior_cov = blend_views(cov_matrix, prior, views)

        scenarios = {}
        for profile in RISK_PROFILES:
            weights = optimize_portfolio(posterior_returns, posterior_cov, profile, settings.max_weight_pct)
            ret, vol, sharpe = evaluate_scenario(weights, posterior_returns, posterior_cov)
            var_95, cvar_95 = historical_var_cvar(weights, daily_returns, settings.var_confidence_level)
            scenarios[profile] = {
                "weights": weights,
                "expected_return": ret,
                "volatility": vol,
                "sharpe": sharpe,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "sector_exposure": exposure_by_tag(weights, DEFAULT_SECTOR_MAP),
                "factor_exposure": exposure_by_tag(weights, DEFAULT_FACTOR_MAP),
            }

        existing_baseline = session.execute(select(BaselineAllocation).limit(1)).scalars().first()
        if existing_baseline is None:
            session.add(BaselineAllocation(weights_json=json.dumps(scenarios[active_profile]["weights"])))
            logger.info("froze buy-and-hold baseline at this run's %s weights", active_profile)

        current_equity = get_account_equity(trading_client)
        current_peak = peak_equity(session, fallback=current_equity)
        gate = drawdown_breaker(current_equity, current_peak, settings.max_drawdown_circuit_breaker_pct)

        rebalance_event = RebalanceEvent(
            active_risk_profile=active_profile,
            scenarios_json=json.dumps(scenarios),
            prior_returns_json=json.dumps(prior.to_dict()),
            posterior_returns_json=json.dumps(posterior_returns.to_dict()),
            latest_prices_json=json.dumps(latest_prices),
            executed=False,
        )
        session.add(rebalance_event)
        session.flush()
        for vr in view_records:
            vr.rebalance_event_id = rebalance_event.id

        if gate.rebalancing_allowed:
            active_weights = scenarios[active_profile]["weights"]
            current_positions = positions_market_value(get_open_positions(trading_client))
            trades = compute_rebalance_trades(active_weights, current_positions, current_equity, settings.min_trade_usd)
            submit_rebalance_trades(trading_client, trades)
            rebalance_event.executed = True
            logger.info("executed rebalance: %d order(s)", len(trades))

            for symbol, delta_usd in trades.items():
                view = views_by_symbol.get(symbol)
                rationale = view.rationale if view else "Rebalance toward target weight (no fresh view this cycle)."
                session.add(
                    Transaction(
                        rebalance_event_id=rebalance_event.id,
                        symbol=symbol,
                        asset_class=_asset_class_for(symbol),
                        side="buy" if delta_usd > 0 else "sell",
                        notional_usd=abs(delta_usd),
                        price=float(latest_prices.get(symbol, 0.0)),
                        rationale=rationale,
                    )
                )
        else:
            rebalance_event.skip_reason = gate.reason
            logger.warning("rebalance skipped: %s", gate.reason)

        snapshot = build_snapshot(session)

    write_snapshot(snapshot, settings.snapshot_path)
    logger.info("weekly rebalance complete, snapshot written to %s", settings.snapshot_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=RISK_PROFILES, default=None, help="Override the active risk profile for this run only (persists as the new active profile going forward).")
    args = parser.parse_args()
    run_weekly_rebalance(risk_profile_override=args.profile)


if __name__ == "__main__":
    main()
