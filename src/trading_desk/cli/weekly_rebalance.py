"""Entrypoint: `python -m trading_desk.cli.weekly_rebalance`

The full weekly pipeline: fetch prices + fundamentals, compute the CAPM
equilibrium prior, get a Claude view for every asset in the universe, blend
via Black-Litterman, optimize all three risk-profile scenarios, execute the
active one (unless the drawdown breaker is tripped), and write the
dashboard snapshot. Called by GitHub Actions — see
.github/workflows/weekly-rebalance.yml.
"""

import json
import logging

import anthropic
import httpx
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from trading_desk.config import DEFAULT_FACTOR_MAP, DEFAULT_SECTOR_MAP, settings
from trading_desk.data.fundamentals import dollar_volume_proxy_weights, fetch_market_caps
from trading_desk.data.market_data import fetch_price_panel, fetch_volume_panel
from trading_desk.data.news import fetch_finnhub_headlines
from trading_desk.engine.black_litterman import blend_views, compute_covariance, compute_market_prior, optimize_portfolio
from trading_desk.engine.portfolio_risk import evaluate_scenario, exposure_by_tag, historical_var_cvar
from trading_desk.engine.schemas import PortfolioView
from trading_desk.engine.views import request_portfolio_view
from trading_desk.execution.broker import get_account_equity, get_open_positions
from trading_desk.execution.rebalance import compute_rebalance_trades, positions_market_value, submit_rebalance_trades
from trading_desk.persistence.db import get_session, init_db
from trading_desk.persistence.models import RebalanceEvent, ViewRecord
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


def gather_views(anthropic_client, http_client, universe) -> list:
    views: list[PortfolioView] = []
    for symbol in universe:
        headlines = []
        if _asset_class_for(symbol) == "equity" and settings.finnhub_api_key:
            try:
                headlines = fetch_finnhub_headlines(symbol, settings.finnhub_api_key, http_client)[:3]
            except httpx.HTTPError as exc:
                logger.warning("%s: finnhub fetch failed: %s", symbol, exc)
        view = request_portfolio_view(
            anthropic_client, symbol, _asset_class_for(symbol), headlines, sector=_sector_or_factor(symbol)
        )
        views.append(view)
    return views


def run_weekly_rebalance() -> None:
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

    views = gather_views(anthropic_client, http_client, universe)

    with get_session(settings.db_path) as session:
        for view in views:
            session.add(
                ViewRecord(
                    symbol=view.symbol,
                    asset_class=view.asset_class,
                    expected_return_annualized=view.expected_return_annualized,
                    confidence=view.confidence,
                    rationale=view.rationale,
                    key_signals=json.dumps(view.key_signals),
                )
            )

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

        current_equity = get_account_equity(trading_client)
        current_peak = peak_equity(session, fallback=current_equity)
        gate = drawdown_breaker(current_equity, current_peak, settings.max_drawdown_circuit_breaker_pct)

        rebalance_event = RebalanceEvent(
            active_risk_profile=settings.active_risk_profile,
            scenarios_json=json.dumps(scenarios),
            prior_returns_json=json.dumps(prior.to_dict()),
            posterior_returns_json=json.dumps(posterior_returns.to_dict()),
            executed=False,
        )

        if gate.rebalancing_allowed:
            active_weights = scenarios[settings.active_risk_profile]["weights"]
            current_positions = positions_market_value(get_open_positions(trading_client))
            trades = compute_rebalance_trades(active_weights, current_positions, current_equity, settings.min_trade_usd)
            submit_rebalance_trades(trading_client, trades)
            rebalance_event.executed = True
            logger.info("executed rebalance: %d order(s)", len(trades))
        else:
            rebalance_event.skip_reason = gate.reason
            logger.warning("rebalance skipped: %s", gate.reason)

        session.add(rebalance_event)
        session.flush()

        snapshot = build_snapshot(session)

    write_snapshot(snapshot, settings.snapshot_path)
    logger.info("weekly rebalance complete, snapshot written to %s", settings.snapshot_path)


if __name__ == "__main__":
    run_weekly_rebalance()
