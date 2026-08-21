"""Generates a realistic sample dashboard_snapshot.json (schema v2) for the
dashboard mockup, run through the project's own engine functions
(compute_covariance, compute_market_prior, blend_views, optimize_portfolio,
evaluate_scenario, historical_var_cvar, exposure_by_tag) so the displayed
numbers are exactly what the real pipeline would produce for this data —
not hand-typed placeholders. A curated 16-asset subset of the full
29-asset production universe (config.DEFAULT_EQUITY_UNIVERSE +
DEFAULT_ETF_UNIVERSE) is used here to keep the hand-written investment
committee rationale readable; the real engine runs the full universe."""

import json
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from trading_desk.config import DEFAULT_FACTOR_MAP, DEFAULT_NAME_MAP, DEFAULT_SECTOR_MAP  # noqa: E402
from trading_desk.engine.black_litterman import blend_views, compute_covariance, compute_market_prior, optimize_portfolio  # noqa: E402
from trading_desk.engine.portfolio_risk import evaluate_scenario, exposure_by_tag, historical_var_cvar  # noqa: E402
from trading_desk.engine.schemas import PortfolioView  # noqa: E402
from trading_desk.metrics import stats  # noqa: E402
from trading_desk.reporting.holdings import build_holdings_detail  # noqa: E402

random.seed(11)
np.random.seed(11)

EQUITIES = ["AAPL", "NVDA", "JPM", "V", "JNJ", "UNH", "XOM", "PG", "HD", "CAT"]
ETFS = ["XLK", "XLF", "XLV", "MTUM", "VLUE", "USMV"]
UNIVERSE = EQUITIES + ETFS
BENCHMARK = "SPY"

MARKET_CAPS = {
    "AAPL": 3.5e12, "NVDA": 3.2e12, "JPM": 620e9, "V": 560e9, "JNJ": 400e9,
    "UNH": 470e9, "XOM": 470e9, "PG": 380e9, "HD": 400e9, "CAT": 190e9,
}
ETF_PROXY_WEIGHTS = {"XLK": 4.2e9, "XLF": 2.1e9, "XLV": 1.4e9, "MTUM": 0.6e9, "VLUE": 0.4e9, "USMV": 0.9e9}


def make_price_panel(n_days=280):
    dates = pd.date_range(end=datetime(2026, 8, 20), periods=n_days, freq="B")
    base_prices = {
        "AAPL": 225, "NVDA": 128, "JPM": 215, "V": 285, "JNJ": 155, "UNH": 520,
        "XOM": 115, "PG": 168, "HD": 380, "CAT": 355,
        "XLK": 235, "XLF": 45, "XLV": 148, "MTUM": 210, "VLUE": 175, "USMV": 84,
        "SPY": 560,
    }
    vols = {
        "AAPL": 0.017, "NVDA": 0.032, "JPM": 0.015, "V": 0.013, "JNJ": 0.010, "UNH": 0.018,
        "XOM": 0.016, "PG": 0.009, "HD": 0.014, "CAT": 0.016,
        "XLK": 0.014, "XLF": 0.012, "XLV": 0.011, "MTUM": 0.013, "VLUE": 0.011, "USMV": 0.008,
        "SPY": 0.010,
    }
    drifts = {
        "AAPL": 0.0004, "NVDA": 0.0009, "JPM": 0.0004, "V": 0.0004, "JNJ": 0.0002, "UNH": 0.0001,
        "XOM": 0.0002, "PG": 0.0002, "HD": 0.0003, "CAT": 0.0004,
        "XLK": 0.0005, "XLF": 0.0003, "XLV": 0.0002, "MTUM": 0.0004, "VLUE": 0.0003, "USMV": 0.0002,
        "SPY": 0.0003,
    }
    panel = {}
    market_factor = np.random.normal(drifts["SPY"], vols["SPY"], n_days)
    drawdown_window = set(range(int(n_days * 0.55), int(n_days * 0.60)))
    for i in drawdown_window:
        market_factor[i] -= 0.012

    for symbol in UNIVERSE + [BENCHMARK]:
        beta = 1.0 if symbol == BENCHMARK else np.random.uniform(0.7, 1.3)
        idio = np.random.normal(drifts[symbol] - beta * drifts["SPY"], vols[symbol] * 0.6, n_days)
        daily_returns = beta * market_factor + idio
        prices = base_prices[symbol] * np.cumprod(1 + daily_returns)
        panel[symbol] = prices
    return pd.DataFrame(panel, index=dates)


RATIONALES = {
    "AAPL": (0.10, 0.62, "Services revenue growing ~14% YoY now outweighs hardware cyclicality; iPhone "
             "upgrade cycle in China stabilizing after two soft quarters.",
             ["services revenue +14% YoY", "China iPhone demand stabilizing"]),
    "NVDA": (0.16, 0.58, "Data-center capex guidance from hyperscalers still expanding for next fiscal year, "
             "but valuation already prices in a lot of that — conviction is real but not maximal.",
             ["hyperscaler capex guidance raised", "valuation already reflects strong demand"]),
    "JPM": (0.07, 0.55, "Net interest income holding up better than peers as rate cuts arrive slower than "
             "expected; credit quality metrics remain benign.",
             ["NII resilient vs. peers", "credit quality stable"]),
    "V": (0.08, 0.60, "Cross-border payment volume reaccelerating with travel demand; take-rate stable.",
          ["cross-border volume reaccelerating"]),
    "JNJ": (0.04, 0.45, "Pharma pipeline solid but patent-cliff exposure on two mid-size drugs caps upside "
            "near-term.",
            ["patent cliff exposure on two drugs"]),
    "UNH": (0.02, 0.35, "Medical cost ratio came in above guidance last quarter; regulatory overhang on "
            "Medicare Advantage reimbursement unresolved.",
            ["medical cost ratio above guidance", "Medicare Advantage reimbursement uncertainty"]),
    "XOM": (0.05, 0.48, "Crude inventories drawing down as OPEC+ holds supply discipline, but demand growth "
            "forecasts have been trimmed twice this year.",
            ["OPEC+ supply discipline", "demand growth forecasts trimmed"]),
    "PG": (0.05, 0.50, "Volume growth turned positive after two quarters of price-driven declines; input cost "
           "inflation easing.",
           ["volume growth turned positive", "input costs easing"]),
    "HD": (0.06, 0.47, "Housing turnover still depressed by mortgage rates, but pro-customer segment "
           "outperforming DIY.",
           ["pro segment outperforming DIY"]),
    "CAT": (0.09, 0.52, "Infrastructure-linked order backlog remains elevated; dealer inventory normalized "
            "after last year's destock.",
            ["backlog elevated", "dealer inventory normalized"]),
    "XLK": (0.11, 0.55, "Sector-wide AI infrastructure spend still the dominant earnings driver across the "
            "largest constituents.",
            ["AI infrastructure spend broad-based"]),
    "XLF": (0.06, 0.50, "Yield curve steepening supports net interest margins across the sector into next "
            "year.",
            ["yield curve steepening"]),
    "XLV": (0.03, 0.40, "Sector laggard year-to-date on policy uncertainty around drug pricing reform.",
            ["drug pricing policy uncertainty"]),
    "MTUM": (0.09, 0.45, "Momentum factor has been crowded into a narrow set of mega-cap tech names — "
             "conviction moderate given concentration risk.",
             ["momentum concentrated in mega-cap tech"]),
    "VLUE": (0.06, 0.42, "Value spread vs. growth remains historically wide, a mean-reversion setup without a "
             "clear near-term catalyst.",
             ["value-growth spread historically wide"]),
    "USMV": (0.04, 0.48, "Low-volatility sleeve provides ballast; opportunity cost has been real in a "
             "risk-on tape but that's the point of holding it.",
             ["defensive ballast against risk-on tape"]),
}


def build_views():
    views = []
    for symbol in UNIVERSE:
        expected_return, confidence, rationale, signals = RATIONALES[symbol]
        asset_class = "equity" if symbol in EQUITIES else "etf"
        views.append(
            PortfolioView(
                symbol=symbol,
                asset_class=asset_class,
                expected_return_annualized=expected_return,
                confidence=confidence,
                rationale=rationale,
                key_signals=signals,
            )
        )
    return views


def build_equity_and_benchmark_curves(n_days=180):
    start = datetime(2026, 2, 23, tzinfo=timezone.utc)
    equity = 100_000.0
    benchmark = 100_000.0
    equity_curve, benchmark_curve = [], []
    drawdown_window = set(range(95, 108))
    for i in range(n_days):
        t = start + timedelta(days=i)
        market_move = random.gauss(0.00025, 0.006)
        portfolio_idio = random.gauss(0.0002, 0.004)
        daily_return = 1.05 * market_move + portfolio_idio  # beta slightly above 1
        if i in drawdown_window:
            daily_return -= 0.012
            market_move -= 0.009
        equity *= (1 + daily_return)
        benchmark *= (1 + market_move)
        equity_curve.append({"t": t.isoformat(), "equity": round(equity, 2)})
        benchmark_curve.append({"t": t.isoformat(), "equity": round(benchmark, 2)})
    return equity_curve, benchmark_curve


def build_risk_profile_history():
    start = datetime(2026, 3, 2, tzinfo=timezone.utc)
    history = []
    for week in range(25):
        profile = "conservative" if week < 6 else "balanced"
        history.append(
            {
                "date": (start + timedelta(weeks=week)).isoformat(),
                "active_risk_profile": profile,
                "executed": True,
                "skip_reason": None,
            }
        )
    return history


def main():
    price_panel = make_price_panel()
    universe_prices = price_panel[UNIVERSE]
    market_prices = price_panel[BENCHMARK]
    daily_returns = universe_prices.pct_change().dropna()

    cov_matrix = compute_covariance(universe_prices)
    market_caps = {**MARKET_CAPS, **ETF_PROXY_WEIGHTS}
    prior = compute_market_prior(market_prices, market_caps, cov_matrix)

    views = build_views()
    posterior_returns, posterior_cov = blend_views(cov_matrix, prior, views)

    scenarios = {}
    for profile in ["conservative", "balanced", "aggressive"]:
        weights = optimize_portfolio(posterior_returns, posterior_cov, profile, max_weight_pct=0.15)
        ret, vol, sharpe = evaluate_scenario(weights, posterior_returns, posterior_cov)
        var_95, cvar_95 = historical_var_cvar(weights, daily_returns, confidence_level=0.95)
        scenarios[profile] = {
            "weights": {k: v for k, v in weights.items() if v > 0.001},
            "expected_return": ret,
            "volatility": vol,
            "sharpe": sharpe,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "sector_exposure": exposure_by_tag(weights, DEFAULT_SECTOR_MAP),
            "factor_exposure": exposure_by_tag(weights, DEFAULT_FACTOR_MAP),
        }

    latest_prices = price_panel.iloc[-1].to_dict()
    views_by_symbol = {v.symbol: v for v in views}

    random.seed(25)  # chosen to land a presentable but not fantastical demo: +12% vs SPY +4%, Sharpe ~1.3
    equity_curve, benchmark_curve = build_equity_and_benchmark_curves()
    equity_values = [p["equity"] for p in equity_curve]
    benchmark_values = [p["equity"] for p in benchmark_curve]
    returns = stats.returns_from_equity_curve(equity_values)
    benchmark_returns = stats.returns_from_equity_curve(benchmark_values)
    alpha, beta = stats.alpha_beta(returns, benchmark_returns)

    performance_stats = {
        "sharpe_ratio": stats.sharpe_ratio(returns),
        "sortino_ratio": stats.sortino_ratio(returns),
        "max_drawdown": stats.max_drawdown(equity_values),
        "total_return": (equity_values[-1] - equity_values[0]) / equity_values[0],
        "alpha_annualized": alpha,
        "beta": beta,
        "benchmark_total_return": (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0],
    }

    committee_notes = [
        {
            "created_at": "2026-08-17T14:05:00+00:00",
            "symbol": v.symbol,
            "asset_class": v.asset_class,
            "expected_return_annualized": v.expected_return_annualized,
            "confidence": v.confidence,
            "rationale": v.rationale,
            "key_signals": v.key_signals,
        }
        for v in views
    ]

    # --- fabricate a transaction history: one buy per currently-held
    # position (at a plausible historical entry price), plus two fully
    # closed round-trips (bought then later exited) to show real turnover.
    tx_rng = random.Random(77)
    balanced_weights = scenarios["balanced"]["weights"]
    current_equity = equity_values[-1]
    purchase_dates = [
        "2026-04-13T14:05:00+00:00", "2026-04-20T14:05:00+00:00", "2026-04-27T14:05:00+00:00",
        "2026-05-04T14:05:00+00:00", "2026-05-11T14:05:00+00:00",
    ]
    transactions = []
    for i, (symbol, weight) in enumerate(balanced_weights.items()):
        current_price = latest_prices[symbol]
        entry_offset = tx_rng.uniform(-0.12, 0.09)
        entry_price = current_price / (1 + entry_offset)
        view = views_by_symbol.get(symbol)
        transactions.append(
            {
                "executed_at": purchase_dates[i % len(purchase_dates)],
                "symbol": symbol,
                "name": DEFAULT_NAME_MAP.get(symbol, symbol),
                "asset_class": "equity" if symbol in EQUITIES else "etf",
                "side": "buy",
                "notional_usd": weight * current_equity,
                "price": entry_price,
                "rationale": view.rationale if view else "Initial position build toward target weight.",
            }
        )

    closed_round_trips = [
        ("XOM", "2026-03-02T14:05:00+00:00", "2026-05-18T14:05:00+00:00", -0.04,
         "Crude demand forecasts were cut twice within the quarter and OPEC+ supply discipline proved less supportive than expected — exited energy exposure ahead of the sector-wide downgrade."),
        ("PG", "2026-03-09T14:05:00+00:00", "2026-06-01T14:05:00+00:00", 0.03,
         "Two consecutive quarters of volume weakness reversed the original volume-recovery thesis; reallocated the capital to higher-conviction technology names."),
    ]
    for symbol, buy_date, sell_date, exit_return, exit_rationale in closed_round_trips:
        buy_price = latest_prices[symbol] / 1.05
        sell_price = buy_price * (1 + exit_return)
        buy_notional = 0.08 * current_equity
        shares = buy_notional / buy_price
        view = views_by_symbol.get(symbol)
        transactions.append(
            {
                "executed_at": buy_date, "symbol": symbol, "name": DEFAULT_NAME_MAP.get(symbol, symbol),
                "asset_class": "equity" if symbol in EQUITIES else "etf", "side": "buy",
                "notional_usd": buy_notional, "price": buy_price,
                "rationale": view.rationale if view else "Initial position build.",
            }
        )
        transactions.append(
            {
                "executed_at": sell_date, "symbol": symbol, "name": DEFAULT_NAME_MAP.get(symbol, symbol),
                "asset_class": "equity" if symbol in EQUITIES else "etf", "side": "sell",
                "notional_usd": shares * sell_price, "price": sell_price,
                "rationale": exit_rationale,
            }
        )

    transactions_by_symbol = {}
    for tx in sorted(transactions, key=lambda t: t["executed_at"]):
        transactions_by_symbol.setdefault(tx["symbol"], []).append(tx)

    holdings_detail = build_holdings_detail(
        balanced_weights, current_equity, latest_prices, DEFAULT_NAME_MAP, transactions_by_symbol
    )
    holdings_detail.sort(key=lambda h: h["weight"], reverse=True)

    transaction_history = sorted(transactions, key=lambda t: t["executed_at"], reverse=True)

    snapshot = {
        "schema_version": 3,
        "generated_at": "2026-08-20T21:15:00+00:00",
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
        "performance_stats": performance_stats,
        "active_risk_profile": "balanced",
        "rebalance_generated_at": "2026-08-17T14:05:00+00:00",
        "risk_profile_history": build_risk_profile_history(),
        "scenarios": scenarios,
        "latest_prices": latest_prices,
        "holdings_detail": holdings_detail,
        "transaction_history": transaction_history,
        "investment_committee_notes": committee_notes,
    }

    with open("dashboard/sample_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    print(json.dumps(performance_stats, indent=2))
    print(json.dumps({k: {"expected_return": v["expected_return"], "volatility": v["volatility"], "sharpe": v["sharpe"], "var_95": v["var_95"]} for k, v in scenarios.items()}, indent=2))


if __name__ == "__main__":
    main()
