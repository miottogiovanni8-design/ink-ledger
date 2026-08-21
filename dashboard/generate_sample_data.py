"""Generates a realistic sample dashboard_snapshot.json (schema-accurate) for
the dashboard mockup, computed through the project's own stats functions so
the displayed numbers are exactly what the real pipeline would produce for
this data. Not part of the shipped package — a one-off content generator."""

import json
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "src")

from trading_desk.metrics import stats  # noqa: E402

random.seed(25)

start = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)
n_days = 32

# Mean/vol chosen to land Sharpe ~1-1.5 and a genuine ~10% drawdown mid-series
# — realistic for an active but not fantastical paper-trading strategy.
daily_mean = 0.0022
daily_vol = 0.021
bench_mean = 0.0009
bench_vol = 0.011
correlation = 0.55

equity_curve = []
benchmark_curve = []
equity = 1000.0
benchmark = 1000.0

drawdown_window = set(range(11, 17))  # force a visible mid-series drawdown

for i in range(n_days):
    t = start + timedelta(days=i)
    z1 = random.gauss(0, 1)
    z2 = random.gauss(0, 1)
    bench_z = z1
    strat_z = correlation * z1 + (1 - correlation**2) ** 0.5 * z2

    bench_return = bench_mean + bench_vol * bench_z
    strat_return = daily_mean + daily_vol * strat_z
    if i in drawdown_window:
        strat_return -= 0.018
        bench_return -= 0.006

    equity *= (1 + strat_return)
    benchmark *= (1 + bench_return)
    equity_curve.append({"t": t.isoformat(), "equity": round(equity, 2)})
    benchmark_curve.append({"t": t.isoformat(), "equity": round(benchmark, 2)})

equity_values = [p["equity"] for p in equity_curve]
benchmark_values = [p["equity"] for p in benchmark_curve]
returns = stats.returns_from_equity_curve(equity_values)
bench_returns = stats.returns_from_equity_curve(benchmark_values)

trade_journal = [
    dict(created_at="2026-08-20T15:32:00+00:00", symbol="NVDA", asset_class="equity", direction="long",
         confidence=0.78, rationale="RSI(14) at 27.4 with a bullish MACD crossover confirming; Finnhub headline "
         "flagged a data-center capacity expansion announced this morning. Sized at the fixed-fractional cap "
         "given ATR-based stop distance of 2.1%.",
         key_signals=["RSI 27.4 oversold", "MACD bullish crossover", "fresh headline: capacity expansion"],
         risk_flags=[], skipped_by_risk_layer=False, skip_reason=None),
    dict(created_at="2026-08-20T09:05:00+00:00", symbol="BTC/USD", asset_class="crypto", direction="short",
         confidence=0.61, rationale="Price broke below the lower Bollinger band on declining volume with RSI "
         "still neutral at 46 — reading this as a low-conviction mean-reversion setup, not a trend break, so "
         "sized below the risk cap and confidence reflects that.",
         key_signals=["price broke below lower Bollinger band"], risk_flags=["low volume confirmation"],
         skipped_by_risk_layer=False, skip_reason=None),
    dict(created_at="2026-08-19T19:47:00+00:00", symbol="TSLA", asset_class="equity", direction="hold",
         confidence=0.35, rationale="RSI overbought at 74 but MACD histogram still expanding and no bearish "
         "crossover yet — signals are mixed, holding rather than fading a trend that hasn't turned.",
         key_signals=["RSI 74.0 overbought"], risk_flags=[], skipped_by_risk_layer=False, skip_reason=None),
    dict(created_at="2026-08-19T13:20:00+00:00", symbol="AMZN", asset_class="equity", direction="long",
         confidence=0.0, rationale="Skipped by risk layer: max concurrent positions reached: 5/5",
         key_signals=["RSI 29.1 oversold"], risk_flags=["max concurrent positions reached: 5/5"],
         skipped_by_risk_layer=True, skip_reason="max concurrent positions reached: 5/5"),
    dict(created_at="2026-08-18T16:11:00+00:00", symbol="ETH/USD", asset_class="crypto", direction="long",
         confidence=0.69, rationale="MACD bullish crossover on the 1h chart plus a positive funding-rate shift; "
         "RSI recovering through 34 from oversold territory. Stop placed at 1.5x ATR below entry.",
         key_signals=["MACD bullish crossover", "RSI 34.2 recovering from oversold"], risk_flags=[],
         skipped_by_risk_layer=False, skip_reason=None),
    dict(created_at="2026-08-18T10:02:00+00:00", symbol="MSFT", asset_class="equity", direction="short",
         confidence=0.58, rationale="Price broke above the upper Bollinger band into an RSI of 71 with no "
         "supporting headline — treating this as an overextension in a range-bound name.",
         key_signals=["RSI 71.2 overbought", "price broke above upper Bollinger band"], risk_flags=[],
         skipped_by_risk_layer=False, skip_reason=None),
    dict(created_at="2026-08-11T11:40:00+00:00", symbol="AAPL", asset_class="equity", direction="short",
         confidence=0.64, rationale="RSI 73 into an upper Bollinger breakout with no headline support — faded "
         "the extension; stopped out same day as the name kept running.",
         key_signals=["RSI 73.4 overbought", "price broke above upper Bollinger band"], risk_flags=[],
         skipped_by_risk_layer=False, skip_reason=None),
]

positions = [
    dict(symbol="NVDA", asset_class="equity", direction="long", entry_price=131.42, stop_loss_price=128.66,
         take_profit_price=136.26, size_eur=15.0, opened_at="2026-08-20T15:32:00+00:00"),
    dict(symbol="ETH/USD", asset_class="crypto", direction="long", entry_price=4820.15, stop_loss_price=4690.30,
         take_profit_price=5047.80, size_eur=15.0, opened_at="2026-08-18T16:11:00+00:00"),
    dict(symbol="BTC/USD", asset_class="crypto", direction="short", entry_price=98240.0, stop_loss_price=100680.0,
         take_profit_price=94480.0, size_eur=10.0, opened_at="2026-08-20T09:05:00+00:00"),
]

closed_pnls = [4.62, -2.10, 6.85, -1.35, 3.90, 2.15, -3.40, 5.55, 1.20, -1.80, 4.05, 2.65, -2.95, 3.15, -1.60, 2.30]

computed_stats = {
    "sharpe_ratio": stats.sharpe_ratio(returns),
    "sortino_ratio": stats.sortino_ratio(returns),
    "max_drawdown": stats.max_drawdown(equity_values),
    "win_rate": stats.win_rate(closed_pnls),
    "profit_factor": stats.profit_factor(closed_pnls),
    "closed_trades_count": len(closed_pnls),
    "total_realized_pnl_eur": round(sum(closed_pnls), 2),
    "equity_eur": equity_values[-1],
}
alpha, beta = stats.alpha_beta(returns, bench_returns)
computed_stats["alpha_annualized"] = alpha
computed_stats["beta"] = beta

snapshot = {
    "schema_version": 1,
    "generated_at": (start + timedelta(days=n_days - 1)).isoformat(),
    "equity_curve": equity_curve,
    "benchmark_curve": benchmark_curve,
    "positions": positions,
    "trade_journal": trade_journal,
    "stats": computed_stats,
}

print(json.dumps(snapshot, indent=2))
