# AI Paper Trading Desk

An autonomous long/short trading agent that combines a technical/news
screening pipeline with an LLM (Claude) reasoning layer, a deterministic risk
management module, and real broker execution — running on **paper trading**
(simulated money, real market mechanics) via [Alpaca](https://alpaca.markets).

Built as a portfolio project to demonstrate Finance + AI engineering: every
trade carries a documented, LLM-generated rationale, and performance is
tracked with the same statistics a quant desk would report (Sharpe, Sortino,
max drawdown, win rate, profit factor, alpha/beta vs. a benchmark).

## Why paper trading, not real money

This system executes automatically, without a human approving each trade. No
real broker or bank account is connected — every order is a paper (simulated)
order against Alpaca's paper trading API. That's a deliberate constraint, not
a shortcut: it means the whole autonomy/risk-management/reasoning pipeline can
be demonstrated and stress-tested with zero financial risk, while still using
real market data, real broker mechanics, and real execution.

## Architecture

```
watchlist (equity + crypto)
        │
        ▼
 technical screening (RSI / MACD / Bollinger / fresh headline)  ─── no LLM call for non-candidates
        │ candidates only
        ▼
 risk gate check (daily loss / max drawdown / max positions)    ─── blocks the LLM call entirely if tripped
        │ cleared
        ▼
 Claude Sonnet 5 decision engine (structured tool-use output)
        │ direction, size, stop/take-profit, rationale
        ▼
 risk-layer clamp (re-derives stop/TP if missing, caps size)    ─── the model proposes, this disposes
        │
        ▼
 Alpaca bracket order (entry + stop-loss + take-profit, one call)
        │
        ▼
 SQLite ledger (decisions, trades, equity snapshots)
        │
        ▼
 dashboard snapshot JSON ──► Artifact dashboard / weekly email / on-demand chat recap
```

The **LLM never has unchecked authority over capital** — even simulated
capital. The risk layer evaluates circuit breakers *before* the LLM is
called (skipping the call entirely if a breaker is tripped), and re-validates
every proposed trade's size and stop/take-profit *after* the LLM responds,
before anything reaches the broker.

## Cost-aware model tiering

- **Claude Sonnet 5** for per-cycle trade decisions — high call volume (every
  candidate, every ~15 minutes during market hours), so this is a Sonnet-tier
  job, not a frontier-tier one.
- **Claude Opus 5** only for the weekly recap's narrative synthesis — low
  frequency, recruiter-facing prose, worth the premium tier.
- A rule-based pre-filter (`engine/prefilter.py`) screens the entire watchlist
  for free before any LLM call happens, so spend scales with genuinely
  interesting market moves, not with watchlist size.
- Prompt caching on the static system prompt/tool schema in every decision
  call.

## Repository layout

```
src/trading_desk/
├── config.py                  settings (env-driven), risk defaults, watchlists
├── data/
│   ├── indicators.py           RSI, MACD, Bollinger, ATR (pure pandas, no pandas-ta)
│   ├── market_data.py          Alpaca bars → IndicatorSnapshot
│   └── news.py                 Finnhub headlines + Alpha Vantage sentiment
├── engine/
│   ├── prefilter.py             rule-based candidate screening
│   ├── schemas.py               TradeDecision / PortfolioState / tool schema
│   └── decision.py              Claude Sonnet 5 call, structured tool-use, caching
├── risk/
│   ├── sizing.py                 fixed-fractional sizing + LLM-decision clamping
│   └── circuit_breakers.py       daily loss / max drawdown / max positions gates
├── execution/
│   ├── broker.py                  Alpaca bracket order submission, account reads
│   └── reconcile.py               closes DB trades when their bracket leg fills
├── persistence/
│   ├── models.py                  SQLAlchemy: decisions, trades, equity_snapshots
│   ├── db.py, queries.py
├── metrics/stats.py               Sharpe, Sortino, drawdown, win rate, alpha/beta
├── reporting/
│   ├── snapshot.py                 builds the dashboard JSON (single source of truth)
│   ├── recap.py                    shared weekly/on-demand recap builder
│   └── email_sender.py             Resend HTML email
└── cli/
    ├── run_cycle.py                one trading cycle — called by GitHub Actions cron
    ├── weekly_recap.py             email + narrative — called by a scheduled routine
    └── on_demand_recap.py          regenerates the snapshot on request

.github/workflows/                 equity (market hours) + crypto (24/7) cron jobs
tests/                              94 tests, no live API keys required
```

## Scheduling — two mechanisms for two different needs

- **Trading cycles** (timing-critical, high frequency): GitHub Actions cron.
  Free, runs independently of any local machine, and more reliable than a
  scheduler that depends on an app being open. `trading-cycle-equity.yml`
  runs every 15 minutes during US market hours; `trading-cycle-crypto.yml`
  runs every 15 minutes, 24/7. Both write to the same SQLite ledger, so they
  share a concurrency group and never run in parallel.
- **Weekly recap** (low frequency, delay-tolerant): a Claude Code scheduled
  routine that runs `weekly_recap.py`, since it needs to call the Artifact
  tool to republish the dashboard — something only a Claude Code session can
  do, not a plain CI job.
- **On-demand recap**: just ask, in chat, any time.

## Setup

You'll need to create these accounts yourself (I can't create accounts on
your behalf) and provide the API keys:

1. [Alpaca](https://alpaca.markets) — enable paper trading, generate an API
   key/secret.
2. [Anthropic](https://console.anthropic.com) — API key for the decision
   engine (separate from any Claude Code usage).
3. [Finnhub](https://finnhub.io) — free-tier API key for news headlines.
4. [Alpha Vantage](https://www.alphavantage.co) — free-tier API key
   (optional; secondary sentiment signal, not wired into the default cycle
   yet — see `data/news.py`).
5. [Resend](https://resend.com) — free-tier API key for the weekly email.
6. A GitHub repository to host this code and run the Actions workflows —
   copy `.env.example` values into the repo's Actions secrets (`Settings →
   Secrets and variables → Actions`), and set `DAILY_BUDGET_EUR` as a repo
   variable.

Locally:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your keys
pytest
```

## Known simplifications (documented, not hidden)

- Alpha Vantage sentiment is implemented and tested but not called from
  `run_cycle.py` by default — its free tier (25 req/day) is too thin for a
  multi-times-daily cycle across a watchlist; wiring it in as a once-daily
  cached signal is a natural next step.
- The equity/crypto GitHub Actions workflows share one concurrency group so
  they never write the SQLite ledger concurrently; at high cadence this can
  queue runs behind each other. Acceptable at a 15-minute cadence for this
  project's scale.
- Realized P&L on reconciled trades is computed from notional-sized returns
  (`execution/reconcile.py`), not from Alpaca's own fill-level P&L reporting
  — a reasonable approximation given orders are submitted by notional value,
  not share quantity.
