# Ink Ledger — AI-Augmented Black-Litterman Investment Desk

An autonomous investment desk that combines LLM-generated fundamental/macro
research views with **Black-Litterman portfolio construction**, running on
**paper investing** (simulated capital, real broker mechanics) via
[Alpaca](https://alpaca.markets). The LLM never proposes a trade or a
weight — it proposes an expected-return view with a confidence level, which
is then blended with the market's equilibrium prior and optimized
deterministically.

Built as a portfolio project to demonstrate Finance + AI engineering for an
investment-management audience: every allocation traces back to (1) a CAPM
equilibrium prior, (2) a documented AI research view per asset, and (3) a
risk-profile-constrained mean-variance optimization — the same vocabulary
(VaR/CVaR, Sharpe, sector/factor exposure) a quant or IB recruiter reads
daily.

## Why this design, not a technical trading bot

An earlier version of this project ran short-horizon long/short trades off
technical indicators. It was rebuilt around Black-Litterman because:

- **It's the right audience fit.** IB/asset-management recruiters recognize
  Markowitz/Black-Litterman/VaR instantly; RSI/MACD reads as retail trading.
- **It's academically grounded, not invented.** The approach of feeding
  LLM-generated views into Black-Litterman follows a published method —
  [*Integrating LLM-Generated Views into Mean-Variance Optimization Using the
  Black-Litterman Model*](https://arxiv.org/pdf/2504.14345) (ICLR 2025
  workshop). The optimization math itself is delegated to
  [PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) rather than
  reimplemented — a well-known library is a credibility signal to anyone
  reading the code, not a shortcut.
- **It matches "investment," not "trading."** Weekly rebalancing, long-only,
  risk managed at the portfolio level (a volatility target and a drawdown
  circuit breaker) rather than a stop-loss on every position — the way an
  asset manager actually runs a book.

## Why paper investing, not real money

No real broker or bank account is connected. Every order is a paper
(simulated) order against Alpaca's paper trading API — real market data,
real broker mechanics, real weekly execution, zero financial risk.

## Architecture

```
equity + ETF universe (29 assets: sector-diverse large caps + sector/factor ETFs)
        │
        ▼
 price/volume panel + market caps (Alpaca + Finnhub)
        │
        ▼
 Ledoit-Wolf shrinkage covariance ──► CAPM-implied equilibrium prior (Pi)
        │
        ▼
 Claude Sonnet 5 research view per asset (expected return + confidence)  ─── never a trade, never a weight
        │
        ▼
 Black-Litterman blend (Idzorek's method — confidence becomes Ω directly)
        │
        ▼
 EfficientFrontier optimization × 3 risk profiles (conservative/balanced/aggressive)
        │
        ▼
 drawdown circuit breaker check ──► execute the active profile's weights (plain rebalance orders)
        │
        ▼
 SQLite ledger (views, rebalance events, daily equity marks)
        │
        ▼
 dashboard snapshot JSON ──► Artifact dashboard / weekly email / on-demand chat recap
```

All three risk-profile scenarios are computed and persisted every week, not
just the active one — that's what lets the dashboard's risk-profile selector
switch live with zero backend calls.

## Cost-aware model tiering

- **Claude Sonnet 5** for the weekly per-asset research view — moderate
  volume (once per asset per week, ~29 calls), fundamentals/macro reasoning
  doesn't need frontier-tier depth.
- **Claude Opus 5** only for the weekly recap's narrative synthesis — low
  frequency, recruiter-facing prose, worth the premium tier.
- Prompt caching on the static system prompt/tool schema in every view call.

## Repository layout

```
src/trading_desk/
├── config.py                    universe (equities/ETFs), sector & factor tag maps, risk settings
├── data/
│   ├── market_data.py            Alpaca price/volume panels
│   ├── fundamentals.py           Finnhub market cap (equities) + dollar-volume ETF proxy
│   └── news.py                   Finnhub headlines feeding the view prompt
├── engine/
│   ├── schemas.py                 PortfolioView + record_portfolio_view tool schema
│   ├── views.py                   Claude Sonnet 5 call → PortfolioView
│   ├── black_litterman.py         covariance, CAPM prior, BL blend, risk-profile optimization
│   └── portfolio_risk.py          expected return/vol/Sharpe, historical VaR/CVaR, exposure-by-tag
├── risk/circuit_breakers.py       drawdown-from-peak breaker (the one that survived the pivot)
├── execution/
│   ├── broker.py                   Alpaca account/position reads
│   └── rebalance.py                diffs holdings vs. target weights, submits notional orders
├── persistence/
│   ├── models.py                   SQLAlchemy: ViewRecord, RebalanceEvent, EquitySnapshot
│   └── db.py, queries.py
├── metrics/stats.py                Sharpe, Sortino, max drawdown on the equity curve
├── reporting/
│   ├── snapshot.py                  builds the dashboard JSON (single source of truth)
│   ├── recap.py                     shared weekly/on-demand recap builder
│   └── email_sender.py              Resend HTML email
└── cli/
    ├── daily_mark.py                cheap no-LLM equity mark — GitHub Actions, every market day
    ├── weekly_rebalance.py          the full BL pipeline — GitHub Actions, weekly
    ├── weekly_recap.py              narrative + email — Claude Code scheduled routine
    └── on_demand_recap.py           regenerates the snapshot on request

.github/workflows/                  daily-mark.yml + weekly-rebalance.yml
dashboard/dashboard.html            the published Artifact — risk selector, IT/EN, history scrubber
tests/                              77 tests, no live API keys required
```

## Scheduling — two mechanisms for two different needs

- **Daily mark + weekly rebalance** (GitHub Actions): free, runs
  independently of any local machine. `daily-mark.yml` reads account equity
  every market day (no LLM call, near-zero cost) so the dashboard's history
  scrubber has daily granularity. `weekly-rebalance.yml` runs the full
  pipeline once a week. Both write the same SQLite ledger, so they share a
  concurrency group and never run in parallel.
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
2. [Anthropic](https://console.anthropic.com) — API key for the view engine
   (separate from any Claude Code usage).
3. [Finnhub](https://finnhub.io) — free-tier API key for headlines and
   equity market cap.
4. [Resend](https://resend.com) — free-tier API key for the weekly email.
5. A GitHub repository to host this code and run the Actions workflows —
   copy `.env.example` values into the repo's Actions secrets (`Settings →
   Secrets and variables → Actions`), and set `ACTIVE_RISK_PROFILE` as a
   repo variable.

Locally:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your keys
pytest
```

## Known simplifications (documented, not hidden)

- ETFs don't have a traditional market cap, so their Black-Litterman
  equilibrium weight is proxied by trailing average dollar trading volume
  (`data/fundamentals.py::dollar_volume_proxy_weights`) — liquidity-weighted
  rather than AUM-weighted. A documented approximation, not a hidden one.
- Factor exposure grouping is a static tag map
  (`config.DEFAULT_FACTOR_MAP`), not a regression-based factor-loading
  model — equities and sector ETFs fall into "Other" on the factor view.
  A real factor-loading model (e.g. Fama-French regression) is a natural
  extension.
- The daily-mark and weekly-rebalance GitHub Actions share one concurrency
  group so they never write the SQLite ledger concurrently; at higher
  cadence this could queue runs behind each other. Not a concern at
  daily/weekly frequency.
- The dashboard's sample data uses a curated 16-asset subset of the full
  29-asset production universe, so every investment-committee rationale
  could be individually hand-written rather than templated — see
  `dashboard/generate_sample_data.py`.

## License

© 2026 Giovanni Miotto. All rights reserved.

This project — architecture, code, methodology, and design — is the
author's original work. Reproduction, copying, distribution, or creation
of derivative works, in whole or in part, is prohibited without the
author's written permission. See [LICENSE](LICENSE).
