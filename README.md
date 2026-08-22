# Ink Ledger — AI-Augmented Black-Litterman Investment Desk

An autonomous investment desk that operationalizes portfolio theory taught
in university finance coursework — CAPM equilibrium, Black-Litterman,
mean-variance optimization, risk-adjusted performance measurement — into a
live system that runs every week with real (paper) capital and produces
auditable results, not a backtest. An LLM contributes exactly one thing to
the process: a per-asset expected-return view with a confidence level.
Everything downstream — the equilibrium prior, the Bayesian blend, the
optimization, the execution — is deterministic, textbook math.

## Quick look

**[→ Live dashboard](https://miottogiovanni8-design.github.io/ink-ledger/)** —
the real portfolio, updating automatically every week straight from this
repo's own data, no manual step.

If you have five minutes: read [Why this design](#why-this-design-academic-theory-not-a-trading-bot)
below, then [Financial metrics used](#financial-metrics-used) for the math.
If you have one: open the dashboard above and look at the **Analytics**
page — it's the part most projects like this don't have (see
[Is the AI's view actually any good?](#is-the-ais-view-actually-any-good)).

Core files, if you're reading code rather than prose:
[`engine/black_litterman.py`](src/trading_desk/engine/black_litterman.py)
(portfolio construction), [`cli/weekly_rebalance.py`](src/trading_desk/cli/weekly_rebalance.py)
(the full weekly pipeline), [`dashboard/dashboard.html`](dashboard/dashboard.html)
(the front end).

## Why this design: academic theory, not a trading bot

An earlier version of this project ran short-horizon long/short trades off
technical indicators (RSI/MACD-style signals). It was rebuilt around
Black-Litterman for one reason above the others: **it's the framework
actually taught for this problem**, not a heuristic invented for a demo.

- **CAPM and Black-Litterman are the standard toolkit**, not a novel
  invention — this project applies them, it doesn't reinvent them. The
  optimization math is delegated to
  [PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) rather
  than reimplemented, so anyone who knows the library can verify the
  method directly against the calls in the code.
- **Using an LLM's output as a Black-Litterman view is itself a published
  method**, not something improvised for this project — see
  [*Integrating LLM-Generated Views into Mean-Variance Optimization Using
  the Black-Litterman Model*](https://arxiv.org/pdf/2504.14345) (ICLR 2025
  workshop).
- **It matches "investment," not "trading."** Weekly rebalancing,
  long-only, risk managed at the portfolio level (a volatility target and
  a drawdown circuit breaker) rather than a stop-loss on every position —
  the way an asset manager actually runs a book, and the vocabulary
  (VaR/CVaR, Sharpe, sector/factor exposure, attribution) an investment
  team uses to talk about it.

The LLM's role is deliberately narrow: it never proposes a trade or a
weight, only an expected-return view and how confident it is. That view is
one input among several (the market's own equilibrium prior weighs in too)
to a deterministic optimizer — the same separation of "research" from
"portfolio construction" a real investment desk maintains between its
analysts and its PM.

## Why paper investing, not real money

No real broker or bank account is connected. Every order is a paper
(simulated) order against Alpaca's paper trading API — real market data,
real broker mechanics, real weekly execution, zero financial risk. The
point of the project is the process and its measurable output, not
speculating with actual capital.

## Architecture

```
equity + ETF universe (29 assets: sector-diverse large caps + sector/factor ETFs)
        │
        ▼
 price/volume panel + market caps (Alpaca + Finnhub)
        │
        ▼
 Ledoit-Wolf shrinkage covariance ──► CAPM-implied equilibrium prior (Π)
        │
        ▼
 per-asset headlines + market-wide macro headlines (Finnhub) + sentiment (Alpha Vantage, equities)
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
 SQLite ledger (views, rebalance events, daily equity marks) ──► frozen buy-and-hold baseline (control arm)
        │
        ▼
 dashboard snapshot JSON ──► live dashboard / weekly email / on-demand chat recap
```

All three risk-profile scenarios are computed and persisted every week, not
just the active one — that's what lets the dashboard's risk-profile
selector switch live with zero backend calls. A frozen copy of the very
first week's weights is kept forever, untouched, as a buy-and-hold control
arm — the dashboard's equity chart shows it alongside the actively managed
portfolio, so "did the ongoing rebalancing actually help" is a number, not
a claim.

## Financial metrics used

Every number on the dashboard maps to a specific formula in the code, not
a hand-wave. Grouped by what question each one answers.

### Building the portfolio

**CAPM equilibrium prior (Π)** — [`black_litterman.py::compute_market_prior`](src/trading_desk/engine/black_litterman.py)

```
Π = δ · Σ · w_mkt          δ = (E[R_mkt] − R_f) / Var(R_mkt)
```

`Σ` is the Ledoit-Wolf shrinkage covariance (not raw sample covariance,
which PyPortfolioOpt's own docs advise against), `w_mkt` are market-cap
weights (ETFs use trailing dollar volume as a liquidity-weighted proxy —
see [Known simplifications](#known-simplifications-documented-not-hidden)),
`δ` is the market's reverse-optimized risk aversion. This is "what would
the market's own returns imply everyone already expects," before any AI
view is added — the neutral starting point Black-Litterman is built to
correct, not replace.

**Black-Litterman posterior** — [`black_litterman.py::blend_views`](src/trading_desk/engine/black_litterman.py)

```
E[R] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
```

`Q` is the vector of Claude's per-asset expected-return views, `P` maps
each view to its asset (identity matrix here, since every view is
absolute), and `Ω` — the view uncertainty — comes from **Idzorek's
method**, which converts each view's stated 0–1 confidence directly into a
variance. This is the exact mechanism that gives the LLM's "confidence"
output a real mathematical consequence: a low-confidence view barely moves
the posterior away from the market prior; a high-confidence view moves it
further.

**Efficient Frontier optimization** — [`black_litterman.py::optimize_portfolio`](src/trading_desk/engine/black_litterman.py):
`conservative`/`aggressive` target an explicit annualized volatility band
(8% / 22%) via `efficient_risk`; `balanced` maximizes the Sharpe ratio (the
tangency portfolio) via `max_sharpe`. All three are long-only with a 15%
single-name cap.

### Measuring performance

**Sharpe ratio** — [`metrics/stats.py::sharpe_ratio`](src/trading_desk/metrics/stats.py)

```
Sharpe = mean(R_p − R_f) / std(R_p − R_f) · √252
```

Return per unit of *total* volatility, annualized. The standard first
question: is the return worth the risk taken to get it.

**Sortino ratio** — [`metrics/stats.py::sortino_ratio`](src/trading_desk/metrics/stats.py)

```
Sortino = mean(R_p − R_f) / std(min(R_p − R_f, 0)) · √252
```

Same idea as Sharpe, but the denominator only counts downside deviation.
Sharpe penalizes big up-days exactly like big down-days, which understates
a strategy with a favorable skew; Sortino doesn't make that mistake.

**Max drawdown** — [`metrics/stats.py::max_drawdown`](src/trading_desk/metrics/stats.py)

```
MaxDD = max over t of (peak_≤t − equity_t) / peak_≤t
```

The worst peak-to-trough loss actually experienced. Volatility is
symmetric and abstract; drawdown is what an investor actually feels and
what the circuit breaker below reacts to.

**Alpha and beta vs. S&P 500** — [`metrics/stats.py::alpha_beta`](src/trading_desk/metrics/stats.py)

```
β = Cov(R_p, R_b) / Var(R_b)
α = [mean(R_p) − β · mean(R_b)] · 252     (OLS on excess returns)
```

Beta isolates how much of the portfolio's movement is just "the market
went up or down." Alpha is the annualized return left over after removing
that — the actual test of whether the AI's views and the active
rebalancing added anything beyond a passive market bet.

**Historical VaR 95% / CVaR 95%** — [`engine/portfolio_risk.py::historical_var_cvar`](src/trading_desk/engine/portfolio_risk.py)

```
VaR_95  = −Percentile_5(daily portfolio returns)
CVaR_95 = −mean(returns ≤ VaR cutoff)      (expected shortfall)
```

VaR: how bad a bad day gets, 95% of the time. CVaR: *if* that bad day
happens, how bad on average — the more conservative companion metric,
since VaR says nothing about severity beyond its own cutoff.

### Is the AI's view actually any good?

This is the question most AI-trading demos never ask, because most don't
keep enough history to answer it. Once a view's asset has a subsequent
price, this project scores it — see the dashboard's **Analytics** page.

**Information Coefficient (IC)** — [`engine/skill_analysis.py::information_coefficient`](src/trading_desk/engine/skill_analysis.py)

```
IC = Pearson correlation(predicted return, realized return)
```

The standard metric (Grinold & Kahn, *Active Portfolio Management*) for
whether a forecasting signal has any real skill. IC ≈ 0 means the views
are noise; consistently above ~0.05–0.10 is considered a genuine edge in
quant equity research.

**Calibration** — [`engine/skill_analysis.py::calibration_buckets`](src/trading_desk/engine/skill_analysis.py):
views are grouped by their stated confidence into bands, and each band's
actual hit rate is reported. A well-calibrated model's hit rate should
climb with its stated confidence; a flat or inverted curve means the
confidence number isn't meaningful, however convincing it sounds.

**Beta vs. skill decomposition** — [`engine/skill_analysis.py::decompose_return`](src/trading_desk/engine/skill_analysis.py)

```
beta_contribution  = β · R_benchmark
skill_contribution = R_total − beta_contribution
```

Splits the realized return into "what simply being long the market at this
beta would have earned" and whatever's left — the part attributable to the
views and the optimizer's tilts, not the market's own drift.

**Brinson-Fachler attribution** — [`engine/attribution.py::brinson_attribution`](src/trading_desk/engine/attribution.py),
for each sector `s`:

```
Allocation_s  = (w_p,s − w_b,s) · (R_b,s − R_b,total)
Selection_s   = w_b,s · (R_p,s − R_b,s)
Interaction_s = (w_p,s − w_b,s) · (R_p,s − R_b,s)
```

The textbook three-effect decomposition a fund's performance team reports
to an investment committee. Allocation: did it overweight the sectors that
did well? Selection: within a sector, did it pick the better performers?
Interaction: the cross-term between the two.

### Circuit breaker

**Drawdown breaker** — [`risk/circuit_breakers.py`](src/trading_desk/risk/circuit_breakers.py):
pauses rebalancing (requires manual reset) if drawdown-from-peak exceeds a
configured threshold. Risk is managed at the portfolio level, not with a
stop-loss on every position — consistent with weekly, long-only investing
rather than active trading.

## Cost-aware model tiering

- **Claude Sonnet 5** for the weekly per-asset research view — moderate
  volume (once per asset per week, ~29 calls), fundamentals/macro reasoning
  doesn't need frontier-tier depth.
- **Claude Opus 5** only for the weekly recap's narrative synthesis — low
  frequency, recruiter-facing prose, worth the premium tier.
- Prompt caching on the static system prompt/tool schema in every view call.
- Estimated spend is self-tracked from each response's token usage and
  checked against a monthly threshold every run (see
  [`metrics/cost_tracking.py`](src/trading_desk/metrics/cost_tracking.py)) —
  a second layer on top of the hard spend cap set directly in the
  Anthropic console.

## Repository layout

```
src/trading_desk/
├── config.py                    universe (equities/ETFs), sector & factor tag maps, risk settings
├── data/
│   ├── market_data.py            Alpaca price/volume panels
│   ├── fundamentals.py           Finnhub market cap (equities) + dollar-volume ETF proxy
│   └── news.py                   Finnhub headlines (per-asset + general market) and Alpha Vantage sentiment feeding the view prompt
├── engine/
│   ├── schemas.py                 PortfolioView + record_portfolio_view tool schema (strict mode)
│   ├── views.py                   Claude Sonnet 5 call → PortfolioView
│   ├── black_litterman.py         covariance, CAPM prior, BL blend, risk-profile optimization
│   ├── portfolio_risk.py          expected return/vol/Sharpe, historical VaR/CVaR, exposure-by-tag
│   ├── skill_analysis.py          Information Coefficient, calibration, beta/skill decomposition
│   └── attribution.py             Brinson-Fachler sector attribution
├── risk/circuit_breakers.py       drawdown-from-peak breaker
├── execution/
│   ├── broker.py                   Alpaca account/position reads
│   └── rebalance.py                diffs holdings vs. target weights, submits notional orders
├── persistence/
│   ├── models.py                   SQLAlchemy: ViewRecord, RebalanceEvent, EquitySnapshot, BaselineAllocation, ApiSpendLog
│   └── db.py, queries.py
├── metrics/
│   ├── stats.py                    Sharpe, Sortino, max drawdown, alpha/beta
│   └── cost_tracking.py            self-reported Claude API spend estimate
├── reporting/
│   ├── snapshot.py                  builds the dashboard JSON (single source of truth)
│   ├── holdings.py                  weighted-average cost basis from the transaction log
│   ├── recap.py                     shared weekly/on-demand recap builder
│   └── email_sender.py              Resend HTML email
└── cli/
    ├── daily_mark.py                cheap no-LLM equity mark — GitHub Actions, every market day
    ├── weekly_rebalance.py          the full BL pipeline — GitHub Actions, weekly
    ├── weekly_recap.py              narrative + email — GitHub Actions, weekly
    └── on_demand_recap.py           regenerates the snapshot on request

.github/workflows/                  daily-mark.yml + weekly-rebalance.yml + weekly-recap.yml
dashboard/dashboard.html            risk selector, IT/EN, history scrubber — reads live data when served (e.g. GitHub Pages), embedded sample data otherwise
tests/                              140+ tests, no live API keys required
```

## Scheduling — all on GitHub Actions

- **`daily-mark.yml`** (every market day): reads account equity from
  Alpaca, no LLM call — near-zero cost, gives the dashboard's history
  scrubber daily granularity even though rebalancing itself is weekly.
- **`weekly-rebalance.yml`** (Monday): the full pipeline — prices,
  fundamentals, Claude views, Black-Litterman, optimization, execution.
- **`weekly-recap.yml`** (Monday, an hour later): Claude Opus 5 narrative
  + the recap email, reading that run's fresh numbers.

All three write the same SQLite ledger and share one concurrency group, so
they never run into each other. Nothing here depends on a local machine
being on, or a chat session being open — the schedule runs whether or not
anyone is watching.

## Setup

You'll need to create these accounts yourself and provide the API keys as
GitHub repo secrets/variables (`Settings → Secrets and variables →
Actions`):

1. [Alpaca](https://alpaca.markets) — enable paper trading, generate an API
   key/secret.
2. [Anthropic](https://console.anthropic.com) — API key for the view engine
   and the recap narrative (separate from any Claude Code usage). Set a
   spend limit in the console — it's the real hard cap.
3. [Finnhub](https://finnhub.io) — free-tier API key for headlines (per-asset
   and general market) and equity market cap.
4. [Alpha Vantage](https://www.alphavantage.co) — optional, free-tier API key
   for a secondary sentiment signal on equities (thin 25 requests/day limit,
   the system degrades gracefully and just skips it if unset or rate-limited).
5. [Resend](https://resend.com) — free-tier API key for the weekly recap
   email.
6. Repo variables: `ACTIVE_RISK_PROFILE` (conservative/balanced/aggressive),
   `EMAIL_FROM`, `SPEND_ALERT_THRESHOLD_USD`.

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
- The three scheduled GitHub Actions share one concurrency group so they
  never write the SQLite ledger concurrently; at higher cadence than
  daily/weekly this could queue runs behind each other. Not a concern at
  the cadence this project actually runs.
- The dashboard's sample data uses a curated 16-asset subset of the full
  29-asset production universe, so every investment-committee rationale
  could be individually hand-written rather than templated — see
  `dashboard/generate_sample_data.py`. It's clearly flagged as sample data
  in the UI and never shown once real weekly data exists.
- Claude API spend shown on the dashboard and checked against the alert
  threshold is a **self-reported estimate** from public per-token pricing,
  not Anthropic's own billing figure (which requires an Admin API key,
  unavailable on an individual account) — the real backstop is the hard
  spend limit set directly in the Anthropic console.

## License

© 2026 Giovanni Miotto. All rights reserved.

This project — architecture, code, methodology, and design — is the
author's original work. Reproduction, copying, distribution, or creation
of derivative works, in whole or in part, is prohibited without the
author's written permission. See [LICENSE](LICENSE).
