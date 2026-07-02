# Pairs Trading: Coca-Cola (KO) vs PepsiCo (PEP)

A statistical arbitrage strategy that trades the spread between two cointegrated stocks,
built up from a static baseline to an adaptive Kalman-filter approach — with an honest
account of a lookahead bug that materially changed the results.

**Start here:** [`notebooks/pairs_trading.ipynb`](notebooks/pairs_trading.ipynb) — a single
consolidated notebook walking through the full story end to end (data → cointegration test
→ three strategies → comparison → conclusions). Everything below is a summary of that
notebook plus how the code is organized.

## Why this pair, why this problem

Coca-Cola and PepsiCo are close economic substitutes — same industry, overlapping demand
drivers, similar market-cap tier — making them a textbook candidate for mean-reversion /
statistical arbitrage: if the two prices drift apart, that gap is more likely to be noise
than a fundamental repricing, and it should tend to close.

Pairs trading is a good vehicle for showing a few things end to end: a real statistical
hypothesis test (cointegration), a signal-processing technique (Kalman filtering) applied
outside its usual textbook context, a stochastic-process model (Ornstein-Uhlenbeck) used
for parameter estimation rather than just simulation, and a backtest that treats
transaction costs as a first-class citizen rather than an afterthought.

## Method

**1. Test for cointegration, not just correlation.** Two trending stocks will look highly
correlated in price levels almost by default — that's not a tradeable signal. What matters
is whether a linear combination of the two prices is *stationary*. This is tested with the
Engle-Granger two-step method: regress `log(PEP)` on `log(KO)` via OLS, then run an
Augmented Dickey-Fuller (ADF) test on the residuals.

- Engle-Granger cointegration test: p ≈ 0.030 (rejects "no cointegration" at 5%)
- ADF on the residual spread: p ≈ 0.007 (rejects a unit root)
- Estimated static hedge ratio: β ≈ 1.24

**2. Three ways to turn that into a trading signal:**

| # | Hedge ratio estimation | Entry/exit signal |
|---|---|---|
| 1 | Rolling OLS, re-fit every 60 days | Rolling z-score, fixed window |
| 2 | Kalman filter (adapts every bar) | Z-score window set by rolling OU-estimated half-life |
| 3 | Kalman filter (adapts every bar) | Rolling z-score, fixed window |

Approach 1 is the baseline: a discrete, periodically-refit hedge ratio. Approach 2 pushes
adaptivity further by also letting the *lookback window* track the estimated speed of mean
reversion (fit an Ornstein-Uhlenbeck process to the spread, use its half-life as the z-score
window). Approach 3 keeps the adaptive hedge ratio but reverts to a fixed-window z-score,
isolating whether the OU-derived window was actually helping.

**3. Backtest with realistic frictions.** Every trade pays 2bps commission + 5bps slippage
on notional traded (both legs), sized as ±1 unit of PEP hedged by −β units of KO.

## Results

| Strategy | Total PnL | Sharpe | Max Drawdown | Trades | Cost drag |
|---|---|---|---|---|---|
| 1. Rolling OLS (baseline) | −$8.19 | −0.12 | −$23.52 | 50 | — |
| 2. Kalman + OU | $5.60 | 0.08 | −$17.63 | 194 | 77.7% |
| 3. Kalman only | **$9.35** | **0.14** | −$16.33 | 230 | 71.0% |

*KO/PEP, 2015–2022 daily bars, 5bps slippage + 2bps commission.*

**Takeaways:**
- A continuously-adapting Kalman hedge ratio beats a periodically-refit rolling OLS one.
- Making the *entry signal* adaptive too (OU-derived half-life window) backfires: the
  half-life for this pair is very short (median ~2 bars), so the window is noisy and
  jumpy, driving up turnover until transaction costs eat ~78% of gross PnL.
- None of the three strategies clears a strong risk-adjusted return once costs are honest.
  For a fast-reverting, high-liquidity pair like KO/PEP, the binding constraint isn't
  finding mean reversion — it's finding enough of it, cheaply enough, to net out to profit.

## A bug worth mentioning

An earlier version of `signals_rolling.generate_positions` had a same-bar lookahead: it
recorded the day's position *after* reacting to that day's z-score, so a signal computed
from today's close could trade at today's close — effectively using information before it
was tradeable. `signals_kalman_ou.py`'s position generator already avoided this (positions
are recorded *before* acting on the current bar, so a signal only takes effect from the next
bar). Applying the same fix to `signals_rolling.py` dropped the rolling baseline's Sharpe
from 0.33 → **−0.12**, and the Kalman-only strategy's from 0.44 → **0.14**. The Kalman+OU
strategy was unaffected since it never had the bug.

The table above already reflects the fix. It's included here rather than smoothed over
because catching it is a more useful signal of rigor than the original (inflated) numbers
were.

## Repo structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── data.py               # price download + log-price transform
│   ├── signals_rolling.py    # rolling OLS hedge ratio, rolling z-score, position logic
│   ├── signals_kalman_ou.py  # Kalman filter hedge ratio, OU half-life, position logic
│   ├── backtest.py           # positions -> PnL with transaction costs
│   └── metrics.py            # Sharpe, max drawdown, trade count
└── notebooks/
    ├── pairs_trading.ipynb          # consolidated walkthrough — start here
    ├── exploration.ipynb            # correlation, cointegration, ADF exploration
    ├── backtest_rolling.ipynb       # Approach 1 + hyperparameter grid search
    ├── backtest_kalman_only.ipynb   # Approach 3
    └── backtest_kalman_ou.ipynb     # Approach 2
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/pairs_trading.ipynb
```

Requires internet access on first run (`yfinance` pulls KO/PEP daily prices from Yahoo
Finance for 2015–2022).

## Limitations & next steps

- All three strategies were tuned and evaluated on the same 2015–2022 sample — there's no
  out-of-sample or walk-forward validation, so even the post-fix Sharpes are likely
  optimistic.
- The rolling-OLS hyperparameters came from a grid search on this same data; post-fix,
  every combination in that grid is Sharpe-negative — the "best" one is just the least bad.
- Position sizing is fixed at 1 unit of PEP per trade; no volatility targeting.
- Only a single pair is traded — no portfolio diversification to smooth the equity curve.

Natural next steps: walk-forward validation, a small basket of consumer-staples pairs
evaluated at the portfolio level, volatility-targeted position sizing, and testing whether
flooring the OU half-life (e.g. at 10–20 bars instead of down to 1) recovers some of
Approach 2's theoretical appeal without the turnover.
