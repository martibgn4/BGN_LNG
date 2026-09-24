# StockPicker — weekly US equity selection

A survivorship-bias-free, walk-forward stock selection model over the S&P 500,
built on the Bloomberg Desktop API. Produces five long-only names each week,
inverse-volatility weighted, with full factor attribution.

---

## Why the plumbing matters more than the model

Most stock-picking backtests fail for one of four reasons, all of which are
addressed structurally here rather than assumed away.

**1. Survivorship bias.** Selecting from *today's* S&P 500 means a 2010
backtest only ever considers companies that survived to 2026. It silently
knows which banks would not fail and which firms would not be acquired.

We pull point-in-time membership with `INDX_MWEIGHT_HIST` and an
`END_DATE_OVERRIDE`, which returns constituents *as they stood* on a past
date — including names now delisted, which appear under Bloomberg's dead-ticker
convention (`1436513D UN Equity`) and still carry full price history.

> Measured on this terminal: **959 unique names** have been S&P 500 members
> since 2006, of which **503 are members today**. The other **456 are exactly
> the names a naive backtest deletes.**

**2. Look-ahead in the features.** Deciding a Monday trade using Monday's own
close assumes information you only have after the bell. Every feature is
lagged one trading day before it touches a rebalance bar, and reported
fundamentals carry an additional 45-day publication lag.

**3. Unobtainable execution prices.** Close-to-close backtests assume a price
you can see but not reliably get. Execution here is at **VWAP**
(`EQY_WEIGHTED_AVG_PX`), which a worked retail order can actually achieve.

**4. Delistings that quietly vanish.** If a holding stops trading and the row
is simply dropped, capital teleports into the survivors. Here the final week is
marked to the last price the security ever printed — so an acquisition premium
is earned and a bankruptcy is taken in full.

---

## Layout

```
StockPicker/
  config.py            universe, fields, strategy and model parameters
  bbg/
    session.py         hardened blpapi client: chunking, retries, field exceptions
    universe.py        point-in-time index membership + dead-ticker handling
  data/
    store.py           incremental parquet cache (one file per security)
    panel.py           joins everything into the weekly cross-sectional panel
  features/
    technical.py       momentum, risk, volume, trend  (30 signals)
    fundamental.py     revisions, value, quality, crowding, sentiment
    crosssec.py        winsorise -> sector-neutralise -> rank-normalise
    registry.py        the signal catalogue + expected signs from the literature
  model/
    combine.py         IC-weighted blend (default) and a LightGBM challenger
    portfolio.py       selection, sector caps, inverse-vol sizing, vol targeting
  backtest/
    engine.py          weekly rebalance simulator with costs
    metrics.py         performance, regimes, drawdown, t-stats
    validate.py        look-ahead, shuffle and lag-sensitivity tests
  report/
    weekly.py          dark-theme HTML report with factor attribution
  scripts/
    build_universe.py  point-in-time membership (run monthly)
    ingest_history.py  price + fundamental pull (incremental)
    run_research.py    IC study, backtest, concentration sensitivity
    run_weekly.py      live picks + report
```

---

## Usage

Requires the Bloomberg Terminal running and logged in.

```bash
# one-off setup (~5 min + ~13 min)
python -m StockPicker.scripts.build_universe
python -m StockPicker.scripts.ingest_history

# research: factor ICs, backtest, sensitivity
python -m StockPicker.scripts.run_research --rebuild

# weekly run: refresh data, pick five names, write the report
python -m StockPicker.scripts.run_weekly
```

Both ingest scripts are incremental and interruptible — re-running only fetches
what is missing, so a weekly run costs seconds rather than minutes.

---

## The signal set

Standardisation is applied per date: winsorise → demean within sector →
rank-normalise. Sector-neutralising matters — without it a "momentum" score in
2020 is mostly a bet on being in Technology, and the sector bet belongs in
portfolio construction, not in the signal.

| Family | Signals | Rationale |
|---|---|---|
| Momentum | 12-1, 6-1, residual, Frog-in-the-Pan, 52w high | Jegadeesh-Titman; residual momentum strips the beta bet (Blitz-Huij-Martens); FIP says smooth trends persist and jumpy ones do not |
| Reversal | 1w, 1m, RSI, Bollinger | Short-horizon mean reversion, opposite sign to momentum |
| Risk | realised vol, idio vol, beta, downside dev | Low-volatility and betting-against-beta; also the main lever on the "minimise volatility" objective |
| Revisions | EPS/target/rating changes over 4-26w | Fastest-decaying and among the most reliable equity signals — analysts anchor and revise in the same direction for months |
| Value | earnings/book/EBITDA/FCF yield | Weak alone recently, but a genuine diversifier against momentum |
| Quality | ROE, gross margin, margin trend, net issuance | Novy-Marx gross profitability; buybacks vs dilution |
| Crowding | short interest, its change | High days-to-cover predicts weakness on average |
| Sentiment | Bloomberg news NLP: level, change, dispersion | The change matters more than the level |
| Volume | dollar volume, volume trend, Amihud | Confirms moves and flags fragile names |
| Sector | 13w/26w relative sector momentum | The rotation dimension |

Each signal carries an **expected sign** from the literature. The model learns
its own weights, but a signal that prices opposite to forty years of evidence is
surfaced rather than buried — it usually indicates a bug or a data alignment
error.

---

## How the five names are chosen

1. **Screen** — point-in-time index member, ≥ $20m ADV, ≥ $5 price, ≥ $2bn cap.
2. **Score** — blend standardised signals, weighted by each one's own
   exponentially-decayed historical IC, shrunk toward zero by its standard
   error so an unstable factor cannot buy a large weight with a few lucky
   quarters. Weights are lagged, so they only use returns already observed.
3. **Select** — top 5 by score, at most 2 per sector.
4. **Damp turnover** — a challenger must beat the incumbent by a score margin
   before displacing it, otherwise weekly rebalancing churns between names
   whose rank difference is noise but whose cost is real.
5. **Size** — inverse-volatility, capped at 35% and floored at 5%.
6. **Scale** — gross exposure scaled so forecast portfolio volatility meets the
   target; the residual sits in cash.

---

## Measured results

Walk-forward, survivorship-bias free, net of 8bps on turnover.
Sample 2008-2026 (18.7y, 970 weekly rebalances — the first two years are
consumed establishing IC weights before any position is taken).

| | Strategy (5 names) | SPY |
|---|---|---|
| CAGR | **10.25%** | 10.69% |
| Volatility | 19.0% | 19.4% |
| Sharpe | **0.54** (t = 2.33) | 0.55 |
| Max drawdown | −38.6% | — |
| Alpha (ann) | **+3.09%** (t = 1.01) | — |
| Beta | 0.71 | 1.00 |
| Up / down capture | 0.78 / 0.73 | — |
| Turnover | 49%/week | — |

**Read this honestly.** The strategy roughly *matches* the index return while
carrying 29% less market beta. The +3.1% alpha is directionally right but
**t = 1.01 — not statistically distinguishable from zero** over 18.7 years. What
is significant is the Sharpe (t = 2.33) and the defensive profile: down-capture
0.73, and 2008 finished −8.4% against the index's −35.1%.

### Which signals actually carry information

Per-feature IC over 1,074 weekly cross-sections:

| Signal | mean IC | t | Verdict |
|---|---|---|---|
| `short_interest` | −0.0133 | **−6.28** | strongest single signal; crowding works |
| `mom_sharpe` | +0.0145 | +2.70 | risk-adjusted momentum |
| `resid_mom` | +0.0143 | +2.65 | beats raw momentum, as Blitz-Huij-Martens predicts |
| `bollinger_z` | −0.0132 | −3.90 | short-horizon reversal |
| `eps_rev_26w` | +0.0123 | +3.40 | *stronger in the second half* (0.018 vs 0.007) |
| `mom_x_fip` | +0.0126 | +3.08 | Frog-in-the-Pan path interaction earns its place |
| `margin_trend` | +0.0069 | +3.10 | quality |

18 of 59 features clear |t| > 2 *and* hold their sign across both halves of
the sample.

**Signals that did not work, reported rather than buried:**

* **Value** — every value feature (`earnings_yield`, `book_to_price`,
  `fcf_yield`, `ebitda_yield`) came in with the *wrong* sign. 2006-2026 was a
  growth regime and the model correctly shrinks value's weight to ~0.
* **Options-implied** — all eight signals are weak (|t| < 1.3) and
  sign-unstable across halves, despite being the most "outside the box" data in
  the system. IV skew, the volatility risk premium and put/call positioning
  added nothing at this horizon. Kept in the codebase, weighted to zero by the
  model.
* **News sentiment** — t = 0.95, sign-unstable.

The blend weights the model actually learned track the ICs closely: crowding
0.0116, reversal 0.0108, revisions 0.0104, momentum 0.0098, quality 0.0045,
and everything else shrunk to roughly zero.

### Validation

| Test | Result |
|---|---|
| Point-in-time membership | **PASS** — 0 non-member rows in 513,782 |
| Survivorship | **PASS** — 419 of 921 names (45%) stop trading mid-sample |
| Leak scan | **PASS** — 0 of 59 features exceed \|corr\| 0.15 with forward return (max 0.014) |
| Shuffle test | Random selection gives Sharpe 0.20 / 0.00 / −0.09 vs 0.54 |
| Lag sensitivity | Decays monotonically: 0.54 → 0.36 → 0.28 with each week of delay |

The lag test is the one that matters most: a signal that survives delay
unchanged is usually a look-ahead artefact, and one that decays smoothly is
behaving like real, perishable information.

### Concentration

| n | CAGR | Vol | Sharpe | Turnover |
|---|---|---|---|---|
| 3 | 12.2% | 20.3% | 0.60 | 55% |
| **5** | **10.3%** | **19.0%** | **0.54** | **49%** |
| 10 | 10.1% | 18.2% | 0.55 | 39% |
| 20 | 9.8% | 17.1% | 0.57 | 27% |
| 30 | 10.3% | 16.4% | **0.63** | 19% |

Return is flat across concentration; Sharpe improves modestly with more names,
almost entirely through lower volatility. Five names is defensible, but 20-30
gives a better risk-adjusted result and a third of the turnover.

## Honest limitations

- **Five names is a lot of idiosyncratic risk.** Position sizing equalises
  *risk* contribution, but single-stock event risk — an earnings miss, a failed
  acquisition, a fraud — is not diversifiable at n=5. `run_research.py` prints a
  concentration sensitivity table for 3/5/10/20/30 names precisely so this
  trade-off is visible rather than assumed.
- **Weekly rebalancing is expensive.** Costs are charged at 8bps on turnover;
  the research output reports gross and net side by side so the cost drag is
  explicit.
- **Not entitled on this terminal:** Twitter/social sentiment
  (`TWITTER_SENTIMENT_DAILY_AVG`) and news heat/publication counts. "Social
  trends" is therefore covered by Bloomberg's news NLP sentiment only.
- **Backtest ≠ forecast.** `validate.py` runs shuffle and lag-sensitivity tests
  to establish that the measured edge comes from selection rather than from
  mechanics — but a real out-of-sample track record only accumulates from the
  first live run, which `run_weekly.py` records in `cache/live_picks.parquet`.
