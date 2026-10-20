# Research specification — TTF forecasting platform

Version 0.1, 2026-10-16. Status: **specification only; no result exists yet.** Changes to this
document are recorded in `modelling_log.md`.

## 1. Purpose and framing

Test whether European gas fundamentals, LNG flows, weather and cross-commodity information
improve forecasts of Dutch TTF prices **beyond what the TTF forward curve already prices**, at which
horizons, for which target, in which regimes, and whether any improvement survives realistic
publication delays, execution timing and costs.

The forward curve is the null hypothesis. Under market efficiency the futures price is the
risk-neutral expectation of the settlement it converges to, so for every futures-based target the
**market benchmark forecast of the excess return is zero** and the market benchmark forecast of a
delivery-period price is the contemporaneous futures price. A model is interesting only if it beats
that benchmark out of sample. Beating a historical mean, a random walk in levels, or an in-sample
fit is not evidence of edge.

## 2. Hypotheses

Each is tested separately, per target and horizon, with the same validation design.

| ID | Hypothesis | Primary target(s) | Rejection evidence |
|---|---|---|---|
| H1 | Storage surprise (inventory vs seasonal normal and vs scenario-projected path) predicts front-month excess return at 5–20 trading days. | T3b, T3c, T5 | DM test vs zero-return benchmark not significant in ≥ 3 of 4 outer test years |
| H2 | Weather **forecast revisions** (not levels) predict 1–5 day front-month returns in the heating season. | T3a, T3b | as H1, heating-season subsample |
| H3 | LNG arrival forecast surprises and freight-adjusted JKM–TTF arbitrage predict 5–20 day returns and the M1–M2 spread. | T3b, T3c, T7a | as H1 |
| H4 | Norwegian unplanned outages predict prompt returns and M1–M2 spread widening, with larger effect when storage is tight (interaction). | T3a, T7a | interaction coefficient sign/size unstable across folds |
| H5 | Fundamentals predict realised **volatility** and tail-move probability better than price-only volatility models even if they do not predict direction. | T8, T9 | QLIKE / Brier vs HAR-type and price-only logit |
| H6 | Any predictability is regime-dependent (tight/crisis vs normal/oversupplied). | all | performance indistinguishable across regimes, or entirely from 2021-H2–2022 |
| H0 | (Control) Price-only features (momentum, curve shape) explain any apparent fundamental signal. | all | ablation: fundamentals add nothing over price-only |

H5 is listed deliberately: a fundamentals model that cannot predict direction can still be useful
for risk and option desks. It must not be relabelled as directional skill.

## 3. Targets and horizons

Full definitions, contract and roll rules are in `target_definitions.md`. Summary:

| ID | Target | Horizon(s) | Benchmark(s) |
|---|---|---|---|
| T1 | TTF day-ahead price level | 1d | no-change; current M1/BOM futures |
| T2 | TTF front-month settlement level | 1, 5, 20d | no-change (= the forward) |
| T3a/b/c | Log return of a **fixed** front-month contract | 1, 5, 20 trading days | zero (forward); AR; rolling mean |
| T4 | Monthly average day-ahead price of delivery month M+k | k = 1, 2 | futures for month M+k at origin |
| T5 | Realised delivery price minus forward at origin (excess return to delivery) | k = 1, 2, 3 months | zero |
| T6 | Quarterly and seasonal contract log returns | 20, 60d | zero |
| T7a–d | Curve spreads: M1–M2, Q1–Q2, Summer–Winter, Winter–next Summer (changes) | 5, 20d | no-change of the spread |
| T8 | Realised volatility of T3 returns over next 5 / 20 days | 5, 20d | trailing realised vol; HAR; EWMA |
| T9 | Direction and tail labels (sign, \|r\| > threshold, regime transition) | 1, 5, 20d | base rate; price-only logit |

**Primary targets** (the ones that answer question A): T3b, T3c and T5. T1/T2 level targets are
reported for completeness but a level forecast that only reproduces persistence is not a result.

## 4. Benchmarks (Level 0)

1. Zero excess return (the forward-curve benchmark) — **principal**.
2. No-change / random walk in levels (identical to 1 for fixed-contract futures levels).
3. Rolling mean of past returns (60d, 250d) — tests "drift" models.
4. AR(p) on the target's own history, p chosen on the inner fold.
5. Seasonal historical average (by calendar month) — for T1/T4 only.
6. Simple storage-deviation OLS — the "obvious fundamental" model every richer model must beat.
7. For T8: trailing realised vol, EWMA, HAR(1,5,22).

A model is compared to the **best** of the benchmarks appropriate to its target, not the weakest.

## 5. Initial features (families)

Each family is a unit for ablation (`features.yaml`, `features/registry.py`).

| Family | Examples | PIT risk |
|---|---|---|
| `price` | lagged returns, rolling vol, momentum, z-scores | low (settlement times known) |
| `curve` | M1–M2, slope, curvature, seasonal spreads, prompt vs deferred | low |
| `cross_commodity` | JKM–TTF, HH netback, Brent, API2, EUA, NBP/THE/PEG basis, FX | medium (asynchronous closes) |
| `storage` | % full, deviation from seasonal normal, injection vs normal, projected end-of-season level | **high** (AGSI revisions, publication lag) |
| `flows` | Norwegian flows, capacity utilisation, outage-adjusted capacity, LNG send-out | high (nominations vs actuals) |
| `lng` | expected EU arrivals 7/14/30/45d, on-water, diversions, freight-adjusted arb | **high** (realised destinations) |
| `weather` | pop-weighted HDD anomaly, **revisions**, ensemble spread | **high** (realised vs forecast) |
| `power` | wind/solar forecasts, French nuclear availability, clean spark/dark, switching | high |
| `calendar` | weekday, month, gas year, heating season, days to expiry, holidays | none |
| `interactions` | cold × low storage, outage × low storage, low wind × demand, JKM premium × falling arrivals, terminal spare capacity × arrivals, expiry × prompt stress | inherits parents |
| `text` | LLM-extracted outage/maintenance events (reviewed) | high; **excluded from clean backtest until provenance exists** |

## 6. Data availability assumptions

- Market data (Bloomberg): TTF month/quarter/season/year settlements back to at least 2015 are
  assumed available on the desk Terminal (roots in production use). Day-ahead, API2, EUA and power
  identifiers are unknown. Bid/ask/OI/volume entitlement unknown.
- Storage: GIE AGSI via Bloomberg `CGIE*`, daily from ~2011, **revised**, Bloomberg history does not
  preserve vintages. Backtests apply a publication-lag rule (see PIT policy) and flag the series
  `pit_grade = B` (lag-approximated, revisions not reproducible).
- Weather: point-in-time forecasts exist from mid-2021 (Open-Meteo archive, 7-day max lead,
  non-commercial licence) — **only ~5 heating seasons**. Longer leads must be accumulated live.
- LNG: Bloomberg `LNGG*`/`AHOY*` aggregates are daily from 2017 but their revision behaviour is
  unknown; Spark netbacks from 2021-09. No PIT vessel/destination history is known to be available,
  so destination-based arrival features are `pit_grade = C` (excluded from the clean backtest)
  until a vintaged source is confirmed.
- Norwegian flows / outages: Gassco UMM is public but no connector exists here. Unknown.

Consequence: the **clean backtest window for the full feature set starts mid-2021**, which contains
the 2021–2023 crisis. That is the single largest threat to valid inference here (section 9).

## 7. Validation design

Detailed in `point_in_time_policy.md` §6 and implemented in `evaluation/splits.py`.

- **Walk-forward, expanding window** by default; rolling window (3y) as a sensitivity.
- **Outer folds:** contiguous test blocks of 6 months, origin dates only, refit at each block start.
  Each outer fold reports its own metrics; the headline is the distribution across folds, not a
  pooled number.
- **Inner folds:** inside each outer training set, expanding walk-forward with the last 25 % of the
  training window as validation blocks, for hyperparameters only.
- **Purge** of training samples whose label window overlaps the first test origin (≥ horizon
  trading days) and **embargo** of `h` trading days after each test block when it is re-used for
  training.
- **Locked final holdout:** the last 12 months of available data are not touched by any experiment
  until the model specification is frozen and signed off. Its single evaluation is reported as is.
- Retraining frequency (daily / weekly / monthly / quarterly) is an experiment variable.
- Random K-fold is forbidden; `splits.py` refuses any non-monotone split.

### Metrics
MAE, RMSE, median AE, bias, correlation (returns), out-of-sample R² **vs the forward benchmark**
(`1 − MSE_model / MSE_bench`), Diebold–Mariano with Newey–West HAC (lag = h − 1) and Harvey–
Leybourne–Newbold small-sample correction; classification: balanced accuracy, MCC, Brier, calibration;
probabilistic: pinball, interval coverage and width, CRPS (empirical, from quantiles); trading: net
P&L, Sharpe, max drawdown, turnover, hit rate, P&L concentration (share of P&L from top 5 % days),
all by year, regime and horizon.

## 8. Success criteria (all must hold for a "useful" claim)

For a given target × horizon:

1. **Beats the forward benchmark:** OOS MSE reduction > 0 in at least 75 % of outer folds **and**
   pooled HLN-corrected DM p < 0.05 (one-sided), on the clean-PIT feature set.
2. **Not one regime:** OOS R² vs benchmark > 0 both including and **excluding** 2021-07..2023-06.
3. **Fundamentals matter:** the full model beats the price-only (`price`+`curve`+`calendar`) model
   with the same estimator (DM p < 0.10) — otherwise H0 stands.
4. **Survives execution:** signal acted on at the next settlement after `publication_time` (not
   the same settlement), net Sharpe > 0.5 at 2× the base cost assumption, max drawdown < 3× annual
   vol.
5. **Calibrated:** 80 % interval coverage within ±5 pp in every regime with ≥ 60 test origins.
6. **Interpretable:** sign of the main fundamental effects matches the physical story signed off by
   the gas-fundamentals review (e.g. tighter storage → higher prompt, steeper backwardation).
7. **Locked holdout** does not contradict 1–4.

### Amendments (2026-10-17, before any real-data result)

- **A1 (D-032 F7).** Criteria 1–3 are evaluated on the fixed-contract label **and** on the executable
  label r(t+1, t+1+h). A pass requires the executable label.
- **A2 (D-031).** Criterion 1 needs at least 4 outer folds; otherwise it is "not evaluable".
- **A3 (D-020/D-032 F4).** HLN-DM is the gate. Clark–West is supplementary only: it is oversized with
  shrinkage estimators.
- **A4 (D-028).** Criterion 1 is also reported against the best benchmark (zero, rolling mean, AR on
  own known lags). Criterion 3's twin includes the curve family, own lags and price-derived spreads
  (`docs/feature_signs.md`).
- **A5 (D-032 F6).** A failed criterion 3 is reported with a power estimate (positive-control
  simulation at the same n and feature count).
- **A6.** Criterion 6 is judged against the pre-registered `docs/feature_signs.md`.

## 9. Stop criteria

Stop the line of research for a target × horizon (and report it as a negative result) if:

- after the Level 1 and Level 2 models and family ablations, **no** family adds DM-significant
  value over price-only at any horizon; or
- the only improvement comes from one exceptional year (removing it flips the sign); or
- the improvement disappears when publication lags are applied as specified; or
- it requires a feature graded `pit_grade = C`; or
- (A5) the history behind a criterion-3 failure is shorter than 5 years — then the result is "inconclusive", not a stop.

Do not proceed to Level 3+ (regimes, deep models, ensembles) for a target that fails Levels 1–2.
Deep models (Level 5) are not built in v1 at all.

## 10. Known limitations (initial)

- Short PIT history for weather (2021+) and LNG; the clean sample is crisis-dominated.
- Storage and many fundamentals lack preserved vintages on Bloomberg; lag approximation only.
- Overlapping multi-day targets → effective sample far smaller than the number of rows.
- Structural break in 2022 (Russian pipeline loss, EU storage regulation): pre-2022 relationships
  may not transfer. Regime-conditional evaluation is mandatory, not optional.
- Bid/ask and volume entitlement unknown; cost assumptions are parameters until confirmed.
- TTF futures settle ~17:30 CET; JKM, HH, Brent and US data close later — same-date cross-commodity
  features are lagged one session by default (PIT policy §4).

## 11. Role of the LLM

The LLM writes code and documents, organises research, and (later) extracts structured events from
authorised text into the `text/` schema with provenance and mandatory human review. It never
produces a numeric price forecast and never fills a numeric impact the source does not state.
