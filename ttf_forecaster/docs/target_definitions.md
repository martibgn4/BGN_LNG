# Target definitions

Version 0.1, 2026-10-16. Implemented in `src/ttf_forecaster/targets/` and configured in
`configs/targets.yaml`. Every target is a separate experiment; no model is trained on a mix of them.

Items marked **[CONFIRM]** are assumptions about exchange rules that must be checked against the
ICE Endex contract specification or Bloomberg contract metadata before any live result is quoted.
The code reads them from configuration so a correction is not a code change.

## 0. Common conventions

| Item | Convention |
|---|---|
| Exchange | ICE Endex, Dutch TTF Natural Gas Futures (monthly, quarterly, seasonal, calendar) |
| Bloomberg roots (in production use in `Data/lng_report/config.py`) | month `TZT`, quarter `QZT`, season `QQT`, year `QTT` (EUR/MWh). USD/MMBtu twins `TMR/TQR/TSR/TYR` are **not** used for targets |
| Currency / unit | **EUR/MWh**, unadjusted settlement prices. Conversions to USD/MMBtu only for cross-commodity features: `1 MWh = 3.412141633 MMBtu`, EURUSD at the same session (see PIT policy §4) |
| Settlement field | Bloomberg settlement price (`px_settle` is in production use in `NWEBasisModel/config/sources.yaml`) — **[CONFIRM]** that it equals the exchange settlement and not last trade on all dates |
| Price kind | unadjusted futures settlement; no back-adjustment of levels, ever |
| Time zone | all market timestamps in **Europe/Amsterdam**; stored as UTC with `source_timezone` recorded |
| Trading date | the local date of the settlement |
| Publication time of a settlement | conservatively **19:00 Europe/Amsterdam on the trading date** (settlement window is late afternoon **[CONFIRM]**); a feature from date *t* is usable for an origin at or after that time |
| Forecast origin | the close of trading date *t*, after settlement publication. Any action is assumed at the **next** settlement (*t+1*) unless an experiment states otherwise |
| Holiday calendar | historical: the set of dates on which TTF M1 settled (empirical, PIT-safe). Future dates (expiry projection): configured holiday list **[CONFIRM ICE Endex calendar]** |
| Gas day | 06:00–06:00 Europe/Amsterdam. Day-ahead price for gas day *D* is assessed on the last trading day before *D* |
| Seasons | Summer = Apr–Sep, Winter = Oct–Mar; gas year starts 1 Oct; quarters calendar |
| Missing prices | never interpolated for targets. A missing settlement on a trading day makes every target whose window touches it `NaN` with `quality_flag = missing_input`. Rows with NaN targets are dropped per experiment and the count reported |
| Non-positive prices | TTF futures have not settled ≤ 0, but the code does not assume it: log returns on a non-positive price raise `NonPositivePriceError`; spread targets use **differences in EUR/MWh**, never logs |

## 1. Contract rules

**Monthly expiry.** Trading in month *M* ceases at close of business **two business days before
the first calendar day of *M*** **[CONFIRM]**. The rule is configured (`expiry.business_days_before_delivery: 2`)
and every computed expiry is cross-checked against vendor contract metadata when available; a
mismatch is a hard error, not a warning.

**Quarter / season / calendar contracts** cascade into their constituent contracts before delivery
**[CONFIRM]**. To avoid cascade gaps, returns on these targets are computed on the **hours-weighted
strip of constituent monthly settlements** when every constituent month has a settlement on both
dates; otherwise on the listed contract, flagged `construction = listed`.

**Roll rule (for "front month").** The *front contract on date t* is the nearest monthly contract
whose last trading day is **more than `R` trading days after t**, with `R = 5` by default
(sensitivity: 1, 3, 5, 10). Rolling before the final days avoids expiry-week liquidity and the
convergence-to-spot effect.

**Held contract for an h-day return.** For origin *t* and horizon *h* the held contract is the
**front contract under the roll rule R**, unless it stops trading before the exit date *t + h + R'*
(R' = 0 by default), in which case it is the next contract that still trades at exit
(`held_respects_roll_rule: true`, corrected 2026-10-16, D-027). The earlier literal rule ("nearest
contract trading at exit") held the expiring contract through expiry week for short horizons — the
convergence effect R exists to avoid. It remains available with `held_respects_roll_rule: false`. The return is always measured on **one contract**:

    r(t, h) = ln F_c(t+h) − ln F_c(t)

**Continuous series** are built only for *features*, never for targets:
- level series: unadjusted, switching on the roll date (used for levels/spreads only);
- return series: daily within-contract log returns spliced at the roll (equivalent to a ratio
  back-adjusted series). A raw difference across a roll is never a return.

## 2. Targets

| ID | Name | Definition | Unit | Kind | Horizons |
|---|---|---|---|---|---|
| T1 | `da_level` | the **next** day-ahead assessment: DA assessed on session *t+1* (for the next working gas day after it). The DA for gas day *t+1* is assessed on *t* and is already known at the origin (corrected 2026-10-16, D-027) | EUR/MWh | spot assessment level | 1 |
| T2 | `m1_settle` | settlement of the *held* front contract at *t+h* | EUR/MWh | futures level | 1, 5, 20 |
| T3a/b/c | `m1_logret_{h}` | `r(t, h)` on the held contract (§1) | log | futures return | 1, 5, 20 trading days |
| T4 | `da_month_avg_k` | arithmetic mean of day-ahead prices over all gas days of calendar month *M+k* (month of *t* = *M*) | EUR/MWh | average spot | k = 1, 2 |
| T5 | `delivery_excess_k` (not emitted for origins on/after the contract's last trading day) | (a) **futures convergence:** final settlement of month *M+k* − its settlement at *t*; (b) **physical:** T4 − settlement of month *M+k* at *t*. Reported in EUR/MWh and as log ratio | EUR/MWh, log | excess return vs forward | k = 1, 2, 3 |
| T6 | `q_logret_h`, `season_logret_h` | `r(t, h)` on the prompt quarter / next season, strip construction | log | futures return | 20, 60 |
| T7a | `spread_m1m2_chg_h` | Δ over *h* of (held front − next month), both contracts fixed at *t* | EUR/MWh | spread change | 5, 20 |
| T7b | `spread_q1q2_chg_h` | Δ of (prompt quarter − next quarter) strips | EUR/MWh | spread change | 5, 20 |
| T7c | `spread_sum_win_chg_h` | Δ of (next Summer − following Winter) strips | EUR/MWh | spread change | 20, 60 |
| T7d | `spread_win_sum_chg_h` | Δ of (next Winter − following Summer) strips | EUR/MWh | spread change | 20, 60 |
| T8 | `rv_h` | `sqrt(252/h · Σ r_d²)` over daily within-contract log returns on (t, t+h] | annualised | realised vol | 5, 20 |
| T9a | `dir_h` | `1[r(t,h) > 0]` | binary | direction | 1, 5, 20 |
| T9b | `tail_h` | `1[|r(t,h)| > k · σ_t · sqrt(h)]`, σ_t = 60d EWMA daily vol known at *t*, k = 1.5 (sensitivity: fixed 10 %) | binary | tail event | 1, 5, 20 |
| T9c | `regime_change_h` | `1[regime(t+h) ≠ regime(t)]`, regimes per `models/regimes.py` evaluated with information at each date | binary | transition | 20, 60 |

### Benchmarks per target

| Target | Market benchmark (principal) | Other benchmarks |
|---|---|---|
| T1 | front BOM/M1 settlement at *t* (**[CONFIRM]** BOM availability) | no-change DA, seasonal mean |
| T2 | F_c(t) (no change) | — |
| T3, T6 | 0 | rolling mean, AR(p) |
| T4 | settlement of month *M+k* at *t* | seasonal mean, no-change DA |
| T5 | 0 | rolling mean of past T5 realisations (known at *t*) |
| T7 | no change in spread | AR(1) on spread changes |
| T8 | trailing realised vol (h-matched) | EWMA, HAR(1,5,22) |
| T9 | base rate in training window | price-only logistic |

## 3. Label availability (for purging)

A label for origin *t* and horizon *h* is **known at the close of t+h** (T4/T5: after the last gas
day of the delivery month and its DA publication). Training at origin *t₀* may only use labels whose
known-time ≤ *t₀*. `targets/` returns a `label_known_time` column with every target; `evaluation/splits.py`
purges on it. This is the mechanism that prevents overlapping-label leakage.

## 4. Open confirmations

1. ICE Endex TTF monthly expiry rule and holiday calendar.
2. Whether Bloomberg `px_settle` on `TZT` generics/contracts is the official settlement on all dates
   since 2015, including the 2022 margin/liquidity episodes.
3. Quarter/season cascade mechanics and listing depth of monthly contracts.
4. Source and identifier of the TTF day-ahead assessment (exchange spot index vs broker vs PRA).
5. Whether bid/ask, OI and volume are entitled for `TZT`.
