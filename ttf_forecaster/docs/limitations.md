# Limitations and model-risk register

Version 0.1, 2026-10-17. Owner: model-risk review. Read with `research_specification.md` (§8 success
criteria, §9 stop criteria), `point_in_time_policy.md` and `modelling_log.md`. The harness that
backs the statements below is `src/ttf_forecaster/evaluation/diagnostics.py`; the probes are in
`tests/unit/model_risk/`. **No result on real data exists.** Every number quoted here is from
generated data (`data/synthetic.py`) and describes the *machinery*, never TTF.

## 1. What a reader must NOT conclude from the synthetic run

`outputs/synthetic_e2e_report.md` and every model-risk test are SYNTHETIC. In particular:

1. **Not evidence of skill.** "ridge PASS criterion 1" in the planted world means the harness can
   detect a signal that was put there on purpose. It says nothing about storage, LNG or weather
   moving TTF.
2. **Not evidence that fundamentals fail either.** The planted world fails criterion 3 because its
   planted driver is persistent and leaks into price momentum (D-026). The positive control in
   `test_positive_control.py` shows the harness *can* pass criterion 3 when the driver is white.
3. **Not a power statement.** The control signal is large on purpose (about 30 % of daily variance
   at h = 1). With a realistic size (beta = 0.01, about 5 % of daily variance) and 2.5 years of
   history, a storage-only OLS finds it (OOS R² ≈ +5 %, HLN-DM p ≈ 0.006), but the 37-column
   ridge and its price-only twin are indistinguishable (DM p ≈ 0.47): **criterion 3 is under-powered
   at the sample sizes this project will have**. A failed criterion 3 on real data is weak
   evidence of "no fundamental effect"; a passed one is strong.
4. **The trading numbers are not P&L.** Costs are assumptions (D-021), there is no market impact,
   no margin, and the synthetic curve has no bid/ask.
5. **Clark–West significance is not a pass.** In the null world CW gave p = 0.014 for full vs
   price-only (D-026). The harness's Monte Carlo (`nested_test_size`, 100 replications, ridge,
   10 noise regressors) rejects at **11 %** at nominal 5 % for CW with ridge, against 5 % with OLS
   and 3 % for HLN-DM: shrinkage makes the CW adjustment term over-compensate. CW stays
   supplementary (D-020).

## 2. Data limitations

| Topic | Limitation | Consequence |
|---|---|---|
| PIT grades | Storage (AGSI via Bloomberg `CGIE*`), LNG aggregates, flows and ERA5 are grade B: only the latest revised value exists, publication time comes from a lag *rule*. | The canary can only prove features respect the *recorded* publication time; it cannot prove the rule is right. The +1/+3-day lag sensitivity (`diagnostics.shift_publication`) is mandatory, and a result that dies at +1 day is a stop criterion (spec §9). |
| Revisions | Revision behaviour of `LNGG*`/`AHOY*` and of Bloomberg storage history is unknown. | A revised history used with a lag rule still contains the revision's information at the first-release time. Grade B is an optimistic label for any series whose revisions are large relative to its signal. |
| Grade C | Destination-based LNG arrivals, hindsight-assembled text events. | Excluded from clean runs (verified: `test_lng_grade_c_arrivals_never_reach_clean_matrix`). Their "upper bound" experiments are contaminated by construction. |
| Short, crisis-dominated history | Weather PIT archive from mid-2021; clean full-feature window ≈ 2021-07 onwards; 2021-07..2023-06 is the crisis. | Few independent observations: h = 20 overlapping labels on ~5 years give ~60 non-overlapping periods. Criterion 2 (ex-crisis R² > 0) may have < 2 years of ex-crisis test data. `skill_concentration(...).attrs["fragile"]` implements spec §9's "one exceptional year". |
| Unknown identifiers | DA assessment, API2, EUA, power, Norwegian flows/UMM: `requires_user_input` / `requires_discovery`. | H4 is blocked on data. Families without data are reported as absent by the ablation harness, never silently skipped. |
| Timezones | Weather valid days are UTC days; gas days run 06:00–06:00 local; origins are 19:00 Amsterdam (17:00 UTC in summer, 18:00 UTC in winter). | A 12z weather run published at issue + 6 h is visible at a winter origin but not a summer one (`test_weather_12z_visibility_flips_with_dst`); in winter the decision rests on `pub <= as_of` at the exact instant. Evidence real availability times before treating same-day runs as clean. |
| DA event time | `price.ttf.da_eur_mwh` uses gas-day start 06:00 as event time with a 0-day lag, while D-016 says D 00:00. | Conservative (a day late), not a leak; inconsistent with D-016. |

## 3. Methodology limitations

- **DM vs CW, nesting.** Every return model nests the zero benchmark and the full model nests its
  price-only twin. DM is biased against the larger model under nesting (conservative); CW corrects
  for that under OLS assumptions but is oversized with shrinkage (above). Neither accounts for
  model search across families, horizons, targets and estimators.
- **Data snooping.** Ten families × three horizons × several targets × L0–L2 estimators is
  hundreds of comparisons. The spec has no multiple-testing correction. A single DM p < 0.05 among
  them is expected by chance. Any claim should state how many configurations were tried; the
  locked holdout (D-019) is the only clean defence and can be used once.
- **Overlapping labels.** The outer purge uses `label_known_time` and is verified independently
  (`test_walk_forward_training_labels_known_at_refit`). Model-internal tuning now also purges by
  `label_known_time` (passed by `walk_forward`, fixed during this review). For T3 a row gap of h
  is sufficient on a session grid, also when NaN rows are dropped (`TestOverlapAudits`). **For T4/T5
  the `horizon` field is a month count**: anything that converts it into a row count — the model
  `horizon` parameter, `make_target_lags`, the DM/CW HAC lag `h − 1` — is wrong by ~20× per month
  (`test_t5_horizon_field_is_a_month_count_not_an_overlap`). Use
  `diagnostics.label_overlap_rows(targets) + 1` as the model horizon and HAC length for those
  targets.
- **The statistical label starts at the signal's own settlement.** Criteria 1–3 score
  `r(t, t+h)` from the settlement at *t*, which no order can fill (fills are at t+1, D-021). A
  feature whose information arrives before the settlement but which the market prices only on the
  next day (exactly the synthetic planted mechanism: storage published 07:00 on *d* moves the
  *d→d+1* return) passes criterion 1 with skill that is not executable. `variant_execution_delay(1)`
  re-scores on `r(t+1, t+1+h)`; in the toy world a one-session signal's OOS R² falls from > 10 %
  to ≈ 0 under it. Report both.
- **Short-window estimator instability.** In the positive control (seed 13, ~260 training rows,
  37 columns) the ridge inner CV chose alpha = 3e-4 and produced forecasts of ~8 daily sigmas;
  pooled OOS R² ≈ −1.2 although a strong white storage signal was present (strict-xfail
  `test_full_model_not_catastrophic_in_short_window_seed_13`). Near-collinear curve columns with
  disjoint NaN patterns (`curve__sum_minus_win` / `curve__win_minus_sum`, corr −0.99, 58 %
  coverage) make this worse.
- **Missing-not-at-random.** Feature builders return NaN, never 0, for unavailable inputs (audited:
  no `fillna(0)` on a feature path except the reviewed-event coding below). Inside models, NaN is
  median-imputed without an indicator and an all-NaN training column becomes 0
  (`keep_empty_features`, D-011): "not applicable this season" reads as "typical". Text features
  code a missing reviewed event as 0 after `coverage_start` — valid only if the review feed is
  complete. `prices.price_features_from_series` treats a missing return as 0 inside the
  back-adjusted level used for z-scores (minor).
- **Regime features.** Rule-based regimes use trailing percentiles, which stop looking extreme
  during a long crisis (accepted in D-018). Only filtered Markov probabilities reach features
  (canary passes with Markov on). The regime builder is not a registered family, so it is outside
  the default ablation set.
- **Interactions.** Built from parent families and kept when a parent is dropped in the ablation;
  read the `interactions` ablation row with that in mind. The matrix-stage expanding z-score was
  probed and does not depend on the sample start.
- **Feature-shift test is a heuristic.** A genuine fresh signal also collapses when lagged one
  session; the leak flag needs an implausibility prior (OOS R² > 5 % on a return). The canary is
  the gate; the shift test is a cross-check.
- **Benchmark choice.** The zero-return benchmark is correct for futures excess returns only if
  the risk premium is ~0 over the horizon. A model that learns a time-varying risk premium (e.g.
  from curve slope or volatility) beats zero without forecasting fundamentals.

## 4. Economic limitations

- **Risk premium vs forecast.** Positive OOS R² vs zero may be compensation for risk (storage
  risk, winter tail risk), not mispricing. It is harvestable only with matching risk appetite.
- **Persistent fundamentals are in the price.** Any slowly-moving fundamental is partly visible
  in past returns and curve shape, so price-only twins absorb it (D-026). Criterion 3 is the
  binding test and is under-powered (§1.3).
- **Liquidity and costs.** Cost parameters are assumptions; regime-scaled costs are a required
  sensitivity (`execution_cost_grid` covers delay {1, 2} × cost {0, 1, 2, 4}). 2022 spreads were a
  multiple of normal.
- **Structural breaks.** Pre-2022 relationships (Russian pipeline flows, no storage mandate) may not
  transfer. The storage policy target is itself a regime change in the meaning of `full_pct`.

## 5. Operational limitations

- No live feed exists; every publication lag is an assumption from config, not a measurement.
- `strict_ingestion` (ingestion_time ≤ as_of) is not used in backtests; live performance will be
  worse by any ingestion delay.
- Sample-dependent artefacts (feature matrix, splits) are reproducible only with the same config,
  data vintage and origin set; the derived-holdout path warns but is not a lock.
- Parallel agents changed `models/interface.py`, `evaluation/backtest.py` and
  `features/registry.py` during this review; tests written against the old behaviour were converted
  to regression tests.

## 6. Model-risk register

Severity: **H** = can produce a false "useful" claim; **M** = can bias a result materially; **L** =
nuisance or reporting risk.

| # | Risk | Where | Severity | Mitigation in code | Residual risk | Owner |
|---|---|---|---|---|---|---|
| R1 | Look-ahead in a feature builder | `features/*`, `models/regimes.py`, `text/to_features.py` | H | `leakage_canary` (perturb + drop, rebuild on sub-origins); run on every family incl. regime and text, with DST check origins (`test_leakage_audit.py`) | canary trusts recorded `publication_time`; cannot detect a wrong lag rule | feature owners; model-risk |
| R2 | Wrong publication-lag rule (grade B) | `configs/sources/*.yaml`, `data/base.apply_publication_lag` | H | `shift_publication(+1/+3)` sensitivity; grade B flagged | rules unmeasured until live snapshots | data-architecture, gas-fundamentals |
| R3 | Revised data used before the revision's publication | `features/point_in_time.py`, storage | H | resolver by revision + publication; revision probe at ±1 s | Bloomberg history holds only latest vintage | data-architecture |
| R4 | Overlapping-label leakage, outer folds | `evaluation/splits.py`, `backtest.walk_forward` | H | purge by `label_known_time`, re-verified in runner and in tests | none known | backtesting |
| R5 | Overlapping-label leakage, model-internal tuning | `models/*`, `models/interface.py` | H (T4/T5), L (T3) | `fit(label_known_time=)` purge, now passed by `walk_forward`; row gap = h sufficient for T3 | callers that pass `horizon = k` months to models / `make_target_lags` / DM for T4/T5 | forecasting, backtesting |
| R6 | HAC lag too short for month-based targets | `statistical_tests.diebold_mariano/clark_west`, `evaluate_success_criteria(horizon)` | H | `label_overlap_rows` gives the correct lag | not enforced: the function trusts the caller's `h` | backtesting |
| R7 | Non-executable skill counted in criteria 1–3 | `targets/futures_targets.py` label from settlement t | M | `delay_targets` / `variant_execution_delay(1)` | criteria as written still score `r(t, t+h)` | orchestrator (spec) |
| R8 | CW false positives with shrinkage | `statistical_tests.clark_west` with ridge | M | DM gates (D-020); `nested_test_size` measures size (ridge 11 %) | CW still printed next to DM | backtesting |
| R9 | Criterion 3 under-powered | spec §8.3 with short windows and many price columns | M | positive control + power numbers; add-one ablation | a real fundamental effect can fail criterion 3 | orchestrator |
| R10 | Short-window ridge overfit (tiny alpha) | `models/linear.py`, `configs/models.yaml` alpha grid | M | strict-xfail documents it; `coefficient_stability` and `skill_concentration` expose it | unfixed | forecasting |
| R11 | Data snooping across families/horizons/targets | research process | H | locked holdout (D-019); ablation reports every variant | no multiple-testing correction in spec | orchestrator |
| R12 | Crisis/one-year concentration | evaluation | H | criterion 2; `skill_concentration` leave-one-year-out | ex-crisis sample short | backtesting |
| R13 | MNAR imputation hides "not applicable" | `models/interface.TabularPreprocessor`; text coverage coding | L–M | NaN never 0 in features; documented | model cannot distinguish seasonal NaN | forecasting, llm-news |
| R14 | Same-day weather run availability flips with DST | `features/weather.py`, weather lag config | M | documented probe | real availability time unmeasured | weather-power |
| R15 | Roll artefacts in features/targets | `contracts/rolls.py`, `features/prices.py` | M | fixed-contract targets verified against raw settles; spliced returns equal factor return on roll days | real contracts lack common factor — check on real data | market-data |
| R16 | Regime family outside registry | `features/registry.FAMILIES` | L | custom builder path works; canary passes | excluded from default ablation | feature-engineering, regime |
| R17 | Persistent fundamentals proxied by momentum | economics | M | criterion 3, add-one ablation | cannot be removed, only measured | gas-fundamentals |
| R18 | Cost / liquidity assumptions | `configs/backtests.yaml` | M | `execution_cost_grid`, 2× cost criterion | no spread history | user / desk |
| R19 | Registry path errors disable a family silently at enable time | `features/registry.py` | L | was: text path wrong (fixed during review); regression test | — | feature-engineering |
