# Modelling decision log

Format per entry: date · decision · alternatives · rationale · evidence · consequences · revisit?

---

### D-001 · 2026-10-16 · Project location and name
- **Decision:** new self-contained project `BGN_LNG/ttf_forecaster/`, package `ttf_forecaster`, `src/` layout.
- **Alternatives:** extend `NWEBasisModel` or `GasImbalanceModel`; shared library across siblings.
- **Rationale:** siblings are each self-contained research projects with their own config and
  cache; coupling would make reproduction depend on another project's fetch state.
- **Consequences:** some adapter code duplicates sibling patterns. Identifiers are re-used by
  reference (documented provenance), not by import.
- **Revisit:** when productionising — a shared `bgn_marketdata` package would remove duplication.

### D-002 · 2026-10-16 · Python environment
- **Decision:** run in the existing `..\env_local` (Python 3.14.2); `requires-python = ">=3.11"`.
- **Alternatives:** create a 3.11 venv as the brief suggests.
- **Rationale:** env_local already holds pandas, scikit-learn, statsmodels, LightGBM, pyarrow,
  pytest, blpapi/xbbg; creating a new environment means installing packages, which the brief says
  not to do without justification. Code avoids 3.12+ only syntax so 3.11 remains supported.
- **Revisit:** at productionisation (pin a lockfile; CI on 3.11 and current).

### D-003 · 2026-10-16 · pandas, not polars
- **Decision:** pandas 2.x for all frames.
- **Rationale:** installed; scikit-learn/statsmodels/LightGBM interop; every sibling uses it; data is
  small (daily, tens of thousands of rows per series). xbbg 1.x returns polars — we do not use 1.x.
- **Revisit:** if feature building on intraday or vessel-level data becomes slow.

### D-004 · 2026-10-16 · Storage: Parquet via pyarrow; DuckDB optional, not installed
- **Decision:** Parquet files behind a `Store` interface; no DuckDB dependency in v1.
- **Rationale:** pyarrow installed; DuckDB not; nothing in v1 needs SQL. The interface allows a
  DuckDB or database backend later.

### D-005 · 2026-10-16 · Schema validation in-house, not pandera; no new installs
- **Decision:** a small typed column-contract validator in `data/base.py` / `data/validation.py`.
  ruff, mypy, pre-commit, xgboost, catboost, hmmlearn, optuna, mlflow are declared as *optional*
  extras in `pyproject.toml` but not installed.
- **Rationale:** brief forbids installs without justification; the validator needed is ~100 lines.
  XGBoost/CatBoost add nothing to a first test that LightGBM cannot answer.
- **Consequences:** lint/type-check cannot be run until the user approves installing ruff/mypy
  (requested at the data boundary). Tests run.
- **Revisit:** at the data boundary.

### D-006 · 2026-10-16 · Primary targets are excess returns vs the forward, on fixed contracts
- **Decision:** primary targets T3b/T3c (5/20d front-month log return on a single held contract) and
  T5 (delivery excess vs forward). Benchmark = zero excess return.
- **Alternatives:** price levels; continuous-series returns.
- **Rationale:** levels are dominated by persistence; continuous-series returns contain roll jumps
  that are not P&L.
- **Revisit:** no.

### D-007 · 2026-10-16 · Spark credentials via client id/secret env vars
- **Decision:** `.env.example` uses `SPARK_CLIENT_ID`, `SPARK_CLIENT_SECRET`, `SPARK_BASE_URL`;
  the brief's `SPARK_API_KEY` is not used.
- **Evidence:** existing desk code authenticates with OAuth2 client credentials
  (`Utils/utils_spark.py`), which reads exactly these env var names.
- **Consequences:** the tracked `adhoc_scripts/client_credentials.csv` is not read by default
  (security finding in `environment_audit.md`).

### D-008 · 2026-10-16 · Clean backtest = PIT grades A and B only
- **Decision:** grade C features are excluded from the headline backtest; grade B features require a
  publication-lag sensitivity.
- **Rationale:** see `point_in_time_policy.md` §3.

### D-009 · 2026-10-16 · No Level 5 (deep) models in v1
- **Decision:** deep sequence models are out of scope until a Level 1/2 model passes the success
  criteria on at least one primary target.
- **Rationale:** ~5 PIT heating seasons cannot support them; spec §9 stop criteria.

### D-010 · 2026-10-16 · LLM text events: extraction only, reviewed, excluded from clean backtest
- **Decision:** the LLM only extracts and classifies events. A numeric impact stays null unless a verbatim
  quote with a literal mcm/d unit, or a registered asset-capacity reference, supports it. Human review is
  mandatory in v1. Text PIT grades: live = A; historical back-fill = B (24 h review lag, assumed);
  assembled with hindsight = C (the default until shown otherwise). The `text` family is ablation-only
  (`clean_backtest_eligible: false`).
- **Alternatives:** convert MSm3/d, GWh/d or kWh/h to mcm/d with a configured GCV (open question for the user).
- **Rationale:** spec §8; a historical event set assembled after the fact cannot be point-in-time.
- **Governance:** no proprietary text goes to any external model, including Anthropic's API, without a
  recorded user/compliance approval.
- **Source:** llm-news-agent report.
- **Revisit:** once a live UMM archive exists.

### D-011 · 2026-10-16 · Model interface and in-fit tuning
- **Decision:** one `ForecastModel` fit/predict interface (`models/interface.py`); every tuning step
  (alpha, AR order, early stopping, calibration, conformal residuals) happens inside `fit`, using
  time-ordered inner splits from the training tail with a gap of `horizon` rows. `SimpleImputer` always runs
  with `keep_empty_features=True`.
- **Rationale:** information from outside the training window cannot enter a fit; the desk lesson on
  silently dropped all-NaN columns.
- **Consequence / open:** gaps are counted in rows, so the runner must pass a contiguous trading-day grid,
  or purge by `label_known_time` itself (it does). Revisit if rows are dropped inside the training window.
- **Source:** forecasting-agent.

### D-012 · 2026-10-16 · ARIMAX not implemented; ECM gated
- **Decision:** no ARIMAX. With overlapping h-day labels, its autoregressive terms use y(t−1, h), which is
  not known at the origin when h > 1. An AR(p) on lags known at the origin replaces it. The error-correction
  model fits only if an Engle–Granger test on the training window gives p < 0.05; otherwise it refuses.
- **Revisit:** no.

### D-013 · 2026-10-16 · Classifier predictions
- **Decision:** for classification models the PREDICTION_SCHEMA `prediction` column holds P(event), and
  metrics read `prediction`. The optional `prob` column is not used. The threshold and calibration method
  are identified through `model_id`, which hashes the parameters.
- **Source:** forecasting-agent challenge to the contract.

### D-014 · 2026-10-16 · Level 6 ensembles deferred
- **Decision:** the stacking and weighting code exists, with leakage guards, but stays unused until a
  single model passes Levels 1–2 on a primary target (spec §9).

### D-015 · 2026-10-16 · Contract fixes from the data-architecture review
- **Decisions:**
  - `.gitignore` uses `/data/`. Plain `data/` also ignored the source package `src/ttf_forecaster/data/`.
  - In `apply_publication_lag`, an ambiguous wall time (the repeated hour at the autumn DST change)
    resolves to the later, standard-time instant: the conservative choice. It no longer becomes NaT.
  - The resolvers raise `DataError` when rows match in key, revision and `publication_time` but differ in
    value. Row order must never decide the as-of view. `ingestion_time` is the last sort key, and the sort
    is stable.
  - New `resolve_curves_as_of`, so curve reads also go through the point-in-time module (challenge from the
    regime agent).
- **Evidence:** strict-xfail tests in `tests/unit/data_arch/test_pit_edge_cases.py`, now passing.

### D-016 · 2026-10-16 · Daily event_time convention
- **Decision:** a daily fact labelled with date D (settlement date, gas day D, storage day D) has
  `event_time` = D 00:00 Europe/Amsterdam, stored in UTC. The gas-day 06:00 start is a labelling convention
  of the source, recorded in its config, not shifted into `event_time`. Hourly and forecast `valid_time`
  values keep their true instant.
- **Consequence:** the synthetic weather generator (UTC midnight) is aligned during integration.

### D-017 · 2026-10-16 · Observations dated after as_of
- **Decision:** `resolve_as_of` keeps observation rows with `event_time > as_of` when they were published by
  `as_of` (for example, published maintenance schedules). Value-of-today feature helpers
  (`latest_value_per_series`) take only `event_time <= as_of`. The policy wording "data error" applies only
  to series whose config says they cannot be forward-dated.

### D-018 · 2026-10-16 · Regimes
- **Decision:** rule-based score regimes with thresholds fixed in advance in `configs/regimes.yaml`.
  Anti-flicker is a band of 1 plus a minimum duration of 3 days, applied forward-only. Markov and HMM
  features use filtered probabilities only; smoothed output is a diagnostic. The 2021-07..2023-06 crisis
  window exists only as an evaluation label and is rejected on the feature path. T9c uses the hysteresis
  `code`.
- **Accepted weakness:** trailing-percentile indicators stop looking extreme during a multi-year crisis.
  This is accepted and will not be "fixed" with hindsight.
- **Pending:** the gas-fundamentals challenge pass must sign off thresholds and signs before any real-data
  run. The regime CUSUM `h_var` = 12 was set on synthetic data.

### D-019 · 2026-10-16 · Locked holdout fixed in advance
- **Decision:** `holdout.start = 2025-10-01` (`configs/backtests.yaml`), the ~12 months before the audit
  date. It was set before any real data was pulled, so no result can have influenced it.
- **Alternative:** derive the holdout from the data range. Rejected: a derived holdout drifts as data
  accrues and is not a lock.
- **Revisit:** only if the real history ends before 2026-06, in which case the user decides. Never after
  seeing results.

### D-020 · 2026-10-16 · Success criterion 1 tests: DM and Clark–West
- **Decision:** spec §8 criterion 1 stays as written: the per-fold win share must be ≥ 75%, and the
  pass/fail reads the HLN-corrected DM test. Clark–West (CW) is always reported next to it. A result with
  CW p < 0.05 but DM not significant is reported as "population predictability, not demonstrated as
  exploitable". It never counts as a pass.
- **Evidence:** in the backtesting agent's synthetic planted world (β = 0.004, h = 1), OOS R² was +0.1%,
  HLN-DM p = 0.47 and CW p = 0.009. A real but weak signal is invisible to DM.
- **Rationale:** changing the criterion to CW before any result exists would be legitimate. Keeping DM
  as the gate is the more conservative choice for a trading claim, because CW measures predictability,
  not exploitable skill.

### D-021 · 2026-10-16 · Evaluation conventions
- **Fill timing:** next-settlement fill (delay 1); same-settlement execution is refused.
- **P&L:** marked on the held contract only, and a roll trades both legs.
- **Criterion 4 (net Sharpe > 0.5 at 2× cost)** uses NON-overlapping h-day trades (`rebalance_every = h`)
  and must be reported with the top-5% P&L concentration. On the synthetic null world, a daily-rebalanced
  tilt reached a net Sharpe of 0.53–0.70.
- **Costs** (0.025 half bid-offer + 0.005 fee EUR/MWh per side) are ASSUMPTIONS until the user supplies a
  spread history. A regime-scaled cost run is a required sensitivity.
- **Pending:** one tuning-split implementation. Models tune on row-based tails of what the runner passes
  them; the runner purges by `label_known_time` before calling fit. That holds while the training grid is
  contiguous. The software-quality review should check it.

### D-022 · 2026-10-16 · Weather conventions
- **Degree days:** Eurostat convention, HDD = 18 − T when T ≤ 15 °C, else 0; CDD = T − 21 when T ≥ 24 °C,
  else 0. Computed per point, then weighted. The sibling project's "15.5 °C Eurostat" is the UK base.
- **Normals:** for gas year G, resolved as of the start of G (confirmed as the reading of PIT §5.3).
- **Open-Meteo `_previous_dayN` publication:** end of day V−N + 8 h. Conservative; it costs one day of
  lead.
- **Revisions:** NaN, never 0, when no newer run exists.
- **Region weights:** `requires_user_input`.
- **Ensemble features:** live accumulation only; there is no historical archive.
- **CSS/CDS efficiencies:** null until the desk supplies them.

### D-023 · 2026-10-16 · LNG data rules
- **Bloomberg LNG aggregates (`LNGG*`, `AHOY*`):** grade B, usable from D+1 23:00 Amsterdam (transits from
  D+2), with a ±1-day sensitivity. Revision behaviour is unknown until live snapshotting measures it.
- **Spark assessments:** 19:00 London on the release date, grade A [CONFIRM].
- **Arrival forecasts:** built only from destinations and ETAs as declared at the issue time. They are
  grade C until a vintaged vessel source exists, so they are excluded from the clean backtest and appear
  only in a labelled contaminated experiment.
- **Realised arrivals:** kept separately as grade-B observations and used only for the lagged
  forecast-error feature.
- **Usage limits:**
  - An empty netback payload is a hard error.
  - AHOY legs and LNGGTR counts are used only as rolling sums and shares.
  - IKA1/IKD1/IKI1 are never used as raw returns, because they roll.
- **Contract:** realised arrivals and destination statistics are documented backend extensions of
  `LngDataAdapter`, not part of the protocol. `PROVENANCE_COLUMNS` is now public.

### D-024 · 2026-10-16 · Fundamentals and balance
- **Note on references:** section references "§4A–D" in the agent briefs mean the user's original brief
  (§4 Data universe), not `research_specification.md`.
- **Storage:** AGSI via Bloomberg `CGIE*` is grade B at D+2 07:00, with a +1/+3-day sensitivity. Seasonal
  normals use only gas years before min(origin gas year, last-data gas year).
- **Balance:** one GCV, 10.55 kWh/m³ (the GasImbalanceModel decision). The residual is explicit; a missing
  component makes it NaN with a reason; linepack is absorbed in the residual.
- **Projection:**
  - A profile in % of working gas volume, from prior gas years only, adjusted per scenario, plus a seeded
    same-season 10-day block bootstrap.
  - Scenario magnitudes are assumptions awaiting desk sign-off.
  - Scenario-variant columns are deterministic transforms of the normal projection, so they are excluded
    as model inputs and kept for reporting only.
  - H1's "vs scenario-projected path" feature is `storage__proj_eos_dev_vs_normal_pct_pts`.
- **Uninformative or unstable columns:** `storage__normal_*` (pure calendar) and
  `flows__no_utilisation_pct` (sign not identifiable) are excluded from models.
  `storage__policy_required_injection_gwhd` must be capped or logged near the deadline.
- **Flows:** implemented but inert on real data until a Norwegian flow and UMM source exists. H4 is
  blocked on data, not on code.
- **Contract:** the `sign` conversion in `bloomberg_fundamentals.yaml` is applied once, in
  `data/fundamentals.py`, never in the Bloomberg adapter.
- **Seasonal NaNs:** NaN in season-specific features (e.g. `prob_eos_below_*` in winter) is meaningful and
  must not be imputed as 0.

### D-025 · 2026-10-16 · Feature assembly
- **Computation:** features are computed per trading date, with `available_time` = cumulative maximum of
  the inputs' publication times, and joined onto origins with a 7-day staleness cap.
- **Revised settlements:** a revision is used only from its own publication time, never back-filled.
- **Cross-commodity pairing:** pairs are formed by session date on the same delivery month and become
  usable at the latest component publication. Never aligned by calendar date.
- **Interactions:**
  - The feature matrix uses an expanding z-score.
  - `InteractionTransformer` is refitted on training rows inside each fold.
  - Positive means bullish tightness.
  - Grade = worst of the inputs.
- **Leakage canary:** `pipelines.build_features.leakage_canary` (perturb and drop modes, rebuild on
  origins ≤ o, bit-identical required) is the gate a family must pass before it is enabled in
  `configs/features.yaml`.
- **One roll definition:** `contracts/rolls.py` is the canonical roll. `features/prices.py` uses it through
  `return_series_provider`; its own reference splice is a fallback only. Aligned at integration.
- **Placeholders:** the HH netback constants (1.15, 3.0) and the maintenance-season months are
  illustrative [CONFIRM with desk].

### D-026 · 2026-10-16 · Synthetic end-to-end run: what it showed
- **Built:** `pipelines/synthetic_e2e.py` plus the `ttf_forecaster.cli synthetic-e2e` command.
  - Synthetic vendor frames go through the real ingest path into the Parquet store and are read back.
  - Then: point-in-time features, T3 h=5 fixed-contract targets, purged walk-forward, §8 criteria 1–3,
    trading at 1× and 2× cost, immutable run artefacts.
  - Report: `outputs/synthetic_e2e_report.md`.
- **Null world:** every model fails criterion 1, as it should (ridge: HLN-DM p ≈ 0.48, OOS R² ≈ 0.001).
  **But** Clark–West for full vs price-only gave p = 0.014: a false positive in a world with no signal.
  This is direct support for D-020: Clark–West alone never counts as a pass.
- **Planted world:** ridge and LightGBM pass criteria 1–2. So does the **price-only twin**, and
  criterion 3 fails (HLN-DM p = 0.98). Cause: both planted drivers tried, the storage level and the
  flow change, are serially correlated in the synthetic world, so past returns carry them through
  momentum. This is a property of the generator, not a leak (the price features fail in the null world).
- **Consequences:**
  1. The planted world validates criteria 1–2 detection, but it is **not** yet a positive control for
     criterion 3. The model-risk review owns building one (a white, storage-only driver).
  2. The lesson transfers to real data: any persistent fundamental is partly visible in price history,
     so criterion 3 (beat the price-only twin) is the binding test, not criterion 1.
- **Fixes made along the way:** `walk_forward` adds a default zero benchmark to every call, so the
  pipeline keeps benchmarks from the first call only. The software-quality review should decide whether
  `walk_forward` should do this itself.

### D-027 · 2026-10-16 · Target definition corrections (from the market-data review)
- **T1:** the label is the *next* day-ahead assessment (session t+1). As first written, the label was
  already known at the origin.
- **Held contract:** the held contract respects the roll rule (front under R, unless it expires before
  exit). The literal rule held expiring contracts during expiry week.
- **T5:** not emitted for origins on or after the last trading day.
- **Strips:** a strip is used only when every constituent month settles on both dates; otherwise the
  listed contract, otherwise NaN.
- **Expiry rule:** the TTF last trading day is cross-checked against the vendor's `LAST_TRADEABLE_DT`.
  The field is marked known: it is used via `bdp` in `adhoc_scripts/build_india_options_xlsm.py:130`.
  A wrong field fails loudly through `ExpiryMismatchError`.
- **Config:** Bloomberg roots live in `configs/sources/bloomberg.yaml`, not `instruments.yaml`.
- **Open [CONFIRM]:** ICE Endex expiry rule and holidays, `px_settle` = official settlement, the T4
  weekend gas-day treatment, and the NBP quote unit.

### D-028 · 2026-10-17 · Backtesting challenge: decisions
- **`walk_forward(benchmark_factories={})` now means no benchmarks.** Only `None` adds the default zero
  benchmark.
- **Pipeline changes:**
  - Benchmarks are zero, rolling mean (60) and AR on the target's own known lags. Criteria are reported
    both vs zero and vs the best benchmark (lowest pooled OOS MSE, chosen ex post, so conservative
    against the model).
  - Feature coverage selection uses only rows before the first test fold. Previously it used the full
    sample (a PIT §5.4 breach, small in effect).
  - `clean_pit` is derived from the features' PIT grades, never asserted.
  - Breakdowns by year and by PIT rule regime are added.
  - The price-only twin gets its own `feature_hash`.
  - Report caveats: the ex-crisis column means nothing in the synthetic world; planted P&L scales
    with β; a positive DM statistic means the full model is worse.
- **Forecasting agent asked to make `horizon` mandatory and to purge inner, early-stopping and
  calibration splits on `label_known_time`.** The row gap is wrong for T4/T5. This affects model
  selection, not the outer OOS metrics.
- **Clark–West:** before it is quoted even as supplementary, its size must be checked with the actual
  estimators over many null seeds (model-risk).
- **Minimum outputs for any real-data result:**
  - backtesting challenge §4: per fold vs zero and vs best benchmark, DM + CW and the number of specs
    tried;
  - by year and regime, with and without the crisis, and with the best year dropped;
  - lag sensitivity, canaries, PIT grades;
  - trading at 1× and 2× cost plus regime-scaled cost;
  - calibration by regime;
  - full provenance.

### D-029 · 2026-10-17 · Gas-fundamentals physical review: decisions
- **Interaction sign bug** (products of z-scores make warm-and-full "bullish"): every leg becomes
  one-sided (relu) before multiplying. The real input columns replace the placeholders.
  Low wind × demand is replaced by residual-load revision × gas-in-merit. The terminal interaction is
  redefined as no spare capacity × high arrivals. Expiry × prompt stress uses a deseasonalised M1–M2.
  (Feature-engineering agent.)
- **Regimes:**
  - storage z in % of working gas volume with a 2.5 pp floor and ≥ 3 prior gas years;
  - the gas-year boundary bug is fixed;
  - M1–M2 is scored as a deviation from the same-month normal;
  - price + vol contribution is capped at +2, and tight or crisis requires a physical indicator;
  - the price percentile is relabelled "market stress" (504 days);
  - the freight-adjusted Spark arb replaces raw JKM–TTF. (Regime agent.)
- **Expected feature signs are pre-registered** in `docs/feature_signs.md`.
- **Price-derived columns count as price information in the criterion-3 twin:** spark, dark and switch
  spreads, and cross-commodity spreads against TTF.
- **The price-only twin always includes the curve family.** Storage levels are expected to be priced,
  so the incremental candidates are flow surprise vs normal, tail probabilities, the policy gap, and
  storage × weather-revision.

### D-030 · 2026-10-17 · Runner passes label_known_time into model fits
- **Decision:** `walk_forward` passes `label_known_time` (aligned to the training rows) to every model whose
  `fit` accepts it. Inner CV, early stopping, calibration and conformal tails then purge by when labels
  are known, not only by a row gap. `horizon` is now mandatory in every model.
- **Effect on the synthetic planted world:** ridge's fold win share fell from 0.83 to 0.67, so ridge no
  longer passes criterion 1; LightGBM and the price-only twin still pass. Reported as measured.
- **Invariant check:** across 4,662 full-run predictions, `train_end > origin` occurred 0 times and
  `label_known_time ≤ origin` occurred 0 times.

### D-031 · 2026-10-17 · Software-quality review: decisions
- **`success_criteria.min_folds = 4`.** Below that, criterion 1 is "not evaluable" and fails.
  Evidence: a 2-fold synthetic NULL world "passed" (HLN-DM p = 0.004).
- **Run manifest config hash** now covers every resolved config (pipeline, features, models,
  backtests, targets, instruments, regimes).
- **Still open, assigned:**
  - SQ-01 credential file tracked in the parent repo: **user action**.
  - SQ-04 thread `config_dir` through every loader in `run_world`: orchestrator.
  - SQ-05 reconcile the power/coal/EUA/FX ids between `power.yaml` and `bloomberg.yaml`: market-data and
    weather-power. Until then, power price features cannot resolve.
  - SQ-06 wire `return_series_provider` to `contracts/rolls.py`, with holidays from `instruments.yaml`.
    Today the two rolls agree on 1,565 synthetic sessions.
  - SQ-07 one tuning-split owner.
  - SQ-09 run_id derived from the config hash.
  - Git SHA and package versions in the manifest; a lockfile frozen from env_local before the first
    real result.
- **Accepted:** no module rejected; two runs are bit-identical; 0 Future/DeprecationWarnings.

### D-032 · 2026-10-17 · Model-risk review: decisions
- **No feature leak found.** The leakage canary (perturb and drop modes, check origins on DST days)
  passes on every family:
  - price, curve, cross_commodity, calendar, interactions;
  - storage, flows, lng, weather, power;
  - regime, including filtered Markov;
  - text.
- **F1, month-based horizons (High).** T4/T5 store `horizon` in months, while overlap is ~21 sessions per
  month. Until fixed, **T4/T5 must not be run.** The model horizon, `make_target_lags` and the DM/CW HAC
  lag must use `label_overlap_rows + 1`, never the month count. Owners: backtesting and forecasting
  agents. T3 (session horizon) is unaffected.
- **F4, Clark–West size.** ridge 11% vs OLS 5% at nominal 5% (100 null replications). This confirms D-020:
  DM is the only gate, and CW is printed with a size caveat.
- **F5, short-window ridge overfit.** An inner-CV alpha of 3e-4 at p/n ≈ 0.14 gave a pooled OOS R² of −1.2.
  Required before real data:
  - a minimum penalty or one-standard-error alpha rule (forecasting);
  - drop one of each near-duplicate curve pair, e.g. `curve__sum_minus_win` / `curve__win_minus_sum`
    (feature engineering).
- **F6, criterion 3 is under-powered.** Same 5%-of-variance signal: it fails over 2.5 years (DM
  p ≈ 0.47), passes over 7 years. Spec amendment: a failed criterion 3 is reported with a power estimate
  and is not by itself evidence of "no fundamental effect" (stop criterion §9 then needs ≥ 5 years).
- **F7, executable label (Medium; adopted).** Criteria 1–3 score r(t, t+h), but trading fills at t+1.
  A signal priced at the next settlement scores as skill and cannot be traded. Spec amendment:
  criteria 1–3 are reported on both labels, and **a pass requires the executable label
  r(t+1, t+1+h)** (`diagnostics.delay_targets(1)`). The synthetic planted world is exactly this case.
- **F8, weather run visibility flips with DST at the 19:00 origin.** Real availability times must be
  evidenced (weather-power agent).
- **F9:**
  - add missingness indicators for MNAR features;
  - register the regime family for ablation;
  - DA `event_time` convention vs D-016.
- **Positive control for criterion 3:** a white storage driver passes for 2 of 3 seeds; the failing seed
  is the F5 overfit.

### D-033 · 2026-10-19 · Pre-real-data remediation (D-031 / D-032 must-close items)
Implements the blockers and majors that had to close before any real-data run. Nothing live was
called, no identifier was added, nothing was installed or committed. Synthetic results are reported
as measured; the `null` world still has no criterion-1 pass.

- **F1, session horizon (D-032).** `targets/common.py` now owns `label_overlap_rows` (re-exported by
  `evaluation/diagnostics.py`) and `session_horizon(targets, declared)` = overlap + 1.
  `session_horizon` refuses a declared horizon that disagrees with the overlap, which is what a
  month count looks like. Used by:
  - `pipelines/synthetic_e2e.py`: the model horizon is derived after the targets are built;
  - `evaluation/statistical_tests.py:session_hac_lag`: DM / CW HAC lag = h − 1;
  - `models/baselines.py:lag_horizon`: the `make_target_lags` horizon.
  T4/T5 stay unrun. T3 is unchanged (overlap = h − 1, so session horizon = h).
- **F5, short-window ridge (D-032).** The active fix is the minimum penalty.
  - `models/linear.py`: `min_alpha` floors the penalised-linear grid (values below it are dropped; an
    all-below grid raises). Default 0 in code; `configs/models.yaml` sets `min_alpha: 1e-2` for ridge
    and ridge_family_pca. The F5 failing alpha (3e-4) is below the floor.
  - Curve pair: `curve__win_minus_sum` dropped; `curve__sum_minus_win` kept (corr −0.99, disjoint
    NaN patterns).
  - The strict xfail in `tests/unit/model_risk/test_positive_control.py` is removed. The probe now
    passes as a plain test with the floor.
  - **One-standard-error rule: implemented, tested, NOT the default.** `grid_search_time_cv(rule="one_se")`
    exists and is unit-tested, but `selection` defaults to `"min"` everywhere. Reason: with
    `one_se` and `min_alpha`, the **short null world** (integration test, 4 outer folds) gave ridge a
    criterion-1 pass (fold win share 0.75, HLN-DM p 0.042), a false positive. With the floor alone,
    ridge fails (p 0.46), as it did before. The review allowed either remedy; the floor is the minimal
    change that addresses the documented failure. One-SE was not kept because it was not needed for
    it and it adds shrinkage that the null world punishes. Single draw, single seed: this is evidence
    of how the rule behaves here, not a size study.
  - No test threshold was changed for F5. An intermediate threshold change in
    `test_diagnostics.py` was reverted when the floor-only fix kept the toy signal above 0.1.
- **F7 / amendment A1, executable label.** `pipelines/synthetic_e2e.py`:
  - `_executable_predictions` re-scores the same forecasts on `delay_targets(targets, 1)`. Models are
    not re-fitted on the shifted label; this is a conservative check, not a re-estimation.
  - `WorldResult.criteria_exec`, `price_only_test_exec` and `headline_pass` are new.
  - `headline_pass` = criteria 1 and 2 pass on BOTH labels. Criterion 3 is reported on both labels.
  - `reporting/research_report.py` shows the headline table, the executable criteria table and the
    executable criterion-3 line.
  - Synthetic result (`outputs/synthetic_e2e_report.md`, regenerated after the F5 change). `null`:
    no headline pass (ridge and lightgbm fail); criterion 3 HLN-DM p = 0.53 on r(t, t+h) and 0.51
    on r(t+1, t+1+h). `planted`: lightgbm passes the headline (both labels); ridge fails; the
    price-only ridge twin passes criterion 1 on r(t, t+h) (it is not in the headline table, which covers
    the full models only); criterion 3 HLN-DM p = 0.96 on r(t, t+h) and 0.93 on r(t+1, t+1+h), i.e. the full model
    does not beat its price-only twin in this planted run. All numbers are SYNTHETIC and are not
    evidence about TTF.
- **SQ-04, config_dir threaded.** `run_world` passes `config_dir` to `CurveContext.from_curve`,
  `make_model` (pipeline, benchmarks, price-only twin), `SplitConfig.from_config`,
  `TradingConventions.from_config`, `load_regime_config`, the feature config
  (`features.yaml` from the directory) and the price family's roll provider. The manifest hash
  (`full_cfg`) is unchanged in this entry; widening it is SQ-02, not done here.
- **SQ-05, source ids (single owner = `bloomberg.yaml`, no tickers invented).**
  - `power.yaml` no longer defines the five duplicated series. `features.prices` now references
    `price.power_de.base_m1_eur_mwh`, `price.ttf.m1_eur_mwh`, `price.coal_api2.m1_usd_t`,
    `price.eua.front_eur_t` and `fx.eurusd.spot_usd_per_eur`.
  - `bloomberg.yaml` gains `price.ttf.m1_eur_mwh` (`requires_discovery`, no ticker). It is the TTF M1
    settle from the curve store; no adapter serves it yet.
  - **EUA decision:** front, not Dec. `bloomberg.yaml` has no Dec EUA id and none is invented; a Dec
    series needs a user-supplied identifier.
  - Open: until an adapter serves these series, power price features still cannot resolve live. The
    config test enforces the single owner and that the references resolve in the config, not that a
    vendor serves them.
  - Test: `tests/unit/model_risk/test_review_remediation.py`.
- **SQ-06, roll splice.** `contracts/rolls.py:canonical_front_series` wraps `targets.common.CurveContext`:
  - the same chain, roll rule R (`targets.yaml`) and session calendar (`instruments.yaml`) as the targets;
  - availability and PIT grade come from `features.prices.prepare_curve`, so the reference
    availability semantics are kept;
  - `front_month` is the delivery start of the front contract on each observed trade date.
  `configs/features.yaml: price.params.return_series_provider` now points to it. The reference
  splice in `features/prices.py` remains as a cross-check (set the key to null to run it).
  `features.common.holidays` equals `instruments.calendars.TTF.future_holidays` (both `[]`); a test
  pins the equality, since the price family's `holidays` no longer affects the provider.
- **Tests.** Changed test code:
  - `test_prices.py::test_curve_features_and_strip_weights` now checks `sum_minus_win` against an
    origin whose following winter is in the fixture (its old check used `win_minus_sum`, removed);
  - `tests/unit/weather/test_power_features.py` builds its synthetic price observations under the
    owned ids (SQ-05);
  - `test_positive_control.py`: xfail marker removed.
  New tests: `tests/unit/model_risk/test_review_remediation.py`.
  Result after the final change: full suite **685 passed, 0 xfailed** (673 passed + 1 xfail before; +11 new
  tests, the xfail is now a plain test). `outputs/synthetic_e2e_report.md` regenerated with
  `python -m ttf_forecaster.cli synthetic-e2e` (run with `PYTHONPATH=src`).
- **Not done (out of this brief):** SQ-02 manifest hash and code version, SQ-07 single tuning split,
  SQ-09 run_id from the config hash, SQ-01 credential (user action), F6 power analysis.

### D-034 · 2026-10-19 · User-supplied tickers, Bloomberg discovery, power sources
- **Inputs:**
  - User instruction (2026-10-19): use the found API2 and EUA candidates; German power `DEDAILY`;
    Dutch power `DFB`; TTF day-ahead = exchange spot index; no bid/ask; volume and OI via the API.
  - The user asked for Bloomberg and online research on power fundamentals, and for "decent"
    assumptions elsewhere.
- **Method:** read-only `//blp/instruments` discovery plus reference-data checks (NAME, PX_LAST,
  LAST_UPDATE_DT, QUOTE_UNITS) on 2026-10-19. **No history was pulled**, so history depth is
  unverified everywhere.
- **Accepted as given:**
  - API2 = `XA1 Comdty` (live, OI 2,267).
  - EUA = `MO1 Comdty`: this is the Dec26 benchmark (OI 308k), so it covers the Dec-vs-front question.
  - TZT `px_volume` and `open_int` are live.
  - Bid/ask is marked NOT AVAILABLE; costs stay assumptions.
- **User tickers that did not work as given, and the substitutions:**
  - `DEDAILY` resolves to Nasdaq "NOMX DE Base Futures Day", but there is **no price and no update date**
    on the index or on its dated contracts: not entitled, or dormant. Kept on record as
    `price.power_de.daily_eur_mwh` (requires_user_input).
    - German M1 = `FPA2 Comdty`. FPA is EEX German Base Month, OI 4,951, listed 2016. FPA1 is the
      current delivery month, so M1 is FPA2 (assumption, to confirm).
    - German day-ahead = `ELGBDAHD Index`.
  - `DFB` is the EEX Dutch Power Base **Week** future. Registered as `price.power_nl.base_w1_eur_mwh` =
    `DFBAW Comdty`. `DFB1 Comdty` is a USD/RIN future.
    - Dutch M1 = `ELHB1MON Index`, a Bloomberg OTC assessment chosen over the thin EEX KGR (OI 6) and
      ENDEX DUB (OI 155).
  - The ICE Endex TTF spot index `TTFDDAHD` stopped on 2020-09-29.
    - TTF day-ahead = **EEX EGSI `EGSITFAD Index`**, the exchange spot index, live.
    - Cross-check: `EEDATTF1 Index` (PEGAS day-ahead).
    - Balance-of-month = `TTFGBOM Index`.
    - Implied vol = `FJS1 Comdty` `hist_call_imp_vol` (assumption: approximates front ATM).
- **Power fundamentals:** new `configs/sources/bloomberg_power.yaml` with 15 live series:
  - DE wind on/offshore EEX forecasts and actuals; DE solar day-ahead for 3 of 4 TSOs and actual;
  - DE load actual; FR nuclear generation;
  - weekly reservoirs for NO, SE, CH, AT, FR.
  - All grade B. Forecasts are conservatively usable only from D 08:00, which loses the day-ahead
    lead until Bloomberg's update time is measured.
- **Not on Bloomberg:**
  - German load forecast;
  - French nuclear availability (only generation, used as a labelled proxy);
  - TransnetBW solar (dead since 2021);
  - live capacities.
  - Dead series not used: Nord Pool `NPER*` reservoirs (stopped 2025-02), `TGD DE` gas burn (2011),
    and the `PZ*` capacity series (no data).
- **Best online sources (`power.yaml non_bloomberg_sources`):**
  - ENTSO-E Transparency API: load forecast A65, wind/solar forecast A69, capacity. Free token needed.
  - SMARD (CC BY 4.0).
  - EDF REMIT archive since 2015. Its original timestamps make a **grade-A** history of French nuclear
    availability possible.
  - ENTSO-E and SMARD forecasts are overwritten, so they are grade B unless snapshotted from now on.
- **Assumptions:**
  - gas plant 45% on a GCV basis (50% NCV × 0.901), consistent with GCV-priced TTF;
  - gas emission factor converted to GCV, 0.182 t/MWh;
  - coal plant 38% on an NCV basis, consistent with API2's 6,000 kcal/kg NAR.
- **Open:**
  - `power.yaml derived:` (wind = onshore + offshore, MW→GW, the 3-TSO solar sum, the reservoir sum) is
    declared but **not implemented**: no derivation step exists in the data layer. Until it does, the
    power features stay NaN with a flag.
  - FR nuclear availability, the DE load forecast and full-coverage DE solar need non-Bloomberg
    connectors.

### D-035 · 2026-10-19 · TTF expiry: vendor LAST_TRADEABLE_DT is authoritative
- **Evidence:** from the authorised pull, vendor last-trade dates for 159 TZT contracts (2015-02 to 2028-04).
  - The configured rule (2 business days before delivery, weekdays only) matched **151/159**.
  - The 8 misses all involve England bank holidays: Good Friday, Spring BH, Summer BH. TTF *settles* on
    Spring and Summer BH, but expiry skips them from 2020; in 2015–16 the same holidays were counted
    as business days (2015-09 expired 28 Aug, 2016-06 on 30 May).
  - No single holiday calendar reproduces the history. Tested: an `exchange_calendars` XLON calendar
    also adds year-end closures that are not bank holidays.
  - The 33 weekdays without any TTF settlement since 2015 are all London holidays.
- **Decision:**
  - `instruments.yaml hubs.TTF.expiry.vendor_authoritative: true`. Listed contracts use the vendor
    date. Rule disagreements go into `contract_metadata(...).attrs["expiry_mismatches"]`, not an error.
  - Without vendor data (projection beyond the listed strip) the rule plus the hard cross-check still
    apply.
  - The trading-session calendar stays empirical (dates with a settlement).
- **Effect:** targets and rolls on real data use the true expiry of every contract. For the 8
  contracts, the rule alone would have mis-dated expiry week by one session.

### D-036 · 2026-10-19 · Real-data H1 runner
- **Refactor:** `synthetic_e2e.evaluate_bundle()` now holds pipeline steps 3–7. Synthetic `run_world`
  and the real runner share it unchanged. Behaviour is preserved: 34/34 integration and remediation
  tests, including the null-world and determinism checks.
- **Real runner:** `pipelines/real_h1.py` plus `cli real-h1`. It reads only the git-ignored
  `data/live` store and passes vendor contract metadata to the targets. Settings: `is_synthetic=False`,
  first origin 2016-01-01, horizons 5 and 20 (T3b, T3c). The holdout from 2025-10-01 is excluded by
  the splitter.
- **Report:** the real-data banner replaces the synthetic one, and the synthetic-only caveats are
  suppressed.
- **Storage data note:** EU working gas volume rises 579 → ~1,110 TWh over 2010–2016. That is GIE
  reporting coverage, not capacity, so storage features use % of WGV; TWh normals before 2016 are not
  comparable. The identity ΔInventory vs injection − withdrawal has p95 |residual| ≈ 660 GWh/d
  (perimeter, revisions, GCV); it is not used as an input.

### D-052 · 2026-10-19 · Derived power series implemented (D-034 `derived:`)
_Renumbered 2026-10-20 from a duplicate D-035; all other references to D-035 mean the expiry entry._
- **Decision:** the `derived:` block of `configs/sources/power.yaml` is implemented in
  `src/ttf_forecaster/data/derived.py` and applied inside `features/power.py` (`build_power_features`).
  Each GW / GWh id is built from its raw MW / MWh components, with the rules below. `power.py`
  templates are unchanged.
  - **event_time:** the components' shared event time. A component with no row at an event time
    gives NaN for that event time, never a partial sum.
  - **publication_time:** the latest publication among the component rows of that vintage. A derived
    value is never visible before its last component is published, including revisions.
  - **pit_grade:** the worst grade among the component rows of that vintage. Grade C propagates and
    is excluded by the readers' default `resolve_as_of`.
  - **unit:** from the id suffix (`_gw` -> GW, `_gwh` -> GWh). Components must carry MW / MWh, and
    the scale must be 0.001, or the config is refused.
- **Forecast view:** `power.py` reads wind and solar forecasts from `bundle.forecasts`, so the two
  `schema: forecast` ids are also exposed as FORECAST_SCHEMA rows. `valid_time` is the market-local
  day at UTC midnight (the daily convention of `features/weather.py`). `issue_time` = publication.
  The OBSERVATION_SCHEMA rows are kept as specified.
- **Alternatives:**
  - Derive at ingest and store GW rows: rejected. It would freeze a derivation into stored vintages
    and duplicate ids.
  - Sum at feature time inside each feature: rejected. The as-of logic would be repeated per feature.
  - Stamp the derived vintage at its **earliest** component publication: rejected. It lets the sum be
    seen before the last component is published. The canary catches it (positive control in tests).
- **Method:** the derived vintage at each publication step is the as-of state of the components, the
  same rule as `resolve_as_of`. It is computed in a vectorised form. A test checks it against
  `resolve_as_of` at every 2 hours over random vintages and gaps.
- **Evidence:** `tests/unit/data_arch/test_derived_power.py` (26 tests, all SYNTHETIC): sums and
  scale, NaN for missing components, publication = latest component, revisions, grade, unit guards,
  the forecast view, power-builder wiring, and the leakage canary on the power family with raw MW
  components. The canary passes; the positive control fails as intended.
- **Test fixtures changed:** `tests/unit/weather/test_power_features.py` and
  `tests/unit/model_risk/test_leakage_audit.py` fed the GW ids straight into the bundle. The builder
  now refuses a bundle that already carries a derived id, so those fixtures now supply the raw MW
  components. The forecast components have an earlier vintage (revision 0, two days out) and the
  day-ahead value (revision 1). Expected feature values are unchanged.
- **Consequences / open:**
  - **Publication lag blocks the forecast features on real data.** `bloomberg_power.yaml` stamps the
    EEX day-ahead for day D at D 08:00. The renewables window is day D+1 at origin D-1 19:00, so no
    window origin can see it. The wind, solar and residual-load features therefore stay NaN until
    Bloomberg's update time is measured and the lag is changed (D-034). This is the intended
    conservative behaviour, not a derivation fault.
  - **Revisions need prior vintages.** One day-ahead value per day gives no earlier vintage, so the
    `*_rev_gw` features are NaN. Snapshotting from now on is the only route.
  - **Vintages = revisions.** `OBSERVATION_SCHEMA` has no forecast-run dimension, so successive
    runs for the same day must be stored as successive revisions of that event time.
  - **Timezone convention (not fixed here).** The Bloomberg adapter stamps daily values at Amsterdam
    midnight (22:00 or 23:00 UTC the day before). `features/power.py` reads observations on UTC
    days. The forecast view normalises its valid day, but the actual series (`_gen_gw`,
    `nuclear_avail_gw`, hydro) keep the adapter's stamp and are shifted one day in the normals.
    Needs a decision before real data.
  - Not yet run against live data. No vendor call was made.

### D-037 · 2026-10-19 · H1 RESULT on real data: storage does NOT beat the forward curve (T3, h = 5, 20)
- **Run:** `cli real-h1` (run ids `real_T3_h5_v4`, `real_T3_h20_v4`; v2 and v4 give identical numbers, same
  code before the report-only fix).
  - Origins 2016-01-01..2025-09 (holdout locked); 2,781 origins (h = 5), 2,766 (h = 20); 18 outer folds.
  - Families price + curve + calendar + storage; 55 features, all grade A/B (`clean_pit` true).
- **Criteria 1–2:** no model passes for either horizon, on either label, against any benchmark. The
  zero (forward) benchmark is also the best benchmark at both horizons.

  | h | model | fold win share | HLN-DM p | OOS R² vs zero | OOS R² ex-crisis |
  |---|---|---|---|---|---|
  | 5 | ridge | 0.50 | 0.63 | −0.010 | −0.029 |
  | 5 | lightgbm | 0.39 | 0.79 | −0.009 | −0.014 |
  | 5 | ridge price-only | 0.50 | 0.75 | −0.008 | −0.006 |
  | 20 | ridge | 0.47 | 0.93 | −0.203 | +0.037 |
  | 20 | lightgbm | 0.35 | 0.97 | −0.055 | −0.140 |
  | 20 | ridge price-only | 0.47 | 0.87 | −0.146 | −0.052 |

  AR on own lags and the rolling mean also lose to zero: −0.018 / −0.029 at h = 5.
- **Criterion 3 (storage vs price-only twin):**

  | h | HLN-DM p | DM statistic | executable-label p | Clark–West p |
  |---|---|---|---|---|
  | 5 | 0.52 | 0.05 | 0.46 | 0.26 |
  | 20 | 0.65 | 0.39 (full model worse) | 0.67 | 0.033 |

  - The h = 20 Clark–West p of 0.033 is the pattern D-020/D-032 F4 warn about (ridge CW over-rejects
    at 11% under the null). It does not count.
- **Stability:**
  - Single-year OOS R² swings between about −0.9 and +0.17 at h = 20. The h = 20 ridge loss is
    concentrated in one fold-year (OOS R² −0.90).
  - The positive ex-crisis R² for ridge at h = 20 (+0.037) sits next to −0.20 all-in: not a stable
    edge.
- **Trading** (non-overlapping h-day trades, assumed costs):
  - Net Sharpe 0.01–0.13 at h = 5.
  - h = 20: ridge 0.29, lightgbm −0.05.
  - All far below criterion 4's 0.5, and P&L is concentrated (top-5% days ≫ total).
- **Conclusion (per spec §9):** with daily EU storage features (level, % full, deviation vs normal,
  injection vs normal, projected end-of-season, threshold probabilities, policy gap), H1 is **not
  supported** at 5 and 20 days. This is consistent with D-029: storage levels are public, watched
  daily and largely priced.
  - Per spec amendment A5 this is a **failed result, not merely "inconclusive"**: the history is
    ~10 years, more than the 5-year power threshold.
  - The stop criterion applies to H1 in this form. Do not tune features, add interactions or switch
    estimators on this sample to rescue it; any further storage test needs a NEW pre-registered
    hypothesis (e.g. storage × weather-revision, H2 × H1), judged on the same harness.
- **Caveats:**
  - storage is grade B (no vintages);
  - costs are assumptions;
  - executable-label scores reuse the fixed-label forecasts;
  - the holdout is untouched and stays locked.

### D-038 · 2026-10-19 · Spark credential file stays in Git (user decision)
- The user decided that `adhoc_scripts/client_credentials.csv` stays tracked in the parent repo for now.
  SQ-01 is closed as accepted risk. This project still reads Spark credentials from env vars and never
  reads that file.

### D-039 · 2026-10-19 · Bloomberg discovery and authorised live pull (Task 1 and Task 2)
- **Authority:** user instruction 2026-10-19: live pulls of `known` series and read-only discovery. The
  pulls used a temporary copy of `configs/` with `live_calls_authorised: true`. The committed
  `configs/local_data_sources.yaml` stays `false`. Data went only to the git-ignored `data/live/` store.
- **Pulled (all returned data unless noted):**
  - 128 observation series from 2011 (or from when the series exists): 103 existing `known` series, 16
    fundamentals and LNG series added the same day, 6 weather station series, 2 Amprion zone load series,
    1 Bacton IUK flow. Earlier-year gaps are empty responses before each series' first date, not errors.
  - Monthly curves 2015 to 2026, monthly root only: TTF volume and open interest (129,279 rows),
    JKM, HH, Brent, NBP, TTF_USD settle (51k to 70k rows each).
  - Contract metadata (vendor LAST_TRADEABLE_DT) for HH, NBP, TTF_USD (159, 159, 130 contracts).
    JKM and Brent have no expiry rule, so their 159 vendor dates each are saved raw under
    `data/live/vendor_expiry/`, not as contract metadata.
  - **Failed:** THE, PEG, PSV, PVB, VTP monthly settle (no data for any year). See Evidence 2.
- **Evidence:**
  1. **Expiry (D-035 applied to HH, NBP, TTF_USD).** Vendor LAST_TRADEABLE_DT against the configured rule:
     TTF 8 of 159 (already authoritative), HH 14 of 159, NBP 10 of 159, TTF_USD 21 of 130. The HH and NBP
     misses are one-session holiday effects of the same kind as D-035. The TTF_USD misses are larger
     (2015-03 contract: rule 26 Feb, vendor 16 Mar), so that root's rule does not describe it.
  2. **THE, PEG, PSV, PVB, VTP.** The configured monthly ticker rule (2-digit year, period suffix, e.g.
     `NCGF26M Comdty`) returns "Unknown/Invalid Security". The Terminal's working form is 1-digit year plus
     `M` (e.g. `NCGX6M Comdty` = Nov26, verified, LAST_TRADEABLE_DT 2026-10-29). That form has data only for
     the listed contract: no rows for 2011, 2016 or 2021. Other forms tried (`NCGX16M`, `NCGX26M`,
     `NCGX16`, `NCGX6`) returned nothing either.
  3. **Revisions.** Bloomberg revised 18 TTF volume rows (last trade day) between two pulls on the same day.
     The store keeps the first vintage (`on_conflict="keep_existing"`, reported in the pull log).
  4. **Anomaly, Amprion load.** `power.de.amprion_load_fc_mw` peaks at 31,863 MW against 19,279 MW for the
     actual of the same zone; 5 outlier-suspect rows. Not used until the unit or period is checked.
- **Decisions:**
  1. `configs/instruments.yaml`: `vendor_authoritative: true` for HH, NBP and TTF_USD (the same policy as
     TTF). Mismatches are reported in this entry. The rule itself is unchanged.
  2. JKM and Brent: no expiry rule (`requires_user_input`). Raw vendor dates only. Not used for targets.
  3. THE, PEG, PSV, PVB, VTP: **not pulled and no ticker rule changed.** The 1-digit-year form cannot give
     history without a decision on how to map the year, and changing the rule for five hubs is a
     vendor-identifier decision. A user action is listed in `docs/bloomberg_discovery.md` section 3, item 6.
  4. New `known` series, each with cited provenance, unit, `pit_grade: B` and a conservative lag:
     - `bloomberg_fundamentals.yaml`: 8 flow series (Gassco totals, Norway to Germany, UK and others;
       Dornum; Russia to CE and SE Europe; Bacton IUK), 2 production (NL daily; Troll monthly), 1 demand
       (German SLP weekly), 6 WSI station temperatures (observed only).
     - `bloomberg_lng.yaml`: 6 terminal series (Montoir, Dunkerque, Swinoujscie, Rovigo, Bilbao regas,
       Zeebrugge net entry).
     - `bloomberg_power.yaml`: 2 Amprion zone series (forecast and actual), labelled zone-only.
  5. Known series with no usable history are kept `known` in config, with the history start recorded in
     `docs/bloomberg_discovery.md`: DFBAW weekly (20 rows, active contract only), FJS1 implied vol (from
     2025-10-31), NLGDMINH (from 2022-07).
  6. `tests/unit/fundamentals/test_data_fundamentals.py`: the provenance assertion accepts "Bloomberg
     discovery" as well as "GasImbalanceModel". The intent (a known series must cite where it came from)
     is unchanged.
- **Effect:** the live store now holds the full known-series set, except the five hubs above. Nothing was
  modelled: no model, feature, target or backtest was run or changed, and the stored H1 outputs (D-037) are
  not regenerated.
- **Open:** the 14 gaps and the to-do list in `docs/bloomberg_discovery.md` section 3 (German national load
  and solar forecasts, French nuclear availability, European HDD and temperature forecasts, historical
  THE/PEG/PSV/PVB/VTP tickers, DFBAW history, FJS1 history, Grain and Huelva send-out, UK production,
  TurkStream, Medgaz/Transmed, pre-2020 Gassco totals).

### D-040 · 2026-10-19 · THE, PEG, PSV, PVB, VTP monthly history: ticker forms resolved, pulled, post-expiry rows dropped
- **Trigger:** orchestrator follow-up. D-039 had recorded "no monthly history" for these five hubs. The cause
  was ticker naming, not missing data. Re-verified here with short bdh windows before any change.
- **Evidence (bdh PX_SETTLE, 2026-10-19):**
  - THE is ICE Endex root `LFT` with the TZT 2-digit rule: `LFTX16` 2016-10-03..07 (5 rows, 15.82), `LFTX21`
    2021-10-04..08 (5 rows, 96.84), `LFTX26` 2026-10-01..08 (6 rows, 74.98). `LFTX6` also returns the live
    contract (6 rows) but nothing for 2016. The 2-digit form works in every window.
  - PNX/EEX hubs: the listed contract is root + month + 1-digit year + `M` (`NCGX6M`, `PSRX6M`: 2026 only).
    An expired contract is root + month + 1-digit year + `M` + 2-digit year (`PNGX6M16` 15.88, `PSRX1M21` 96.68,
    `NCGX6M16` 15.65). The listed form returns nothing for 2016 or 2021.
- **Decisions:**
  1. `configs/sources/bloomberg.yaml`: THE moved to root `LFT`, `period_suffix: false`, monthly root only. The
     PNX `NCG` form is kept as a documented alternative. Q/S/Y LFT roots are not verified and are not configured.
  2. PEG, PSV, PVB, VTP: `ticker_form: pnx`, monthly roots only (PNG, PSR, PXB, CEG). Q/S/Y are refused for the pnx form.
  3. `contracts/futures.py`: `BloombergCurveIds.form`. `bloomberg_ticker_candidates` returns the listed and
     expired tickers for a pnx contract. `parse_bloomberg_ticker` accepts only the expired form (the listed form
     has no decade). `bloomberg_ticker` refuses pnx hubs and points to the candidates.
  4. `data/bloomberg.py` (live adapter). Curve pulls request both candidates. A contract with data under one
     form uses that form. Under both: if the two series agree on every common day, one security, the expired
     form is kept. If they disagree, the pull raises. Static reference data (`LAST_TRADEABLE_DT`) uses the expired
     form whenever it has a value, because the listed 1-digit form can name a later security once expired
     (PEG.M.2015-12 disagreed). `last_ticker_counts` records which ticker resolved each contract.
  5. **Post-expiry rows dropped before ingest (counts):** THE 1, PEG 2,134, PSV 2,157, PVB 1,685, VTP 2,288. Rows
     dated after the vendor last-trade date carry a flat, carried-forward value (PEG.M.2017-12: 13 rows at 20.78
     after 2017-11-28). TTF, TTF_USD, HH, NBP, JKM and Brent have none. The five curve partitions were rebuilt
     from this session's pulls only (the raw pulls stay in the raw layer).
  6. Vendor-authoritative expiry (D-035) for the five hubs in `configs/instruments.yaml`. Mismatches vs the
     configured rule: THE 9 of 159, PEG 19 of 158, PSV 4 of 151, PVB 2 of 95, VTP 7 of 122.
  7. Not changed: the committed `local_data_sources.yaml` (still false), targets, features and models.
- **Pulled (store totals):** THE 71,139 settle rows, 159 contracts; PEG 19,850 rows, 158 contracts; PSV 16,958, 151;
  PVB 13,748, 95; VTP 16,476, 122. Contract metadata: THE 159, PEG 158, PSV 151, PVB 95, VTP 123.
- **Open (not resolved here):**
  - PNX hubs carry many settles below 0.5 EUR/MWh (PEG 2,205 rows across 61 contracts; PSV 3,425 / 62; PVB
    1,665 / 42; VTP 1,682 / 45) and repeated values. These are not removed. They need a liquidity or price-floor
    rule decided before any PNX curve feature.
  - VTP metadata lists 123 contracts, the curve has 122: one contract has only post-expiry rows.
  - THE Q/S/Y LFT roots are unverified.
- **Effect:** the five hubs are now pullable with a verified form and can feed the curve store. Nothing was
  modelled. The H1 result (D-037) and all stored features are unchanged.

**D-040 addendum (2026-10-19, user decision):** the daily report (`Data/lng_report`) keeps THE on root `NCG`
(EEX PNX H-Cal). The research store and `configs/sources/bloomberg.yaml` use root `LFT` (ICE Endex). The two are
different contracts and will not match. Do not compare report THE prices with store THE settles without that
caveat. Report change: the year part of PNX-hub tickers is one digit (`_year_code`), so five-year strips no longer
build `NCGX10M`.

### D-041 · 2026-10-19 · BloombergNEF workbook: 278 series registered, pulled and cross-checked; German day-ahead repointed
- **Source:** the user-supplied BNEF European Power Market Monitor workbook. Only its extracted ticker list and
  long daily table were used (git-ignored under `data/raw/extracts/bnef/`). The `.xlsm` was not opened.
- **Registered:** 278 series in `configs/sources/bloomberg_bnef.yaml`, one entry each, provenance "BloombergNEF European
  Power Market Monitor workbook ticker list (user-supplied 2026-10-19)". Each was reference-checked on 2026-10-19
  (NAME, PX_LAST, LAST_UPDATE_DT). Known only where live. Two dead tickers were excluded (GT1CBECO, GT1CDENU).
  Breakdown: generation by fuel 184, load 22, capacity 16, gas consumption and gas for power 5, power day-ahead
  prices 41, gas day-ahead prices 9, carbon 1.
  The three existing ids that match an already-registered ticker were reused, not duplicated: `power.de.gen_solar_mw`
  (GT1CDESL), `power.de.gen_wind_on_mw` (GT1CDEWS), `power.de.load_act_mw` (LD1CDE).
- **Pulled:** all 278 series, 2000 to 2026 in year chunks, through a temporary authorised config. Stored rows
  1,245,058. Every series returned data. Earliest history 2000 (33 series), median start 2015 for power and 2012
  for gas. Full table in `docs/bloomberg_discovery.md` section 6.
- **Cross-check against the BNEF long table:** 246 of 278 overlap. 242 match on at least 99% of overlap days.
  Four disagree and are annotated in the registry (not removed):
  - `price.power_gb.da_apx_gbp_mwh` (PXPXBASE): not the same series or averaging (mean 114 vs 134 GBP/MWh).
  - `price.power_pl.da_pln_mwh` (PXPEIDTB): BNEF values are EUR-scale under a PLN label. The Bloomberg PLN series is kept.
  - `price.gas_gb.da_nbp_gbp_therm` (NBPGDAHD): unit or definition differs (GBp vs GBP/therm).
  - `carbon.gb.price_support_gbp_t` (CARBPRIC): five overlap days only.
  32 registered series are in the workbook list but not in the long table, so they have no BNEF check.
- **Decision (user-directed: "point the DE day-ahead at it if its history is better; record the choice"):**
  `price.power_de.da_eur_mwh` now points at **LPXBHRBS** (EPEX Germany day-ahead baseload index). Its history
  starts 2000-06-15 (9,623 rows). The previous target **ELGBDAHD** (OTC forward day-ahead assessment) starts
  2011-02-01. The probe returned LPXBHRBS rows for 2001, 2008 and 2011-01 windows, and ELGBDAHD none before 2011-02.
  ELGBDAHD is kept as `price.power_de.da_otc_elgb_eur_mwh`, and its stored partition was moved to that id (same file,
  series_id column rewritten). DEDAILY stays `requires_user_input`.
- **Not registered:** interconnector flows (257 rows, deferred per request), spark and dark spreads and carbon or coal
  calculated fields (no Bloomberg ticker; derive them from registered prices), pumped-storage consumption, other fuels,
  and gas subtypes (CCGT/OCGT). Listed in the doc.
- **Pit:** exchange auction and gas hub prices grade A, lag delivery day 19:00 (conservative: the auction publishes at
  D-1 noon). Generation, load and gas flows grade B, D+1 12:00. Capacity grade B, D+7 12:00.
- **Gaps closed:** gas-for-power for ES, IT, GB and BE (BE is a burn figure, not confirmed power-only); load actuals
  for 22 more countries and zones; national solar generation for other countries; German day-ahead power.
  **Not closed:** national German load and solar forecasts, French nuclear availability, European HDD or
  temperature forecasts. The workbook has no forecast metrics.
- **Effect:** configs and store gain 278 verified series. The DE day-ahead id changes its source, so any feature
  that reads `price.power_de.da_eur_mwh` now sees LPXBHRBS. No model, feature, target or backtest was run or changed.
  The H1 result (D-037) and the stored outputs are unchanged.
- **Test change:** `tests/unit/fundamentals/test_data_fundamentals.py` accepts "BloombergNEF" in provenance, as it
  already accepted "Bloomberg discovery". The intent (cite the source of every known series) is unchanged.

### D-043 · 2026-10-19 · BSRCH power-plant outages: snapshot adapter, daily snapshot; nuclear availability wired in D-044
- **Source:** BSRCH `ASSET_OUTAGE:POWER_PLANT` (user-supplied domain), via the Python API (`xbbg.blp.bsrch`), live path, `known`
  series only, through a temporary authorised config. The Excel route was not needed.
- **Measured (docs/bloomberg_discovery.md section 8):** FR, DE and NORDPOOL return data. AS_OF_DATE is honoured from 2026-10-13.
  Vintages are real: the same future day differs between as-of dates, and realised days can be revised. History is one rolling year.
  A window longer than about three years returns nothing, so each snapshot is two requests.
- **Stored:** series `power.<geo>.outage_<fuel>_unavail_mw` (32 series, `configs/sources/bloomberg_outages.yaml`, ticker null, so the
  ticker adapter refuses them). Forward plan = FORECAST rows: issue = as-of 00:00 UTC, publication = request time (conservative), grade A.
  Realised days = OBSERVATION rows, grade B, publication = max(D+1 12:00 UTC, request time), revision = as-of offset from 2026-10-13.
  Snapshots written: as-of 2026-10-13 and 2026-10-19 (today).
- **Deviation from the brief, stated:** the brief said issue = publication = the as-of, using the request time. Using the request time as
  issue made two as-of snapshots taken together share a key with different values (a store conflict), so issue is the as-of date and
  publication is the request time. Publication is still the conservative stamp.
- **Consequence for backtests:** no outage vintage is visible before 2026-10-13, and each snapshot is visible only from its request time.
  Daily accumulation builds the vintage history from now on. No backtest was run.
- **Gas-asset outages:** no domain found; none invented.
- **Daily job:** `scripts/run_daily_snapshot.py` (outages always; BQL weather only while Excel is open). Idempotent per request date.
  Not scheduled: the user decides (command in docs/bloomberg_discovery.md section 8).


### D-042 · 2026-10-19 · Bloomberg weather through BQL via Excel: vintaged forecasts, actuals, normals
- **Route:** `weather()` through the user's Excel BQL add-in (scratch books only, closed unsaved; the user's workbooks are never
  touched or recalculated). The Python API login is not entitled to BQL, so this route is the only one. Section 7 of
  docs/bloomberg_discovery.md has the detail.
- **Lead mapping proved:** `as_of_date = D`, `offset = range(1,16)`: row i = valid day D+i. Pinned in tests; dates audited on every pull.
- **Stored:** forecasts for every daily as-of 2025-01-01 to 2026-10-19 (650 issue days, ECMWF and GFS, 0z), 4,168,500 rows after
  dedup. Actuals 2011-01-01 to 2026-10-16. Normals NORMALS_10Y and NORMALS_30Y, grade C.
- **Publication-time assumption (ASSUMED):** forecast publication = issue (as-of) 00:00 UTC + 10 h. Sensitivity to this assumption
  was NOT run. It matters for any backtest that reads a forecast at its publication time.
- **Normals grade C:** they use hindsight, so they never enter a clean backtest.
- **Lead 0 null for HDD, CDD and solar:** a data fact, not a bug. Models must not use lead 0 for those fields.
- **Bundled queries:** several model types per query (one `let` alias each), so 2 queries per as-of day instead of 7. Verified against
  single-model queries.
- **Config refusal:** the 204 weather series have no ticker, so the ticker adapter refuses them (tested).
- **Sanity:** DE HDD error grows with lead (RMSE 1.73 at 24 h, 2.61 at 336 h). Lead 0 has the smallest temperature error (consistent
  with row 0 = the as-of day). DE HDD January 15.5, July 0.5.
- **Backtest consequence:** about 1.8 heating seasons of vintaged history. Enough for a short H2 backtest only; no backtest was run.
- **Daily accumulation:** `scripts/run_daily_snapshot.py` runs this extractor for today's as-of (BQL needs Excel open; it is skipped otherwise).

### D-044 · 2026-10-19 · Derived series: signed terms, a constant and forecast components; French nuclear availability wired
- **Rule:** a derived series = scale x (constant + sum of coef x component). Components are observations (as before) or, with
  `components_kind: forecast`, FORECAST rows joined on (issue, valid). A derived forecast exists only where every component has a row
  for that issue; it is published at the latest component publication and carries the worst grade. A missing observation component gives
  NaN, as before. Plain `sum` declarations behave exactly as before.
- **Wired:** `power.fr.nuclear_avail_gw` (observation) and `power.fr.nuclear_avail_fc_gw` (forecast) = (63000 minus the BSRCH
  `power.fr.outage_nuclear_unavail_mw`) x 0.001 GW. The former generation proxy (`power.fr.nuclear_gen_mw`) is no longer used.
- **Constant, ASSUMED:** 63,000 MW, French nuclear installed capacity (about 63.0 GW incl. Flamanville 3). Recalled, not verified.
  Marked `constant_status: assumed`, with provenance "CONFIRM". No FR nuclear capacity series was found in the registry, and a
  live Bloomberg lookup was NOT run. If one exists, it should replace this constant.
- **Tests:** subtraction, the constant applied once, missing component NaN, forecast join on the same issue only, no join across issues,
  publication and grade rules, a leakage canary (a later component revision moves the derived publication forward), config refusals,
  and the committed config.
- **Effect:** the nuclear feature and its test fixtures now read the outage components. The fixtures were repointed to MW outage
  components with the same implied availability, so their expected values are unchanged. The full suite passes (770 tests). No model was run.


### D-045 · 2026-10-19 · BUG: BQL actuals stored on the requested day, one day early for DE; fixed and re-pulled
- **Bug:** `observation_frame` stamped row i of an actual block with the requested start + i. BQL labels the value of a requested
  day with the observed day, and that label is one day later than the request for DE Country (GB Country is labelled on the request).
  So every DE actual was stored one day early. With the rule publication = D+1 12:00, the real day's value became visible at noon on
  that same day: a future leak.
- **Evidence (2026-10-19, Amsterdam dates):** BQL DE t2m vs the WSI regional series `weather.de.regional_t2m_obs_degc`: correlation
  0.9995 when the BQL value of D is paired with WSI D+1, 0.964 at the same date. MAE 0.06 vs 1.47. The forecast cross-check (ECMWF
  mean, 24 h lead vs WSI) gave MAE 0.50 at the same date versus 1.37-1.54 at +-1 day, so the WSI dating is the reference.
- **Fix:** a dates-only actuals query for every location, labels per location. `actual_event_dates` accepts a location's first label
  at the request or the next day, strictly increasing, within the period. `observation_frame` uses the labels as observed days and
  refuses a count mismatch between values and labels (no trimming). Each year's block is checked against WSI before storage.
- **Permanent data check** (`pipelines/bql_weather.check_actuals_alignment`, run in every actuals pull): each BQL actual must correlate
  best with its WSI reference at lag 0. Pairs: DE (regional), GB (Heathrow), NL (Schiphol). It raises otherwise.
- **Result on the full stored actuals (2011-2026):** DE best lag 0, correlation 1.000. GB best lag 0, 0.995. NL best lag 0, 0.998.
- **Missing observations:** 2026-08-05 is absent from the BQL result for most locations (DE, NL, FR, IT and others, 290 rows against 291 days).
  The gap is kept and reported, not filled.
- **Quarantine, not deleted:** the wrong-day raw files are in `data/live/quarantine/bql_weather_obs_wrong_day_2026-10-19/`. The
  first fixed pull (shared labels from DE, not per location) is in `data/live/quarantine/bql_weather_obs_shared_labels_2026-10-19/`.
  The store was rebuilt from the per-location pull; the 48 BQL actual partitions were replaced, not re-ingested as revisions.
- **Normals:** NOT shifted. The 10Y HDD normal matches the 2016-2025 actual climatology by day of year (correlation 0.990, MAE 0.60 HDD
  days). Monthly means agree to 0.05 (January 15.75 vs 15.76). Caveat: climatology is smooth, so a one-day shift cannot be excluded
  by this test alone. The normals' dating matches the verified forecast dating (the normals dates query labels row 0 as the block start).
- **Forecast vs actual, recomputed (ECMWF mean, DE, 2025-01-01 to 2026-10-19):** HDD MAE rises with lead: 0.46 at 24 h, 0.50 at 48 h,
  1.16 at 168 h, 1.83 at 336 h. Temperature MAE: 0.39 at lead 0, 0.51 at 24 h, 2.39 at 336 h. (RMSE, before the fix at 24 h: 1.69; after: 0.64.)
- **Consequence:** every backtest or feature built on BQL actuals before this fix must be rerun. Nothing was rerun here (no model run).


### D-046 · 2026-10-20 · Gas/LNG outage spreadsheet: snapshot loader, aggregates, side table
- Source: user-maintained Bloomberg export `data/raw/extracts/outages/LNG_Gas_Outages_20Sep26.xlsx` (name wrong; rows to 2026-10-19).
  Snapshot time = file modification time (user rule). Each snapshot writes forward-plan aggregates (grade A) and the event side table.
- Result on the current file: 125 events. Units: 110 mcm/d, 2 mtpa (not converted, side table only), 13 with no quantity.
  Aggregates from 2026-10-20: planned 82.3 mcm/d on 23 Oct, unplanned 5.2 on 20 Oct. LNG terminal (Hammerfest) rows: 58.
- Decisions: aggregate issue = snapshot time (an aggregate is a snapshot total; event Last Update kept in the side table). Sign: stored
  reduction positive = capacity lost. Gas days inclusive. GWh/d twins use the balance GCV (10.55, DEFAULT_UNVERIFIED in the source).
- Idempotent by file SHA-256; the side table is written last. Tests: 8 on a synthetic sheet (units, categories, zero fill, reload, revision).

### D-047 · 2026-10-20 · Implied realised Norwegian outage from GAANTOTI: built, NOT validated, feature partly mapped
- Flow: GAANTOTI (Norway total exit intraday nominations), 2149 rows, usable from D+1 07:00 Amsterdam (grade B).
- Rule: proxy = 95th percentile of the previous 30 days (min 20, strictly past); outage = max(0, proxy - flow); event = shortfall > 10.
- **Validation FAILED** against the spreadsheet: correlation -0.86 (1 Sep to 19 Oct) and -0.29 (28 Sep to 19 Oct). The Nyhamna -19.8 mcm/d
  outage from 28 Sep is not visible: implied 3.4 mcm/d against the sheet's 25.8 mcm/d on 28 Sep to 19 Oct. Mean flow rose from 272 (September)
  to 333 (1 to 19 Oct), so the rolling 95th percentile set by late-summer flows (mean 320, p95 335 for 15 Jul to 31 Aug) reads September as an
  outage. This is a regime effect of the method, not a measurement. Not tuned to rescue the result.
- Mapping: `flows_features.norway_flow_series` -> `flow.no.exit_total_intraday_gwhd` (measured flow). `norway_unavailable_series` NOT mapped
  (kept on the placeholder): the implied outage did not validate. Decision for the user.
- Open: a proxy that does not depend on a percentile of past flows (e.g. the sheet's technical capacity, or a nominated-capacity series), and
  an explicit regime check before any outage feature uses the implied series.

### D-048 · 2026-10-20 · LNG cargo imports by country (BSRCH TPD:DEX): NL registered and pulled; side table per cargo
- NL BIKEY 41UT24O9ARLJ60ST27JSQI3N6: 1,377 import cargoes, 2016-12-11 to 2026-10-19, **14 columns** (not the 30 expected). Quantities: tonnes
  (negative for imports, absolute value kept) and mcm.
- Only keys verified to return rows are `known` in `configs/sources/lng_bsrch.yaml`. The other 13 keys return 0 rows (user to explain): not listed.
- Series per country and arrival day: cargo count, tonnes, mcm (zero on days with no arrival in the span). Grade B, published D+1 12:00 UTC.
- Side table keyed (country, IMO, arrival). Idempotent: a second pull adds no cargo. Tests: 5 (cargo rows, zero fill, publication, known-only, idempotence).

### D-049 · 2026-10-20 · LNG fleet snapshot (BSRCH COMDTY:SHIP) to vessel snapshots, arrival forecasts and destination observations
- Source: user-supplied BSRCH COMDTY:SHIP (LNG tankers in service), verified 2026-10-20: 883 rows, 17 columns.
- Mapping to VESSEL_SNAPSHOT_SCHEMA (`data/fleet.py`): snapshot = request time (= publication time, conservative); vessel = IMO;
  laden from Cargo status; destination = normalized port (FOR ORDERS / Unknown = null); region from the destination country via config
  (NWE: NL, BE, FR, DE, UK; EU: other EU countries listed; all else unmapped); ETA before the snapshot = unknown; vessels with Last Seen older
  than 3 days are excluded from forecasts (kept in the side table); origin null (no departure-port field).
- **Volume unit ASSUMED t_LNG** (magnitude ~70 kt per laden cargo). Not confirmed from a DES field description. A conversion to m3 would need a
  stated density, so none is applied.
- PIT: grade A from the first snapshot date; no row before it (NaN, never back-filled); later snapshots never change earlier forecasts (tested).
- Forecasts (`lng.arrivals.<nwe|eu>.cargoes_{7,14,30,45}d` and `volume_<h>d_t_lng`) and destination observations (`lng.diversions.eu.*`,
  `lng.destination.global.undeclared_share`) stored from today's snapshot. Daily job: `scripts/run_daily_snapshot.py` (idempotent per UTC day).
- Limits: a single current snapshot per day gives no history for the destination features, and EU here excludes NWE countries. The
  diversion observations need at least two days of snapshots to be non-trivial.
- Tests: 7 on a synthetic result (mapping, ETA unknown, staleness, no back-fill, vintage, missing column, daily idempotence).

### D-050 · 2026-10-20 · LNG imports: 15 countries with new keys, truncation for ES and FR, regional totals, AHOY cross-check
- The user supplied new BIKEYs, all 15 verified with TPD:DEX on 2026-10-20 (NL unchanged). The old NL-only config is superseded.
- ES and FR return the BSRCH cap (2,000 rows). `history_truncated_before` (ES 2018-07-16, FR 2020-02-19): earlier days are NOT stored, so a
  feature gets NaN, not zero imports. Regional totals containing them start on the truncation date.
- Regions (config): NWE (NL, BE, FR, GB, DE), IBERIA (ES, PT), MED (IT, HR, GI), NORDIC_BALTIC (FI, SE, PL, LT), TURKEY.
- Cross-check: NWE total vs AHOYIMEU, 2,433 days: ratio 0.87 in sums, same-day correlation 0.08. Shifting AHOY by one day gives 0.94. So the
  AHOY series is dated one day before the port arrival date (or uses another convention). Not corrected: the arrival date is the port date. Decision
  for the user before comparing the two series. All 15 countries vs AHOYIMTL (global): ratio 0.26 (expected: global includes Asia).
- Side table: 13,019 cargoes, keyed (country, IMO, arrival). Multi-key merge and de-duplication are tested.
- Tests: 9 in the LNG section (cargo rows, zero fill, truncation, merge, regional NaN propagation, configuration, idempotent pull).

### D-051 · 2026-10-20 · LNG import dates: port-arrival convention kept; AHOY timing documented (orchestrator)
- **Evidence (D-050 cross-check):** cargo-level NWE imports from the per-country BSRCH searches against
  `AHOYIMEU`.
  - Totals: 355 vs 407 Mt over 2,433 days, a ratio of 0.87.
  - Daily correlation is 0.08 on the same day and 0.94 with AHOY shifted back one day. AHOY dates the same
    flows one day earlier than our berth-arrival date.
- **Decision:**
  - The cargo series keep `event_time` = the berth-arrival date from the vessel's port call. It is the
    later and more conservative date, and it comes straight from the source.
  - AHOY series are kept as published. Their existing publication lag (D+1 23:00, D-023) already makes
    them visible only after our arrival date, so there is no leak in either series.
  - Rule: never compare or combine the cargo series with AHOY on the same day without the one-day
    alignment. A feature mixing both must apply it explicitly and test it.
- **Level gap (13%):** most likely a perimeter difference; AHOYIMEU covers more countries than NL/BE/FR/GB/DE.
  Not used to rescale either series.
- **Revisit:** if the vendor documents AHOY's date convention.

### D-053 · 2026-10-20 · Daily snapshot: Terminal preflight with a clear failure, and a logging wrapper
- **User decision:** the daily job may fail when the Bloomberg Terminal is not logged in, but the message must say so plainly.
- **Change:** `pipelines/daily_snapshot.check_terminal` opens a desktop-API session (localhost:8194, one attempt, 5 s timeout)
  and `//blp/refdata`, then closes it. It requests no data. On failure `scripts/run_daily_snapshot.py` prints
  "Bloomberg Terminal not connected: log in to the Terminal and re-run. Nothing was pulled today ..." to stderr and exits 2,
  before any config copy, store write or pull.
- **Wrapper:** `scripts/run_daily_snapshot.bat` appends stdout/stderr and the exit code to `data\live\logs\daily_snapshot.log`
  (git-ignored). The Task Scheduler command in `docs/bloomberg_discovery.md` §8 now points at it. No task is created.
- **Not changed:** a failure inside one part still stops the later parts (user accepted). The LNG fleet snapshot of a missed
  day is lost for good; outages and BQL weather can be re-pulled late.
- **Tests:** 5 (connected, not started, refdata refused, exception wrapped, script exits 2 with nothing touched).
