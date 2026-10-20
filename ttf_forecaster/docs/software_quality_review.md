# Software-quality review (Phase C, agent 12)

Date: 2026-10-17. Reviewer: software-quality-agent. Scope: reproducibility, testability, static
quality, API consistency, tooling. **Read-only review of every module.** The only files written are
`tests/integration/`, `.pre-commit-config.yaml` and this document. No installs, no network, no
commits. Interpreter: `..\..\env_local\Scripts\python.exe` (CPython 3.14.2, pandas 2.3.3,
numpy 2.3.5, scikit-learn 1.9.0, lightgbm 4.7.0).

The code base was being changed while the review ran: the `ar` benchmark, `evaluation/diagnostics.py`
and `tests/unit/model_risk/` landed at ~02:35 on 2026-10-17. The findings below are against the tree
as of 02:45.

## 1. Verdict summary

The platform is **reproducible and testable at the module level.** Every unit-tested module passes,
there are no FutureWarnings or DeprecationWarnings, every file parses as Python 3.11, and two
identical synthetic runs give bit-identical predictions. The live backends refuse safely.

There is **one blocker**, outside the code: the parent repo still tracks the Spark credentials
(SQ-01). There are **four majors** that would let a real-data run be mislabelled or
under-specified:

- the run config hash does not cover the inputs that produced the run (SQ-02);
- criterion 1 can pass on two folds (SQ-03);
- `run_world(config_dir=...)` silently mixes two config trees (SQ-04);
- series ids diverge between `power.yaml` and `bloomberg.yaml` (SQ-05).

None of these touches PIT correctness of features or targets.

## 2. Findings

Severity: **blocker** = must fix before any real-data result is quoted; **major** = fix before the
first real-data run; **minor** = hygiene.

| ID | Sev | File | Issue | Evidence | Proposed fix | Owner |
|---|---|---|---|---|---|---|
| SQ-01 | blocker | `../adhoc_scripts/client_credentials.csv` (parent repo) | Spark client id and secret are tracked in Git (already flagged in the environment audit; still true). | `git -C .. ls-files adhoc_scripts/client_credentials.csv` returns the path. | User action: rotate the Spark secret, `git rm --cached` the file, add `*credentials*` to a root `.gitignore`. The project never reads it (integration test `test_no_credential_or_absolute_paths_in_project_files`). | user / orchestrator |
| SQ-02 | major | `pipelines/synthetic_e2e.py` L≈200, `data/storage.py:config_hash` | The run manifest's `config_hash` covers only `{"world", **evaluation}`. It leaves out the synthetic `seed`/`start`/`end`/`planted_beta`, `features.yaml`, `models.yaml` (seed, hyper-grids), `backtests.yaml` (splits, holdout, costs) and `targets.yaml`. Two runs with different seeds or model params get the **same** config hash. The manifest also has no code version (git SHA or package version) and no environment record. | Read `run_world`: `write_run_artifact(..., config={"world": world, **ev})`. `RunManifest` fields: layer, run_id, created_time, config_hash, is_synthetic, artifacts. | Hash the fully resolved config set: `{name: load_config(name, config_dir) for name in (pipeline, features, models, backtests, targets, instruments)}` plus the fundamentals registry. Add `code_version` (git describe, or "dirty") and `env` (python, pandas, numpy, sklearn, lightgbm versions) to `RunManifest`. | orchestrator (pipeline) + data-architecture (manifest fields; frozen contract change, log it) |
| SQ-03 | major | `evaluation/statistical_tests.py:evaluate_success_criteria` | Criterion 1 has **no minimum number of outer folds**. With 2 folds the fold-share condition is trivially met (2/2 ≥ 0.75), so criterion 1 reduces to the DM test alone. On a 2-fold synthetic **null** world, ridge (with `ar__` lags) "passed" criterion 1: HLN-DM p = 0.004, CW p = 0.001. That is a false positive in a world with no signal. With 4 folds (end 2023) and 6 folds (default end 2024) every model fails, as it should. | Integration run, `pipeline.yaml` end 2022-12-30: `ridge {"passed": true, "fold_win_share": 1.0, "n_folds": 2, "dm_p_one_sided": 0.0042}`. The same seed with end 2023-12-29 (4 folds): ridge p = 0.50, fails. I checked the `ar__` lags for leakage and they are clean: 0 of 1,033 lag sources have a `label_known_time` after their origin (`make_target_lags` uses shift h+k-1). | Add `min_folds` (e.g. 4) to `backtests.yaml: success_criteria`. Criterion 1 should return `passed=False, reason="too few folds"` below it. Also report the DM test's effective sample (n/h) next to p. Log as a modelling_log entry: this is a size problem of HLN-DM with h=5 overlap and ~250 test origins. | backtesting-agent |
| SQ-04 | major | `pipelines/synthetic_e2e.py:run_world` | The `config_dir` argument is only partly honoured. `pipeline.yaml` and the fundamentals registry come from `config_dir`. Features (`load_feature_config(None)`), models (`make_model(n, horizon=h)` without `config_dir`), splits (`make_splits` → `SplitConfig.from_config()`), trading (`TradingConventions.from_config()`), targets and instruments are always read from the repo `configs/`. A run "with config X" silently mixes two trees. | Grep: `load_config` / `config_dir` call sites in `synthetic_e2e.py`, `splits.py:115`, `backtest.py:389`, `interface.py:586`. | Thread `config_dir` through every loader (`make_model(..., config_dir=)`, `SplitConfig.from_config(load_config("backtests", config_dir))`, `TradingConventions.from_config(load_config("backtests", config_dir))`, `load_feature_config(Path(config_dir) / "features.yaml")`, `CurveContext.from_curve(..., config_dir=)`). Add an integration test that a temp `backtests.yaml` change alters the folds. | orchestrator |
| SQ-05 | major | `configs/sources/power.yaml` vs `configs/sources/bloomberg.yaml` | One concept has two internal ids, so power features can never find the Bloomberg series. EUA even differs in **contract** (Dec benchmark vs front). Separately, `fx.eurusd.spot_usd_per_eur` is defined in both files (the only exact duplicate among 155 ids). `BloombergConfig.load` refuses duplicates only across `bloomberg*.yaml`, so this one is not caught. | Pairs (power.yaml → bloomberg.yaml): `coal.api2.m1_usd_t` vs `price.coal_api2.m1_usd_t`; `carbon.eua.dec_eur_t` vs `price.eua.front_eur_t`; `power.de.base_m1_eur_mwh` vs `price.power_de.base_m1_eur_mwh`; `gas.ttf.m1_eur_mwh` vs the TTF curve (power.yaml itself says "align id"). | Make `bloomberg.yaml` the single owner of market prices and FX; `power.yaml: prices` should **reference** those ids, not define new series. Decide Dec vs front for EUA explicitly. Add a config test: every series id is defined in exactly one `configs/sources/*.yaml`, and every id referenced in a `prices:`/`inputs:` block exists. | market-data-agent (ids) + weather-power-agent (references) |
| SQ-06 | major | `features/prices.py` (reference splice) vs `contracts/rolls.py` | Two roll implementations are live. D-025 says `rolls.py` is canonical, but `features.yaml: price.params.return_series_provider: null`, so features use their own splice. They also use different calendars: features use `common.holidays: []` (weekday calendar), while targets use the empirical session calendar in `instruments.yaml`. On synthetic weekday data they agree on all 1,565 sessions. On real data with ICE Endex holidays they can disagree around a roll. | Integration test `test_roll_consistency.py` (passes on synthetic; it cannot see holidays). | Wire `return_series_provider` to a `contracts.rolls` adapter and delete the fallback, or derive `features.common.holidays` from `instruments.yaml: calendars.TTF`. Keep the integration test either way. | feature-engineering-agent (consumer) + market-data-agent (owner of the roll) |
| SQ-07 | major | `models/interface.py:inner_time_splits` vs `evaluation/splits.py:inner_folds` | Two tuning-split implementations. `inner_time_splits` purges by **row count** (`gap` rows), which assumes one row per trading day; with NaN-dropped or multi-horizon rows a row gap is not a time gap. `inner_folds` purges by `label_known_time`, which is the correct rule (PIT policy §6). `models.interface.tune_*` uses the row-based one. | Read both functions. `inner_time_splits` docstring: "training rows are `[0, b0 - gap)`". | One owner: `evaluation/splits.inner_folds`. Make `interface.tune_*` accept fold index pairs produced from `inner_folds`. Keep `inner_time_splits` only as a thin wrapper (or delete it). | backtesting-agent (owner) + forecasting-agent (caller) |
| SQ-08 | minor | `evaluation/backtest.py:walk_forward` L≈304 | `walk_forward` always adds a benchmark. `benches = dict(bf) if bf else {"zero": ZeroForecast}`, so even `benchmark_factories={}` gets a zero benchmark, and callers cannot opt out. The pipeline works around this by filtering. `backtest.ZeroForecast` also duplicates `models.baselines.ZeroReturn`. | `synthetic_e2e.py` comment: "walk_forward always adds a default zero benchmark; keep benchmarks from the first run only". | Default `benchmark_factories=None` → `{"zero": ZeroReturn}`, with `{}` meaning **no** benchmarks (`if bf is None`). Remove `ZeroForecast` in favour of `models.baselines.ZeroReturn`. | backtesting-agent |
| SQ-09 | minor | `pipelines/synthetic_e2e.py` run_id | The default `run_id` uses the wall clock (`%Y%m%dT%H%M%S`), so two reproductions get different ids and cannot be compared by id. The manifest's `created_time` is wall clock too (acceptable as metadata). | Read `run_world`. | Default `run_tag` = first 12 hex of the full config hash (SQ-02), so identical inputs → identical run_id. | orchestrator |
| SQ-10 | minor | `settings.Paths.from_env` | Resolving the data root calls `get_secret("TTF_DATA_ROOT")`, which parses the whole `.env` (and so loads every secret into memory) whenever `data_root` is None. `TTF_DATA_ROOT` is not a secret. | Read `settings.py`. | Read `TTF_DATA_ROOT` from `os.environ` / a non-secret config; keep `get_secret` for secrets only. | orchestrator (frozen contract) |
| SQ-11 | minor | `models/regimes.py` L653, 671, 702 | `warnings.simplefilter("ignore")` around statsmodels Markov fits suppresses **all** warnings, including FutureWarning/DeprecationWarning, so upstream API breakage there would be invisible. | Grep. | Filter only `ConvergenceWarning` / `RuntimeWarning` by category, and count convergence failures into the regime diagnostics. | regime-agent |
| SQ-12 | minor | 4 modules | Private names imported across modules: `features/panel.py:43 _raise_on_conflicting_ties` (point_in_time), `features/weather.py:51 _deep_merge` (settings; a public `deep_merge` exists), `models/baselines.py:25 _linear_predict` (linear), `models/probabilistic.py:18 _require_lightgbm` (boosting). | ast scan. | Make each public in its owner module (or use `deep_merge`). | owners of the importing modules |
| SQ-13 | minor | `pipelines/ingest.py` logs | Structured logs carry absolute local paths (`raw_path: C:\Users\<user>\...`), which leaks the user name into any shared log. | Integration-run log lines. | Log paths relative to the store root. | data-architecture-agent |
| SQ-14 | minor | `settings.load_config` | `lru_cache` returns the **same mutable dict** to every caller. A caller that mutates it poisons the cache for the rest of the process. `load_feature_config` deep-copies; other callers don't. | Read `settings.py`. | Return `copy.deepcopy` (or a `MappingProxyType`) from the cached loader. | orchestrator (frozen contract) |
| SQ-15 | minor | `features/*` builder | Performance: the feature matrix takes ~80 % of an e2e run (≈20 s for 516 origins, ≈40 s for 770). This is not a correctness issue, but it limits the size of the ablation / sensitivity harness. | `WorldResult.timings`. | Profile `build_feature_matrix`. The as-of resolution per origin is the likely hotspot (`features/panel.py` already vectorises one path). | feature-engineering-agent |
| SQ-16 | minor | `.pre-commit-config.yaml` location | Git's root is `BGN_LNG/`, so pre-commit will not pick this file up automatically. | `git -C .. rev-parse --show-toplevel`. | Run with `-c ttf_forecaster/.pre-commit-config.yaml`, or reference it from a root config once the user agrees. | user |
| SQ-17 | minor | environment | There is no lockfile; `pyproject.toml` has lower bounds only (D-002). The system interpreter has pandas 3.0.3, which the `<3` bound would reject. That is fine, but a fresh install today would resolve different minor versions than `env_local`. | `docs/environment_audit.md`. | Before the first real-data result, `pip freeze > requirements-lock.txt` from `env_local` and record its sha256 in each run manifest (SQ-02). | orchestrator |

### Static-quality counts (no ruff/mypy installed; ast-based scan of `src/`, 44 modules, 558 public functions)

| Check | Result |
|---|---|
| `python -m compileall src tests` | OK |
| `ast.parse(..., feature_version=(3, 11))` | 137/137 files OK (src + tests) |
| Unused imports (ast, excluding `__init__` re-exports) | **3, all in tests**: `tests/unit/lng/test_spark.py: np`, `tests/unit/market/test_bloomberg.py: MockBloombergAdapter`, `tests/unit/weather/test_weather_adapter.py: MockWeatherAdapter`; 0 in src |
| Lines > 100 chars | 140 total (src ≈ 85, tests ≈ 55). Largest: `data/spark.py` 8, `data/storage.py` 7, `data/base.py` 6, `data/fundamentals.py` 5, `data/validation.py` 5, `models/regimes.py` 5; tests: `weather/test_weather_features.py` 9, `data_arch/test_pit_edge_cases.py` 8, `lng/test_spark.py` 7 |
| Public functions without full type hints | **0** |
| Public functions/methods without a docstring | 137. Largest: `data/spark.py` 26, `data/bloomberg.py` 14, `contracts/futures.py` 12, `evaluation/diagnostics.py` 6, `features/weather.py` 6, `features/power.py` 5, `models/baselines.py` 5, `models/interface.py` 5. Mostly adapter methods that implement a documented protocol (acceptable) and small properties. |
| Hard-coded absolute paths in src/configs | none (also enforced by an integration test and a pre-commit pygrep) |
| Broad `except Exception` in src | 1 (`evaluation/backtest.py:198`, re-raised with context: acceptable) |

### Reproducibility checklist

| Item | Status |
|---|---|
| Seeds | OK. Synthetic generators are seeded from `pipeline.yaml`; LightGBM uses `random_state`, `n_jobs=1`, `deterministic=True`, `force_row_wise=True` (`models/boosting.py:70`); the model seed is in `models.yaml`. |
| Determinism | **Verified:** two null-world runs into separate stores give bit-identical predictions and feature matrices (`assert_frame_equal(check_exact=True)`), identical artefact content hashes and identical criteria. |
| Ordering | `groupby(sort=False)` is used in `validation.py` and `panel.py` on frames that are pre-sorted. The determinism test covers these. No `set` iteration feeds an output order. `feature_hash` sorts columns. `config_hash` uses `sort_keys`. |
| Warnings | **0** FutureWarning / DeprecationWarning. The full suite passes with `-W error::FutureWarning -W error::DeprecationWarning`, and `-W default` shows no warnings. Caveat: SQ-11 hides the Markov-fit warnings. |
| Timezones | Schemas require tz-aware UTC (`validate_frame` refuses naive or non-UTC timestamps). The `tz_convert(local).tz_localize(None)` pattern is used only to derive local session dates (correct). One unchecked use: `data/spark.py:285` `tz_localize(None)` without `tz_convert` on release dates; harmless for London 19:00 releases, but it should convert first. |
| Config hashing / manifests | Present, but incomplete (SQ-02); `run_id` is wall-clock by default (SQ-09). |
| Environment pinning | Lower bounds only, no lock (SQ-17). |
| Hard-coded paths | None in src/configs. `PROJECT_ROOT` is derived from `__file__`. |

## 3. Test inventory

| Directory | Tests | Notes |
|---|---|---|
| `tests/unit/` (top level: `test_core_contract.py`) | 25 | frozen contract |
| `tests/unit/data_arch` | 134 | store, validation, ingest, PIT edge cases |
| `tests/unit/evaluation` | 39 | splits, metrics, DM/CW, backtest |
| `tests/unit/features` | 58 | incl. leakage canary on real builders |
| `tests/unit/fundamentals` | 51 | |
| `tests/unit/lng` | 36 | |
| `tests/unit/market` | 53 | |
| `tests/unit/model_risk` | 18 | added during review |
| `tests/unit/models` | 73 | |
| `tests/unit/regimes` | 36 | slowest unit tests (≈5 s each for the canary/Markov) |
| `tests/unit/text` | 54 | |
| `tests/unit/weather` | 36 | |
| **`tests/integration`** (new) | **23** | 18 fast (≈3 s) + 5 `slow` e2e (≈100 s, one module-scoped fixture running the null world twice) |
| **Total** | **636** | **636 passed in 183 s** with `-W error::FutureWarning -W error::DeprecationWarning` (unit ≈ 75-80 s, integration ≈ 100 s) |

Modules with no direct test: `reporting/research_report.py` and `logging_config.py`. Both are
exercised only indirectly; `cli.py` and `pipelines/synthetic_e2e.py` are now covered by
`tests/integration`.

How to run:

    ..\..\env_local\Scripts\python.exe -m pytest tests/integration              # all (≈105 s)
    ..\..\env_local\Scripts\python.exe -m pytest tests/integration -m "not slow" # fast (≈3 s)
    set TTF_SKIP_SLOW=1 & ..\..\env_local\Scripts\python.exe -m pytest           # suite without e2e

### What the integration tests pin down

- `test_cli.py`: `check-config` and `identifiers` return 0 offline. Every identifier status is one
  of known / requires_discovery / requires_user_input.
- `test_live_backends_refuse.py`: Bloomberg, Spark, fundamentals and weather live backends raise
  `BackendUnavailable` without authorisation, with the socket layer patched to raise. Specifically:
  - an injected Bloomberg client and Spark transport are never touched before the check;
  - fundamentals refuses unknown ids even when authorised;
  - weather refuses on licence even when authorised;
  - every factory's default backend is `mock`.
- `test_no_secrets_by_default.py`:
  - **Static check.** No file under src/configs/tests/scripts/notebooks mentions
    `client_credentials` (except "never read" disclaimers) or an absolute `C:\Users` / `C:\Marti`
    path. No package code mentions `adhoc_scripts`. `.gitignore` covers `/data/`, `.env`,
    `*credentials*` and `configs/local*.yaml`.
  - **Runtime check.** `check-config`, `identifiers` and all mock adapters run with `get_secret` and
    `.env` loading patched to raise, and no `.env` or credential file is opened.
- `test_synthetic_e2e.py`: a reduced-window run, with a temp copy of `configs/` (end 2023-12-29,
  4 folds). Checks:
  - artefacts plus `is_synthetic` manifests and config hash, with content-hash verified read-back;
  - predictions validate against `PREDICTION_SCHEMA`;
  - all models are scored on identical rows, with no duplicate forecasts;
  - `train_end ≤ origin < label_known_time`;
  - the null world fails criterion 1 for every model;
  - **determinism**: two runs are bit-identical;
  - run immutability: an identical re-write is a no-op, and a changed re-write raises.
- `test_roll_consistency.py`: the features' front month equals `contracts.rolls` on every
  synthetic session, and `roll_offset_bdays` equals `RollRule.roll_days`.

## 4. Module verdicts

**Accept** = reproducible and tested, with findings at most minor. **Accept with conditions** = usable
now, but the named finding must close before a real-data result. **Reject** = not reproducible or
not testable as is. No module is rejected.

| Module | Verdict | Reason |
|---|---|---|
| `settings.py` | accept with conditions | SQ-10, SQ-14 (frozen contract: cache mutability, `.env` read for the data root) |
| `logging_config.py` | accept | small; no direct test (minor) |
| `cli.py` | accept | offline, now integration-tested |
| `data/base.py` | accept | strict UTC schema validation; 6 long lines |
| `data/storage.py` | accept with conditions | immutability and content hashes verified; manifest lacks code/env version (SQ-02) |
| `data/validation.py` | accept | |
| `data/synthetic.py` | accept | seeded, labelled SYNTHETIC |
| `data/bloomberg.py` | accept | live refusal verified offline; 14 undocumented protocol methods |
| `data/spark.py` | accept | live refusal verified, transport and secrets injectable; 26 undocumented methods, 8 long lines; `tz_localize(None)` at L285 |
| `data/fundamentals.py` | accept | refusal ordering correct (unknown id → authorisation → connector) |
| `data/weather.py` | accept | refuses on authorisation and on licence |
| `contracts/*` | accept | `rolls.py` is the canonical roll (SQ-06 is a consumer issue) |
| `targets/*` | accept | fixed held contract; `label_known_time` is consistent with the AR lags (checked) |
| `features/point_in_time.py` | accept | |
| `features/prices.py` | accept with conditions | SQ-06 (second roll definition, holiday calendar) |
| `features/panel.py`, `calendar.py`, `interactions.py`, `registry.py`, `storage.py`, `flows.py`, `lng.py`, `weather.py` | accept | private import (SQ-12) in panel/weather; performance (SQ-15) |
| `features/power.py` | accept with conditions | SQ-05: its price inputs reference ids no adapter serves |
| `fundamentals/*` | accept | |
| `models/interface.py` | accept with conditions | SQ-07 (row-based inner splits) |
| `models/baselines.py`, `linear.py`, `boosting.py`, `probabilistic.py`, `ensemble.py` | accept | deterministic LightGBM; private imports (SQ-12) |
| `models/regimes.py` | accept | SQ-11 (blanket warning suppression) |
| `evaluation/splits.py` | accept | canonical purge/embargo |
| `evaluation/backtest.py` | accept | SQ-08 (implicit zero benchmark, duplicate ZeroForecast) |
| `evaluation/statistical_tests.py` | accept with conditions | **SQ-03**: criterion 1 has no minimum fold count |
| `evaluation/metrics.py`, `evaluation/diagnostics.py` | accept | diagnostics landed during review; unit-tested by model_risk |
| `pipelines/ingest.py`, `build_features.py` | accept | SQ-13 (absolute paths in logs) |
| `pipelines/synthetic_e2e.py` | accept with conditions | **SQ-02, SQ-04, SQ-09**: config hash, partial `config_dir`, wall-clock run_id |
| `reporting/research_report.py` | accept | no direct test (minor) |
| `text/*` | accept | no LLM calls, schema-validated |
| `configs/sources/power.yaml` / `bloomberg.yaml` | accept with conditions | SQ-05 |

## 5. Tooling

`.pre-commit-config.yaml` (config only, nothing installed) contains:

- **pre-commit-hooks:** check-yaml, end-of-file-fixer, trailing-whitespace, check-toml, merge-conflict,
  large-files (500 kB), detect-private-key, mixed-line-ending.
- **Local `system` hooks:** ruff, ruff-format --check, mypy (src). These use the env's own versions
  and `pyproject.toml` settings.
- **`fail` hooks:** they block anything under `data/` (except `.gitkeep`), `*credentials*` / `.env` /
  `*.secret` / `*.pem` / `*.key`, `.parquet` / `.feather` / `.duckdb` anywhere, and
  `configs/local*.yaml`.
- **pygrep:** blocks absolute `C:\Users` / `C:\Marti` paths in src/configs/scripts.

Installing ruff, mypy and pre-commit (the `dev` extra) needs the user's approval.
