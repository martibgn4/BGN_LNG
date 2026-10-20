# Agent plan (STEP 2)

The orchestrator owns the contract that everyone builds against, and every file not listed below.
Each implementation agent owns a disjoint set of paths and may **read** anything but **write only**
its own paths. Where an agent needs a change in someone else's file it says so in its report; the
orchestrator decides and makes the change. No two agents write the same file.

All agents work in the shared checkout (no worktrees: the project is new and uncommitted, and a
worktree cut from `HEAD` would not contain the frozen contract).

## Frozen contract (orchestrator, written before any agent starts)

| Path | Content |
|---|---|
| `docs/research_specification.md`, `docs/target_definitions.md`, `docs/point_in_time_policy.md` | what is being tested and the rules |
| `docs/modelling_log.md` | decision log — agents propose entries in their reports, orchestrator writes them |
| `pyproject.toml`, `.gitignore`, `.env.example`, `CLAUDE.md`, `README.md` | project shell |
| `src/ttf_forecaster/settings.py`, `logging_config.py` | config loading, env, paths, logging |
| `src/ttf_forecaster/data/base.py` | canonical schemas (column contracts), adapter protocols, error types |
| `src/ttf_forecaster/features/point_in_time.py` | the as-of resolver every feature goes through |
| `src/ttf_forecaster/contracts/units.py` | energy/volume units |
| `tests/conftest.py`, `tests/fixtures/synthetic.py` | shared synthetic generators (SYNTHETIC-labelled) |

## Phase B — implementation agents (parallel)

| # | Agent | Owns (write) | Delivers | Depends on |
|---|---|---|---|---|
| 1 | data-architecture-agent | `data/storage.py`, `data/validation.py`, `pipelines/ingest.py`, `tests/unit/data_arch/`, `docs/data_dictionary.md` | Parquet layered store (raw/normalised/features/predictions/benchmarks) with vintage append, schema validation, ingest pipeline over any adapter | frozen contract |
| 2 | market-data-agent | `data/bloomberg.py`, `contracts/futures.py`, `contracts/calendars.py`, `contracts/rolls.py`, `targets/`, `configs/instruments.yaml`, `configs/targets.yaml`, `tests/unit/market/` | Bloomberg adapter (mock / local-file / live skeleton), TTF contract calendar, expiry & roll rules, fixed-contract return construction, all target builders | frozen contract |
| 3 | gas-fundamentals-agent | `data/fundamentals.py`, `fundamentals/`, `features/storage.py`, `features/flows.py`, `tests/unit/fundamentals/` | fundamentals adapter, transparent daily balance with explicit residual, scenario storage projection, storage & flow features | frozen contract |
| 4 | lng-flows-agent | `data/spark.py`, `features/lng.py`, `tests/unit/lng/` | Spark adapter (mock / local-file / live skeleton), PIT-safe arrival forecast schema and features | frozen contract |
| 5 | weather-power-agent | `data/weather.py`, `features/weather.py`, `features/power.py`, `tests/unit/weather/` | forecast-vintage weather adapter, HDD/CDD/revision/ensemble features, power-economics features | frozen contract |
| 6 | feature-engineering-agent | `features/prices.py`, `features/calendar.py`, `features/interactions.py`, `features/registry.py`, `pipelines/build_features.py`, `configs/features.yaml`, `tests/unit/features/` | price/curve/calendar families, interpretable interactions, family registry for ablations, as-of feature-matrix builder | 2 (column names only) |
| 7 | forecasting-agent | `models/baselines.py`, `models/linear.py`, `models/boosting.py`, `models/probabilistic.py`, `models/ensemble.py`, `configs/models.yaml`, `tests/unit/models/` | model ladder L0, L1, L2, L4 behind one fit/predict interface | frozen contract |
| 8 | regime-agent | `models/regimes.py`, `tests/unit/regimes/` | rule-based physical regimes (PIT), optional HMM behind a guard, regime labels for conditional evaluation | frozen contract |
| 9 | backtesting-agent | `evaluation/splits.py`, `evaluation/metrics.py`, `evaluation/statistical_tests.py`, `evaluation/backtest.py`, `configs/backtests.yaml`, `tests/unit/evaluation/` | nested walk-forward with purge/embargo, metrics, Diebold–Mariano (HAC), trading simulation with explicit execution conventions | frozen contract |
| 11 | llm-news-agent | `text/`, `docs/llm_event_schema.md`, `tests/unit/text/` | event schema + validator, provenance & human-review gate; **no populated data, no LLM calls** | frozen contract |

## Phase C — challenge, review and integration (after Phase B returns)

| # | Agent | Owns | Role |
|---|---|---|---|
| 9b | backtesting-agent (challenge pass) | review only | attacks the forecasting agent's evaluation claims |
| 10 | model-risk-agent | `evaluation/diagnostics.py`, `tests/unit/model_risk/`, `docs/limitations.md` | leakage, overfitting, false-conclusion hunt; ablation and sensitivity harness |
| 12 | software-quality-agent | `tests/integration/`, `.pre-commit-config.yaml`, review notes | reproducibility and test review; rejects untestable code |
| 3b | gas-fundamentals-agent (challenge pass) | review only | physical interpretability of feature families and interactions |

Orchestrator then: `pipelines/` (other than ingest/build_features), `scripts/`, `reporting/`,
synthetic end-to-end run, `docs/live_data_requirements.md`, `docs/model_card.md`, final report.

## Required report format (every agent)

assumptions · files inspected/created · methodology · data requirements (one row per series:
purpose, source, frequency, unit, history, publication delay, revision behaviour, PIT need, internal
field name, identifier status known/requires-discovery/requires-user-input) · limitations · tests run
with results · unresolved questions · proposed `modelling_log.md` entries · recommended next action ·
**challenges to other agents' assumptions**.

## Rules every agent receives

- Never invent vendor identifiers, endpoints, field names or dataset names. Unknown → `null` with
  `status: requires_user_input` or `requires_discovery` in YAML.
- No live vendor calls, no network, no package installs. Live backends validate config and raise.
- Synthetic data only in tests; every synthetic frame carries `source="SYNTHETIC"`.
- Every feature function takes `as_of` (or a PIT-resolved frame) and must not read beyond it.
- Type hints, docstrings, deterministic seeds, explicit units, meaningful errors.
- Run tests with `..\env_local\Scripts\python.exe -m pytest` from `ttf_forecaster/`.
