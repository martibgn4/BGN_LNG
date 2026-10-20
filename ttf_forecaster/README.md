# ttf_forecaster

Research platform that tests whether European gas fundamentals, LNG flows, weather and
cross-commodity information forecast **Dutch TTF** returns better than the **forward curve** already
does — honestly, point-in-time, with purged walk-forward validation.

**Status (2026-10-17): architecture complete and verified on synthetic data only.** No real-data
result exists. No claim about TTF predictability is made. See `docs/live_data_requirements.md` §8 for
the smallest set of inputs that unblocks the first real test.

## Install / environment

Developed in the desk venv `..\..\env_local` (Python 3.14, pandas 2.3, scikit-learn, statsmodels,
LightGBM, pyarrow, xbbg/blpapi). Code targets Python ≥ 3.11. No packages were installed for this
project; ruff/mypy/pre-commit are declared in the `dev` extra but not yet installed.

```bash
cd ttf_forecaster
..\..\env_local\Scripts\python.exe -m pytest -q
```

## Configure

| What | Where | Committed? |
|---|---|---|
| Vendor identifiers + status (known / requires_discovery / requires_user_input) | `configs/sources/*.yaml` | yes — identifiers only |
| Targets, features, models, backtests, regimes, scenarios, pipeline | `configs/*.yaml` | yes |
| Machine-specific overrides (e.g. LDZ weights, `live_calls_authorised: true`) | `configs/local_<name>.yaml` | **no** (git-ignored) |
| Secrets (`SPARK_CLIENT_ID`, `SPARK_CLIENT_SECRET`, `SPARK_BASE_URL`, Bloomberg host/port) | environment or `.env` (see `.env.example`) | **no** |
| Data store (raw / normalised / features / predictions / benchmarks) | `data/` or `TTF_DATA_ROOT` | **no** |

Live vendor calls are refused unless `live_calls_authorised: true` is set in a git-ignored local
config **and** the series' identifier status is `known`.

## Run

```bash
python -m ttf_forecaster.cli check-config      # parse every config, load adapter registries
python -m ttf_forecaster.cli identifiers       # identifier status per source file
python -m ttf_forecaster.cli synthetic-e2e     # SYNTHETIC end-to-end run, null + planted worlds
```

(With `PYTHONPATH=src`, or after `pip install -e .`.) The synthetic run writes
`outputs/synthetic_e2e_report.md` and immutable, synthetic-labelled artefacts under
`data/synthetic_runs/`. It verifies the pipeline: in the **null** world nothing should pass; in the
**planted** world the planted signal should be found.

## Layout

```
configs/            YAML: sources/ (vendor ids), targets, features, models, backtests, regimes, ...
docs/               research_specification, target_definitions, point_in_time_policy, modelling_log,
                    data_dictionary, live_data_requirements, model_card, limitations, ...
src/ttf_forecaster/
  data/             base.py (frozen contract), storage.py, validation.py, synthetic.py,
                    bloomberg.py, spark.py, fundamentals.py, weather.py (mock / local_file / live)
  contracts/        units, calendars, futures (expiry, tickers), rolls
  targets/          T1–T9 builders with label_known_time
  features/         point_in_time (as-of resolvers), panel, price/curve/cross-commodity, calendar,
                    storage, flows, lng, weather, power, interactions, registry
  fundamentals/     daily balance (explicit residual), scenarios, storage projection
  models/           ladder L0 baselines, L1 linear, L2 boosting, L3 regimes, L4 probabilistic, L6 ensemble
  evaluation/       splits (purge/embargo/locked holdout), metrics, DM/CW tests, backtest, diagnostics
  text/             LLM event schema, validation, review gate (no LLM calls)
  pipelines/        ingest, build_features, synthetic_e2e
  reporting/        research_report
tests/              unit/<area>/, integration/
```

## Rules that matter

Read `CLAUDE.md`. In short: point-in-time reads only through `features/point_in_time.py`; the forward
curve is the benchmark; targets are fixed-contract returns; no invented identifiers; no secrets or
licensed data in Git; negative results are reported as found; every decision goes in
`docs/modelling_log.md`.
