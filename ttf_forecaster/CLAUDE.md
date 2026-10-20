# ttf_forecaster — working rules for Claude

Research platform testing whether fundamentals beat the TTF forward curve. Read
`docs/research_specification.md`, `docs/target_definitions.md` and `docs/point_in_time_policy.md`
before changing anything that touches data, features, targets or evaluation.

## Run

    ..\..\env_local\Scripts\python.exe -m pytest            # from ttf_forecaster/
    ..\..\env_local\Scripts\python.exe -m ttf_forecaster.cli synthetic-e2e

## Non-negotiables

- **Point-in-time.** Feature code reads stored data only through
  `features/point_in_time.py` (`resolve_as_of`, `resolve_forecasts_as_of`). Never filter on
  `event_time` alone. Grade C data never enters a clean backtest.
- **Benchmark is the forward curve.** Report skill as OOS R² / DM test against the zero-excess-return
  (or futures-price) benchmark, never only against a historical mean.
- **Targets on fixed contracts.** No return is ever computed across a roll on a continuous series.
- **No invented identifiers.** Vendor tickers, fields, endpoints, dataset ids live only in
  `configs/sources/*.yaml` with a `status` of known / requires_discovery / requires_user_input.
  Only `known` may be requested live, and only on the user's explicit instruction.
- **No secrets or licensed data in Git.** Secrets from env / `.env` via `settings.get_secret`.
  `data/` is git-ignored. Tests use `data/synthetic.py` only; synthetic output is labelled
  SYNTHETIC and is never evidence of skill.
- **Negative results are results.** Do not tune, sign-flip or re-define a target to rescue a model.
  Record decisions in `docs/modelling_log.md`.
- **No package installs** without telling the user why.

## Layout

`data/base.py` is the frozen contract (schemas, adapter protocols, errors). Change it only
deliberately and log the change. One module per concern; notebooks call package functions.
