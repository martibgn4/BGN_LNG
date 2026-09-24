# WeatherReturnsModel

Predicting 1- to 30-day returns on natural gas and agricultural futures from
weather, at the front and front+1 contract.

`SPEC.md` carries the design and the phased acceptance gates. This file is the
short version and the operating instructions.

Conventions follow `GasImbalanceModel` and `NWEBasisModel`: config-driven, no
float literals in `.py`, vintaged cache, fetchers raise rather than invent, and
a failed gate is reported as failed.

---

## The one idea

**Forecast levels are priced. Forecast revisions are the news.**

A cold January is not bullish; a January that turns colder than the curve
already assumed is. Every participant sees the same 06z run at the same moment,
so the level of forecast heating demand is in the price by construction. What
this model regresses on is the *change* between two consecutive runs in the
forecast for the same target day.

That choice drives everything else: the point-in-time discipline, the coverage
rule, and the source selection.

---

## Why not Google Weather API

It was named in the brief and it is wired up (`data/google_weather.py`) as a
live cross-check. It cannot be the primary source: it serves the current
forecast and keeps **no archive of what past forecasts said**. Without that
there is no revision series before the day collection starts, so nothing to
backtest on.

Open-Meteo is primary because its Historical Forecast API does answer "what did
the forecast say last January". The Google fetcher raises rather than answering
a past issue date with the current run — that substitution would be
undetectable lookahead and would make a backtest look superb.

---

## What was verified against the live API (7 Sep 2026)

Probed rather than read off documentation, because three of these are
load-bearing and two were surprises.

| Finding | Consequence |
|---|---|
| `_previous_dayN` works on the **multi-year archive**, not just live | Historical revisions are reconstructable. The project is viable. |
| Max lead is **7 days**; `previous_day8+` returns **all-null, not an error** | Far-lead buckets are marked `historical: false` and excluded from backtests |
| Previous-run archive starts **~mid-2021** | ~4 years, 4 heating seasons. That is the honest denominator. |
| `_previous_dayN` is **hourly-only** | Daily aggregates are built client-side |
| Ensemble models **pad the time axis with nulls** past their real horizon | `gfs025` answers a 35-day request with 10 days of data. Only **`gfs05`** truly reaches 30 days. |
| NG front delivery window is **0/102 dates reachable** at lead 7 | `window_mode: balance_of_month`; see SPEC §3.1 |

The last two would each have produced a confident, wrong model. Both are now
guarded by tests.

---

## Layout

```
config/      sources, regions, targets, features, model   <- all tunables
src/wxreturns/
  store.py       vintaged cache; three time axes, see below
  data/          openmeteo, google_weather, bloomberg (+ roll resolution)
  weather/       degree_days, normals, revisions, crop
  market/        contracts (roll, windows), returns (roll-gap aware)
  model/ridge.py closed-form ridge, Newey-West, block bootstrap
  backtest.py    walk-forward with purge and embargo
tests/
```

### Three time axes, and why

- `vintage` — when *we* fetched it. Open-Meteo restates its archive on model
  upgrades; this stops that changing a finished backtest.
- `issue_date` — when the *forecast* was produced. Stops a forecast being used
  before it existed.
- `target_date` — the day being forecast.

Use `store.read_forecast_as_of` for anything carrying `issue_date`. Using
`read_as_of` instead gives a backtest that trades on forecasts issued days into
its own future, and it will look excellent.

---

## Running it

Runs in the `GasImbalanceModel` venv unchanged (numpy-only estimation).

```bash
PYTHONPATH=src ../GasImbalanceModel/.venv/Scripts/python.exe -m wxreturns.cli info
```

| Command | Does |
|---|---|
| `info` | config hash, archive limits, what is cached |
| `audit-weights` | lists regions whose demand weights are still estimates |
| `backfill --region us_gas` | walks the archive from mid-2021 in chunks |
| `fetch-weather` | archived forecasts at leads 1–7 for a window |
| `fetch-actuals` | ERA5 reanalysis, for normals |
| `fetch-live --ensemble` | today's run plus ensemble spread |
| `fetch-prices --commodity NG` | Bloomberg settles (needs a Terminal) |
| `check-revisions` | Gate 1: builds the panel and checks the sign |
| `roll-calendar` | constructed roll and delivery windows |

### Start the daily collection now

The single highest-value action, and it costs nothing. History caps lead at 7
days; the 8–30 day buckets can only be *accumulated*. Running `fetch-live`
daily from today builds them at one day per day, which makes the
contract-delivery-window feature backtestable during 2027. Every day it is not
running is a day of that history not being created.

---

## State as of 7 Sep 2026 (Phase 1 complete)

Gate 1 passes:

- 64 tests green, including the no-float-literals rule.
- A real fetch: 94,224 rows of archived US gas forecasts, Nov 2024 – Mar 2025.
- **Sign check passed on real data.** The mid-January 2025 Arctic outbreak
  shows as +9.3 gas-weighted HDD of colder revision over 15–21 Jan — right
  sign, plausible magnitude.
- End to end on real Bloomberg NG settles: roll calendar (6 front contracts, 12
  roll-day returns correctly blanked) → balance-of-month revisions → a 10-column
  design matrix joined to forward returns.

**No return-predictability result is claimed, and none should be read into
this.** One winter is not a sample; the walk-forward needs 750 training days
before it will run at all, which the full backfill supplies and this slice does
not. Phase 2 is the backfill and the weather-side validation gates.

The honest prior remains that most of this signal is arbitraged away within the
session. The evaluation harness is built to be capable of saying so: the
benchmark is a zero forecast, costs are charged on position change, and every
scorecard carries the effective sample size next to the nominal one.

---

## Known gaps

1. **Open-Meteo licensing.** Free tier is CC-BY non-commercial. Desk use needs
   the paid tier. A decision for the user, not an engineering matter.
2. **All nine region weightings are estimates.** `audit-weights` lists them. The
   EU gas weights should come from the repo's own
   `20260825-EuropeLocalDistributionZoneGasDemandMonitor.xlsx`.
3. **Only the NG Bloomberg root is verified.** Every other contract is
   `verified: false` and the fetcher refuses it until confirmed from a Terminal.
   MATIF `CA` also needs real expiry dates; its rule is not constructible.
4. **Four heating seasons** is thin for a seasonal signal. Any claim resting on
   fewer than three independent winters is a hypothesis.
5. **Close-to-close only.** The 06z revision is traded well before the close, so
   a daily model measures a decayed version of the signal.

This project also fills the weather-forecast-revision gap that `NWEBasisModel`
and `GasImbalanceModel` both list as blocking; the revision panel is reusable by
both.
