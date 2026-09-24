# WeatherReturnsModel — specification

Predict 1- to 30-day returns on natural gas and agricultural futures from
weather, at the front and front+1 contract.

Built 7 Sep 2026. Conventions inherited from `GasImbalanceModel` and
`NWEBasisModel`: config-driven, no float literals in `.py`, vintaged cache,
fetchers raise rather than invent, and a failed acceptance gate is reported as
failed rather than tuned until it passes.

---

## 0. The thesis, and the one that would not work

The naive version of this project is: forecast HDDs, regress returns on HDDs.
It does not work, and it is worth being explicit about why before building
anything, because every design decision below follows from it.

**The weather forecast is not private information.** Every participant sees the
same 06z run at the same time. A cold January is not bullish — a cold January
that is *colder than the curve already assumed* is bullish. The level of
forecast heating demand is in the price by construction.

So the object of interest is the **forecast revision**:

```
revision(issue_t, target_d) = HDD_forecast(issue_t, target_d)
                            − HDD_forecast(issue_{t−1}, target_d)
```

the change, between two consecutive model runs, in the forecast for the *same*
target day. That is the news. It is what the desk reacts to at 07:00, and it is
the only part of a weather forecast with a defensible claim on the return
between yesterday's close and today's.

Three corollaries that shape the build:

1. **Levels enter only as anomalies against a climatological normal**, and even
   then as controls, never as the signal.
2. **The revision must be measured over the contract's own delivery window**,
   not a fixed 1..15 day horizon. See §3.
3. **Ensemble spread predicts volatility, not direction.** It is carried as a
   separate target, not thrown into the mean equation.

The honest prior is that most of this is arbitraged away within the first
session, and that whatever survives is (a) small, (b) concentrated in the front
contract during the heating season, and (c) possibly not net of costs. The
backtest is built to be capable of returning that answer. `NWEBasisModel`
already returned a negative result on its own question and it was recorded as
the finding; the same rule applies here.

---

## 1. Data sources

### 1.1 Weather — Open-Meteo (primary)

The request named Google Weather API. It is a reasonable live-forecast source
and is supported as a pluggable provider (§1.2), but it **cannot be the primary
source**, for one disqualifying reason: it serves only the current forecast. It
keeps no archive of *what the forecast said last January*. Without that, the
revision series of §0 cannot be reconstructed for any date before the day the
project starts collecting, so there is nothing to backtest on for two or three
years.

Open-Meteo is used instead because it is the only free source that solves this:

| Endpoint | What it gives | Why it is needed |
|---|---|---|
| **Historical Forecast API** | Archived forecasts *as they were issued*, back to 2016 | The revision panel. This is the whole project. |
| **Archive API** (ERA5 / ERA5-Land) | Reanalysis actuals, 1940 onward | Realised degree days, normals, forecast-error attribution |
| **Ensemble API** | Per-member forecasts, GEFS / ECMWF-ENS / ICON-EPS | Spread, and so the volatility signal |
| **Forecast API** | Live deterministic runs, ECMWF IFS / GFS / ICON | Today's production signal |

All four were reachable from this machine on 7 Sep 2026 and are wired up.
Licence is CC-BY-4.0; the free tier is non-commercial, and commercial desk use
requires the paid tier. **That is a licensing decision for the user, not an
engineering one** — flagged in the README, not silently assumed.

### 1.2 Weather — pluggable providers

`data/base.py` defines the provider contract. `google_weather.py` implements it
against the Google Maps Platform Weather API for live forecasts only, and
raises `SourceUnavailable` on any request for a historical issue date rather
than quietly returning the current run under a past timestamp — which would be
lookahead of the most damaging kind, because it would make the backtest
brilliant.

### 1.3 Market — Bloomberg

`xbbg==0.12.2` against a logged-in Terminal, as everywhere else in this repo.
Settles via `bdh`, and crucially `blp.fut_ticker(root + "1 Comdty", dt=...,
freq="M")` to resolve *which contract was actually front on a past date*. The
roll calendar has to be observed, not assumed; see §3.

---

## 2. Point-in-time discipline

Two separate time axes, and conflating them is the failure mode this project is
most exposed to:

- **`issue_date`** — when the forecast was produced.
- **`target_date`** — the day being forecast.

Every weather row carries both. `lead = target_date - issue_date`. A revision is
a difference along `issue_date` at fixed `target_date`. A realised degree day is
`lead <= 0`.

On top of that sits the `store.py` vintage stamp (when *we* fetched it),
inherited unchanged from the sibling projects. Three timestamps sound
redundant; they are not. Open-Meteo restates its own archive when a model is
upgraded, so `vintage` protects against the archive changing under a backtest,
while `issue_date` protects against using a forecast before it existed.

**Publication lag.** A 00z run is not actionable at 00:00. Each model carries an
`available_after_hours` in `config/sources.yaml`, and features are stamped with
the first market close at which the run could have been traded. Getting this
wrong by one session produces a spectacular and entirely fake backtest.

---

## 3. Contract alignment

The generic `NG1 Comdty` is not a tradeable return series. Two problems, both
handled explicitly:

**Roll gaps.** The jump from expiring to next contract is not a return. Returns
are computed *within* a contract's life and chained, never across a roll.

**Window drift.** This is the subtle one. `NG1` on 20 December references
January gas; on 28 December, after expiry, it references February. A fixed
"next 15 days of HDDs" feature therefore points at the wrong gas for part of
every month, and the misalignment is worst in the last week — exactly when the
front contract is most weather-sensitive.

So weather is aggregated over **the delivery window of the specific contract
being traded**, resolved per date from the observed roll calendar. For NG1 on
20 December that is 1–31 January; for NG2 it is 1–28 February.

**Coverage.** Early in December, a 30-day forecast does not reach the end of
January, so the January aggregate is partly forecast and partly normal. Two
rules follow. Coverage fraction is carried on every row. And a revision is
computed **only over the target days common to both issue dates**, so that a
forecast horizon rolling forward one day cannot masquerade as news.

---

### 3.1 Reachability — a structural constraint, found on real data

Probing the live API on 7 Sep 2026 turned up a limit that changes what Phase 1
can deliver, so it is recorded here rather than in a commit message.

Open-Meteo's previous-run archive stops at **lead 7**. NG expires three
business days before its delivery month begins, and this calendar rolls five
business days before that, so the front contract always references a month
starting at least eight business days out.

Measured over Nov 2024 – Mar 2025: the front contract's delivery window sits
**10 to 41 days** ahead, and **0 of 102** observation dates have it within
seven days. Front+1 sits 38 to 72 days out.

So a contract-delivery-window revision for front NG is empty *always* — by
construction, not for want of a longer sample. Three consequences:

1. `config/features.yaml` sets `window_mode: balance_of_month`. The feature is
   the revision to weather between tomorrow and month end. This is defensible
   on its own terms rather than a fallback: near-term weather sets the storage
   draw, and the storage trajectory is what the front contract prices.
2. `contract_delivery` mode is implemented and refuses loudly, naming the
   reason, until leads beyond ~15 days exist.
3. Those far leads can only be **accumulated live**. Running the daily job from
   today builds them at roughly one day of history per calendar day, so the
   contract-window feature becomes available for real backtesting in 2027.
   Starting that collection immediately is the highest-value action available,
   and it costs nothing.

## 4. Markets

### 4.1 Natural gas
- **Henry Hub (`NG`)** — the deep, liquid, weather-driven benchmark. Primary.
- **TTF** — the desk's own market. Ticker root requires Terminal verification
  before use; carried in config as `verified: false`.

Signal: gas-weighted HDD/CDD revisions over the contract delivery window,
weighted across demand regions (§5).

### 4.2 Agriculture

The mechanism is *not* degree days, and the calendar gate is not optional. Corn
is insensitive to July rain in April and acutely sensitive to it in the third
week of July. Weather outside a phenological window carries close to zero
information, and including it is the main way an ag weather model manufactures
noise.

`config/features.yaml` therefore carries a **crop calendar** per commodity and
growing region, with hemispheres kept separate (Brazilian pod-fill is January).
Features are gated to the window and are: accumulated precipitation anomaly,
heat-stress days above a threshold, and soil-moisture anomaly.

- **Corn (`C `)**, **Soybeans (`S `)** — US Midwest; South America Dec–Feb.
- **Wheat (`W `, `KW`)** — US HRW and spring belts; MATIF `CA` for EU.

---

## 5. Demand weighting

An unweighted average temperature over a country is not a demand signal. Gas
HDDs are weighted by where the gas is actually burned; ag weather is weighted
by where the crop is actually grown.

- US gas: population/gas-weighted city composite, lower 48.
- EU gas: country LDZ demand shares. The repo already carries
  `20260825-EuropeLocalDistributionZoneGasDemandMonitor.xlsx`, which is the
  right provenance for these and supersedes any estimate.
- Ags: production-share weights by state / province.

Weights ship with a `provenance` field. Anything marked `estimate` is a
placeholder to be replaced, and `wxreturns audit-weights` lists them.

---

## 6. Estimation

Deliberately conservative, and numpy-only so every coefficient traces to an
expression in the source — the same choice `NWEBasisModel` made.

- **Ridge** on standardised revision features, penalty chosen by walk-forward
  CV. The features are collinear (a cold revision at lead 3 correlates with one
  at lead 4) and unpenalised OLS flips signs between folds.
- **Newey–West / HAC** standard errors with lag at least the return horizon.
- Separate fits per commodity, contract slot (front, front+1) and horizon.

### The overlapping-returns trap

Sampling 30-day returns daily gives about 30x overlap. Naive t-statistics are
inflated by roughly sqrt(30) = 5.5. A model can look strongly significant and be
nothing. Three defences, all mandatory rather than optional:

1. HAC lag truncation at least the horizon.
2. Stationary block bootstrap for all p-values, mean block length at least the
   horizon.
3. Effective sample size reported next to nominal N on every scorecard.

---

## 7. Evaluation

**The benchmark for a return forecast is zero.** Under an efficient market the
futures price is a martingale, so the honest competitor is "no forecast".
Beating a seasonal-mean or persistence benchmark proves nothing here.

Reported per commodity, slot and horizon:

- Out-of-sample R² against a zero forecast (negative is the common answer)
- Directional hit rate, with the binomial interval at the effective N
- Mean P&L of a unit signal, **net of a configured cost per round turn**
- Skill decay across horizons 1, 5, 10, 21, 30

Walk-forward only: expanding train, periodic refit, standardiser fitted on train
alone.

---

## 8. Phases and acceptance gates

Stop for sign-off at each gate.

### Phase 1 — Infrastructure *(this delivery)*
Config, vintaged store, all four Open-Meteo fetchers, Google provider stub,
Bloomberg fetcher with roll resolution, degree days, normals, revisions, crop
calendar, contract alignment, ridge + HAC, walk-forward harness, CLI, tests.

**Gate 1:** `pytest` green; `wxreturns fetch-weather` writes a real vintaged
forecast panel; the revision panel reproduces a known cold snap with the right
sign; no float literals in `.py`.

### Phase 2 — Weather validation
Backfill 2016 onward historical forecasts for all regions. Verify forecast skill
decays with lead as it should, and that reconstructed HDDs match ERA5 actuals at
lead 0.

**Gate 2:** lead-0 forecast HDD equals ERA5 HDD to within tolerance; skill decay
is monotone in lead. If it is not, the archive is being misread and nothing
downstream is trustworthy.

### Phase 3 — Market panel
Bloomberg settles, roll calendar, chained returns, contract delivery windows.

**Gate 3:** the chained return series has no roll-gap outliers, and reconstructed
front-contract identity matches `fut_ticker` on a sample of past dates.

### Phase 4 — Signal
Fit and walk-forward the gas model.

**Gate 4:** report OOS R² and net-of-cost P&L against the zero benchmark, with
bootstrap intervals. **A negative result passes this gate.** What fails it is a
result reported without its effective sample size or without costs.

### Phase 5 — Agriculture
Crop-calendar-gated features, same harness.

### Phase 6 — Production
Daily job, artefact for the desk, integration with the LNG report.

---

## 9. Known open questions

1. Open-Meteo commercial licensing for desk use — user decision.
2. TTF Bloomberg root needs Terminal verification.
3. EU LDZ weights should come from the repo's own demand monitor.
4. Whether ECMWF-ENS via Open-Meteo has enough archive depth; GEFS is the
   fallback and is weaker.
5. Intraday: this is a close-to-close model. The 06z revision is traded well
   before the close, so a daily model measures a decayed version of the signal.
