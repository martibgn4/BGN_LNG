# LNG freight model

Short-horizon forecasting of **Spark30S** — the Atlantic round-voyage freight
assessment (US Gulf → Continent, 174k cbm 2-stroke), in USD per charter day —
at **1 day, 1 week, 2 weeks and 3 weeks** ahead (1 / 5 / 10 / 15 business days).

Target is the forward **log change**, not the level. Freight averaged 207k/day
in 2022, 40k in 2025 and printed 278k in Feb-2026 and 9k in Aug-2026. A model
fitted to the level is fitted mostly to whichever regime it saw most of.

---

## Headline result, stated up front

**The fundamental features do not add statistically significant skill over
freight's own lagged returns, at any of the four horizons.**

| horizon | best model | RMSE skill vs own-lags | Diebold-Mariano p | verdict |
|---------|-----------|------------------------|-------------------|---------|
| 1 bd  | lightgbm | +0.014 | 0.65 | not significant |
| 5 bd  | lightgbm | +0.060 | 0.26 | not significant |
| 10 bd | lightgbm | +0.025 | 0.68 | not significant |
| 15 bd | lightgbm | +0.032 | 0.61 | not significant |

The sign is consistently positive and never significant. That is a weak
positive hint, not a result — and the linear models land at **zero**
(family-PCA ridge: −0.005 / +0.004 / −0.003 / −0.000), so the sign is not
robust to changing estimator.

This holds after three separate rounds of adding data: fleet-supply proxies,
the routing-adjusted tonne-mile complex, and the Baltic traded routes. See
[The supply side](#the-supply-side-what-was-added-and-what-it-changed) and
[The Baltic traded routes](#the-baltic-traded-routes-ika1-ikd1-iki1).

### Why "vs own-lags" is the number to read, not "vs random walk"

Against a random walk the table looks much better — ridge scores +0.076 skill
at h=1, lightgbm +0.119 at h=15, several with p < 0.05. Almost all of that is
an artefact, and it was measured **before** any model was fitted:

```
AR(1) of the daily log change in Spark30S = +0.46
```

That is assessment smoothing. A price-reporting agency walks its published
number toward the market over several days rather than jumping to it, so the
published series is far more forecastable than the market it describes. Any
model holding lagged freight will harvest that and post a large, real and
almost entirely uninteresting "skill".

Two pieces of evidence that this is what is happening:

1. A ridge fitted on **freight's own past alone** (`own_lags`) gets a 72% hit
   rate and 0.52 IC at h=1 — matching the full 134-feature model to three
   decimals.
2. Delaying every feature by **one extra business day** collapses own-lags
   skill against the random walk from +0.038 to +0.0004 at h=5. The edge lives
   entirely in the most recent print.

So `own_lags` is the benchmark the headline is quoted against. The question
this project actually answers is *"do the arb, on-water and routing features
add anything on top of the assessment's own inertia?"* — and the answer today
is no, not significantly.

### The one finding with commercial content

**The smoothing edge is decaying.** Own-lags skill against a random walk, by
calendar year:

| horizon | 2024 | 2025 | 2026 |
|---------|------|------|------|
| 1 bd  | 0.134 | 0.107 | 0.055 |
| 5 bd  | 0.066 | 0.100 | −0.002 |
| 10 bd | 0.070 | 0.061 | 0.002 |
| 15 bd | 0.059 | 0.030 | 0.016 |

Spark's assessment has become materially more responsive as the FFA screen
tightened. Anything built on assessment inertia is standing on a shrinking
base, and by 2026 at a one-week horizon it is gone.

### The FFA forward has no skill either

The de-biased FFA curve scores +0.011 / +0.008 / −0.011 / −0.007 against a
random walk, with **negative IC at 10 and 15 days** (−0.15, −0.08). Over a
1-to-3 week horizon the freight curve carries essentially no information about
where the spot assessment goes, and at the longer end it points slightly the
wrong way. That is worth knowing on its own: it means the desk is not giving
up a good forecast by ignoring the near curve at these horizons.

---

## The supply side: what was added, and what it changed

The first version of this model was entirely cargo-side, and the obvious
objection was that a demand numerator without a supply denominator is
uninterpretable. That gap has now been searched for properly. Three findings.

### 1. Direct fleet counts exist on Bloomberg and are dead

The `VESL*` family — "Vessel Fleet Status LNG Tanker" — is exactly the right
data: `VESLLNIS` total in service, `VESLLNTC` in timecharter, `VESLLNOO` on
order, `VESLLNUC` under construction, `VESLLNBU` broken up, `VESLLNPC`
percentage orderbook.

**It stops on 2021-10-15.** Weekly from 2014, then nothing. This is not an
LNG-specific outage: `VESLAFIS` (Aframax), `VESLCTPC` (containership),
`VESLVLBU` (VLCC) and `VESLSZTC` (Suezmax) all stop on the same date, so the
whole vendor feed was switched off. The model sample starts 2022-01-03, so it
is unusable even as a partial overlay. Recorded in `sources.yaml` so the next
search does not repeat this one.

Poten's per-route shipping costs (`POTNGUSC` US Gulf→UK, `POTNGJSC` →Japan,
`POTNGSSC` →Spain, `POTNGCSC` →China) are live but **monthly, and six months
stale** — last print 2026-03-31. Useless at a 1–15 day horizon.

### 2. Tonne-miles is the better variable anyway

Over a 1-to-15 business-day horizon the LNG fleet is **fixed**. Newbuild
deliveries run a handful of ships a month against roughly 600 in service, and
they are scheduled years ahead. What varies at this horizon is not how many
ships exist but how many ship-days the trade pattern is consuming — which is
tonne-miles, and which *is* computable.

`AHOY JOURNEY` publishes daily LNG tons by origin→destination region, daily
from 2017-01-01, all live (`AHOYAMEU` Americas→NWE, `AHOYAMNA` →North Asia,
`AHOYAMSA` →South/SE Asia, `AHOYAMMD` →Med, `AHOYAMAM` →Americas, plus
`AHOYEXTL`/`AHOYIMTL` global totals). A ton to North Asia occupies a ship
roughly three times as long as a ton to NWE, so the **mix** of these legs sets
ship-day demand.

Validated before use: the five Americas legs sum to 1.08× `AHOYEXUS`, which is
correct — Americas includes Canadian, Mexican, Peruvian and Trinidadian
liftings. The resulting average-haul series behaves exactly as it should:

| | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| mean voyage length (nm) | 6,471 | 6,496 | **8,043** | 6,764 | 7,455 |

The 2024 peak is the Panama drought and Red Sea diversions forcing long
routings — the tonne-mile spike everyone in the market lived through, showing
up in the feature without being told about it. The Asian legs are
**routing-adjusted**: distance is interpolated between direct and
Cape-of-Good-Hope using the observed `LNGGTRCA` transit share, so the feature
responds to rerouting rather than assuming a fixed voyage.

### 3. It still does not help

Holding the estimator fixed and varying only the information set — the clean
incremental test, ridge throughout:

| feature set | cols | skill vs own-lags (h=5) | DM p | skill vs own-lags (h=15) | DM p |
|---|---|---|---|---|---|
| freight own lags | 21 | 0.000 | — | 0.000 | — |
| + tonne-miles | 45 | −0.013 | 0.13 | −0.002 | 0.91 |
| + utilisation | 69 | −0.008 | 0.40 | −0.020 | 0.49 |
| + both (supply) | 93 | −0.013 | 0.27 | −0.032 | 0.29 |
| + on-water | 75 | +0.004 | 0.64 | −0.006 | 0.75 |
| + arb spreads | 39 | −0.005 | 0.23 | −0.021 | 0.22 |
| + routing | 33 | −0.000 | 0.90 | −0.004 | 0.74 |
| + FFA curve | 40 | +0.004 | 0.57 | −0.001 | 0.94 |
| everything | 206 | −0.009 | 0.69 | −0.074 | 0.11 |

**Not one family adds anything.** Every incremental skill sits between −0.03
and +0.004 and no p-value is below 0.10. The supply features are, if anything,
mildly negative.

### The dilution problem was real, and separate

Note the last row: all 206 columns together score worse than any pair. That is
the sibling StockPicker's failure mode — 206 columns against ~600 observations,
where `tm__avg_haul_nm` at two windows, z-scored and differenced four ways, is
one economic quantity getting eight votes.

Its documented fix is applied here as `family_pca_ridge`: each fundamental
family is compressed to its two leading principal components, fitted inside
the training window, with the freight family passed through raw. It repairs
the damage —

| horizon | raw 206-column ridge | family-PCA ridge |
|---|---|---|
| 10 bd | −0.033 | −0.002 |
| 15 bd | −0.074 | +0.001 |

— and lands on **zero**. Which is the point worth taking away: the negative
result is not "too many columns". Fix the dilution properly and the
fundamentals still add nothing.

---

## The Baltic traded routes (IKA1, IKD1, IKI1)

`IKA1` = BLNG1 (Australia→Japan), `IKD1` = BLNG2 (US Gulf→Continent),
`IKI1` = BLNG3 (US Gulf→Japan). These are the desk's Baltic route reads and
replace the unentitled `BLNG*174` assessment tickers.

**BLNG2 is the same route as Spark30S**, which makes `IKD1` an independent,
exchange-settled read on this model's own target. Two measurements made before
any model was fitted:

```
AR(1) of daily log return:   IKD1 = -0.07      Spark30S = +0.44
```

The traded contract has **no serial correlation**; the assessment has a great
deal of it. That is the cleanest confirmation available of this project's
central finding: `IKD1` is the unsmoothed twin of `Spark30S`, and the
assessment's forecastability really is a reporting artefact rather than a
property of the freight market.

### They roll, and the roll is most of their big moves

| series | share of large moves at a month turn | vs baseline |
|---|---|---|
| `IKD1` | 69% | 2.53x |
| `IKA1` | 62% | 2.28x |
| `IKI1` | 60% | 2.22x |
| Spark30S (does not roll) | 35% | 1.29x |

These are generic 1st futures: on a roll the level steps by the calendar
spread. Raw returns are therefore unusable, and excluding roll days drops the
IKD1-to-Spark30S same-day return correlation from 0.225 to 0.134 — meaning a
third of the apparent co-movement was two series stepping on the same dates.
Features use levels, spreads, and returns with a 3-day roll mask.

### The basis predicts — with the wrong sign

The `log(IKD1 / Spark30S)` basis correlates with the subsequent Spark30S
change at **−0.25** (h=1), −0.27 (h=5), −0.17 (h=10), −0.10 (h=15), all
significant.

The sign is negative, which is *not* the "assessment catches up to the traded
price" story the feature was built to test — a positive basis precedes the
assessment falling further, not rising. It is recorded as measured rather than
reinterpreted. The most likely mundane explanation is that it proxies "spot is
low relative to its own recent range", which `frt__z21` already carries.

### And it still does not add skill

`lngfreight baltic-test` restricts to 2024-02 onward, where these series
actually exist, and varies only the feature set:

| horizon | own lags + baltic | DM p | baltic alone | IC: own → +baltic |
|---|---|---|---|---|
| 1 bd  | −0.023 | 0.31 | −0.071 | 0.47 → 0.37 |
| 5 bd  | **+0.038** | 0.36 | −0.017 | 0.46 → 0.48 |
| 10 bd | −0.010 | 0.75 | −0.021 | 0.31 → 0.34 |
| 15 bd | **+0.019** | 0.54 | +0.005 | 0.13 → **0.27** |

Mixed, and nothing significant. The one genuinely interesting line is h=15,
where adding the Baltic routes roughly **doubles the IC** (0.13 → 0.27) and
lifts the hit rate from 51.6% to 60.3% while barely moving RMSE. That is a
ranking improvement without a magnitude improvement — the Baltic basis helps
say *which way* three weeks out, not *how far*. With n=381 it is not
significant, and it is the one thread here worth pulling if any is.

Note these numbers use a 252-day training window against the headline table's
378, because the restricted sample is only ~665 days total. They are not
directly comparable to the headline.

---

## What the features say when you look inside

Sum of |standardised ridge coefficient| by family:

| family | h=1 | h=5 | h=10 | h=15 |
|--------|-----|-----|------|------|
| `ut` utilisation (afloat/exports, global flows) | 0.019 | **0.111** | **0.180** | **0.222** |
| `ow` on water | 0.021 | 0.102 | 0.153 | 0.182 |
| `frt` freight's own past | **0.025** | 0.098 | 0.137 | 0.138 |
| `blt` Baltic traded routes | 0.017 | 0.066 | 0.111 | 0.122 |
| `crv` FFA curve shape | 0.011 | 0.049 | 0.076 | 0.089 |
| `tm` tonne-miles / voyage length | 0.010 | 0.049 | 0.074 | 0.089 |
| `arb` JKM-HH, TTF-HH, JKM-TTF | 0.006 | 0.025 | 0.043 | 0.060 |
| `rt` routing / chokepoints | 0.005 | 0.031 | 0.053 | 0.061 |
| `sea` seasonality | 0.002 | 0.011 | 0.019 | 0.022 |
| `ux` US exports | 0.001 | 0.010 | 0.012 | 0.012 |

The supply-side families are where the fit puts **most** of its weight from a
week out — `ut` is now the largest family at every horizon past h=1, ahead of
freight's own past. The model is doing exactly what the economics say it
should, leaning on fleet utilisation and voyage length.

And it gains nothing for it. That juxtaposition is the cleanest statement of
the result: these features are not being ignored by the fit, they are being
used and they do not pay. Read this table as *where the fit puts weight*, never
as evidence of skill — the out-of-sample tables above are the evidence, and
they are flat.

### Where the non-linear model's advantage comes from

LightGBM's gain over own-lags is concentrated in one year:

| horizon | 2024 | 2025 | 2026 |
|---------|------|------|------|
| 5 bd  | −0.017 | +0.007 | **+0.138** |
| 10 bd | −0.173 | −0.017 | **+0.159** |
| 15 bd | −0.283 | +0.050 | **+0.196** |

In 2024 it was materially worse. It wins only in 2026 — the year with the
17.5k → 278k February spike and the August collapse to 9k. The honest reading
is that the tree model handles violent regime moves better once it has seen
two years of data, not that it has a stable edge. One good year is not a
result, and this is the single most likely thing to disappear out of sample.

---

## Data

Every ticker and contract was executed live from this machine on 2026-09-17 and
its payload inspected before being written into `config/sources.yaml`.
38 series, 97,078 rows. `IKD1` shares a route with the target.

| block | source | coverage | lag |
|-------|--------|----------|-----|
| Spark30S / Spark25S spot | Spark API | 1,315 releases from 2019-07-30 | 0 |
| Spark30 FFA M+0..M+3 | Spark API | 1,402 releases from 2021-02-04 | 1 |
| HH / TTF / JKM | Bloomberg `NG1`, `TMR1`, `TZT1`, `JKL1` | 2016/2018 → | 1 |
| **Baltic BLNG1/2/3** | Bloomberg `IKA1`, `IKD1`, `IKI1` | from 2024-02 | 1 |
| **LNG on water** | Bloomberg `LNGG20D*`, `LNGG30D*` | daily from 2017-01-01 | 1 |
| Chokepoint transits | Bloomberg `LNGGTRPA/SU/CA/NS` | daily from 2018-12-04 | 2 |
| US LNG exports | Bloomberg `AHOYEXUS` | daily from 2017-01-01 | 1 |
| **Trade flows by O-D leg** | Bloomberg `AHOYAM*`, `AHOYEX*`, `AHOYIM*` | daily from 2017-01-01 | 1 |

### LNG on water needs no Kpler key

The sibling `NWEBasisModel` records Kpler on-water as its **top blocking gap**.
Bloomberg redistributes it directly:

- `LNGG20DT/DC Index` — volume (t LNG) and vessel count afloat ≥20 days
- `LNGG30DT/DC Index` — the same at ≥30 days
- `LNGG20DU/DQ/DA/DN/DR/DO` — vessel count by origin (USA, Qatar, Australia,
  Nigeria, Russia, other)

Daily, calendar-dated, from 2017-01-01, no extra credential. The ≥20-day cut is
the useful one: a normal USG-to-Europe transit is under 20 days, so this is
cargo *not* on a direct voyage — floating storage, slow steaming, or a cargo
still looking for a home. **This unblocks NWEBasisModel's stated gap too.**

Found by `//blp/instruments` discovery rather than by guessing tickers.

### Identified and not available

`BLNG1174/2174/3174 Index` (Baltic spot assessments) and `NALF*` (Platts LNG
freight) both resolve — `name` comes back — but `px_last` is NaN and
`last_update_dt` empty. That is an entitlement wall, not an outage. Recorded in
`sources.yaml` so nobody re-derives them and assumes a bug. Spark covers the
same need, so this costs the model nothing today.

---

## How the backtest is protected

Walk-forward, expanding window, 18-month minimum train, monthly refit, ~31
refits per horizon.

- **Publication lag applied before windows are rolled, not after.** Rolling a
  21-day mean on unshifted data and shifting the result still leaks the newest
  observation into every window containing it.
- **Purge of h days + 5-day embargo.** The label for origin *t* is not known
  until *t+h*, so training rows within h days of the test block have seen part
  of it. Without this every horizon past h=1 reports inflated skill.
- **Imputation and scaling fitted inside the training window.** A full-sample
  median is information about a future that has not happened.
- **Targets require a genuine print at both ends.** Spark was weekly-only from
  2023-06-27 to 2023-12-19 with a 14-day hole at new year. Forward-filling
  through that and differencing would manufacture runs of exactly-zero returns.
  So the horizons do *not* share a sample (n = 611/643/634/631) and n is
  reported per horizon.
- **Every model scored on identical origins**, benchmarks included.
- **Diebold-Mariano with Newey-West h−1 lags.** With overlapping h-day targets
  the naive t-stat on a difference of squared errors is badly oversized.

### Sanity checks (`lngfreight sanity`)

| check | expected | observed |
|-------|----------|----------|
| shuffled target | skill → 0 | ridge −0.008, lightgbm −0.032, own-lags −0.002, IC ≈ 0 |
| one extra day of lag | skill degrades | own-lags 0.038 → 0.0004; lightgbm 0.119 → 0.070 |

The pipeline finds nothing where there is nothing, and degrades rather than
improves when information is withheld. Both are also enforced as tests.

---

## Usage

Run from `C:\Marti\python_tests_2\env_local` (xbbg 0.10.3 + the ML stack)
against a logged-in Terminal.

```bash
cd C:/Marti/python_tests_2/BGN_LNG/LNGFreightModel
```

```bash
PYTHONPATH=src C:/Marti/python_tests_2/env_local/Scripts/python.exe -m lngfreight.cli backtest
```

Other commands: `fetch --refresh`, `sanity`, `lag-sensitivity`, `importance`,
`predict`, `baltic-test`. Tests: `python -m pytest tests -q` (21 passing).

One runtime note: `family_pca_ridge` fits a PCA per family per refit, so a full
backtest takes a couple of minutes rather than seconds.

Live forecast as of 2026-09-17, Spark30S at 24,500 USD/day:

| model | 1bd | 5bd | 10bd | 15bd |
|-------|-----|-----|------|------|
| lightgbm | 24,817 | 30,017 | 33,207 | 41,630 |
| own_lags | 25,295 | 27,710 | 31,511 | 34,149 |
| family_pca_ridge | 25,036 | 27,701 | 31,368 | 33,280 |
| ridge | 25,229 | 27,863 | 30,040 | 30,965 |
| ffa_forward | 24,624 | 24,698 | 26,366 | 28,835 |
| random_walk | 24,500 | 24,500 | 24,500 | 24,500 |

Everything points up, which is the FFA curve's own shape (M+1 = 42,750 against
a 24,500 spot) showing through. Given the table at the top, treat these as a
structured way of reading the curve rather than as an independent forecast.

`huber` is **disabled**, and the reason is worth recording because it was
nearly written off as a modelling result. It was there as a robustness check
on the linear fit. At 206 features it scored −1.0 to −2.0 skill. That is not
the loss-function mismatch it was first assumed to be — measured at h=15 on
450 training rows, its train RMSE is **0.047 against a test RMSE of 1.07**, a
22x gap. It is interpolating 206 columns through 450 observations, because
sklearn's `HuberRegressor` takes one fixed `alpha` with no CV path while
`RidgeCV` searches to 10,000. Even at alpha=100 it is twice as bad as
predicting no change.

Tuning its alpha until the number looked acceptable would have meant choosing
a hyperparameter by looking at the test block. `family_pca_ridge` serves the
same robustness role correctly. The full diagnosis is in `model.yaml`.

---

## What would actually move this forward

In priority order.

1. **Model the FFA, not the spot assessment.** This is the biggest one. The
   spot assessment is not tradable and its forecastability is mostly the
   reporting agency's own smoothing. `spark30ffa-monthly` M+1 *is* tradable,
   and a real edge there would be worth money. The pipeline already carries
   the curve — this is a change of target, not a rebuild.

2. **Vessel supply has now been searched for, and the honest answer is that
   Bloomberg cannot supply it.** The fleet-count family (`VESL*`) died in
   October 2021. What could be built instead — routing-adjusted tonne-miles
   and fleet-utilisation ratios from the `AHOY*` flow matrix — was built, gets
   the largest ridge weight of any family, and adds nothing. A real fleet-side
   feed (idle/available tonnage, the sublet-relet market, drydock schedules)
   would have to come from a ship broker — Clarksons SIN, Affinity, Fearnleys
   — not from this Terminal. That is a subscription decision, and given that
   tonne-miles delivered zero, it is worth sizing before buying.

3. **Per-route on-water.** `LNGG20D*` aggregates globally with an origin split.
   What matters for Spark30 specifically is USG→Europe versus USG→Asia laden
   tonnage. Kpler would give voyage-level detail that Bloomberg aggregates
   away — still worth having, just not a blocker any more.

4. **Weather forecast revisions**, which this, `NWEBasisModel` and
   `GasImbalanceModel` all want and none has.

5. **Re-check the 2026 LightGBM advantage in a year.** If it holds through
   2027 it is real; if it does not, it was the spike.

---

## Layout

```
config/    sources.yaml  target.yaml  features.yaml  model.yaml
src/lngfreight/
  config.py store.py dataset.py features.py backtest.py cli.py
  data/     base.py bloomberg.py spark.py
  model/    baselines.py estimators.py
tests/      test_leakage.py test_config_discipline.py
```

Conventions follow `NWEBasisModel` and `GasImbalanceModel`: config-driven with
no tunable numeric literals in `.py` (enforced by test), vintaged cache,
fetchers that raise rather than invent, and a partial panel treated as a
failure rather than fitted on.
