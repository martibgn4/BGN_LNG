# SPEC.md — European Gas Balance Model

## 0. How to use this file

Read this file in full before writing any code. Do not build everything at once.
Build **one phase at a time**, stop at each acceptance gate, show me the output, and
wait for my sign-off before moving to the next phase.

Before you start, ask me the questions in Section 10.

---

## 1. Context and objective

I am a gas/LNG trader. I want a European gas supply-demand balance model that
produces a **daily and monthly balance for EU27 + UK + Switzerland**, and from it
derives two tradeable outputs:

1. **Call on LNG** — the LNG import volume required to make the balance close.
2. **Storage trajectory** — projected inventory path, with end-March and
   end-October inventory as the headline numbers.

The model must run both **deterministically** (single weather/supply scenario) and
**stochastically** (Monte Carlo over weather, wind, and supply outages) to produce a
distribution of end-of-winter storage.

This is a decision-support tool that will inform real positions. Accuracy and
auditability matter more than features. I would rather have a model that closes to
history within 2% on three terms than one that models fifteen terms badly.

---

## 2. The core identity

Everything in this model serves one accounting identity, evaluated over a defined
perimeter and period:

```
Indigenous production
+ Pipeline imports (crossing the perimeter)
+ LNG sendout
+ Storage withdrawals
= Total demand
+ Storage injections
+ Exports (crossing the perimeter)
+ Balancing residual
```

The `balancing residual` term is a diagnostic, not a plug. When the model is run
against **historical actuals**, this term should be small and mean-reverting. If it
is systematically non-zero, there is a bug or a missing term — surface it loudly,
do not absorb it silently into another line.

---

## 3. Tech stack and engineering constraints

- **Python 3.11+**, `uv` for dependency management (fall back to `pip` + `venv` if
  `uv` is unavailable).
- **Data layer:** DuckDB, single local file. All series stored long-format with a
  `(date, country, series_id, value, unit, vintage)` schema.
- **Compute:** pandas + numpy. Use `polars` only if you can show a real speed need.
- **Config:** all assumptions in YAML under `config/`. **No magic numbers in code.**
  If a number appears in a `.py` file, it is a bug.
- **Testing:** pytest. Every phase ships with tests. See Section 8.
- **CLI:** `typer`. Commands: `fetch`, `build`, `run`, `backtest`, `report`.
- **No notebooks as production code.** Notebooks under `notebooks/` are for
  exploration only and must not be imported by the package.
- **Every number written to the database must be traceable to a source and a
  vintage.** Never fabricate, interpolate, or "estimate" a data point without
  flagging it with a `is_estimated` boolean column and logging a warning.

---

## 4. Repository structure

```
gasbalance/
├── SPEC.md
├── pyproject.toml
├── config/
│   ├── perimeter.yaml          # countries in/out, interconnection points
│   ├── conversions.yaml        # unit conversions, GCV/NCV, calorific values
│   ├── demand_params.yaml      # regression coefficients, holiday calendar
│   ├── supply_params.yaml      # decline rates, contract volumes, outage stats
│   ├── storage_params.yaml     # working gas capacity, ratchet curves
│   └── scenarios/              # named scenario overlays
├── src/gasbalance/
│   ├── data/                   # fetchers, one module per source
│   ├── model/
│   │   ├── demand_ldz.py
│   │   ├── demand_power.py
│   │   ├── demand_industry.py
│   │   ├── supply_pipeline.py
│   │   ├── supply_indigenous.py
│   │   ├── storage.py
│   │   └── balance.py          # assembles the identity, solves for call on LNG
│   ├── stochastic/             # Monte Carlo engine
│   ├── validation/             # backtest + reconciliation
│   └── cli.py
├── tests/
└── output/
```

---

## 5. Units — get this right first, everything else depends on it

Convert **at the point of ingestion**. Store everything internally in a single unit
system. Never let a raw-unit value propagate past the fetcher.

- Internal daily unit: **mcm/d**
- Internal energy unit: **TWh** (for storage and seasonal aggregates)
- Reference: 1 bcm ≈ 10.55 TWh (GCV); 1 mcm/d ≈ 11 GWh/d

**Mandatory checks to encode:**

- **GCV vs NCV.** European gas volumetrics are quoted GCV. Some power-sector
  efficiency figures are NCV. The gap is ~10%. Every series in the config must
  declare which basis it is on, and the conversion layer must handle the crossover
  explicitly. Add a test that fails if a series lacks a declared basis.
- **Calorific value varies by source.** A Norwegian mcm ≠ an Algerian mcm ≠ a
  Qatari mcm. Hold per-source CVs in `conversions.yaml`, defaulted but overridable.
- Where a source publishes in kWh/d or GWh/d (ENTSOG does), convert, don't assume.

---

## 6. Data sources

Build one fetcher module per source, each with local caching to Parquet and a
**vintaged** write to DuckDB (see Section 8 on why).

| Source | What | Access |
|---|---|---|
| ENTSOG Transparency Platform | Physical flows at every interconnection point | Public API |
| GIE AGSI+ | Daily storage inventory, injection, withdrawal, by country and facility | Public API, key required |
| GIE ALSI | LNG terminal inventory and sendout | Public API, key required |
| ENTSO-E Transparency | Power load, generation by fuel, wind/solar | Public API, key required |
| Gassco | Norwegian planned and unplanned outages | Scrape / manual feed |
| National TSOs | National Gas (UK), GRTgaz, Snam, Enagás | Varies |
| ECMWF / Meteostat | Temperature, wind — **ensembles, not just the mean** | Varies |
| Kpler / Vortexa / Spark | Cargo-level LNG flows | **Paid — no free substitute** |

For paid sources: build the interface and a clearly-marked mock/CSV-loader
implementation. **Do not simulate the data and present it as real.** If I have not
given you credentials, the fetcher must raise, not return synthetic values.

If any public endpoint has changed or a key is required, tell me — do not work
around it by inventing data.

---

## 7. Build phases

### Phase 1 — Perimeter, units, data spine

- Define the perimeter in `perimeter.yaml`: which countries are inside the ring, and
  the explicit list of **interconnection points that cross it**.
- Build the DuckDB schema and the ENTSOG + AGSI fetchers.
- Build the unit conversion layer with full test coverage.

**Critical rule to encode: only flows crossing the perimeter count.** Norwegian gas
landing at Easington and then moving through the IUK to Zeebrugge is *one* import
and *one internal transfer*. Ukraine/Moldova transit and Swiss transit to Italy must
be netted. Build a test that asserts internal flows sum to zero net across the
perimeter. Double-counting here is the single most common failure mode.

**Acceptance gate:** reproduce EU+UK aggregate storage inventory from AGSI for the
last three gas years, and reconcile daily flows at every perimeter point with no
unexplained gaps.

---

### Phase 2 — Demand, in three separate buckets

Model these **separately**. Do not fit a single aggregate demand curve.

**2a. LDZ (residential + commercial)** — a temperature function.

```
HDD = max(0, 15.5 − T_avg)
Demand_LDZ = a + b × HDD + day_of_week_effect + holiday_effect
```

- Fit per country on population-weighted temperature (not capital-city temperature).
- Include day-of-week dummies (weekends softer) and a Christmas/holiday calendar.
- Consider a non-linear term at extreme cold — the response steepens.
- **Sanity benchmark:** 1°C below normal across NW Europe for a full winter should
  produce roughly **5–6 bcm** of additional demand. If your fitted model doesn't
  reproduce that order of magnitude, it is wrong. Add this as an explicit test.

**2b. Power sector** — model **residual load**, never a fixed gas burn.

```
Residual load = Power demand − wind − solar − nuclear − hydro − must-run
```

Fill residual load in merit order between gas and coal. Convert to gas burn at a
**fleet-average efficiency of ~50–52%**, not nameplate — plants part-load and cycle.
Make efficiency a config parameter with a comment explaining why it is below
nameplate.

**Sanity benchmark:** a 10 GW drop in wind for one day should produce roughly
**44 mcm/d** of extra gas burn (240 GWh electricity ÷ 50% efficiency ÷ 11 GWh per
mcm). Encode as a test.

**Coal-to-gas switching** — implement the switching price:

```
P_gas_switch = η_gas × [P_coal/η_coal + P_CO2 × 0.34/η_coal] − P_CO2 × 0.20
```

All parameters in config. This sets the elastic band on power demand. **Important
caveat to document in the code:** Europe's remaining coal fleet is now small enough
that this band is much thinner than it was pre-2022. The dominant elastic buffer in
the current market is industrial shut-in and LNG re-export, not coal switching. Do
not let the switching logic carry more weight than it deserves.

**2c. Industry** — baseline plus price elasticity.

- Weather-independent baseline with a slow structural decline term.
- Price-response term calibrated off 2022–23, when roughly 15–20 bcm of European
  industrial demand shut in at €200+/MWh.
- Model the response with a **lag** — industry does not switch on and off daily.

**Acceptance gate:** monthly demand backtest against actuals for three gas years,
MAPE under 5% at EU+UK aggregate.

---

### Phase 3 — Supply

- **Norway** (~115 bcm/y) — the largest single dial. Drive off the Gassco
  maintenance calendar. Model unplanned outages stochastically; historical
  unplanned loss runs ~3–5% of annual volume. Treat Troll and Kollsnes as
  individually significant assets.
- **Indigenous** — UKCS, Netherlands (Groningen **closed October 2023**, offshore
  residual only), Romania, Italy, Denmark. Annual decline rate plus seasonal shape
  is sufficient.
- **Pipeline imports** — Algeria (Transmed, Medgaz), Azerbaijan via TAP, Libya.
  Russian pipeline supply is now **TurkStream into SE Europe only**; Ukraine transit
  ended 1 January 2025. Model as contractual base plus a nomination-flexibility band.
  Put all of this in config — the geopolitics changes faster than the code should.
- **Biomethane** — ~5 bcm/y and growing. Small but no longer noise.
- **LNG** — do **not** forecast this in Phase 3. It is the residual (Phase 4).

**Acceptance gate:** non-LNG supply backtest against actuals, monthly, three gas
years, within 3%.

---

### Phase 4 — Close the balance, solve for call on LNG

Assemble the identity. Solve for LNG as the residual:

```
Call on LNG = Total demand + injections + exports
              − indigenous − pipeline imports − withdrawals
```

Produce a monthly and daily table in this shape (illustrative January, mcm/d):

| Line | mcm/d |
|---|---|
| LDZ demand | 800 |
| Industrial | 350 |
| Power | 350 |
| **Total demand** | **1,500** |
| Norway | 350 |
| Indigenous | 100 |
| Algeria + Azeri + TurkStream | 190 |
| Biomethane | 15 |
| **Non-LNG supply** | **655** |
| Storage withdrawal | 400 |
| **Call on LNG** | **445** |

Then add a **feasibility check**: compare required LNG sendout against
(a) regas capacity by terminal from ALSI, and (b) a configurable Atlantic-basin
supply ceiling. Flag when the required call exceeds either.

**Acceptance gate:** run in backtest mode over three historical gas years —
implied call on LNG should track actual LNG sendout within 10% monthly.

---

### Phase 5 — Storage, with ratchets

```
S_t = S_(t−1) + injections_t − withdrawals_t
```

**The constraint that must not be omitted: deliverability ratchets.** Withdrawal
capability is a function of inventory, because cushion gas pressure falls as the
field empties.

- EU aggregate withdrawal ≈ **20–22 TWh/d** near full.
- Falls toward **8–10 TWh/d** below ~30% full.
- Injection is the mirror image — it slows as you approach full.

Implement as a piecewise or interpolated ratchet curve per country in
`storage_params.yaml`. The model must **reject physically impossible withdrawal
paths** and surface them as a constraint breach, not silently allow them. A balance
that requires 500 mcm/d of withdrawal at 25% full is not a forecast — it is a price
signal, and the model should say so.

Also: **check the current EU storage regulation targets at build time rather than
hard-coding them.** The original rigid 90%-by-1-November rule was amended in 2025 to
be more flexible. Put the target in config with a comment noting it must be verified
against the live regulation.

**Headline outputs: end-of-March and end-of-October inventory**, in TWh and % full.

---

### Phase 6 — Stochastic engine

Monte Carlo over:

- **Temperature** — use ECMWF ensemble members directly. Do not fit a normal
  distribution to temperature; the ensemble spread *is* the distribution, and its
  skew is the information.
- **Wind output** — correlated with temperature. Preserve that correlation; cold
  and calm arrive together and that is exactly the tail that matters.
- **Norwegian and other supply outages** — Poisson arrival, empirical duration.
- **Asian LNG demand** — as a parameterised competing-pull scenario.

Output: a **distribution of end-of-winter storage**, which maps onto a price
distribution. Report P10/P50/P90 on end-March inventory and on call on LNG.
Default 1,000 paths, configurable. Must run in under two minutes on a laptop.

---

## 8. Validation — non-negotiable

- **Vintaged data.** ENTSOG and AGSI restate same-day figures for several days after
  publication. Store every observation with its `vintage` (fetch timestamp) and make
  backtests query **as-of** vintages only. Backtesting against restated data produces
  flattering lies. This is a hard requirement, not a nice-to-have.
- **Daily reconciliation job.** Compare model-implied storage against AGSI actuals
  every day. A systematic 20 mcm/d gap means a bug or a missing term. Log it, alert
  on it, and never let it be absorbed into the residual.
- **Linepack.** Several hundred mcm of short-term system flexibility absorbs daily
  imbalances. Add it as an explicit buffer term so that daily balances aren't
  misdiagnosed as errors.
- **Storage is not supply.** It is a time-shifter. Encode this in the reporting:
  every molecule withdrawn this winter must be repurchased next summer, and that
  repurchase must appear as a **demand** term in the following summer's balance.
- Golden-file tests on the unit conversion layer and the ratchet curves.
- A `backtest` CLI command that runs any gas year and outputs error decomposition
  **by term**, so I can see which line is breaking the balance.

---

## 9. Outputs

- `output/balance_YYYYMM.csv` — full monthly balance, all terms.
- `output/storage_path.csv` — daily trajectory with P10/P50/P90.
- A one-page HTML summary: balance table, storage fan chart, call-on-LNG vs regas
  capacity, and a list of any constraint breaches.
- Every output carries the run timestamp, config hash, and data vintage.

---

## 10. Ask me these before you start

1. Which perimeter — full EU27+UK+CH, or NW Europe only for a first cut?
2. Do I have ENTSO-E, AGSI and ALSI API keys, or should you build against cached
   sample files first?
3. Do I have a paid LNG tracking subscription (Kpler/Vortexa/Spark), or should the
   LNG side stay purely residual for now?
4. Which weather source do I have access to for ensembles?
5. How many historical gas years should the backtest cover?

---

## 11. What not to do

- Do not build all six phases before showing me anything.
- Do not invent, interpolate or synthesise data to make something run. Raise instead.
- Do not put numeric assumptions in `.py` files. Config only.
- Do not add features I have not asked for — no dashboards, no ML price prediction,
  no web frontend.
- Do not silently absorb a reconciliation gap into the residual.
- If something in this spec is ambiguous or looks wrong to you, say so before
  building it.
