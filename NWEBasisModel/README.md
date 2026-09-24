# NWE basis model

Forecasting model for the **NWE LNG discount to TTF**.

Sign convention, fixed in `config/target.yaml` and used everywhere:

```
basis = DES_NWE − TTF        USD/MMBtu,  negative = NWE at a discount
```

Same convention as `adhoc_scripts/NWE_option_pricer.py` (`NWE_MEAN = -0.505`) and
as Spark's `sparknwe-fin-monthly`, so nothing needs flipping at a boundary.

---

## Headline result, stated up front

**The model does not beat the market forward.** On a walk-forward backtest over
2023-07-01 to 2026-09-03, with every feature reachable from Spark:

| tenor | horizon | model RMSE | forward RMSE | skill | hit rate |
|-------|---------|-----------|--------------|-------|----------|
| M+2   | 5 bd    | 0.0516    | 0.0377       | −0.37 | 24%      |
| M+2   | 21 bd   | 0.1416    | 0.1126       | −0.26 | 51%      |
| M+3   | 21 bd   | 0.1036    | 0.0719       | −0.44 | 32%      |
| M+4   | 21 bd   | 0.0648    | 0.0548       | −0.18 | 28%      |

Skill is `1 − RMSE_model / RMSE_benchmark`; negative means the benchmark won.
For a fixed delivery month the forward and the random walk are the same number,
so "forward" above is the price you could actually hedge at.

Two things follow, and they are the useful part of this result:

1. **The Spark-reachable features are not enough.** Adding the east-west arb
   moved M+2/21bd skill from −0.41 to −0.26 — a real improvement in the right
   direction, and still short of the market. The three features that would
   plausibly close the gap are precisely the three that are missing:
   LNG on water, storage headroom, and weather forecast revisions. See
   [What data is needed](#what-data-is-needed).

2. **The hit rate is consistently BELOW 50%**, across four tenors and two
   horizons. That is not random underperformance: it says the error-correction
   term has the wrong sign at these horizons. The basis level clearly reverts —
   the OU fit gives a 31-day half-life on M+1 — but the deviation from the
   *fundamentals-based anchor* does not close within a month; if anything it
   extends. Either the anchor is misspecified (most likely: it is missing the
   length variable that actually drives disequilibrium) or the basis carries
   short-horizon momentum the ECM structurally cannot express.

   **This has not been "fixed" by flipping the sign.** A 28% hit rate inverts to
   72%, which would look excellent and would be a backtest artefact, not a
   strategy. It is recorded as the top open question instead.

What the model *is* already good for is the OU parameter set below, which is
fitted rather than assumed and is a direct upgrade to the option pricer.

---

## What is worth using today

`nwebasis pricer-params` fits the seasonal OU and prints the block the desk's
cancellation-option pricer consumes. Fitted on 2023-07-01..2026-09-03,
de-seasonalised:

| tenor | phi | kappa (ann) | sigma (ann) | theta | half-life |
|-------|-----|------------|-------------|-------|-----------|
| M+1 | 0.9782 | 5.55 | 0.522 | −0.380 | 31 d |
| M+2 | 0.9907 | 2.36 | 0.362 | −0.384 | 74 d |
| M+3 | 0.9938 | 1.57 | 0.278 | −0.349 | 112 d |
| M+4 | 0.9950 | 1.26 | 0.206 | −0.312 | 139 d |
| M+5 | 0.9936 | 1.61 | 0.220 | −0.348 | 108 d |
| M+6 | 0.9923 | 1.95 | 0.244 | −0.373 | 89 d |

`adhoc_scripts/NWE_option_pricer.py` currently assumes a **single** set across
every tenor: `NWE_KAPPA = 3.0` (half-life ~84 d), `NWE_VOL = 0.35`,
`NWE_MEAN = -0.505`. Against the fitted term structure:

- **Reversion is roughly twice as fast at the front and half as fast by M+4.**
  One kappa cannot be right for both ends. A cancellation option on a Jan-27
  cargo valued off a 3-month half-life is being valued off M+3 dynamics while
  its decision date sits at the front.
- **Volatility is understated at the front** (0.52 fitted vs 0.35 assumed) and
  overstated further out. The pricer's own sensitivity run reports vega on NWE
  basis vol, so this moves the number directly.
- **The mean is narrower than assumed**, −0.38 vs −0.505, on the stable sample.

The **delivery-month seasonal** is also materially larger than the pricer's:

| month | fitted basis offset | pricer `nwe_seasonals` (as basis) |
|-------|--------------------|-----------------------------------|
| Jan | −0.099 | +0.004 |
| Feb | −0.104 | +0.014 |
| Mar | −0.109 | +0.024 |
| Jul | +0.074 | −0.001 |
| Aug | +0.120 | −0.001 |
| Sep | +0.121 | −0.006 |
| Dec | −0.053 | −0.021 |

Fitted range is 0.23 USD/MMBtu; the hand-set range is 0.045 — about five times
too small, and Jan-Mar carry the opposite sign. The shape is economically
sensible: the discount is widest in late winter, narrowest in Aug-Sep when
Europe is bidding for injection cargoes.

Note the pricer applies its seasonal to the *discount* (`nwe_discount = -nwe +
nwe_seasonal_adj`), so the table above negates the fitted basis offsets to make
the two comparable.

---

## How it works

Three layers, kept separate so that a disagreement between them is visible
rather than averaged away.

### Layer 1 — `model/envelope.py`, the no-arbitrage envelope

Where the basis *can* sit. The **cost floor** is the all-in cost of turning a
DES cargo into hub gas — regas tariff, slot, boil-off, port, working capital.
The **diversion cap** is the east-west arb: the discount cannot narrow past the
point where the marginal US cargo prefers Asia.

The envelope is a **regime classifier, not a clamp**. In Nov-Dec 2022 the M+1
basis reached −10.52 against a cost stack worth a fraction of that, because the
binding constraint stopped being cost and became capacity. Inside the envelope
cost anchors the basis; outside it the basis is the shadow price of a scarce
regas slot and has no cost ceiling.

This layer currently **raises** — `config/regas.yaml` ships with every tariff
unset, deliberately, and the model falls back to an empirical anchor. Filling
it in is the highest-value config change available.

### Layer 2 — `model/ecm.py`, the conditional mean

Engle-Granger in two stages:

```
stage 1   basis_t = gamma' Z_t + u_t                  the anchor, and the gap
stage 2   Δbasis_t = alpha·u_{t-1} + beta'ΔX_t + e_t  how the gap closes
```

Levels and changes are split because the basis and its drivers are all
persistent; a single levels regression gives a high R² and standard errors
wrong by an order of magnitude. Ridge is applied to the short-run block only —
Atlantic and Pacific freight correlate 0.93 and unpenalised OLS flips their
signs between folds — and never to the anchor, which carries the economics.
Standard errors are Newey-West, because the features are overlapping 5-day
changes and plain OLS errors would overstate significance about threefold.

Current fit, M+2, n=803, stable sample:

```
term                       coefficient   std_error    t
alpha (error correction)      -0.0178      0.0074   -2.41
arb_nea_minus_nwe             +0.0023      0.0009   +2.49
d_freight_atlantic            +0.0009      0.0010   +0.91
d_freight_basin               -0.0004      0.0009   -0.48
swe_nwe_spread                -0.0035      0.0020   -1.81
```

The arb is significant and correctly signed: Asia's netback improving relative
to Europe's pushes the basis up, because Europe has to richen to hold cargoes.
Freight contributes nothing once the arb is in — which makes sense, since
Spark's netback is already freight-adjusted, so the raw day rates were only ever
a noisy proxy for it. `alpha` is significant and negative, as the specification
requires; `ecm.fit` raises rather than reporting a positive alpha.

### Layer 3 — `model/seasonal_ou.py`, the distribution

Exact-discretisation AR(1) on the de-seasonalised basis, fitted per tenor. No
optimiser and no starting values: the OU has an exact AR(1) representation at a
fixed step, so a linear regression recovers kappa and sigma outright. Guards
against reporting a half-life for something that does not mean-revert — a
finite sample of a random walk fits phi just below 1 and would otherwise return
a confident several-hundred-day half-life.

---

## What data is needed

Reachable today with the desk's existing credentials, all verified live on
2026-09-04:

| what | source | status | history |
|------|--------|--------|---------|
| **NWE basis to TTF** (the target) | Spark `sparknwe-fin-monthly` | working | 946 releases from 2022-11-29 |
| **DES NWE outright** | Spark `sparknwe-des-fin-monthly` | working | from 2022-11-29 |
| **SWE basis** (congestion tell) | Spark `sparkswe-fin-monthly` | working | from 2022-11-29 |
| **East-west arb, freight-adjusted** | Spark netbacks, US Gulf via COGH | working | 1262 releases from 2021-09-01 |
| **TTF and JKM prices** | same netback payload | working | from 2021-09-01 |
| **Freight, Atlantic + Pacific** | Spark `spark30s` / `spark25s` | working | 1305 releases from 2019-07-30 |
| **Regas slot availability** | Spark terminal-slots, 20 terminals | working | from 2023-11-28 (NWE) |

The netback endpoint deserves a note: it returns TTF, JKM, the route cost to
each basin, and the arb, all in USD/MMBtu against TTF, from one call. It makes
the separate JKM and freight pulls largely redundant for this model. It also
requires `via-point` — omit it and it answers HTTP 200 with an empty list and no
error, which is why the fetcher treats an empty payload as a failure.

### What I still need from you

Ordered by expected value to the forecast.

**1. Kpler LNG on water — the biggest gap.** `KPLER_API_KEY`, or a CSV export.
Every other driver says what the market is worth; on-water volume says how much
cargo has to find a home and by when. In a long market the basis is set by the
marginal cargo that cannot be placed, and that cargo is visible on the water two
to three weeks before it prices. Without it the model reacts to length instead
of anticipating it, which is a fair description of what the backtest shows.
Specifically:
- Atlantic-basin laden volume, daily
- floating storage held >20 days, separated from transit — that is positioning,
  not logistics, and it means something different
- scheduled arrivals into NWE terminals by ETA week
- US Gulf liftings, daily

The interface and a CSV loader are built (`data/kpler.py`); the API
normalisation is deliberately unimplemented until a real payload has been seen,
rather than guessed at.

**2. Storage headroom — a Bloomberg pull you can already do.** The `CGIE*`
tickers are verified and in production in `GasImbalanceModel`, so this needs a
run from the Terminal rather than any new entitlement. `(capacity − inventory) /
capacity` for EU aggregate and the NWE countries, plus days-to-full at the
current injection rate. Late in injection season, with no headroom, the marginal
cargo has nowhere to land and the discount widens — that mechanism is entirely
absent from the model today.

**3. Regas cost stack — config, not data.** `config/regas.yaml` lists eleven NWE
terminals and five adders, all `null`. Filling them switches the anchor from
`empirical` to `cost`, which is what lets the model extrapolate to a tariff
structure it has not observed. Per terminal: the all-in cost a third-party
shipper pays to unload and receive gas at the hub, in USD/MMBtu of delivered
cargo, plus which tariff sheet and what date it is from. Tariffs reset annually
and a stale one is a silent error. `nwebasis cost-anchor` prints exactly which
entries are missing.

**4. Weather forecast revisions.** Not wired anywhere, in this project or
`GasImbalanceModel`, and both need it. NWE population-weighted HDD, day 6-15
window, with the **revision** as the feature rather than the level — the basis
reacts to news, not to weather. Worth building once and sharing.

**5. A JKM decision.** Production JKM on this desk comes from a
manually-maintained CSV built from settlement screenshots. That is fine for a
daily table and unusable for fitting: no vintage, unknown gaps. Either confirm
`JKL` Comdty carries clean daily history back to 2022-11, or the model keeps
taking JKM from the netback payload, which is verified. My recommendation is the
netback payload — it is already consistent with the arb.

### Known limitation: sample length

The binding constraint is not the target, it is the features. The basis goes
back to 2022-11 but regas slot data starts 2023-11-28, and the stable-regime
sample starts 2023-07-01. That leaves roughly 800 daily observations per tenor
covering **one full seasonal cycle and change**. A twelve-dummy seasonal on that
sample is close to fitting the calendar; the sine/cosine encoding is provided
alongside for that reason, and which is better is a question the backtest
should settle rather than a decision buried in code.

---

## Running it

Uses `GasImbalanceModel`'s venv as is — estimation is numpy-only, with no
statsmodels or sklearn dependency, so every coefficient traces to an expression
in the source.

All commands run from `NWEBasisModel/`:

```bash
PYTHONPATH=src ../GasImbalanceModel/.venv/Scripts/python.exe -m nwebasis.cli status
```

```bash
PYTHONPATH=src ../GasImbalanceModel/.venv/Scripts/python.exe -m nwebasis.cli fetch all
```

```bash
PYTHONPATH=src ../GasImbalanceModel/.venv/Scripts/python.exe -m nwebasis.cli pricer-params --tenor "M+1"
```

```bash
PYTHONPATH=src ../GasImbalanceModel/.venv/Scripts/python.exe -m nwebasis.cli backtest --tenor "M+2" --horizon 21
```

A full netback backfill is 1262 sequential calls and takes roughly twenty
minutes; everything else is quick.

### Conventions carried over from GasImbalanceModel

- **No float literal in any `.py`.** Enforced by `tests/test_no_magic_numbers.py`,
  which parses the AST. Anything calibrated lives in `config/`.
- **Vintaged cache.** Spark restates, and slot counts for a delivery month
  update continuously as slots sell. Every write is stamped with when *we*
  fetched it; `store.read_as_of` returns what was actually published at a chosen
  moment. Backtests must go through it — reading the raw frame returns restated
  data and a flatteringly good result.
- **Fetchers raise rather than invent.** No credential means an exception, never
  an empty frame or a plausible number.
- **A dropped feature is reported, never zero-filled.** Every command that
  builds a panel prints the manifest of what went in and what did not.

28 tests pass:

```bash
PYTHONPATH=src ../GasImbalanceModel/.venv/Scripts/python.exe -m pytest tests -q
```

---

## Open questions, in priority order

1. **Why is the hit rate below 50%?** The single most informative thing in the
   backtest. Test whether the anchor is missing the length variable (needs
   Kpler) before concluding the basis has short-horizon momentum.
2. **Is `stable_start: 2023-07-01` the right regime boundary?** It is currently
   a judgement. `model.yaml` calls for a sensitivity run over the date; it has
   not been built yet.
3. **Twelve dummies or sine/cosine for the seasonal?** Both are in the panel;
   the backtest has not been run head-to-head.
4. **Does the SWE spread lead or coincide?** It sits at t = −1.81, close enough
   to matter and not clearly a leading indicator yet. Worth a lead-lag scan.
