# M / M+1 delivery options on HH–TTF margins

Values the right to deliver a cargo in month **M or M+1** (optionally also to
cancel), where the margin is `sell.slope · P_sell + K − buy.slope · P_buy`.
Economically this is a **calendar spread option on the inter-hub spread**.

The work is split into two jobs that communicate only through files:

```
python calibrate.py            # Job 1: Bloomberg -> data/calibration/<date>/
python price.py                # Job 2: files -> output/<timestamp>/   (no Bloomberg)
python price.py --calib 2026-09-23_edited --only NovDec26,DecJan27 --as-of 2026-09-24
python -m pytest tests -q      # 22 tests, no Bloomberg
```

Config: `config/market.yaml` holds the hubs, calibration settings and engine
settings. `config/trades.yaml` holds the list of CSOs, with defaults and
per-trade overrides.

## Job 1: calibration

**Which months.** Every pair used in `trades.yaml`, plus the consecutive pairs
along the next `strip_months` (default 12). A new consecutive CSO inside the
strip can be priced without re-calibrating.

**Vol per contract (`contracts.csv`).**
- **Source.** For each future, read the option chain (TTF options are on `FJS`;
  `TZT` has none). Take the strikes either side of the forward, read the call
  vol (falling back to the put) and interpolate linearly to the forward.
- **Fields.** `IVOL_MID` first, then `IVOL_LAST` (the vol implied from the
  exchange settlement). Far TTF months have only the latter.
- **Traceability.** `vol_source` records the tickers and field used.
- **Missing vols.** Far NG (May‑27 onwards) and far TTF (Nov‑30 onwards)
  return neither field on this terminal. The default
  `missing_vol_fill: samuelson` fills `vol` from the realised vol-by-maturity
  curve scaled to the hub's market vols (see "Long-dated CSOs"). `previous`
  carries the last liquid month instead. Either way the row is marked `FILLED`
  in `note` and `atm_vol` stays blank.
- **Editing.** `vol` is the column the pricer reads. It is the vol to the
  option expiry, before any horizon adjustment.

**Vol profile (`vol_profile.csv`).**
- **What it is.** Realised vol by hub and days-to-expiry bucket, pooled over
  about 5 years of contracts.
- **Method.** Weekly returns; root mean square rather than standard deviation,
  to avoid demeaning bias.
- **Use.** Only its shape matters; see "Horizon vol" below.

**Correlations (`correlations/<M>_<M1>.csv`, one matrix per pair).**
- **Sample.** Each of the last 48 months k supplies its own contracts (hub_k,
  hub_k+1 for each hub).
- **Window.** Only the days when k was as far from fixing as M is between now
  and the decision date: `[0, max(days to M's fixing, 45)]`.
- **Returns.** Non-overlapping weekly returns, stacked across anchors. There
  are no rolled generics and no roll jumps.
- **Daily diagnostic.** Daily correlations are written alongside as
  `_daily.csv`. They cut the HH–TTF correlation roughly in half, because the
  TTF and NG closes are asynchronous.
- **Summary.** `correlation_summary.csv` has n_obs, the window and the headline
  numbers per pair.

The job never overwrites: a second run on the same day gets a time suffix. To
use hand edits, copy the folder to a name that sorts later
(`2026-09-23_edited`) and edit it; `--calib latest` then picks it up.

### Long-dated CSOs (2028-2031)

`config/trades_2028_2031.yaml` has one CSO per M from Jan-28 to Nov-31 (47).

```
python calibrate.py --trades config/trades_2028_2031.yaml   # keep the strip: see below
python price.py     --trades config/trades_2028_2031.yaml
```

Four things matter at these maturities (all verified on this terminal,
2026-09-23).

- **Anchors are counted back from today, not from M.** The 48 months before
  a 2031 M haven't expired, so there would be no history.
  - Windows run up to about 1,860 days before fixing.
  - TTF contracts print for about 6 years before expiry, NG for longer.
  - `max_stale_share` in the correlation summary reports zero weekly returns;
    it is at most 0.6%.
- **Most far vols are filled.**
  - TTF settlement vols (`IVOL_LAST`) run to Oct-30. HH has no listed vol
    past Apr-27.
  - `missing_vol_fill: samuelson` uses the hub's realised vol-by-maturity
    curve (buckets to 5y), averaged to each contract's expiry.
  - That curve is scaled to the hub's 12 longest-dated market vols: TTF x0.60
    (from Nov-29..Oct-30), HH x0.96 (from Oct-26..Apr-27).
  - HH therefore needs the near strip in the calibration. Without it there is
    nothing to scale to, and the far HH vols are left blank with a note.
  - The fill keeps the term decay but carries **no seasonality**. M and M+1
    get near-identical vols, so the value comes almost entirely from
    correlation and the forward curve.
- **FX is a forward per delivery month** (`contracts.csv` `fx`), taken from
  EUR<tenor> BGN pips, the same method as the daily report. It runs from
  1.163 (Jan-28) to 1.225 (Dec-31), against 1.141 spot.
- **The strike cancels**, so only extrinsic value and risk are meaningful.
  Total value depends on the placeholder strike of 21.5.

### Brent as a sell benchmark

Brent is a hub in `market.yaml`: root `CO`, USD/bbl, 1,000 bbl lots,
`contract_month_offset: 2`. Calibration covers every hub in the registry, so
each pair's correlation matrix now spans TTF, HH and Brent. What was verified
on this terminal (2026-09-24):

- Futures are listed to 2032.
- Live option vols are available to 2027. Settlement vols (`IVOL_LAST`) run to
  COF30 (Nov-29 deliveries); later months use the Samuelson fill, scaled x1.09.
- History runs from 2015-18.
- The Brent M/M+1 correlation is 0.995-0.999.

**Which contract.** ICE Brent contract X expires at the end of X-2, so a cargo
delivered in m prices on contract m+2: the prompt Brent during the delivery
month (a Nov-28 cargo prices on COF29). Contracts that average Brent over
earlier months (e.g. 3-0-1) are not modelled; averaging would lower the
effective vol. Change `contract_month_offset` if the indexation differs.

**Flexible legs.** A trade's `sell` and `buy` legs are
`{underlying, weight, constant}`. The underlying is any hub in the registry,
and the weight and constant are a scalar or `[M, M+1]`:

```yaml
sell: {underlying: Brent, weight: [0.166, 0.132], constant: 1.5}   # Dec-28/Jan-29
buy:  {underlying: TTF, weight: 1.0}
```

The margin is (sell weight x price + sell constant) - (buy weight x price +
buy constant). For Brent the weight is the oil slope (0.166 x USD/bbl =
USD/MMBtu). The older form `{hub, slope}` plus `strike_usd_mmbtu` still
works; giving both a `sell.constant` and a `strike_usd_mmbtu` is an error.
`config/trades_brent_2028_2031.yaml` holds the 47 Brent CSOs.

**Haircuts** extend automatically: `Brent_vol_haircut`, `Brent_corr_haircut`,
`TTF_Brent_corr_haircut` and `HH_Brent_corr_haircut`, with flags such as
`--brent-vol-haircut`. `trades_summary.csv` follows each trade's hubs:
`vol Brent_M`, `corr TTF_M/Brent_M`, `corr_sens_usd_mc TTF/Brent all cross
pairs`, and so on.

## Job 2: pricing

For each trade, the pricer:
1. reads the legs (`<HUB>_M`, `<HUB>_M1`) from `contracts.csv` and the matrix
   for that pair;
2. computes each leg's **horizon vol**;
3. applies the trade's overrides;
4. prices with MC (reference), Kirk on moment-matched baskets, and Bachelier;
5. computes risk (`risk_mc`): pathwise deltas, and vegas and correlation
   sensitivities by bumps with common random numbers.
   - Correlation risk covers each hub's M/M+1 pair.
   - It also covers `TTF/HH all cross pairs`: all four TTF/HH pairs bumped
     together. Bumping one cross pair alone would change how the two calendar
     spreads co-move.

A trade that cannot be priced (for example a month that was never calibrated)
is reported and skipped; the others still price.

**Horizon vol (`pricing.vol_horizon`).** An implied vol averages over
[today, option expiry]; the trade needs [today, decision date].
- **`samuelson`** (default) spreads the implied variance over time in
  proportion to the vol profile, and keeps the horizon's share:
  `σ_h² h = σ_imp² T_opt · W(T_fix−h, T_fix) / W(T_fix−T_opt, T_fix)`.
  M legs are left as they are; M+1 legs lose (or gain) the part of their life
  the trade never sees.
- **`implied`** uses the file vol as it is.

**Haircuts (all trades, per hub).** These are set under `pricing:` in
`market.yaml`; 1.0 means as calibrated. For a single run, the flags
`--ttf-vol-haircut 0.8`, `--hh-corr-haircut 1.01`, `--ttf-hh-corr-haircut 0.5` and so on override the config.

- `TTF_vol_haircut` / `HH_vol_haircut` multiply that hub's vols after the
  horizon adjustment. For example, 0.8 prices at 80% of the vol.
- `TTF_corr_haircut` / `HH_corr_haircut` multiply that hub's calibrated M/M+1
  correlation.
  - **Below 1 the option is worth more**, because the months decorrelate: 0.95
    takes 0.985 to 0.936, which is about 4x the spread variance.
  - To be conservative on an option you own, use a value above 1: 1.01 takes
    0.985 to 0.995.
  - A result beyond 1 is an error.
- `TTF_HH_corr_haircut` multiplies all four TTF/HH cross correlations
  (TTF_M/HH_M, TTF_M/HH_M1, TTF_M1/HH_M, TTF_M1/HH_M1), each keeping its own
  level; M/M+1 correlations are untouched.
  - Its effect is small. The option depends on how the TTF and HH calendar
    spreads co-move, and a proportional change to all four pairs largely
    cancels out of that.
  - Below 1 is worth slightly more for the sell-HH / buy-TTF margin.
- Any correlation haircut that leaves the matrix inconsistent (not PSD) is an
  error, rather than a silent repair of other correlations.
- A trade's `vol_scale` multiplies on top. Absolute per-trade `vol` / `corr`
  overrides are taken as given.
- The old single `vol_haircut` key is rejected with a message naming the new
  keys.
- The haircuts used appear in `trades.csv` (`<HUB>_vol_haircut`,
  `<HUB>_corr_haircut`, `TTF_HH_corr_haircut`), in each trade's notes, in the console header and in
  `run.json`.

**Overrides (per trade, in `trades.yaml`).**

| override | example |
|---|---|
| forward | `fwd: {TTF_M: 70}` |
| scale all horizon vols | `vol_scale: 1.1` |
| set a horizon vol | `vol: {TTF_M1: 0.80}` |
| set a correlation | `corr: {TTF_M/TTF_M1: 0.92}` |

**Audit notes.** Each applied override appears in the trade's notes. So does
any leg whose vol differs from the Bloomberg read (filled or hand-edited), and
any repair of a correlation matrix that isn't positive semi-definite.

**Outputs (`output/<timestamp>/`).** `trades.csv` has a units row under the header (USD/MMBtu, USD, probability 0-1, ...); load it with `pd.read_csv(path, skiprows=[1])`. Naming: anything ending in `_mc` is a Monte Carlo number; `_kirk` / `_bachelier` are the closed forms; unsuffixed values (`intrinsic`, `pv_margin_*`, `option_intrinsic`) use today's forwards.

| file | contents |
|---|---|
| `trades.csv` | **`option_value_mc`**: the right to switch from M to M+1 (and to cancel, if allowed), against committing to M today; always ≥ 0, split into `option_intrinsic` + `option_time_value_mc`, with `option_value_usd_mc` and `option_stderr_mc`. **`value_mc`**: the whole delivery, PV(X_M) + option, which can be negative. Also the inputs actually used — `vol <leg>` for TTF/HH M and M+1 (horizon-adjusted, after overrides) and `corr <leg>/<leg>` for all six pairs (after overrides and PSD repair) — `value_kirk` / `value_bachelier` checks, `prob_m1_mc`, `prob_cancel_mc` and `corr_sens_usd_mc` |
| `trades_summary.csv` | the headline subset of `trades.csv` (`SUMMARY_COLUMNS` in `price_job.py`): terms, vols and correlations used, option value MC/Kirk, time value, deal value, P(M+1) and the three correlation sensitivities; same units row |
| `legs.csv` | inputs actually used per leg, `delta_per_mmbtu_mc`, `qty_native_mc`, `lots_mc` (hedge of the whole delivery), `option_lots_mc` (hedge of the option alone), `vega_usd_per_pt_mc` |
| `portfolio.csv` | `lots_mc`, `option_lots_mc` and `vega_usd_per_pt_mc` summed per futures contract across all trades |
| `grids.csv` | extrinsic against TTF M/M+1 correlation and vol scale, per trade; `method` says `kirk` or `mc` (MC when the trade can cancel) |
| `run.json` | calibration folder used, as-of date, settings, notes, errors |

## First run (as of 2026-09-23)

| trade | intrinsic | MC | extrinsic | USD |
|---|---|---|---|---|
| NovDec26 | 0.980 | 1.358 ± 0.004 | 0.377 | 1.36m |
| NovDec26, TTF ρ = 0.92 | 0.980 | 1.927 | 0.946 | 3.41m |
| DecJan27 (3.4 TBtu) | 1.420 | 1.795 | 0.375 | 1.28m |
| MarApr27 with cancel | 4.317 | 6.801 | 2.484 | 8.94m |

## Modelling choices that move the number

1. **The TTF M/M+1 correlation.**
   - Pooled weekly estimate: 0.986–0.993 along the strip.
   - Past Nov/Dec pairs ranged from 0.914 (2022) to 0.997 (2024).
   - For Nov/Dec, each +0.01 is worth about −$0.48m.
   - `anchor_filter: same_pair` restricts the sample to past pairs of the same
     calendar months. That is thinner, but relevant across the Mar/Apr and
     Sep/Oct season boundaries, which the pooled number smooths over.
2. **The TTF M+1 vol over the horizon.**
   - With ρ≈0.99, the spread variance `σ₁²+σ₂²−2ρσ₁σ₂` is driven by the gap
     between σ₁ and σ₂.
   - TTF's realised profile is almost flat in the last two months (0.68 at
     0–30 days, 0.72 at 30–60), so `samuelson` keeps Dec near its implied
     (0.98 vs Nov 0.88).
   - The earlier `σ_M × realised ratio` gave 0.81. That gap alone moved the
     Nov/Dec extrinsic by about 0.1 $/MMBtu.
   - Treat this as a view to set: override `vol: {TTF_M1: …}` or edit
     `contracts.csv`.
3. **Far HH vols are placeholders**, filled from Apr‑27. Replace them in
   `contracts.csv` if a trade depends on them.

## Layout

```
calibrate.py / price.py              entry points (put BGN_LNG on sys.path)
config/market.yaml, trades.yaml
src/deliveryoption/
  market_data.py    Bloomberg: forwards, fixings, ATM vols, FX, vol fill
  calibration.py    anchor-window pair correlations, vol-by-maturity profile
  snapshot.py       the file contract between the two jobs
  vols.py           horizon vol from implied vol + profile
  deal.py           trades.yaml -> engine inputs, overrides, audit notes
  pricer.py         engine: MC, Kirk-basket, Bachelier, risk
  calibrate_job.py / price_job.py
data/calibration/<date>/             Job 1 output (kept: it is the audit trail)
output/<timestamp>/                  Job 2 output (git-ignored)
```
