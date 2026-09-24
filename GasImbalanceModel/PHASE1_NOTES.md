# Phase 1 — perimeter, units, data spine

Status: **built and running against live data. The storage half of the gate passes;
the flow half does not, for documented data reasons.** Nothing was fudged to pass.

Bloomberg is the storage source (`CGIE*`, GIE's AGSI+ redistribution). No GIE key
is needed. `xbbg` + `blpapi` are installed in `.venv` and pull live data from the
Terminal.

Run it:

```bash
gasbalance status
```

```bash
gasbalance resolve-points
```

```bash
gasbalance fetch entsog --start 2025-01-13 --end 2025-01-19
```

```bash
gasbalance fetch bloomberg --start 2022-10-01 --end 2025-09-30
```

```bash
gasbalance reconcile-storage --start 2022-10-01 --end 2025-09-30
```

```bash
gasbalance reconcile --start 2025-01-13 --end 2025-01-19
```

---

## What works

- **Storage from Bloomberg `CGIE*`** — tickers discovered via Bloomberg own instrument
  search and verified against real values — none guessed.
- **Perimeter resolved from live ENTSOG**, not hand-typed. 2,253 point-directions
  classified into: 87 crossing, 50 LNG entry, 41 production (these three are the
  balance terms), 312 internal, 425 storage, 722 internal nodes, 616 outside.
- **Unit layer** with no numeric constant in any `.py` file, enforced by a test that
  parses the AST and rejects float literals outside `{0.0, 1.0, -1.0}`.
- **Vintaged DuckDB store.** A restatement appends at a new vintage; `read_as_of`
  returns what was actually published at a chosen moment. Backtests must use it.
- **Fetchers refuse rather than invent.** AGSI and Kpler raise without
  credentials; ENTSOG needs none; Bloomberg is live.
- 79 tests passing.

## Gate result on a real week (13–19 Jan 2025)

```
days covered              7
internal net, worst day   15.9968 mcm/d
internal breaches         7
one-sided internal points 231 point-days (1454 mcm/d gross, excluded from net)
points expected to report 3087
unexplained missing       1225
expected missing          420
VERDICT                   FAIL
```

### Why it fails, and what each number means

**1. Coverage: 1,225 missing point-days.** ENTSOG returned data for 125 of the 179
perimeter point keys requested. The other 54 have `hasData = true` in the registry
but published nothing for that week. This is a property of ENTSOG, not of the model.
It needs a decision from you: either accept a documented coverage floor, or backfill
those points from national TSOs (National Gas, GRTgaz, Snam, Gasunie).

**2. Internal netting: worst day 16.0 mcm/d, and it is systematically negative.**

The residual is concentrated in four points, and the top two are the exact case
SPEC §7 names:

| Point | Label | 7-day net (mcm) |
|---|---|---|
| ITP-00005 | Bacton (IUK) | −43.7 |
| ITP-00207 | Bacton (BBL) | −19.3 |
| ITP-00227 | RC Lindau | −19.1 |
| ITP-00019 | Überackern (AT/DE) | −7.6 |

All four are cross-border pipes reported by two TSOs. A persistent one-signed bias
rather than random noise points at three candidates, in order of likelihood:

- **Gas-day definition.** The UK gas day runs 05:00–05:00 UTC; the CET gas day runs
  06:00–06:00 local. A one-hour offset on a large flow produces exactly this kind of
  small, persistent, one-signed residual. This is the leading explanation for the two
  Bacton points.
- **Transit fuel gas.** Compressors on IUK and BBL burn gas in transit, so exit on
  one side is genuinely larger than entry on the other. This is real physics and
  should end up as an explicit term, not as residual.
- **Metering differences** between the two operators.

None of these is a coding bug, but the model must not absorb them silently, so they
are reported per-point rather than netted away.

**Correction made during the build:** the first run reported a 72.6 mcm/d worst-day
netting failure. That was misleading. Most of it was internal points where only
*one* of the two TSOs published, so there was nothing to cancel against. Orphan
sides are now split out and excluded from the net, which is why the figure is 16.0.
The orphan volume is reported separately rather than hidden.

---

## Storage gate: PASSES

Three gas years, 1 Oct 2022 – 30 Sep 2025, 1,095 days, 8 countries, 35,072
observations with no missing days. The test is the SPEC §5 identity
`S_t = S_(t-1) + injections_t - withdrawals_t`, rebuilding the inventory path from
the reported flows and comparing it to the reported stock. Inventory and flows are
three independent published series, so this is a real integrity test, not a tautology.

```
aggregate mean error    +0.22 mcm/d
aggregate abs mean       6.48 mcm/d
worst country bias       0.18 mcm/d
alert threshold         20.00 mcm/d
```

| country | mean error mcm/d | abs mean | worst day |
|---|---|---|---|
| DE | +0.08 | 4.16 | 549.83 |
| UK | +0.18 | 1.07 | 229.44 |
| NL | −0.07 | 0.59 | 163.07 |
| AT | −0.12 | 0.54 | 113.75 |
| FR | +0.14 | 0.34 | 40.79 |
| CZ | −0.00 | 0.27 | 85.25 |
| DK | +0.02 | 0.20 | 47.40 |
| BE | +0.00 | 0.01 | 4.62 |

**Every country is two orders of magnitude inside the 20 mcm/d alert on bias.**
Median absolute error is 0.006 mcm/d; the 90th percentile is 0.55 mcm/d.

**The worst-day numbers are a gas-day alignment artefact, not lost gas.** Only 57 of
8,760 country-days (0.65%) breach 20 mcm/d, and the large ones come in adjacent
opposite-signed pairs — a flow booked against the neighbouring gas day:

| date | country | error mcm/d |
|---|---|---|
| 2024-03-31 | DE | −549.83 |
| 2024-04-01 | DE | +315.99 |
| 2024-06-16 | DE | −210.44 |
| 2024-06-17 | DE | +210.61 |
| 2022-12-31 | UK | −221.72 |
| 2023-01-02 | UK | +229.44 |

2024-03-31 is the DST spring-forward date — a 23-hour gas day. About 30% of the
breach days coincide with a revision to reported working gas volume (against 2.5%
of all days), so facility reclassification explains a further slice. Bias stays at
zero throughout, which is why bias rather than worst-day is the honest measure.

This is why the model must keep the residual visible: absorbing a 550 mcm/d
one-day artefact into a balance term would corrupt a whole month.

---

## Decisions taken (you signed these off)

- **Unit conflict** — canonical GCV stays at **10.55 kWh/m³**, so 1 bcm = 10.55 TWh
  and 1 mcm = 10.55 GWh. The `11 GWh/d` figure in SPEC §5 is not used anywhere. A
  test asserts the model never runs two calorific values at once.
- **Netting tolerance** — left at 0.5 mcm/d. It is breached daily by Bacton; the
  breaches are reported rather than suppressed.
- **ENTSOG coverage floor** — accepted as-is for Phase 1.
- **AGSI key** — not needed. Storage comes from Bloomberg. The AGSI fetcher stays in
  the tree as an alternative and still refuses cleanly without a key.

## Known gaps in Phase 1 as built

- **`isPipeInPipe` is captured but not used.** 37 internal points carry the flag.
  Pipe-in-pipe points can double-report the same physical pipe under two operators,
  which would double-count. Needs handling before the balance is trusted.
- **Country is derived from the operator key prefix**, which mislabels a few operators
  (TAP reports under `AL-`). The authoritative `tso_country` from the point registry
  should be joined instead.
- **Only physical flow is fetched.** Nominations and interruptions are not, and will
  be needed for the nomination-flexibility band in Phase 3.
- **`xbbg` must stay pinned to 0.12.2.** 1.x returns polars frames, so `.empty`
  raises `AttributeError` instead of failing a check — a silent break across this
  repo. Pinned in `pyproject.toml` with the reason.
- **Your PyCharm venv is broken**, which is why Bloomberg looked unavailable at
  first: `env_test` was built on Python 3.13.5 and that base install is gone (the
  machine is on 3.14 now). It is unrelated to this project, but the rest of BGN_LNG
  will not run until it is rebuilt.
- **BSRCH/BIKEY series are still empty.** `bloomberg_series.yaml` has a `series: {}`
  block for saved searches. Storage did not need one, but any BSRCH-based feed
  (e.g. cargo or flow panels) will — those keys can only come from your Terminal.
