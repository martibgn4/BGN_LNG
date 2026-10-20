# ttf_forecaster: project status (handoff, 2026-10-20)

Read this first in any new session, then `CLAUDE.md`, `docs/research_specification.md` (§8 criteria +
amendments A1–A6), `docs/point_in_time_policy.md`, and the latest entries of `docs/modelling_log.md`
(D-001 … D-052; D-052 is the renumbered derived-power entry, dated 2026-10-19).

## State in one paragraph
- **Pipeline:** point-in-time research platform. It is synthetic-tested, about 770+ tests pass
  (`..\..\env_local\Scripts\python.exe -m pytest -o addopts=""` from `ttf_forecaster/`).
- **Version control:** NOTHING is committed (branch `feature/7-5-2027/local_branch_continuous_work`).
- **Data:** licensed data lives only in the git-ignored `data/live/` store.
- **First real result:** H1 (EU storage → TTF fixed-contract returns, h = 5/20) FAILED. It is recorded
  as a negative result (D-037, `outputs/real_h1_report.md`).
- **Holdout:** origins from 2025-10-01 are locked and never evaluated.

## Data in the store (all pulled with explicit user authorisation)
| Family | Content | History | Notes |
|---|---|---|---|
| TTF curve | TZT monthly settles, vendor LAST_TRADEABLE_DT, px_volume/open_int | 2015+ | vendor expiry authoritative (D-035) |
| Hubs | JKM, HH, Brent, NBP; THE (ICE `LFT`); PEG/PSV/PVB/VTP (PNX, expired tickers `NCGX6M16`) | 2015+ | PNX curves have many near-zero repeated settles; needs a liquidity filter |
| Storage | CGIEEU ST/WG/IN/WI + 8 countries | 2011+ | grade B; WGV perimeter grows to 2016, so use % of WGV |
| Flows / production / demand | Gassco (GAANTOTL/GAANTOTI 2020-12+, NO→DE 2012+), Russia→CE/SE, IUK, NL production, TSO demand, BNEF gas-for-power BE/IT/ES/GB | varies | registry: `configs/sources/bloomberg_fundamentals.yaml` |
| Power (BNEF, 278 series) | generation by fuel, load, capacity, day-ahead auctions (DE = LPXBHRBS), gas day-ahead hubs | 2014+ (prices 2000+) | `configs/sources/bloomberg_bnef.yaml` |
| Power outages | BSRCH `ASSET_OUTAGE:POWER_PLANT` FR/DE/NORDPOOL | rolling 1y; plan snapshots from 2026-10-13 | FR nuclear availability = 63 GW (ASSUMED) − unavailable (D-044) |
| LNG | LNGG*/AHOY* aggregates (2017+); Spark (freight, netbacks, slots); per-country imports (15 BIKEYs, 13,019 cargoes, `configs/sources/lng_bsrch.yaml`); fleet snapshots `COMDTY:SHIP` | snapshots from 2026-10-20 | ES/FR capped at 2,000 rows; AHOY one day earlier than port arrival (D-051) |
| Weather | BQL `weather()` through the user's Excel (xlwings, `data/bql_excel.py`): ECMWF/GFS forecasts, issue dates 2025-01-01+, leads 1–14d; 8 countries + 4 regions; actuals 2011+; normals | forecasts only 2025+ | BQL API login NOT entitled (user requesting it). Actuals date bug fixed (D-045) |
| Gas/LNG outages | user's Bloomberg sheet in `data/raw/extracts/outages/`, snapshot loader (D-046) | from 2026-10-20 | user updates it manually |

## Known negatives and caveats
- Implied Norwegian outages from flow shortfalls FAILED validation (D-047); not used.
- H4 relies on flow surprises plus outage-sheet snapshots going forward.
- The BQL weather vintage history is under 2 heating seasons, so H2 will be preliminary.
- Clark–West over-rejects with ridge; HLN-DM is the only gate (D-020, D-032).
- Criterion 1 needs at least 4 folds; pass/fail must also hold on the executable label r(t+1, t+1+h)
  (amendments A1/A2).

## Next steps (in order)
1. **Commit the work** on a branch, once the user approves. Data stays ignored.
2. **Schedule** `scripts/run_daily_snapshot.py`: outages, BQL weather (needs Excel) and the LNG fleet.
   The user decides; the Task Scheduler command is in `docs/bloomberg_discovery.md` §8.
3. **Wire the new data into the features:** flows/balance, cross-commodity basis, LNG (imports,
   arrivals), power (generation, load, spreads, nuclear availability), BQL weather; plus the PNX
   liquidity filter and the Nordpool `gen_gas_mw` bad print (44 GW).
4. **Extend `pipelines/real_h1.py`** into a general real runner, then run H3 (LNG/cross-commodity),
   H5 (volatility/tails), H6 (regimes) and H2 (weather revisions, preliminary). Same harness, holdout
   locked.
5. **Housekeeping:** lockfile, code version in manifests, ruff/mypy.

## Open user inputs
- Bloomberg BQL API entitlement (requested by the user).
- Split BIKEYs for ES/FR imports (2,000-row cap).
- A Norwegian technical/nominated capacity field (would let realised outages be reconstructed).
- German load and national solar forecasts; TTF implied vol before 2025-10.
- Confirm: costs, plant efficiencies (gas 45% GCV, coal 38% NCV), FR nuclear capacity 63 GW.

## Working conventions
- **Models:** the orchestrator runs on Opus; subagents and side sessions run on Haiku (memory
  `agents-run-on-haiku`).
- **Linked sessions:** "Bloomberg pull and EU fundamentals discovery"
  (local_a7d08f17-2696-459e-a6b9-d5851a7ca0c4) owns the Bloomberg configs and pulls.
- **Live vendor calls:** only with explicit user authorisation, via a temporary authorised config copy.
  Never set `live_calls_authorised` in the committed configs.
