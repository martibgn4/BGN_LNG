# Live-data requirements (STEP 7)

Version 0.1, 2026-10-16. Compiled by the orchestrator from the data-requirements tables of all
agents. The machine-readable source of truth is `configs/sources/*.yaml`; run
`python -m ttf_forecaster.cli identifiers` for current counts (at writing: **105 known,
25 requires_discovery, 25 requires_user_input**).

**Status legend**
- **known**: in production use or verified elsewhere in this repo; provenance cited in the config.
  Still never requested live without the user's explicit instruction.
- **discovery**: findable with vendor discovery tools (e.g. Bloomberg `//blp/instruments`), or a
  one-off authorised reference call.
- **user**: only the user can supply it (entitlement, saved search, licence decision, desk convention).

**Lag** means publication lag, the delay before data is visible to the model. "Next session" means
the next European trading day at 19:00 Amsterdam. Time zone is Europe/Amsterdam unless stated.

## 1. Market data (Bloomberg) — owner: market-data agent, `configs/sources/bloomberg.yaml`

### 1.1 Futures curves

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| TTF M/Q/S/Y settle (TZT/QZT/QQT/QTT, `px_settle`) | **targets**, price and curve features | daily | EUR/MWh | 2015+ | trade date 19:00 | none expected | A | curve `TTF` | known (`Data/lng_report/config.py`) |
| TTF USD twin (TMR/TQR/TSR/TYR) | cross-check only | daily | USD/MMBtu | 2015+ | 19:00 | none | A | curve `TTF_USD` | known |
| NBP (FN/QR/SA/YAA) | inter-hub basis | daily | GBp/therm [CONFIRM unit] | 2015+ | 19:00 | none | A | curve `NBP` | known |
| THE/PEG/PSV/PVB/VTP (NCG/PNG/PSR/PXB/CEG + period suffix) | inter-hub basis | daily | EUR/MWh | 2015+ | 19:00 | none | A | curves `THE`… | known (suffix rule from `lng_report`) |
| JKM (JKL) | JKM–TTF arbitrage | daily | USD/MMBtu | 2017+ (**depth unverified**) | next session | none | A | curve `JKM` | known; history depth: discovery |
| Henry Hub (NG) | US netback | daily | USD/MMBtu | 2015+ | next session | none | A | curve `HH` | known |
| Brent (CO) | oil link | daily | USD/bbl | 2015+ | next session | none | A | curve `Brent` | known |

### 1.2 Single series

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| EURUSD (`EUR Curncy`) | conversion | daily | USD/EUR | 2015+ | next session | none | A | `fx.eurusd.spot_usd_per_eur` | known |
| EURGBP (`EURGBP BGN Curncy`) | NBP conversion | daily | GBP/EUR | 2015+ | next session | none | A | `fx.eurgbp.spot_gbp_per_eur` | known (`lng_report/violin.py`) |
| **TTF day-ahead** | T1, T4, T5b targets | daily | EUR/MWh | 2015+ | assessment session 19:00 | rare | A | `price.ttf.da_eur_mwh` | **user**: which assessment (exchange index, broker, PRA) and ticker |
| TTF balance-of-month | T1 benchmark | daily | EUR/MWh | 2015+ | 19:00 | none | A | `price.ttf.bom_eur_mwh` | discovery |
| API2 coal (candidate root `XA`) | switching, clean dark spread | daily | USD/t | 2015+ | next session | none | A | `price.coal_api2.m1_usd_t` | **user**: confirm (candidate in GasImbalanceModel) |
| EUA (candidate `MO1 Comdty`) | switching, CSS/CDS | daily | EUR/t | 2015+ | next session | none | A | `price.eua.front_eur_t` | **user**: confirm, plus roll convention |
| DE / NL power base M1 | CSS/CDS | daily | EUR/MWh | 2015+ | 19:00 | none | A | `price.power_{de,nl}.base_m1_eur_mwh` | **user** |
| TTF ATM implied vol (FJS options) | T8/T9 features | daily | % | 2018+ | 19:00 | none | A | `vol.ttf.m1_atm_implied_pct` | discovery (field) |
| TTF bid/ask, volume, OI | costs, liquidity | daily | EUR/MWh, lots | 2015+ | 19:00 | — | A | curve fields | **user**: entitlement |

### 1.3 Contract metadata

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| TTF last trading day (`LAST_TRADEABLE_DT`) | expiry cross-check | static | date | all contracts | n/a | none | n/a | metadata | known (adhoc India script) |

## 2. European gas fundamentals — owner: gas-fundamentals agent

### 2.1 Storage and LNG send-out (Bloomberg)

Source configs: `bloomberg_fundamentals.yaml`, `fundamentals.yaml`.

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| EU storage inventory / WGV / injection / withdrawal (`CGIEEU{ST,WG,IN,WI}`) | storage features, H1 | daily | TWh, GWh/d | 2011+ | D+2 07:00 | **revised, no vintages on Bloomberg** | **B** (+1/+3 d sensitivity) | `storage.eu.*` | known (GasImbalanceModel, verified 2025-01) |
| Country storage ×8 (`CGIE<CC>*`) | perimeter checks | daily | TWh, GWh/d | 2017+ | D+2 | revised | B | `storage.<cc>.*` | known |
| `CGIEEUFU` (% full) | convenience | daily | % | — | — | — | B | `storage.eu.full_pct` | discovery (derived instead) |
| Injection / withdrawal capacity | ratchet, bounds | daily | GWh/d | 2017+ | D+2 | revised | B | `storage.eu.*_capacity_gwhd` | discovery |
| EU LNG send-out / terminal inventory (`LNGATLSO`, `LNGATLIN`) | supply | daily | GWh/d, 1000 m³ (liquid or gas? [CONFIRM]) | 2017+ | D+2 | revised | B | `lng.eu.sendout_gwhd`, `lng.eu.terminal_inventory_1000m3` | known (identified, not pulled) |
| LNG send-out capacity | regas utilisation | daily | GWh/d | 2017+ | D+2 | rare | B | `lng.eu.sendout_capacity_gwhd` | discovery |
| Country TSO demand ×15 (FR, BE, NL, AT, GB) | demand proxies | daily | GWh/d or mcm/d | 2016+ | D+2 assumed, **unevidenced** | unknown | B | `demand.<cc>.*` | known (GasImbalanceModel `demand_sources.yaml`) |

### 2.2 Flows, outages and balance components (other sources)

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| Norway physical flows, nominations, technical capacity | supply, H4 | daily | GWh/d | 2015+ | D+1 | restated | B (A if snapshotted) | `flows.no.*` | discovery (Gassco/ENTSOG; no connector, no API key) |
| Norway unavailable capacity (planned / unplanned) | **H4 outages** | event → daily | GWh/d | 2015+ with original timestamps | message time | each update is a vintage | **A only with a UMM archive** | `flows.no.outage_*` | **user** |
| Pipeline imports (N. Africa, TAP, Russia, GB), exports, domestic production | balance | daily | GWh/d | 2015+ | D+2 assumed | restated | B | `flows.eu.*`, `production.eu.*` | discovery / user |
| EU demand: LDZ, industrial, power burn, total | balance, days of cover | daily | GWh/d | 2016+ | unknown | restated | B | `demand.eu.*` | **user** (BNEF LDZ monitor is licensed) |
| Interconnector / LNG terminal outages | constraints | event | GWh/d | 2018+ | message time | vintaged | A if archived | `constraints.*`, `outages.*` | user |
| EU storage policy target | policy-gap feature | dated config | % | — | — | — | static | `policy_storage_target` | **user**: confirm current regulation text |

## 3. LNG and shipping — owner: LNG-flows agent

### 3.1 Bloomberg aggregates

Source config: `bloomberg_lng.yaml`. Provenance for all rows: desk notes (Bloomberg discovery), verified 2026-09-17.

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| LNG afloat ≥20/30 days (`LNGG20DT/DC`, `LNGG30DT/DC`) | on-water supply | daily | t, vessels | 2017+ | D+1 23:00 | unknown | B | `lng.global.afloat_*` | known |
| Afloat by origin (`LNGG20D{U,Q,A,N,R,O}`) | origin mix | daily | vessels | 2017+ | D+1 | unknown | B | `lng.<origin>.afloat_20d_vessels_count` | known |
| Chokepoint transits (`LNGGTR{PA,SU,CA,NS}`) | routing (rolling sums only) | daily | vessels | 2018-12+ | D+2 | unknown | B | `lng.<route>.laden_transits_count` | known |
| Trade flows (`AHOY*`, incl. `AHOYIMEU`/`AHOYIMNA`) | Americas→NWE vs →Asia | daily | t/d | 2017+ | D+1 | unknown | B | `lng.amer_to_*`, `lng.*.exports_*` | known |
| Baltic LNG futures (`IKA1/IKD1/IKI1`) | freight (levels only; they roll) | daily | USD/d | 2024-02+ | next session | none | A | `freight.blng*` | known |

### 3.2 Spark

Source config: `spark.yaml`. Provenance: NWEBasisModel, verified 2026-09-04.

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| Spark30S / 25S, FFA monthly | freight | daily | USD/d | 2019-07+ | 19:00 London [CONFIRM] | none | A | `freight.*` | known |
| Spark NWE DES/basis monthly | NWE basis | daily | USD/MMBtu | 2022-11+ | 19:00 London | none | A | `gas.nwe.*` | known |
| Netbacks (Sabine, Corpus via Cape of Good Hope) | freight-adjusted JKM–TTF arb | daily | USD/MMBtu | 2021-09+ | 19:00 London | none | A | `route.*` | known; **month-index meaning: discovery** |
| Terminal slots | regas congestion | daily | slots | 2023-11+ | 19:00 London | none | A | `lng.slots.*` | endpoint known; **terminal codes: discovery** |

### 3.3 Vessel-level data

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| **Vessel snapshots** (laden/ballast, declared destination and ETA, volume) | arrival forecasts, diversions | ≥ daily | m³ LNG | ≥ 3 y | minutes–hours | constant | **C unless snapshot-stored** | `VESSEL_SNAPSHOT_SCHEMA` | **user**: no vintaged source known (Spark/Kpler/Vortexa?) |
| Realised arrivals | forecast-error analysis | daily | cargoes, m³ | as snapshots | ~2 d | re-matching | B | `lng.arrivals_realised.*` | user |

## 4. Weather and power — owner: weather-power agent

### 4.1 Weather

Source config: `weather.yaml`.

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| Point 2 m temperature, forecast runs (Open-Meteo historical-forecast, `_previous_day1..7`) | HDD/CDD revisions, H2 | hourly → daily | °C | **mid-2021+**, leads ≤ 7 d | end of issue day + 8 h (conservative) | runs immutable | A | `weather.<pt>.t2m_mean_degc` | known endpoint; **licence: user** (free tier non-commercial) |
| ERA5 observed temperature | normals, observed anomaly | hourly → daily | °C | 30 y | D+5 | ERA5T → final | B | `weather.<pt>.t2m_mean_obs_degc` | known |
| Ensemble members | ensemble spread (T8) | daily | °C | live accumulation only | issue + lag | none | A from snapshots | `weather.<pt>.t2m_mean_ens_degc` | discovery |
| **Region / LDZ weights** | population or gas-demand weighting | static | weight | — | — | — | — | `weather.yaml regions` | **user**: from BNEF LDZ workbook, in git-ignored `configs/local_weather.yaml` |

### 4.2 Power

Source config: `power.yaml`.

| Series | Purpose | Freq | Unit | History | Lag | Revisions | PIT | Internal id | Status |
|---|---|---|---|---|---|---|---|---|---|
| DE wind/solar/load forecasts and actuals, capacity | renewables, residual load | hourly | GW | 5 y | D−1 18:00 (forecast) | overwritten | A only if snapshotted | `power.de.*` | discovery (ENTSO-E token or vendor) |
| FR nuclear availability (REMIT) and actual | nuclear anomaly | message/daily | GW | 5 y | message time | new messages | A with archive | `power.fr.*` | user |
| Hydro reservoir | hydro proxy | weekly | GWh | 6 y | ~7 d | revised | B | `power.alp_nord.*` | user |
| CCGT/coal efficiencies (NCV/GCV basis) | CSS/CDS, switching | static | fraction | — | — | — | — | `power.yaml economics` | **user**: desk convention |

## 5. Text / events — owner: llm-news agent

REMIT/UMM platforms, Gassco UMMs, terminal notices, broker notes and newswires. All are **user**:
- which sources you are entitled to;
- compliance approval before any proprietary text is sent to an external model;
- who maintains the asset-capacity table.

Text features are excluded from the clean backtest until live-archived (D-010).

## 6. Costs and execution — owner: backtesting agent

| Item | Purpose | Status |
|---|---|---|
| TTF bid/offer history (or broker spread by regime) | replace the 0.025 EUR/MWh half-spread assumption | user |
| Exchange, clearing and broker fees | fee per side | user |
| Margin history | funding cost (directional speculation only) | user |

## 7. Exchange-rule confirmations (no identifier; desk knowledge)

1. ICE Endex TTF monthly expiry rule (assumed: 2 business days before delivery) and holiday calendar.
2. Whether `px_settle` on TZT is the official settlement on every date since 2015.
3. Whether quarter, season and year contracts cascade, and their expiry.
4. T4 index methodology: are weekend gas days included in the monthly DA average?

## 8. The smallest set of inputs that unblocks the first real-data test (H1, storage → T3)

H1 can run on **known** identifiers only. It needs:

1. **Authorisation** for one live Bloomberg pull of:
   - TZT monthly settles 2015+ and their `LAST_TRADEABLE_DT`, to validate the expiry rule;
   - `CGIEEU{ST,WG,IN,WI}` 2011+;
   - `EUR Curncy`.
2. **Confirmation** of the ICE Endex expiry rule and holiday list, or acceptance that the
   vendor-metadata cross-check stands in for the expiry rule.

Everything else (day-ahead, Norwegian flows and UMMs, vessel snapshots, weather weights and
licence, power) extends the study to further hypotheses. None of it blocks H1.

## 9. Update 2026-10-19: user tickers and Bloomberg discovery (D-034)

Identifier counts are now known 134, discovery 24, user 16 (`cli identifiers`).

**Resolved:**

| Series | Id |
|---|---|
| API2 | `XA1 Comdty` |
| EUA | `MO1 Comdty` (Dec benchmark) |
| TTF day-ahead | `EGSITFAD Index`, EEX EGSI (cross-check `EEDATTF1 Index`) |
| TTF balance-of-month | `TTFGBOM Index` |
| TTF implied vol | `FJS1 Comdty` `hist_call_imp_vol` |
| German power | M1 `FPA2 Comdty`, day-ahead `ELGBDAHD Index` |
| Dutch power | week `DFBAW Comdty`, M1 `ELHB1MON Index` |
| TZT volume / open interest | `px_volume` / `open_int` |
| Power fundamentals | 15 series in `configs/sources/bloomberg_power.yaml` |

TZT bid/ask is NOT AVAILABLE.

**Not resolvable on this Terminal:** `DEDAILY` (no data).

**Still needed, all optional for H1:**
- an ENTSO-E security token, for the German load forecast and full-coverage solar;
- access to the EDF REMIT archive, for French nuclear availability;
- implementation of the `power.yaml derived:` step.
