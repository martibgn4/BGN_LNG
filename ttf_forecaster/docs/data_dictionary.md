# Data dictionary

**Status: DRAFT v0.1, 2026-10-16** (data-architecture-agent). Other agents' reports extend the
registry in §5; the orchestrator merges them. Vendor identifiers never appear here. They live in
`configs/sources/*.yaml` with a `status`. This file lists **internal** names only.

Authority: `src/ttf_forecaster/data/base.py` (schemas), `docs/point_in_time_policy.md` (rules).
If this file disagrees with `base.py`, `base.py` wins and this file is the one that's wrong.

---

## 1. Canonical schemas

All timestamps are **tz-aware UTC** (`datetime64[ns, UTC]`). Naive or non-UTC timestamps are rejected
by `validate_frame`, not converted. Local-time semantics are carried by `source_timezone` and by
documented conventions (e.g. `trade_date` = local midnight). Extra columns are allowed (no schema is
`strict`).

### 1.1 Provenance block (shared by observation, forecast, curve)

| Column | Kind | Null | Meaning |
|---|---|---|---|
| `publication_time` | ts_utc | no | When the value became knowable to the market or to us. For a revision, when *that* vintage was published. Drives the as-of rule. |
| `ingestion_time` | ts_utc | no | When our system received the row. `pipelines.ingest` stamps it. Enforced only with `strict_ingestion=True` (live). |
| `revision` | int | no | 0 = first release; 1, 2… are later vintages of the same fact. Must be published strictly after its predecessor (`validation.check_revision_monotonicity`). Gaps are allowed. |
| `unit` | str | no | Unit of `value` (e.g. `TWh`, `GWh/d`, `EUR/MWh`). Exactly one unit per series (hard check). |
| `source` | str | no | Vendor/dataset label (e.g. `bloomberg`, `spark`, `gie_agsi`, `SYNTHETIC`). Not an identifier. |
| `quality_flag` | str | no | `ok`, `estimated`, `missing_input`, `outlier_suspect`, `synthetic` (`QualityFlag`). |
| `entitlement` | str | no | `public`, `licensed_bloomberg`, `licensed_spark`, `licensed_other`, `synthetic`. |
| `pit_grade` | str | no | `A` true vintages · `B` lag-approximated · `C` hindsight (excluded from clean runs). |
| `publication_time_estimated` | bool | no | True when `publication_time` came from a `PublicationLag` rule and not from the source. |
| `source_timezone` | str | no | IANA zone the source used before conversion to UTC. |

### 1.2 `OBSERVATION_SCHEMA` (kind `observation`)

| Column | Kind | Null | Meaning |
|---|---|---|---|
| `series_id` | str | no | Internal id (§2). |
| `event_time` | ts_utc | no | When the fact applies. For daily series this is local midnight (Europe/Amsterdam) of the gas day or date, stored in UTC. **Proposed convention, see §6 Q1.** |
| `value` | float | yes | Numeric value in `unit`. NaN is allowed for "published as missing", never as a placeholder for "not yet published". |
| `geography` | str | no | ISO-3166 alpha-2 code (`NL`, `DE`), hub (`TTF`, `NBP`) or region (`EU`, `NWE`), upper case. |
| + provenance | | | §1.1 |

Dedupe key in the store: `(series_id, event_time, revision, publication_time)`. Fact key:
`(series_id, event_time)`.

### 1.3 `FORECAST_SCHEMA` (kind `forecast`)

| Column | Kind | Null | Meaning |
|---|---|---|---|
| `series_id` | str | no | Internal id. |
| `issue_time` | ts_utc | no | Nominal run/forecast creation time (model init). |
| `valid_time` | ts_utc | no | Time the forecast is for. |
| `lead_hours` | float | no | `valid_time − issue_time` in hours. |
| `model` | str | no | Model/provider label (e.g. an NWP model name or `SYNTH_ENS`). |
| `member` | str | no | Ensemble member id or statistic (`mean`, `spread`, `p10`…). |
| `value` | float | yes | Forecast value in `unit`. |
| `geography` | str | no | As in 1.2. |
| + provenance | | | `publication_time` = when the run became available, ≥ `issue_time` (enforced). |

Dedupe key: `(series_id, issue_time, valid_time, model, member, revision, publication_time)`.
Resolver key: `(series_id, valid_time, model, member)`. The latest visible `issue_time` wins.

### 1.4 `CURVE_SCHEMA` (kind `curve`)

| Column | Kind | Null | Meaning |
|---|---|---|---|
| `hub` | str | no | `TTF`, `NBP`, `THE`… |
| `contract_id` | str | no | `base.contract_id()` format: `TTF.M.2026-11`, `TTF.Q.2027-Q1`, `TTF.S.2027-SUM`, `TTF.Y.2027`. |
| `tenor_type` | str | no | One of `DA, WE, BOM, M, Q, S, Y`. |
| `delivery_start` | ts_utc | no | Local midnight at the start of delivery. |
| `delivery_end` | ts_utc | no | **Exclusive** end of delivery (local midnight). |
| `trade_date` | ts_utc | no | Local midnight of the trading date. |
| `field` | str | no | `settle`, `bid`, `ask`, `volume`, `open_interest`. |
| `value` | float | yes | Price in `unit` (EUR/MWh for TTF) or quantity. |
| `currency` | str | no | ISO-4217 (`EUR`). |
| + provenance | | | Settlement `publication_time` = 19:00 Europe/Amsterdam on the trade date (target_definitions §0). |

Dedupe key: `(hub, contract_id, field, trade_date, revision, publication_time)`. Store partition: `hub`.
Unit consistency is checked per `(hub, field)`.

### 1.5 `CONTRACT_METADATA_SCHEMA` (kind `contract_metadata`, not vintaged)

| Column | Kind | Meaning |
|---|---|---|
| `contract_id`, `hub`, `tenor_type` | str | As in 1.4. |
| `delivery_start`, `delivery_end` | ts_utc | Delivery window (end exclusive). |
| `last_trade_date` | ts_utc | Local midnight of the last trading day. |
| `last_trade_date_source` | str | `rule` (computed from config) or `vendor`. |
| `currency`, `unit` | str | |

Dedupe key: `(contract_id, last_trade_date_source)`. If a rule-based value is later contradicted by
a vendor value, that is a separate row. `contracts/` treats any mismatch as a hard error.

### 1.6 `TARGET_SCHEMA` and `PREDICTION_SCHEMA`

| Schema | Column | Meaning |
|---|---|---|
| target | `origin_time` | Forecast origin (UTC), after the settlement publication of trading date *t*. |
| target | `target_id`, `horizon` | E.g. `m1_logret_5`, 5 (trading days). |
| target | `value` | Label. NaN with `quality_flag = missing_input` when an input is missing. |
| target | `label_known_time` | When the label becomes known. Used for purging. |
| target | `contract_id` | Held contract (nullable for DA targets). |
| target | `construction` | `listed` / `strip` / … (target_definitions §1). |
| prediction | `origin_time, target_id, horizon` | Join keys to the target. |
| prediction | `model_id`, `fold_id` | Model and outer fold. |
| prediction | `prediction` | Point forecast. Optional `q<level>` and `prob` columns. |
| prediction | `train_end` | Last origin whose label was used in training. |
| prediction | `label_known_time` | Copied from the target. |
| prediction | `feature_hash` | Hash of the feature matrix used. |
| prediction | `is_synthetic` | True for anything derived from synthetic data. |

Benchmarks are stored with `PREDICTION_SCHEMA` (same keys as predictions; `model_id` names the
benchmark).

### 1.7 Feature matrices

Index is `origin_time` (tz-aware UTC). Columns are `<family>__<feature>`. `attrs["pit_grade"]` maps each
column to its worst input grade. Values not computable at the origin are NaN, never 0. The store keeps
`attrs` in the run manifest and restores them on read.

---

## 2. `series_id` naming convention

    <domain>.<geography>.<measure>_<unit>

* All lower case, `[a-z0-9_]`, exactly three dot-separated segments.
* `domain` is the family of the fact: `price`, `storage`, `flows`, `lng`, `freight`, `weather`,
  `power`, `demand`, `production`, `fx`, `vol`, `outages`, `constraints`.
* `geography` is a lower-case ISO-2 code (`nl`, `de`), hub (`ttf`), region (`eu`, `nwe`, `nasia`), route
  (`amer_to_nwe`) or `point`/`global`.
* `measure_unit`: the measure, then a unit token. Underscores inside the unit are optional and `/`
  may be written `_`, `per` or dropped: `gwhd`, `gwh_d`, `usd_per_eur`, `eur_mwh`, `pct`, `degc`,
  `count`.
* A vendor ticker is never a `series_id`.
* `validation.check_series_id_convention` checks the pattern and the unit suffix against the `unit`
  column. The check is **advisory** (a warning, not a block).

Known deviations: `weather.eu.hdd` (synthetic; no unit suffix; should become `weather.eu.hdd_degc_day`).
§6 Q3 lists cross-config inconsistencies.

---

## 3. Storage layers (`data/storage.py`, root = `settings.Paths`, `TTF_DATA_ROOT` override)

| Layer | Path | Content | Mutability |
|---|---|---|---|
| raw | `raw/<source>/ingest_date=YYYY-MM-DD/<name>-<HHMMSSffffff>-<sha12>.<ext>` + `.meta.json` | payload as received: `.bin` bytes, `.txt` text, `.json` mapping/list (sorted keys), `.parquet` frame. Sidecar holds source, name, ingestion time, sha256, size and call metadata | write-once; an identical payload on the same day is not rewritten |
| normalised | `normalised/<kind>/<partition_col>=<value>/part-<sha16>.parquet` | canonical rows; partition = `series_id` (observation, forecast), `hub` (curve, contract_metadata) | **append-only**: new vintages become new rows. Rows already stored (dedupe key) are skipped. Same key with different values raises `VintageConflictError` |
| features | `features/<run_id>/<name>.parquet` + `manifest.json` | feature matrices | immutable per run |
| predictions | `predictions/<run_id>/…` | `PREDICTION_SCHEMA` frames | immutable per run |
| benchmarks | `benchmarks/<run_id>/…` | `PREDICTION_SCHEMA` frames | immutable per run |

Manifest fields: `layer, run_id, created_time, config_hash (sha256 of sorted-key JSON), is_synthetic`,
and per artefact `n_rows, n_columns, columns, content_sha256` (format-independent frame hash),
`file_sha256, schema, attrs`. On read the file hash is checked and a mismatch raises. A frame with any
SYNTHETIC row cannot be written with `is_synthetic=False`.

Path safety: every segment must match `[A-Za-z0-9][A-Za-z0-9._=-]*` and contain no `..`. The resolved
path must also sit under the data root.

**Reading is not resolving.** `read_normalised(..., publication_cutoff=t)` is a pre-filter. Feature
code must still call `resolve_as_of` / `resolve_forecasts_as_of`.

**Raw vs vendor bytes.** The adapter Protocols return canonical frames, so `pipelines.ingest` can
only store what the adapter returned. To keep the vendor's own bytes, an adapter has to call
`store.write_raw` itself.

**Backend swap.** Callers depend on the `Store` Protocol. A DuckDB or database backend needs the
same six methods, the same dedupe and conflict semantics, and must pass `tests/unit/data_arch/test_storage.py`
when parametrised over the backend.

---

## 4. Data-quality checks (`data/validation.py`)

| Check | Hard by default | Notes |
|---|---|---|
| conflicting duplicate dedupe keys | yes | exact duplicates are only a warning |
| revision non-monotone (rev n published ≤ rev n−1) | yes | gaps allowed |
| unit inconsistency / unit ≠ expected | yes | per series (curve: per hub+field) |
| staleness vs expected frequency | no | only rows published by `as_of` count. `grace` should include the publication lag |
| robust-z outliers | no | trailing median/MAD (window 60, min 20, \|z\| > 6 on first differences). Flag, never drop |
| calendar coverage gaps | no | compares **local** dates (Europe/Amsterdam) |
| series-id convention | no | advisory |

Thresholds are in `QualityParams` (`from_config`). The proposed YAML block is in §6 Q4.

---

## 5. Series registry (internal ids)

This is a draft. "Defined in" points to the config that holds the vendor mapping and its `status`.
This file does not restate identifiers. **Proposed** rows are implied by the research specification
and have no config entry yet.

| Family | series_id (or pattern) | Kind | Unit | Owner agent | Defined in / status |
|---|---|---|---|---|---|
| price | `TTF.M.*`, `TTF.Q.*`, `TTF.S.*`, `TTF.Y.*` contracts, field `settle` | curve | EUR/MWh | market-data | `configs/sources/bloomberg.yaml` / `instruments.yaml` |
| price | `price.ttf.da_eur_mwh` | observation | EUR/MWh | market-data | bloomberg.yaml |
| price | `price.ttf.bom_eur_mwh` | observation | EUR/MWh | market-data | bloomberg.yaml |
| curve | derived from the `TTF.*` curve: M1–M2, slope, seasonal spreads | feature | EUR/MWh | feature-engineering | not stored |
| cross_commodity | `fx.eurusd.spot_usd_per_eur` | observation | USD/EUR | market-data | bloomberg.yaml **and** power.yaml (duplicate, §6 Q3) |
| cross_commodity | `price.coal_api2.m1_usd_t` / `coal.api2.m1_usd_t` | observation | USD/t | market-data / weather-power | bloomberg.yaml / power.yaml (two ids, §6 Q3) |
| cross_commodity | `price.eua.front_eur_t` / `carbon.eua.dec_eur_t` | observation | EUR/t | market-data / weather-power | bloomberg.yaml / power.yaml (two ids) |
| cross_commodity | `price.jkm.m1_usd_mmbtu` | observation | USD/MMBtu | market-data | **proposed** |
| cross_commodity | `price.hh.m1_usd_mmbtu` | observation | USD/MMBtu | market-data | **proposed** |
| cross_commodity | `price.brent.m1_usd_bbl` | observation | USD/bbl | market-data | **proposed** |
| cross_commodity | `price.nbp.m1_gbp_therm`, `price.the.m1_eur_mwh`, `price.peg.m1_eur_mwh` | observation | GBp/therm, EUR/MWh | market-data | **proposed** (basis features) |
| cross_commodity | `vol.ttf.m1_atm_implied_pct` | observation | % | market-data | bloomberg.yaml |
| storage | `storage.<eu,de,nl,be,fr,at,dk,cz,gb>.{inventory_twh, injection_gwhd, withdrawal_gwhd, working_gas_volume_twh}` | observation | TWh, GWh/d | gas-fundamentals | bloomberg_fundamentals.yaml |
| storage | `storage.eu.{full_pct, injection_capacity_gwhd, withdrawal_capacity_gwhd}` | observation | %, GWh/d | gas-fundamentals | bloomberg_fundamentals.yaml |
| flows | `flows.no.{pipeline_exports_gwhd, pipeline_exports_continent_gwhd, nominations_gwhd, technical_capacity_gwhd, outage_unavailable_gwhd, outage_unplanned_unavailable_gwhd}` | observation | GWh/d | gas-fundamentals | fundamentals.yaml |
| flows | `flows.eu.{imports_north_africa, imports_tap, imports_russia, imports_gb, exports_gb, exports_ukraine_moldova, exports_other}_gwhd` | observation | GWh/d | gas-fundamentals | fundamentals.yaml |
| flows | `production.eu.domestic_gwhd`, `demand.eu.{ldz,industrial,power_burn,total}_gwhd`, `constraints.eu.interconnector_unavailable_gwhd`, `outages.eu.lng_terminal_unavailable_gwhd` | observation | GWh/d | gas-fundamentals | fundamentals.yaml |
| flows | `demand.{fr}.*_gwhd`, `demand.{be,nl,at,gb}.*_mcmd` | observation | GWh/d, mcm/d | gas-fundamentals | bloomberg_fundamentals.yaml (mcm/d needs `GasQuality`) |
| lng | `lng.eu.{sendout_gwhd, sendout_capacity_gwhd, terminal_inventory_1000m3}` | observation | GWh/d, 1000 m³ | gas-fundamentals / lng-flows | bloomberg_fundamentals.yaml |
| lng | `lng.<region>.afloat_{20d,30d}_{volume_t,vessels_count}`, `lng.<canal>.laden_transits_count`, `lng.<route>.flow_t_per_d`, `lng.<country>.{exports,imports}_t_per_d` | observation | t, count, t/d | lng-flows | bloomberg_lng.yaml |
| lng | `lng.eu.arrivals_fcst_*` (expected EU arrivals 7/14/30/45d) | forecast | TBD | lng-flows | **proposed**. Grade C unless snapshotted |
| lng | `freight.atlantic.spark30s_usd_per_d`, `freight.pacific.spark25s_usd_per_d` | observation | USD/day | lng-flows | spark.yaml |
| lng | `freight.blng{1,2,3}_*.front_future_usd_per_d` | observation | USD/day | lng-flows | bloomberg_lng.yaml |
| weather | `weather.point.{t2m_mean_degc, t2m_mean_obs_degc, t2m_mean_ens_degc}` | forecast / observation | °C | weather-power | weather.yaml |
| weather | `weather.eu.hdd_degc_day` (pop-weighted HDD), plus anomaly and revision (derived) | forecast | degC·day | weather-power | **proposed** (synthetic uses `weather.eu.hdd`) |
| power | `power.de.{wind,solar,load}_fc_gw`, `power.de.{wind,solar}_gen_gw`, `power.de.{wind,solar}_capacity_gw`, `power.fr.nuclear_avail[_fc]_gw`, `power.alp_nord.hydro_reservoir_gwh` | observation / forecast | GW, GWh | weather-power | power.yaml |
| power | `power.de.base_m1_eur_mwh` / `price.power_de.base_m1_eur_mwh`, `price.power_nl.base_m1_eur_mwh` | observation | EUR/MWh | weather-power / market-data | power.yaml / bloomberg.yaml (two ids) |
| power | `gas.ttf.m1_eur_mwh` | observation | EUR/MWh | weather-power | power.yaml. Continuous M1, **features only, never a target** |
| calendar | none stored. Weekday, month, gas year, days to expiry and holidays are derived from contract metadata and config | feature | — | feature-engineering / market-data | not stored |
| text | events per `docs/llm_event_schema.md` (not numeric series) | own schema | — | llm-news | excluded from clean backtest |

---

## 6. Open questions for the orchestrator

1. **`event_time` convention for daily series.** Synthetic storage uses local midnight in UTC. Synthetic
   weather uses UTC midnight. Gas-day series (06:00–06:00 local) could also use gas-day start. Proposal:
   daily = local midnight of the date in `source_timezone`, and gas-day series record the date, not 06:00.
2. **Observation `event_time > publication_time`.** `resolve_as_of` keeps these rows, but PIT policy §2
   calls them a data error for observation series. One of the two should change.
3. **Duplicate or competing ids across configs.** `fx.eurusd.spot_usd_per_eur` is defined in two files.
   EUA, API2 and DE power each have two ids. One owner per id is needed.
4. **YAML for quality thresholds**, e.g. `configs/data_quality.yaml: {outlier_window: 60,
   outlier_min_periods: 20, outlier_z_threshold: 6.0, outlier_transform: diff, staleness_periods: 3}`,
   plus a per-series `expected_frequency` and `staleness_grace`.
