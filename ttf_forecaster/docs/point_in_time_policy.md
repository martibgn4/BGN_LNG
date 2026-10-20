# Point-in-time data policy

Version 0.1, 2026-10-16. **Non-negotiable.** A forecast that uses information not available at its
origin is invalid, however good its numbers look. Implemented in `data/base.py` (schemas) and
`features/point_in_time.py` (the as-of resolver).

## 1. The bitemporal record

Every stored observation is one row of the canonical long schema (`OBSERVATION_COLUMNS` in
`data/base.py`):

| Column | Meaning |
|---|---|
| `series_id` | internal name, e.g. `storage.eu.inventory_twh` (never a vendor ticker) |
| `event_time` | when the physical/market fact applies (gas day start, settlement date, valid time) |
| `publication_time` | when the value was first (or, for a revision, re-) published by the source |
| `ingestion_time` | when *our* system received it |
| `revision` | 0 for first release, 1.. for each later vintage of the same `(series_id, event_time)` |
| `value`, `unit` | number and its unit (registry in `contracts/units.py`) |
| `source`, `geography` | vendor/dataset label; ISO country, hub or region code |
| `quality_flag` | `ok`, `estimated`, `missing_input`, `outlier_suspect`, `synthetic` |
| `entitlement` | `public`, `licensed_bloomberg`, `licensed_spark`, `licensed_other`, `synthetic` |
| `pit_grade` | `A` true vintage, `B` lag-approximated, `C` not point-in-time (see §3) |
| `publication_time_estimated` | True when `publication_time` came from a lag rule, not the source |
| `source_timezone` | IANA zone the source used before conversion |

All timestamps are stored **tz-aware in UTC**. Conversion to Europe/Amsterdam is done at the edge
(features, reports). A naive timestamp entering the store is a hard error.

Forecast-type data (weather, LNG arrivals) additionally carry `issue_time` (model run / forecast
creation), `valid_time` and `lead` in their own schemas; for those `publication_time` is the time the
run became available, not the run's nominal init time.

## 2. The as-of rule

    visible(as_of) = rows with publication_time <= as_of
    resolved(as_of) = for each (series_id, event_time), the visible row with the highest revision

The resolver (`resolve_as_of`) is the **only** sanctioned way for feature code to read
observations. Feature functions receive an already-resolved frame or call the resolver; they never
filter on `event_time` alone. In live operation `ingestion_time <= as_of` is also enforced
(`strict_ingestion=True`), because in production we cannot use what we have not yet received.

`event_time > as_of` is allowed only for forecast-type schemas (a forecast *for* the future issued
*before* `as_of`). For observation-type series it is a data error.

## 3. PIT grades

| Grade | Meaning | Use |
|---|---|---|
| **A** | genuine vintages: each revision stored with its own publication time (live-accumulated data, vintaged vendor archives, forecast runs with issue time) | clean backtest |
| **B** | only the latest revised value exists; `publication_time` = `event_time` + documented lag rule | clean backtest **with** a revision-sensitivity test; flagged in every report |
| **C** | value known to contain hindsight (realised destinations, ex-post actuals presented as forecasts, revised series with no lag rule) | **excluded** from the clean backtest; may appear only in a separately labelled "upper-bound / contaminated" experiment |

Grade is a property of the *series as sourced*, set in `configs/data_sources.yaml`, and propagated
to every feature built from it (a feature takes the worst grade of its inputs).

## 4. Publication-lag rules (initial; each must be evidenced before live use)

| Source family | Default rule | Grade | Note |
|---|---|---|---|
| TTF and European futures settlements | trading date 19:00 Europe/Amsterdam | A | settlements are not revised in practice; spot-check |
| JKM, HH (NYMEX), Brent (ICE), API2, EUA | **next European trading date 19:00** for features of TTF origin *t* | A | HH/JKM/Brent closes after TTF settlement — same-date use is leakage |
| FX (EURUSD) | same rule as the instrument it converts | A | |
| GIE AGSI storage (direct or via Bloomberg CGIE*) | gas day *D* published **D+1 19:30 CET**, then revisable; Bloomberg history holds the latest value → use **D+2 07:00** as conservative availability | B | revisions not reproducible from Bloomberg; sensitivity: +1, +3 days |
| GIE ALSI LNG send-out | as AGSI | B | |
| Norwegian flows (Gassco) | nominations: real time; physical flows: D+1 | B until a vintaged feed exists | |
| Gassco/REMIT UMMs (outages) | message timestamp | A if message archive kept | raw message only |
| LLM-extracted, human-reviewed events | max(source publication, extraction, review time) | A only if reviewed live; B for historical back-fill; C if the event set was assembled with hindsight | `text/to_features.py` |
| Weather forecasts | run availability time (init + ~6 h per provider) | A | levels vs revisions both from stored runs |
| Observed weather / ERA5 | ≥ D+5 | B | never used as a forecast |
| LNG vessel destinations / ETAs | only snapshot-stored history | A if snapshotted, else **C** | realised destination is C, always |
| Bloomberg `LNGG*`, `AHOY*` aggregates | D+1 assumed | B (revision behaviour unknown) | to be measured by live snapshotting |
| Spark assessments | release date 19:00 London assumed **[CONFIRM]** | A | |

A lag rule lives in config (`publication_lag` block per series). Code never hard-codes a lag.

## 5. Specific prohibitions

1. No realised weather, realised LNG destination, or ex-post storage value as a forecast-time input.
2. No interpolation across a missing value using a later observation (forward-fill only, with a
   maximum staleness per series; stale beyond it → NaN).
3. No full-sample normalisation, scaling, PCA, imputation or "seasonal normal" fitted on data after
   the origin. Seasonal normals use only years strictly before the origin's gas year.
4. No feature selection, hyperparameter choice or threshold calibration on test folds.
5. No same-session cross-commodity price that closes after the TTF settlement.
6. No structural-break or regime dummy defined with hindsight (e.g. "2022 crisis = 1"). Regime
   labels used as features must be computable from data available at the origin.

## 6. Validation mechanics (what enforces the above)

- `features/point_in_time.py::resolve_as_of` — the single read path; unit-tested with revision and
  late-publication fixtures.
- **Leakage canary test** (software-quality / model-risk): perturb all observations with
  `publication_time > as_of` and assert the feature row at `as_of` is bit-identical.
- **Future-shift test**: shift a feature forward by one day and confirm OOS skill collapses if the
  original skill was real; if skill *rises* sharply when a feature is shifted backward, investigate.
- `targets/` emits `label_known_time`; `evaluation/splits.py` purges training rows whose
  `label_known_time` > fold start and embargoes `h` trading days after each test block.
- Every prediction row stores `as_of`, `model_id`, `feature_hash`, `train_end`, `label_known_time`
  so any number in a report can be re-derived.

## 7. Storage layout

    data/raw/         vendor payloads as received (never edited), partitioned by source/ingestion date
    data/normalised/  canonical long schema, Parquet, append-only (new vintages are new rows)
    data/features/    feature matrices keyed by (as_of, feature_set_hash)
    data/predictions/ model outputs with provenance columns
    data/benchmarks/  benchmark forecasts with the same keys as predictions

`data/` is git-ignored in full. Synthetic fixtures live under `tests/fixtures/` and carry
`source="SYNTHETIC"`, `entitlement="synthetic"`, `quality_flag="synthetic"`.
