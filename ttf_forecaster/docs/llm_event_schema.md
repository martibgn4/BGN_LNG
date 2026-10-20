# LLM event schema, review gate and text features

Version 0.1, 2026-10-16. Code: `src/ttf_forecaster/text/`. Config: `configs/text.yaml`.
Prompt draft: `prompts/event_extraction_v0.md` (not executed). No event data is populated; all test
events are SYNTHETIC.

## 1. Purpose and non-purpose

The LLM is **not** a numerical price forecaster (`research_specification.md` §11). Its only
modelling role is to turn **authorised** unstructured text (outage messages, operator notices,
licensed notes the desk may process) into **candidate** structured records:

- structured extraction (what asset, which region, when, how much *if stated*),
- classification (outage / maintenance / policy / weather / LNG diversion / demand),
- conversion into candidate quantitative factors (counts, supported offline volumes).

Every record is a proposal until a named human approves it. Nothing text-derived enters a model
without provenance, and in v1 nothing enters without review.

## 2. The record (`text/schema.py`)

`EventRecord` is a frozen dataclass; `event_json_schema()` is its JSON Schema (draft 2020-12) and
`extraction_output_json_schema()` is the narrower schema the model must return (content fields
only — provenance and review fields are set by our code, never by the model).

| Group | Field | Notes |
|---|---|---|
| content | `event_time` | when the occurrence happened, if stated; else null |
| | `publication_time` | source's publication time (tz-aware); not our ingest time |
| | `event_type` | `outage, maintenance, policy, weather, lng_diversion, demand` |
| | `asset`, `region` | as written in the source; null if not stated; no code mapping by the LLM |
| | `estimated_volume_impact_mcmd` | null unless supported (§3 rule 4) |
| | `expected_start`, `expected_end` | stated timing only; null otherwise |
| | `directional_effect` | `bullish, bearish, uncertain` — for European gas, from the text's own content |
| | `confidence` | [0,1], faithfulness of extraction, **not** a probability of a price move |
| provenance | `event_id` | `<16 hex of document hash>-<index>`, shared by all versions |
| | `document_sha256` | hash of the exact bytes the model saw |
| | `source_publisher`, `source_reference`, `source_entitlement` | from the authorised-source list; archive key in our store |
| | `extracted_at`, `extraction_model_id`, `extraction_model_version`, `prompt_version` | |
| | `impact_quote` | verbatim span containing the number **and** its unit |
| | `capacity_mapping_ref` | key into an independently maintained, versioned asset-capacity table |
| | `timing_quote` | verbatim span supporting the timing fields |
| | `human_review_required` | `const: true` in v1 |
| review | `version`, `parent_version` | 0 = original extraction; each review action appends a version |
| | `reviewer`, `review_status`, `review_time`, `review_note` | status: `pending, approved, rejected, edited` |

## 3. Validation gate (`text/validation.py`)

`validate_event` returns *all* errors and warnings; `assert_valid` raises `EventValidationError`.

1. Enumerations valid; `confidence` finite in [0, 1]; `source_entitlement` is a `data.base.Entitlement`.
2. All timestamps tz-aware; `publication_time <= extracted_at <= review_time`.
3. `expected_end >= expected_start` when both stated (warning if timing has no `timing_quote`).
4. **Numbers are never invented.** A non-null `estimated_volume_impact_mcmd` requires either
   - an `impact_quote` that literally contains the same number immediately followed by an accepted
     mcm/d unit (`quote_supports_number`; decimal comma and thousands separators handled; "12.5
     GWh/d", "12.5" without unit, "12.5 mcm" without per-day, or "112.5 mcm/d" do **not** support
     12.5), or
   - a `capacity_mapping_ref` registered in the capacity table (checked when the registry is passed).
   Otherwise the number must be null. Unit conversion (GWh/d, kWh/h, MSm3/d) is not literal support.
5. `human_review_required` cannot be False.
6. Provenance complete; publisher on the authorised list (when passed); quotes are verbatim
   substrings of the document (when the document text is passed).
7. Review fields consistent with status; the extraction model can never be its own reviewer.

## 4. Review workflow (`text/review.py`)

`ReviewLedger` is append-only. `add_extraction` registers version 0 (`pending`, validated).
`approve`, `reject` and `edit` each **append** a new version with reviewer, tz-aware review time
(non-decreasing) and parent pointer; the original extraction is never modified, so the model's
output and the human's correction can both be audited and the extraction quality measured.
Edits cannot change provenance or review fields, and an edited version must re-pass validation (a
reviewer cannot add an unsupported number either). Only events whose **latest** version is
`approved` or `edited` pass (`ReviewLedger.passing()`); a later rejection withdraws an event.

## 5. Point-in-time treatment (`text/to_features.py`)

Reviewed events become bitemporal OBSERVATION_SCHEMA rows on a gas-day grid (06:00 Europe/Amsterdam):

- `text.<region>.outage_count` — reviewed `outage`/`maintenance` events active on the gas day.
- `text.<asset>.offline_mcmd` — sum of supported offline volumes; **NaN** if any active event on the
  asset has no supported number (unknown is not zero).

Each change in what was knowable (approval, edit, rejection) emits a new `revision` with its own
`publication_time`, so `features/point_in_time.resolve_as_of` sees exactly what was known.
An extra `lineage` column lists the contributing `event_id:vN`.

| Mode | `publication_time` of a row | Versions used | Grade |
|---|---|---|---|
| live | max(source publication, extracted_at, review_time) of the version that changed the state | every reviewed version | **A** |
| historical back-fill | source publication + `historical_review_lag` (config assumption), `publication_time_estimated=True` | final reviewed version only | **B** at best |
| event set assembled with hindsight | as above | as above | **C** — excluded by `resolve_as_of` defaults |

Why historical is B at best: the reviews were done after the fact, so their timestamps say nothing
about availability; the lag is an assumption, and the final reviewed content may embed later
corrections. Why hindsight is C: if the documents were searched for or selected after the outcome
(e.g. "the outages that moved the market"), the *presence* of an event is itself look-ahead. A
further, LLM-specific hindsight channel: a model trained after the event may "know" the outcome
(actual return date, actual volume) and leak it into fields the text does not state. Rules 3–4 and
verbatim quotes are the defence; reviewers must check timing against `timing_quote`.

Other assumptions (in config, flagged in data): an event with a start but no stated end is assumed
active for `open_ended_days` (rows flagged `estimated`); events with neither `expected_start` nor
`event_time` cannot be placed on the grid and produce no rows.

`build_text_features(bundle, origins, params)` conforms to `FeatureBuilder`: columns
`text__<series without prefix>_d<offset>` (default offset 1 = next gas day), `attrs["pit_grade"]` per
column (worst grade of the series' rows; C if unknown), `attrs["family"] = "text"`,
`attrs["clean_backtest_eligible"]` (default False). A missing row is 0 only for origins at/after a
configured `coverage_start` (the date from which the reviewed feed is complete); otherwise NaN.
Pass a fixed `series` list: deriving columns from the data lets later-discovered assets define the
column universe, a mild form of hindsight.

## 6. How text features are evaluated

- Text is its own ablation family, `text` (`research_specification.md` §5), always evaluated
  **separately** from quantitative features: base model vs base + `text`, same folds, DM test
  against the forward-curve benchmark.
- **Excluded from the clean backtest until provenance exists**: i.e. until there is a live-collected
  (grade A) or complete-archive back-filled (grade B, with review-lag sensitivity 6/24/72 h) event set.
  Grade C results may appear only in a labelled "contaminated / upper-bound" experiment.
- Extraction quality is measured separately from forecasting value: precision/recall of reviewed vs
  original extractions, share of numbers nulled or edited, reviewer disagreement.
- A text signal that beats nothing is reported as such; it is not re-specified to rescue it.

## 7. Data governance

- **Authorised sources only.** A source is usable when `configs/text.yaml` lists it with
  `authorised: true`, a recorded licence and access route supplied by the user. Today every source
  is `requires_user_input`, `authorised: false`.
- **No proprietary text to external models.** Sending licensed or proprietary text (broker notes,
  licensed newswires, internal desk notes) to *any* external model or service — including
  Anthropic's API — requires explicit user and compliance approval, recorded in
  `extraction.approval_reference`. Default: `external_model_approved: false`. Public regulatory
  messages (e.g. REMIT UMMs) still need their terms of use checked.
- **Licensing of news sources.** Many news/terminal licences restrict storage, redistribution and
  machine processing (including LLM use). Check before archiving text; store only what the licence
  allows, git-ignored under `data/`.
- **Retention and audit.** Keep the document hash, archive key, model id/version, prompt version and
  every review version, so any feature value can be traced to a document and a reviewer.
- **No populated data in the repo.** Tests use SYNTHETIC events only.

## 8. Not built yet

- An extraction runner (would call a model): deliberately absent.
- The asset-capacity table and its loader (`capacity_mapping.table_path`).
- Persistence of the ledger (the data-architecture store could hold it as JSON lines / Parquet).
- Mapping of source-specific units (MSm3/d, GWh/d, kWh/h) into mcm/d — needs a decision.
