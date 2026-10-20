# Environment and repository audit (STEP 1)

Date: 2026-10-16. Performed by the orchestrator before any file in `ttf_forecaster/` existed.
Nothing outside `ttf_forecaster/` was modified.

## Repository

- Location: `C:\Marti\python_tests_2\BGN_LNG` (Git, branch `feature/7-5-2027/local_branch_continuous_work`,
  main branch `main`, 45 uncommitted changes belonging to other projects — untouched).
- **No root `.gitignore`.** Each sibling project has its own; this project does too.
- Sibling research projects that follow one shared convention (config-driven, vintaged cache,
  fetchers raise rather than invent, negative results reported as found):
  `GasImbalanceModel` (EU gas balance), `NWEBasisModel` (NWE LNG basis vs TTF),
  `WeatherReturnsModel` (weather-revision returns), `LNGFreightModel` (Spark30S freight),
  `DeliveryOptionModel`, `StockPicker`, `VenezuelaOptionality`.
- Production reporting code: `Data/lng_report/` (daily market report, Bloomberg + Spark).

## Python

| Interpreter | Version | Notes |
|---|---|---|
| System | 3.14.2 | pandas 3.0.3, numpy 2.5.0 |
| `..\env_local` (venv) | 3.14.2 | **Chosen.** pandas 2.3.3, numpy 2.3.5, scikit-learn 1.9.0, statsmodels 0.15.0, lightgbm 4.7.0, pyarrow 22.0.0, PyYAML 6.0.3, pytest 9.0.2, exchange_calendars 4.12, requests, xbbg 0.10.3, blpapi 3.25.12.1 |
| `GasImbalanceModel/.venv` | — | has xbbg, no scikit-learn |

Dependency manager: plain `pip` + `requirements.txt` (frozen from env_local 2026-09-24) at repo root,
plus per-project `pyproject.toml` (hatchling) in siblings. No poetry/uv/conda lockfiles.

**Not installed anywhere checked:** ruff, mypy, pre-commit, duckdb, pandera, xgboost, catboost,
optuna, mlflow, hmmlearn, ruptures, shap, polars. None was installed by this project (see
`modelling_log.md` D-004/D-005).

## Bloomberg

- `blpapi 3.25.12.1` and `xbbg 0.10.3` are installed in env_local; `C:\blp\DAPI` is on PATH, so a
  Desktop API Terminal is present on this machine. Not exercised by this project.
- BQL: only inside BQuant; not available here.
- Existing in-repo identifiers (re-used as *known*, not re-verified in this project):
  - `Data/lng_report/config.py` — production hub roots: TTF `TZT/QZT/QQT/QTT` (month/quarter/season/year,
    EUR/MWh), TTF USD `TMR/TQR/TSR/TYR`, NBP `FN/QR/SA/YAA`, THE `NCG`, PEG `PNG`, PSV `PSR`,
    PVB `PXB`, VTP `CEG`, JKM `JKL`, HH `NG`, Brent `CO`.
  - `DeliveryOptionModel/config/market.yaml` — TTF option root `FJS`, EURUSD `EUR Curncy`,
    MWh→MMBtu factor.
  - `GasImbalanceModel/config/bloomberg_series.yaml` — verified GIE AGSI/ALSI redistribution
    (`CGIE<CC><ST|IN|WI|WG|FU> Index`, `LNGATLIN`, `LNGATLSO`).
  - Memory/notes: `LNGG*` LNG-on-water, `AHOY*` trade flows, `IKA1/IKD1/IKI1` Baltic LNG freight.
- **Not found anywhere:** TTF day-ahead, API2 coal, EUA (only an unexplained `MO1 Comdty` in an
  adhoc script), European power, Norwegian flows, LNG send-out by terminal.
- **Correction (gas-fundamentals review):** demand *is* available: `GasImbalanceModel/config/demand_sources.yaml`
  holds 15 verified daily TSO demand tickers (FR, BE, NL, AT, GB) and a generation-by-fuel ticker pattern.
  European gas fundamentals panels on Bloomberg need a user-generated BSRCH BIKEY.

## Spark Commodities

- Auth is **OAuth2 client credentials** (client id + secret → bearer token), base URL
  `https://api.sparkcommodities.com` (`Utils/utils_spark.py`, `NWEBasisModel/src/nwebasis/data/spark_api.py`).
  The proposed `SPARK_API_KEY` env var therefore does not match the real mechanism;
  `.env.example` uses `SPARK_CLIENT_ID` / `SPARK_CLIENT_SECRET` (the names `Utils/utils_spark.py` already reads).
- Verified dataset ids already in use (NWEBasisModel `config/sources.yaml`, 2026-09-04): `spark30s`,
  `spark25s`, `spark30ffa-monthly`, `spark25ffa-monthly`, `sparknwe-fin-monthly`,
  `sparknwe-des-fin-monthly`, netbacks `/v1.0/netbacks/` (needs `via-point`), terminal slots
  `/beta/terminal-slots/terminals/`.
- **No Spark vessel-position, arrival or AIS dataset is known in this repo.** Treated as
  requires-user-input.

## Credentials (values not displayed)

- **Finding — action recommended:** `adhoc_scripts/client_credentials.csv` (two columns, header +
  one row, i.e. a Spark client id/secret) **is tracked in Git** (committed in `2b70999`). Anyone with
  repo access has the secret. Recommend rotating the Spark secret, `git rm --cached` the file, and
  adding it to a root `.gitignore`. This project never reads that path by default; it reads env vars,
  with an optional path override in a git-ignored local config.
- No Bloomberg credentials are needed (Desktop API uses the logged-in Terminal).
- No relevant env vars are set in this shell (only names were listed, no values).

## Existing data files

Vintaged caches exist in siblings' `data_cache/` (Parquet, git-ignored), plus a BNEF LDZ demand
workbook at repo root. This project does **not** read sibling caches: cross-project coupling would
make reproduction depend on another project's fetch state. Re-use is via the identifiers only.

## Reusable market-data framework?

No shared library. Each sibling has its own `data/base.py` + `bloomberg.py` + `spark*.py` pattern.
This project follows the same pattern (adapter interfaces, config-held identifiers, raise on missing
data) but is self-contained.
