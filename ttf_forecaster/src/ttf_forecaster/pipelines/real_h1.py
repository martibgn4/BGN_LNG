"""H1 on real data: does EU storage improve on the TTF forward curve for fixed-contract returns?

Reads ONLY the git-ignored store written by the authorised Bloomberg pull (``data/live``): TZT
monthly settlements, vendor last-trade dates, ``CGIEEU*`` storage and EURUSD. No vendor call is made
here. Evaluation is ``synthetic_e2e.evaluate_bundle`` unchanged (same targets, features, folds,
benchmarks, price-only twin, both labels, trading), with ``is_synthetic=False``. The locked holdout
(``configs/backtests.yaml``) is excluded by the splitter and is never evaluated here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    FORECAST_SCHEMA,
    DataBundle,
    empty_frame,
)
from ttf_forecaster.data.storage import ParquetStore
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.pipelines.synthetic_e2e import WorldResult, evaluate_bundle
from ttf_forecaster.settings import PROJECT_ROOT, load_config

log = get_logger(__name__)


def _read(store: ParquetStore, kind: str, empty: pd.DataFrame) -> pd.DataFrame:
    try:
        df = store.read_normalised(kind)
    except Exception:  # absent layer -> typed empty frame
        return empty
    return df if len(df) else empty


def load_live_bundle(root: str | Path) -> DataBundle:
    """Bundle from the live store. Raises if the curve or storage layer is missing."""
    store = ParquetStore(Path(root))
    curves = store.read_normalised("curve")
    obs = store.read_normalised("observation")
    if curves.empty or obs.empty:
        raise RuntimeError(f"live store {root} has no curve or observation data; run the pull")
    return DataBundle(observations=obs,
                      forecasts=_read(store, "forecast", empty_frame(FORECAST_SCHEMA)),
                      curves=curves,
                      contracts=_read(store, "contract_metadata",
                                      empty_frame(CONTRACT_METADATA_SCHEMA)))


def run_h1(*, config_dir: str | None = None, run_tag: str | None = None) -> list[WorldResult]:
    """Run H1 for every configured horizon; artefacts go to the live store (is_synthetic=False)."""
    cfg = load_config("pipeline", config_dir)["real"]
    root = PROJECT_ROOT / cfg["store"]
    bundle = load_live_bundle(root)
    store = ParquetStore(root)
    tag = run_tag or pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    results = []
    for h in cfg["horizons"]:
        label = f"real_T3_h{h}"
        log.info("real_h1_start", extra={"fields": {"label": label}})
        results.append(evaluate_bundle(bundle, label=label, run_id=f"{label}_{tag}", store=store,
                                       is_synthetic=False, config_dir=config_dir,
                                       first_origin=cfg["first_origin"], horizon=int(h)))
    return results
