"""Build the implied realised Norwegian outage series from the stored total exit flow (user request, 2026-10-20).

Reads ``flow.no.exit_total_intraday_mcmd`` from the store, writes the observation rows of data/implied_outage.py. Re-running
on the same store adds nothing: identical rows are duplicates for the store.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ttf_forecaster.data import implied_outage as io
from ttf_forecaster.data.storage import Store
from ttf_forecaster.pipelines.ingest import ingest
from ttf_forecaster.pipelines.outage_sheet import project_gas_quality
from ttf_forecaster.settings import load_config

SOURCE = io.SOURCE


def build(store: Store, cfg: dict | None = None) -> dict:
    """Compute and store the observation rows. Returns the rows written per series."""
    cfg = cfg if cfg is not None else load_config("sources/flow_outage")["implied_outage"]
    flow = store.read_normalised("observation", series_ids=[cfg["flow_series"]])
    if flow.empty:
        raise ValueError(f"no stored flow rows for {cfg['flow_series']}: pull it first")
    rows = io.observation_rows(flow, cfg, project_gas_quality(), pd.Timestamp.now(tz="UTC"))
    summary = {}
    for sid, grp in rows.groupby("series_id"):
        summary[sid] = len(grp)
    ingest(lambda f=rows: f, store, kind="observation", source=SOURCE, raw_name="implied_outage_build",
           keep_adapter_ingestion_time=True, fail_on_quality=False)
    return summary
