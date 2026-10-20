"""Pull LNG cargo imports per country from the BSRCH TPD:DEX saved searches (user requests, 2026-10-20).

For each ``known`` country in configs/sources/lng_bsrch.yaml, every BIKEY is requested. The cargo rows of all keys are merged
into the side table ``data/live/side/lng_cargoes.parquet`` (one row per country, IMO and arrival). From the side table this writes:
* per-country daily observations (``lng.<cc>.imports_*``), with days before ``history_truncated_before`` left out;
* regional observations (``lng.region.<region>.imports_*``) for each configured region whose members are all present.

Idempotent: a cargo already in the side table is not added again, and identical observation rows are duplicates for the store.
``cross_check_ahoy`` compares the regional totals with the AHOY aggregates already in the store.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data import lng_imports as li
from ttf_forecaster.data.storage import Store
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.pipelines.ingest import ingest
from ttf_forecaster.settings import load_config

log = get_logger(__name__)
Runner = Callable[[str], pd.DataFrame]  # BIKEY -> the saved search's raw result


def side_path(store: Store) -> Path:
    return Path(store.root) / "side" / "lng_cargoes.parquet"  # type: ignore[attr-defined]


def _write(obs: pd.DataFrame, store: Store, raw_name: str) -> int:
    if obs.empty:
        return 0
    s = ingest(lambda f=obs: f, store, kind="observation", source=li.SOURCE, raw_name=raw_name,
               keep_adapter_ingestion_time=True, fail_on_quality=False)
    return int(s.n_written)


def pull_all(store: Store, runner: Runner, cfg: dict | None = None) -> dict[str, Any]:
    """Pull every key of every known country, then write the country and regional observations."""
    cfg = cfg if cfg is not None else load_config("sources/lng_bsrch")
    entries = li.country_entries(cfg)
    if not entries:
        raise ValueError("no known LNG import country in configs/sources/lng_bsrch.yaml")
    side = side_path(store)
    prev = pd.read_parquet(side) if side.exists() else pd.DataFrame(columns=li.CARGO_COLUMNS)
    fresh = []
    keys_done: dict[str, int] = {}
    for cc, entry in entries.items():
        for key in entry["bikeys"]:
            cargoes = li.cargo_rows(runner(key), cc, key)
            fresh.append(cargoes)
            keys_done[f"{cc}:{key[:6]}"] = len(cargoes)
    merged = li.merge_cargoes(prev, *fresh)
    side.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(side, index=False)

    now = pd.Timestamp.now(tz="UTC")
    calendar = li.calendar_of(merged)
    daily: dict[str, pd.DataFrame] = {}
    written: dict[str, int] = {}
    for cc, entry in entries.items():
        d = li.country_daily(merged[merged["country"] == cc], calendar, entry.get("history_truncated_before"))
        daily[cc] = d
        written[cc] = _write(li.country_observations(d, cc, entry.get("geography", cc), now), store,
                             f"lng_imports_{cc.lower()}")
    regions_written: dict[str, Any] = {}
    for region, members in (cfg.get("regions") or {}).items():
        members = [str(m) for m in members]
        if not all(m in daily for m in members):
            regions_written[region] = f"skipped: members not all pulled {[m for m in members if m not in daily]}"
            continue
        total = li.regional_daily(daily, members)
        regions_written[region] = _write(li.region_observations(total, region, now), store,
                                         f"lng_region_{region.lower()}")
    log.info("lng imports: cargoes %d, countries %s, regions %s", len(merged), written, regions_written)
    return {"cargoes_total": len(merged), "keys": keys_done, "country_rows": written, "regions": regions_written,
            "calendar": [str(calendar.min().date()), str(calendar.max().date())] if len(calendar) else []}


def cross_check_ahoy(store: Store, cfg: dict | None = None) -> dict[str, Any]:
    """Our totals against the AHOY aggregates already in the store, over the days both have.

    NWE (our NL+BE+FR+GB+DE regional total) against ``lng.nwe.imports_t_per_d`` (AHOYIMEU); all configured countries
    against ``lng.global.imports_t_per_d`` (AHOYIMTL, global imports, so our total should not exceed it).
    """
    cfg = cfg if cfg is not None else load_config("sources/lng_bsrch")
    countries = list(li.country_entries(cfg))
    tonnes_ids = [f"lng.{c.lower()}.imports_tonnes" for c in countries]
    region_ids = [f"lng.region.{r.lower()}.imports_tonnes" for r in (cfg.get("regions") or {})]
    ahoy_ids = ["lng.nwe.imports_t_per_d", "lng.global.imports_t_per_d"]
    rows = store.read_normalised("observation", series_ids=tonnes_ids + region_ids + ahoy_ids)
    if rows.empty:
        return {"status": "no rows"}
    rows = rows.assign(day=pd.to_datetime(rows["event_time"], utc=True).dt.tz_localize(None).dt.normalize())
    series = {sid: g.drop_duplicates("day").set_index("day")["value"] for sid, g in rows.groupby("series_id")}

    def compare(label: str, ours: pd.Series, ahoy: pd.Series) -> dict[str, Any]:
        j = pd.concat([ours.rename("ours"), ahoy.rename("ahoy")], axis=1, join="inner").dropna()
        if len(j) < 30:
            return {"days": len(j)}
        return {"days": len(j), "corr": round(float(j.ours.corr(j.ahoy)), 4),
                "ours_sum_t": round(float(j.ours.sum())), "ahoy_sum_t": round(float(j.ahoy.sum())),
                "ratio": round(float(j.ours.sum() / j.ahoy.sum()), 4) if j.ahoy.sum() else None}

    out: dict[str, Any] = {}
    nwe = "lng.region.nwe.imports_tonnes"
    if nwe in series and "lng.nwe.imports_t_per_d" in series:
        out["NWE regional vs AHOYIMEU"] = compare("nwe", series[nwe], series["lng.nwe.imports_t_per_d"])
    if all(c in series for c in tonnes_ids) and "lng.global.imports_t_per_d" in series:
        total = sum(series[c] for c in tonnes_ids)  # NaN where any country is missing that day
        out["all countries vs AHOYIMTL (global)"] = compare("all", total, series["lng.global.imports_t_per_d"])
    return out
