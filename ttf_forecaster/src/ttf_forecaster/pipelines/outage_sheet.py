"""Load one outage-spreadsheet snapshot into the store (user request, 2026-10-20).

What one snapshot writes:
* FORECAST_SCHEMA rows for the aggregates: ``outage.no.gassco_planned_mcmd`` and ``..._unplanned_mcmd`` (regions N. Sea
  and Atlantic, every operator), per gas day from the snapshot day to the last end date. Value = the sum of reductions in
  mcm/d; the GWh/d twins use the project GCV (units.GasQuality). issue_time = publication_time = the snapshot time (file
  modification time). Grade A.
* FORECAST_SCHEMA rows for LNG terminal outages by country (``outage.<cc>.lng_terminal_mcmd``), from the terminal map in
  configs/sources/outage_sheet.yaml. A facility with 'LNG' in its name that is not in the map is refused, not dropped.
* A side table ``data/live/side/outage_events.parquet``: one row per event per snapshot (event id, facility, operator,
  region, planned/unplanned, start, end, reduction, last update, raw text), so revisions to an event can be tracked.

Idempotent: a file whose SHA-256 is already in the side table is skipped; re-loading it writes nothing. A different file
with the same snapshot time conflicts loudly (the store refuses a conflicting re-delivery).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ttf_forecaster.data import outage_sheet as osheet
from ttf_forecaster.data.storage import Store
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.pipelines.ingest import ingest
from ttf_forecaster.settings import load_config
from ttf_forecaster.contracts.units import GasQuality

log = get_logger(__name__)

SOURCE = "gassco_outage_sheet"
SEA_REGIONS = ("N. Sea", "Atlantic")
SIDE_NAME = "outage_events.parquet"
SIDE_COLUMNS = ["file_sha256", "snapshot_time", "event_id", "region", "operator", "facility", "category",
                "start", "end", "reduction_mcmd", "unit", "last_update", "raw_text"]


@dataclass
class SheetSummary:
    file: str
    skipped: bool
    events: int = 0
    rows: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def project_gas_quality() -> GasQuality:
    """The project's GCV (configs/sources/fundamentals.yaml gas_quality), so every mcm -> GWh conversion agrees."""
    gq = load_config("sources/fundamentals")["balance"]["gas_quality"]  # the one GCV of the balance
    return GasQuality(gcv_kwh_per_m3=float(gq["gcv_kwh_per_m3"]), reference=str(gq["reference"]))


def side_path(store: Store) -> Path:
    return Path(store.root) / "side" / SIDE_NAME  # type: ignore[attr-defined]


def _read_side(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=SIDE_COLUMNS)


def _event_side_rows(events: list[osheet.Event], sha: str, snap: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame([{
        "file_sha256": sha, "snapshot_time": snap, "event_id": e.event_id, "region": e.region,
        "operator": e.operator, "facility": e.facility, "category": e.category, "start": pd.Timestamp(e.start),
        "end": pd.Timestamp(e.end) if e.end else pd.NaT, "reduction_mcmd": e.reduction_mcmd, "unit": e.unit,
        "last_update": pd.Timestamp(e.last_update) if e.last_update else pd.NaT, "raw_text": e.raw_text,
    } for e in events], columns=SIDE_COLUMNS)


def _forecast_rows(series: str, sums: pd.Series, geo: str, snap: pd.Timestamp, sha: str,
                   unit: str) -> pd.DataFrame:
    """FORECAST_SCHEMA rows: one per gas day, the value of that day's total (issue = publication = snapshot time)."""
    n = len(sums)
    valid = [pd.Timestamp(d, tz="UTC") for d in sums.index]
    return pd.DataFrame({
        "series_id": [series] * n,
        "issue_time": [snap] * n,
        "valid_time": valid,
        "lead_hours": [(v - snap).total_seconds() / 3600.0 for v in valid],
        "model": ["gassco_sheet"] * n,
        "member": [sha[:12]] * n,
        "value": sums.to_numpy(dtype=float),
        "geography": [geo] * n,
        "publication_time": [snap] * n,
        "ingestion_time": [snap] * n,
        "revision": [0] * n,
        "unit": [unit] * n,
        "source": [SOURCE] * n,
        "quality_flag": ["ok"] * n,
        "entitlement": ["licensed_bloomberg"] * n,
        "pit_grade": ["A"] * n,
        "publication_time_estimated": [False] * n,
        "source_timezone": ["UTC"] * n,
    })


def load_snapshot(path: Path, store: Store, *, cfg: dict | None = None,
                  gas_quality: GasQuality | None = None) -> SheetSummary:
    """Store one snapshot file. Returns what was written, or ``skipped=True`` if this exact file was loaded before."""
    cfg = cfg if cfg is not None else load_config("sources/outage_sheet")
    gq = gas_quality or project_gas_quality()
    sha = osheet.file_sha256(path)
    side = side_path(store)
    prev = _read_side(side)
    if (prev["file_sha256"] == sha).any():
        return SheetSummary(file=path.name, skipped=True)

    snap = osheet.snapshot_time(path)
    events = osheet.parse_events(osheet.read_sheet(path, cfg.get("sheet_name", "Worksheet")))
    summary = SheetSummary(file=path.name, skipped=False, events=len(events))
    unconverted = [e for e in events if e.reduction_mcmd is None]
    if unconverted:
        summary.failures.append(f"{len(unconverted)} rows not in mcm/d (kept in the side table only): "
                                f"{sorted({e.unit for e in unconverted})}")

    # aggregates: N. Sea and Atlantic, by category, from the snapshot day to the last end date
    ev_days = osheet.daily_reductions(events)
    snap_day = snap.normalize().date()
    if not ev_days.empty:
        last = ev_days["gas_day"].max()
        days = [snap_day + timedelta(days=i) for i in range((last - snap_day).days + 1)]
        sea = ev_days[ev_days["region"].isin(SEA_REGIONS)]
        for cat in ("planned", "unplanned"):
            part = sea[(sea["category"] == cat) & (sea["gas_day"] >= snap_day)]
            sums = part.groupby("gas_day")["reduction_mcmd"].sum().reindex(days, fill_value=0.0)
            sid = f"outage.no.gassco_{cat}_mcmd"
            frame = _forecast_rows(sid, sums, "NO", snap, sha, "MCM/d")
            n = _write(frame, store, f"outage_sheet_{sha[:12]}_{cat}")
            summary.rows[sid] = n
            gwh = _forecast_rows(f"outage.no.gassco_{cat}_gwhd", sums * gq.gcv_kwh_per_m3, "NO", snap, sha, "GWh/d")
            summary.rows[f"outage.no.gassco_{cat}_gwhd"] = _write(gwh, store, f"outage_sheet_{sha[:12]}_{cat}_gwh")

    # LNG terminal outages by country (facilities with LNG in the name)
    terminal_map: dict[str, str] = cfg.get("lng_terminal_country", {}) or {}
    for ev in events:
        if "LNG" not in ev.facility.upper() or ev.reduction_mcmd is None:
            continue
        cc = terminal_map.get(ev.facility)
        if cc is None:
            summary.failures.append(f"LNG facility not in the country map: {ev.facility} ({ev.event_id})")
            continue
        days = pd.Series(ev.reduction_mcmd, index=osheet.gas_days(ev), dtype=float)
        days = days[[d >= snap_day for d in days.index]]
        if days.empty:
            continue
        sid = f"outage.{cc.lower()}.lng_terminal_mcmd"
        summary.rows[sid] = summary.rows.get(sid, 0) + _write(
            _forecast_rows(sid, days, cc, snap, sha, "MCM/d"), store, f"outage_sheet_{sha[:12]}_lng_{ev.event_id}")
    # the side table is written LAST: it marks the file as loaded, so a failed load is retried, never skipped
    side.parent.mkdir(parents=True, exist_ok=True)
    new_side = _event_side_rows(events, sha, snap)
    (pd.concat([prev, new_side], ignore_index=True) if len(prev) else new_side).to_parquet(side, index=False)
    return summary


def _write(frame: pd.DataFrame, store: Store, raw_name: str) -> int:
    if frame.empty:
        return 0
    s = ingest(lambda f=frame: f, store, kind="forecast", source=SOURCE, raw_name=raw_name,
               keep_adapter_ingestion_time=True, fail_on_quality=False)
    return int(s.n_written)


def load_directory(directory: Path, store: Store) -> list[SheetSummary]:
    """Load every .xlsx snapshot in ``directory``, oldest modification time first."""
    files = sorted((p for p in directory.glob("*.xlsx") if not p.name.startswith("~$")),
                   key=lambda p: p.stat().st_mtime)
    cfg = load_config("sources/outage_sheet")
    gq = project_gas_quality()
    return [load_snapshot(p, store, cfg=cfg, gas_quality=gq) for p in files]
