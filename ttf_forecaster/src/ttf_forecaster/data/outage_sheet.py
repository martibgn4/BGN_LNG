"""Gas and LNG outage spreadsheet: parse one snapshot into events and daily reductions (user request, 2026-10-20).

Each file is a SNAPSHOT of the user's Bloomberg export. The snapshot time is the file's modification time. The filename
date is not used: the user's file name is wrong (it says 20Sep26 and was downloaded 2026-10-20).

Columns (sheet 'Worksheet'): Region, Operator/Source, Facility, Start Date, Duration/End, Prod./Reduction (e.g.
"-19.8 mcm/day"), Last Update, notes (Planned/Unplanned text, "EVENT - <id>", a UMM link).

Sign convention: the sheet's Prod./Reduction is negative for a loss. ``reduction_mcmd`` = -(that number), so a positive
value is capacity lost in mcm/d. Gas days run from Start Date to Duration/End inclusive.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

COLUMNS = ("Region", "Operator/Source", "Facility", "Start Date", "Duration/End", "Prod./Reduction",
           "Last Update", "notes")
_DATE_FMT = "%d-%b-%Y"
_QTY = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*mcm\s*/\s*day\s*$", re.IGNORECASE)
_EVENT = re.compile(r"EVENT\s*-\s*([0-9A-Za-z-]+)")


@dataclass(frozen=True)
class Event:
    """One row of the sheet, parsed."""

    event_id: str
    region: str
    operator: str
    facility: str
    start: date
    end: date | None  # None when the sheet gives no end ('n.a.'): not in the gas-day aggregates
    reduction_mcmd: float | None  # positive = capacity lost (sheet sign reversed); None when not in mcm/d
    unit: str  # "mcm/day", "mtpa" (not converted here) or "none" (the sheet has no quantity)
    last_update: date | None
    category: str  # "planned" | "unplanned" | "unknown", parsed from the notes
    raw_text: str


def parse_date(text: object) -> date | None:
    if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() in ("", "n.a.", "n/a", "--"):
        return None
    if isinstance(text, datetime):
        return text.date()
    if isinstance(text, date):
        return text
    return datetime.strptime(str(text).strip(), _DATE_FMT).date()


def parse_reduction(text: object) -> tuple[float | None, str]:
    """'-19.8 mcm/day' -> (19.8, 'mcm/day'): the reduction, positive = lost. '-12.6 mtpa' -> (None, 'mtpa'): kept in
    the side table but not converted here (no LNG conversion is configured). '--' -> (None, 'none')."""
    text = str(text or "").strip()
    if text in ("", "--", "nan", "None"):
        return None, "none"
    m = _QTY.match(text)
    if m:
        return -float(m.group(1)), "mcm/day"
    m = re.match(r"^\s*[-+]?\d+(?:\.\d+)?\s*(mtpa)\s*$", text, re.IGNORECASE)
    if m:
        return None, "mtpa"
    raise ValueError(f"unparsed Prod./Reduction {text!r}")


def parse_category(notes: object) -> str:
    text = str(notes or "")
    if re.search(r"\bunplanned\b", text, re.IGNORECASE):
        return "unplanned"
    if re.search(r"\bplanned\b|maintenance", text, re.IGNORECASE):
        return "planned"
    return "unknown"


def event_id(notes: object, facility: str, start: date, end: date | None) -> str:
    """The sheet's 'EVENT - <id>' when present, else a stable hash of the event's key fields."""
    m = _EVENT.search(str(notes or ""))
    if m:
        return m.group(1)
    key = f"{facility}|{start.isoformat()}|{end.isoformat() if end else ''}"
    return "h" + hashlib.sha256(key.encode()).hexdigest()[:12]


def read_sheet(path: Path, sheet_name: str = "Worksheet") -> pd.DataFrame:
    """The sheet as a DataFrame of the expected columns. Refuses a file without them."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"{path.name}: no sheet {sheet_name!r} (sheets: {wb.sheetnames})")
    rows = list(wb[sheet_name].iter_rows(values_only=True))
    wb.close()
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    body = [r for r in rows[1:] if any(v not in (None, "") for v in r)]
    return pd.DataFrame(body, columns=header)


def parse_events(df: pd.DataFrame) -> list[Event]:
    """Parse every row. A row without a start, end or quantity is refused (never silently dropped)."""
    out: list[Event] = []
    for i, r in df.iterrows():
        facility = str(r["Facility"]).strip()
        start, end = parse_date(r["Start Date"]), parse_date(r["Duration/End"])
        if start is None or (end is not None and end < start):
            raise ValueError(f"row {i}: bad dates {r['Start Date']!r} .. {r['Duration/End']!r} for {facility}")
        notes = "" if pd.isna(r["notes"]) else str(r["notes"])
        reduction, unit = parse_reduction(r["Prod./Reduction"])
        out.append(Event(
            event_id=event_id(notes, facility, start, end),
            region=str(r["Region"]).strip(),
            operator=str(r["Operator/Source"]).strip(),
            facility=facility,
            start=start,
            end=end,
            reduction_mcmd=reduction,
            unit=unit,
            last_update=parse_date(r["Last Update"]),
            category=parse_category(notes),
            raw_text=notes,
        ))
    return out


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_time(path: Path) -> pd.Timestamp:
    """The file's modification time in UTC, to the second (the user's rule: mtime, not the filename)."""
    return pd.Timestamp(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)).floor("s")


def gas_days(ev: Event) -> list[date]:
    if ev.end is None:
        return []
    return [ev.start + timedelta(days=i) for i in range((ev.end - ev.start).days + 1)]


def daily_reductions(events: list[Event], through: date | None = None) -> pd.DataFrame:
    """One row per (event, gas day): the reduction on that day, mcm/d only (other units are left out here).
    ``through`` (optional) trims later days."""
    rows = []
    for ev in events:
        if ev.reduction_mcmd is None:
            continue
        for d in gas_days(ev):
            if through is not None and d > through:
                continue
            rows.append({"event_id": ev.event_id, "gas_day": d, "reduction_mcmd": ev.reduction_mcmd,
                         "category": ev.category, "facility": ev.facility, "region": ev.region,
                         "operator": ev.operator})
    return pd.DataFrame(rows, columns=["event_id", "gas_day", "reduction_mcmd", "category", "facility", "region",
                                       "operator"])
