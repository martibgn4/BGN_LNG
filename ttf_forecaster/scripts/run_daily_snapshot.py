"""Daily snapshot: today's BSRCH power-outage snapshot (FR, DE, NORDPOOL) and today's BQL weather forecasts.

    python scripts/run_daily_snapshot.py                       # outages + BQL weather (BQL skipped without Excel)
    python scripts/run_daily_snapshot.py --skip-bql            # outages only
    python scripts/run_daily_snapshot.py --extra-as-of 2026-10-13   # also a past as-of snapshot (honoured from 2026-10-13)

Idempotent: a second run on the same day adds nothing (raw batches are named by request date). Writes only to the
git-ignored data/live store.

Authorisation: live calls run through a temporary copy of configs/ with ``live_calls_authorised: true``. The committed
configs are never changed. The user's authorisation covers ``known`` series only (the outage registry and the BQL
weather registry). Scheduling is the user's decision: see docs/bloomberg_discovery.md section 8 for the Windows Task
Scheduler command, which is NOT created by this script.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ttf_forecaster.data import bql_excel  # noqa: E402
from ttf_forecaster.data.bloomberg import BloombergConfig  # noqa: E402
from ttf_forecaster.data.outages import LiveOutageAdapter  # noqa: E402
from ttf_forecaster.data.storage import ParquetStore  # noqa: E402
from ttf_forecaster.pipelines.daily_snapshot import (  # noqa: E402
    TerminalNotConnected,
    check_terminal,
    excel_available,
    run_daily,
)
from ttf_forecaster.pipelines import fleet  # noqa: E402
from ttf_forecaster.pipelines.outages import pull_snapshot  # noqa: E402
from ttf_forecaster.settings import load_config  # noqa: E402


def authorised_config_dir() -> Path:
    """A temporary copy of configs/ with live calls authorised. The committed configs are left unchanged."""
    tmp = Path(tempfile.mkdtemp(prefix="ttf_live_cfg_daily_"))
    cfg_dir = tmp / "configs"
    shutil.copytree(ROOT / "configs", cfg_dir, ignore=shutil.ignore_patterns("local*"))
    (cfg_dir / "local_data_sources.yaml").write_text(
        "live_calls_authorised: true  # temp copy for the daily snapshot; committed configs unchanged\n",
        encoding="utf-8")
    load_config.cache_clear()
    return cfg_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-bql", action="store_true", help="do not pull BQL weather")
    ap.add_argument("--skip-fleet", action="store_true", help="do not take the LNG fleet snapshot (BSRCH COMDTY:SHIP)")
    ap.add_argument("--extra-as-of", action="append", default=[], metavar="YYYY-MM-DD",
                    help="also take an outage snapshot for this past as-of date (repeatable)")
    args = ap.parse_args(argv)

    try:
        check_terminal()
    except TerminalNotConnected as exc:
        print(f"ERROR {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC: {exc}", file=sys.stderr, flush=True)
        return 2

    today = datetime.now(timezone.utc).date()
    request_time = datetime.now(timezone.utc)
    store = ParquetStore(ROOT / "data" / "live")
    cfg_dir = authorised_config_dir()
    outage_cfg = BloombergConfig.load(str(cfg_dir))
    adapter = LiveOutageAdapter(outage_cfg)

    ts = pd.Timestamp(request_time)
    report: dict[str, object] = {}
    for extra in args.extra_as_of:
        as_of = date.fromisoformat(extra)
        report[f"outages as-of {as_of}"] = pull_snapshot(as_of, store, adapter, request_time=ts)
    bql_runner = None
    if not args.skip_bql:
        bql_runner = bql_excel.run_queries if excel_available() else None
    daily = run_daily(today, store, adapter, request_time=ts, bql_runner=bql_runner)
    report.update({f"today {k}": v for k, v in daily.items()})
    if args.skip_bql:
        report["today bql_weather"] = "skipped: --skip-bql"
    if not args.skip_fleet:
        # LNG fleet snapshot: Python API only (no Excel). Idempotent per UTC day inside run_snapshot.
        from xbbg import blp  # noqa: PLC0415 - only needed for this step

        report["today lng_fleet"] = fleet.run_snapshot(store, lambda domain, options: blp.bsrch(domain, options),
                                                       request_time=ts)
    else:
        report["today lng_fleet"] = "skipped: --skip-fleet"
    for k, v in report.items():
        print(f"{k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
