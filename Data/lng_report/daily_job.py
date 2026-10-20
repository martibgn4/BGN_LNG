"""The desk's single morning job: build and send every daily LNG mail.

Three reports go out as three separate emails from one run:

    settlements -- the front-four-years settlement table
                   (``daily_market_data_reporting``)
    products    -- the refined products forward curves
                   (``oil_products_reporting``)
    report      -- the full BGN LNG chart report (``report``)

They are deliberately independent. Each has its own recipients and its own
failure: if Platts is down and the chart report dies, the settlement table
still lands in the desk's inbox, and vice versa. The job exits non-zero when
any task failed, so Task Scheduler shows a red run rather than a silent one.

That independence extends to imports, which is why each task imports its own
module inside its function rather than at the top of this file. The chart
report pulls in Platts and the plotting stack; importing it up here meant
``--only settlements`` died on a missing Platts entitlement before the
settlement table it was asked for ever ran.

Run everything and send:
    python -m Data.lng_report.daily_job

Build them all without sending anything:
    python -m Data.lng_report.daily_job --dry-run

Run just one of them:
    python -m Data.lng_report.daily_job --only settlements
"""

import argparse
import sys
import traceback
from datetime import datetime

DEFAULT_RECIPIENT = "marti.fernandezreal@bgn-int.com"


def _run_settlements(dry_run):
    # Recipients default to what the task module itself declares, so there is
    # still one place to change an address rather than two.
    # out_path is left to default: the dated copy on the shared drive, next to
    # the chart report's own dated copy.
    from Data.lng_report import daily_market_data_reporting

    daily_market_data_reporting.main(
        sending_to=daily_market_data_reporting.DEFAULT_RECIPIENTS,
        dry_run=dry_run,
    )


def _run_products(dry_run):
    # Same shape as the settlement table: the dated shared-drive copy and the
    # workbook attachment are both left to default.
    from Data.lng_report import oil_products_reporting

    oil_products_reporting.main(
        sending_to=oil_products_reporting.DEFAULT_RECIPIENTS,
        dry_run=dry_run,
    )


def _run_report(dry_run):
    from Data.lng_report import report

    report.main(sending_to=report.DEFAULT_RECIPIENTS, send=not dry_run)


# Order matters: the two tables are cheap and are what the desk reads first
# thing, so they go out before the slower chart report is built.
TASKS = [
    ("settlements", _run_settlements),
    ("products", _run_products),
    ("report", _run_report),
]


def run_daily_job(only=None, dry_run=False):
    """Run each task in turn, isolating failures. Returns the list of failures."""
    started = datetime.now()
    tasks = [(name, fn) for name, fn in TASKS if only is None or name == only]
    failures = []

    for name, task in tasks:
        print(f"\n{'=' * 60}\n{name}: starting\n{'=' * 60}")
        task_started = datetime.now()
        try:
            task(dry_run)
        except Exception:
            # One report failing must not stop the other from going out, so the
            # traceback is printed and the job carries on.
            print(f"{name}: FAILED\n{traceback.format_exc()}")
            failures.append(name)
        else:
            elapsed = (datetime.now() - task_started).total_seconds()
            print(f"{name}: done in {elapsed:.0f}s")

    elapsed = (datetime.now() - started).total_seconds()
    ran = [name for name, _ in tasks]
    succeeded = [name for name in ran if name not in failures]
    print(f"\n{'=' * 60}")
    print(f"Daily job finished in {elapsed:.0f}s "
          f"-- sent: {', '.join(succeeded) or 'none'}"
          f"{'; failed: ' + ', '.join(failures) if failures else ''}")
    print("=" * 60)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=[name for name, _ in TASKS], default=None,
                        help="run a single task instead of the whole job")
    parser.add_argument("--dry-run", action="store_true",
                        help="build every report without emailing anything")
    args = parser.parse_args()

    failures = run_daily_job(only=args.only, dry_run=args.dry_run)
    # Non-zero exit so a scheduled run that half-failed is visible as a failure.
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
