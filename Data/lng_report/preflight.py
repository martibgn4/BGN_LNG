"""Check this machine can run the daily settlement mail, before trying to send it.

The settlement snapshot pulls from three places and delivers through a fourth,
and each one fails differently on a fresh clone: Bloomberg needs a non-PyPI
package plus a logged-in terminal, Spark needs credentials, the JKM master is
not in the repo, and delivery goes through the local Outlook. Finding that out
one traceback at a time is slow, so this checks the lot in one pass and prints
the fix next to anything that is missing.

Run it first on any new machine:
    python -m Data.lng_report.preflight

Exit code is 0 when the mail can be built and sent, 1 when something essential
is missing. Warnings do not fail the run -- they mean a column comes back blank
or the copy is published somewhere other than the shared drive.
"""

import importlib
import os
import platform
import sys
from datetime import date
from pathlib import Path

# Checks are grouped by what failing actually costs you. FAIL means no mail at
# all; WARN means the mail still goes out, just missing a column or a file.
OK, WARN, FAIL = "OK", "WARN", "FAIL"

# pip-installable imports the settlement path needs, as (module, pip name).
REQUIRED_PACKAGES = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("openpyxl", "openpyxl"),
    ("requests", "requests"),
    ("xbbg", "xbbg"),
]

BLPAPI_INSTALL = (
    "pip install --index-url="
    "https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi"
)


def _check_python():
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 9):
        return FAIL, f"Python {version}", "Python 3.9 or newer is required"
    return OK, f"Python {version} on {platform.system()}", ""


def _check_packages():
    """Every pip-installable import the settlement path needs."""
    results = []
    for module, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            results.append((FAIL, f"{module} missing ({exc.__class__.__name__})",
                            "pip install -r requirements.txt"))
        else:
            results.append((OK, f"{module} importable", ""))
    return results


def _check_blpapi():
    """Bloomberg's own package -- pip cannot reach it from PyPI."""
    try:
        import blpapi
    except Exception:
        return FAIL, "blpapi not installed", BLPAPI_INSTALL
    version = getattr(blpapi, "__version__", "unknown")
    return OK, f"blpapi {version}", ""


def _check_terminal():
    """A logged-in Bloomberg Terminal on this machine, reachable from Python."""
    try:
        from xbbg import blp
    except Exception as exc:
        return FAIL, f"cannot import xbbg ({exc.__class__.__name__})", \
            "pip install -r requirements.txt, then install blpapi"

    try:
        quotes = blp.bdp(tickers=["CO1 Comdty"], flds=["px_last"])
    except Exception as exc:
        return FAIL, f"Bloomberg request failed ({exc.__class__.__name__}: {exc})", \
            "Open the Bloomberg Terminal and log in, then re-run this check"

    if quotes.empty or "px_last" not in quotes:
        return FAIL, "Bloomberg returned no data for CO1 Comdty", \
            "Check the terminal is logged in and entitled to futures data"
    return OK, f"Bloomberg reachable (Brent front = {quotes.iloc[0, 0]})", ""


def _check_spark():
    """Spark credentials resolve, and the API actually accepts them."""
    try:
        from Data.lng_report.spark import (
            SPARK_CREDENTIALS_PATH,
            LOCAL_CREDENTIALS_PATH,
            SHIPPED_CREDENTIALS_PATH,
        )
        from Utils.utils_spark import retrieve_credentials, get_access_token
    except Exception as exc:
        return FAIL, f"cannot import Spark helpers ({exc.__class__.__name__}: {exc})", \
            "Run from the repo root so `Data` and `Utils` are importable"

    if SPARK_CREDENTIALS_PATH is None and not os.getenv("SPARK_CLIENT_ID"):
        return FAIL, "no Spark credentials found", (
            f"Expected a CSV at {SHIPPED_CREDENTIALS_PATH} (ships with the repo), "
            f"or put your own at {LOCAL_CREDENTIALS_PATH}, or set "
            "SPARK_CLIENT_ID and SPARK_CLIENT_SECRET"
        )

    source = SPARK_CREDENTIALS_PATH or "SPARK_CLIENT_ID / SPARK_CLIENT_SECRET env vars"
    try:
        client_id, client_secret = retrieve_credentials(file_path=SPARK_CREDENTIALS_PATH)
        get_access_token(client_id, client_secret)
    except Exception as exc:
        return FAIL, f"Spark rejected the credentials ({exc.__class__.__name__}: {exc})", (
            "The credentials may have been rotated. Ask the desk for a current "
            "client id/secret and set SPARK_CREDENTIALS_PATH at your own copy"
        )
    return OK, f"Spark authenticated using {source}", ""


def _check_jkm():
    """The JKM settlement master, which is not in the repo."""
    try:
        from Data.lng_report.daily_market_data_reporting import resolve_jkm_master_path
    except Exception as exc:
        return WARN, f"cannot resolve the JKM path ({exc.__class__.__name__})", ""

    missing_fix = (
        "The JKM column will be blank and the mail will say so. To fill it, "
        "set JKM_MASTER_PATH or drop a copy at Data/jkm_settlements/master.csv"
    )

    path = resolve_jkm_master_path()
    if path is None:
        return WARN, "no JKM settlement master on this machine", missing_fix

    # An explicit JKM_MASTER_PATH is honoured even when it is wrong, so that a
    # typo surfaces here rather than silently falling back to another copy.
    if not Path(path).is_file():
        return WARN, f"JKM_MASTER_PATH points at {path}, which does not exist", \
            missing_fix
    return OK, f"JKM master at {path}", ""


def _check_outlook():
    """Delivery goes through the Outlook client on this machine."""
    if platform.system() != "Windows":
        return FAIL, f"{platform.system()} cannot drive Outlook via COM", \
            "The mail is sent through local Outlook, so this needs Windows"
    try:
        import win32com.client
    except Exception:
        return FAIL, "pywin32 not installed", "pip install pywin32"
    try:
        win32com.client.Dispatch("Outlook.Application")
    except Exception as exc:
        return FAIL, f"cannot reach Outlook ({exc.__class__.__name__})", \
            "Install and sign in to the Outlook desktop client, then re-run"
    return OK, "Outlook reachable", ""


def _check_output_dir():
    """Wherever the published copy lands, it has to be writable."""
    try:
        from Data.lng_report.config import (
            ONLINE_DIR, SHARED_REPORT_DIR, LOCAL_REPORT_DIR,
        )
    except Exception as exc:
        return FAIL, f"cannot import config ({exc.__class__.__name__})", ""

    target = Path(ONLINE_DIR)
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / f".preflight_{date.today().isoformat()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return WARN, f"{target} is not writable ({exc.__class__.__name__})", (
            "The mail still goes out; only the published copy is skipped. "
            "Set BGN_LNG_REPORT_DIR to a folder you can write to"
        )

    if str(target) == str(LOCAL_REPORT_DIR):
        return OK, f"publishing to {target} (shared drive not mounted here)", ""
    if str(target) == SHARED_REPORT_DIR:
        return OK, f"publishing to the desk's shared drive", ""
    return OK, f"publishing to {target} (BGN_LNG_REPORT_DIR)", ""


# Order matters: cheap local checks first, so an obvious miss is reported before
# we spend time on a Bloomberg round trip.
CHECKS = [
    ("python", _check_python),
    ("blpapi", _check_blpapi),
    ("bloomberg terminal", _check_terminal),
    ("spark api", _check_spark),
    ("jkm master", _check_jkm),
    ("outlook", _check_outlook),
    ("output dir", _check_output_dir),
]

_MARK = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


def run_preflight():
    """Run every check, print a report, and return the list of failures."""
    print("Preflight for the daily settlement mail\n" + "=" * 60)

    results = []
    for name, check in [("packages", _check_packages)] + CHECKS:
        if name == "packages":
            for status, detail, fix in check():
                results.append((name, status, detail, fix))
            continue
        try:
            status, detail, fix = check()
        except Exception as exc:
            status, detail, fix = FAIL, f"check itself failed ({exc!r})", ""
        results.append((name, status, detail, fix))

    for name, status, detail, fix in results:
        print(f"{_MARK[status]} {name:<20} {detail}")
        if fix:
            print(f"{'':7}{'':<20} -> {fix}")

    failures = [name for name, status, _, _ in results if status == FAIL]
    warnings = [name for name, status, _, _ in results if status == WARN]

    print("=" * 60)
    if failures:
        print(f"NOT READY -- {len(failures)} blocking: {', '.join(failures)}")
        print("Fix the arrows above, then re-run this check.")
    else:
        print("READY -- the settlement mail can be built and sent on this machine.")
        print("Build it without sending:  "
              "python -m Data.lng_report.daily_market_data_reporting --dry-run")
    if warnings:
        print(f"({len(warnings)} warning: {', '.join(warnings)} "
              f"-- the mail still sends, see above for what is degraded)")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run_preflight() else 0)
