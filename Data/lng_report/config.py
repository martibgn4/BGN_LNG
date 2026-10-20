"""Static configuration for the BGN LNG daily report.

Ticker maps, Bloomberg month codes and the FX tenor grid live here so the rest
of the package never has to hard-code a symbol. Machine-specific paths resolve
off ``REPO_ROOT`` or an environment variable for the same reason -- the report
has to run from a fresh clone on someone else's laptop, not just the desk's.
"""

import os
from pathlib import Path

# --- FX / commodity -> Bloomberg root ---------------------------------------

dict_fx_to_tickers = {
    "GBPUSD": "GBP",
    "EURUSD": "EUR",
    "EURGBP": "EURGBP",
}

# A tuple means (month, quarter, season, year) roots differ; a bare string means
# only a monthly root exists.
dict_comm_to_tickers = {
    "HenryHub": "NG",
    "HH": "NG",
    "JKM": "JKL",
    "TTF": ("TZT", "QZT", "QQT", "QTT"),
    "TFU": ("TMR", "TQR", "TSR", "TYR"),
    "Brent": "CO",
    "NBP": ("FN", "QR", "SA", "YAA"),
    "THE": ("NCG", "NCG", "NCG", "NCG"),
    "PEG": ("PNG", "PNG", "PNG", "PNG"),
    "PSV": ("PSR", "PSR", "PSR", "PSR"),
    "PVB": ("PXB", "PXB", "PXB", "PXB"),
    "VTP": ("CEG", "CEG", "CEG", "CEG"),
    "BLNG1": "IKA",
    "BLNG2": "IKD",
    "BLNG3": "IKI",
}

# Underlyings whose Bloomberg tickers need a period-type suffix (M/Q/S/Y).
add_suffix_underlyings = ["THE", "PEG", "PSV", "PVB", "VTP"]

# --- FX forward tenor grid ---------------------------------------------------

bbg_months_available_fx = [
    "SP", "1M", "2M", "3M", "4M", "5M", "6M", "9M", "1Y", "15M",
    "18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y",
]
bbg_months_available_fx_int = [
    0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120,
]
dict_bbg_months_available_fx_to_int = dict(
    zip(bbg_months_available_fx, bbg_months_available_fx_int)
)

# --- Bloomberg futures month codes ------------------------------------------

bbg_month_codes = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
bbg_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

bbg_dict = dict(zip(bbg_months, bbg_month_codes))
bbg_dict_code_to_month = dict(zip(bbg_month_codes, bbg_months))

# --- Repo layout --------------------------------------------------------------

# The checkout this file lives in: <repo>/Data/lng_report/config.py -> <repo>.
# Every default path below is derived from this rather than from one person's
# drive letter, so a fresh clone works wherever it lands.
REPO_ROOT = Path(__file__).resolve().parents[2]


# --- Report output ------------------------------------------------------------

# Where the desk's daily reports are published. Lives here rather than in
# ``report`` so a module needing only the path does not drag in the whole
# plotting stack to get it.
#
# The desk publishes to a OneDrive folder that only exists on a machine signed
# in to the BGN tenant with that library synced. A teammate running this from a
# fresh clone has no such folder, and writing into a parent that does not exist
# used to kill the whole run before the mail went out -- so the path resolves in
# three steps and always ends somewhere writable:
#
#   1. BGN_LNG_REPORT_DIR, if set -- the explicit override
#   2. the shared drive, when it is actually present (the desk's own machines)
#   3. <repo>/output, created on demand (anyone else)
SHARED_REPORT_DIR = (r"C:\Users\marti.fernandezreal\BAYEGAN DIS TIC. A.S"
                     r"\LNG Team - 01. Miscellaneous\17. LNG BGN Reports")

LOCAL_REPORT_DIR = REPO_ROOT / "output"


def resolve_report_dir():
    """The directory today's reports should be written to."""
    override = os.getenv("BGN_LNG_REPORT_DIR")
    if override:
        return override
    if Path(SHARED_REPORT_DIR).is_dir():
        return SHARED_REPORT_DIR
    return str(LOCAL_REPORT_DIR)


ONLINE_DIR = resolve_report_dir()
