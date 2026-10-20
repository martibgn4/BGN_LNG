"""Spark Commodities API access: cargo, freight and NWE-discount quotes."""

import os
from pathlib import Path

from Utils.utils_spark import (
    retrieve_credentials,
    get_access_token,
    fetch_cargo_prices,
    fetch_freight_prices,
    list_contracts,
)

from Data.lng_report.config import REPO_ROOT

# Where the Spark API client credentials come from, in precedence order:
#
#   1. SPARK_CREDENTIALS_PATH  -- an explicit CSV somewhere on this machine
#   2. <repo>/adhoc_scripts/client_credentials.local.csv -- a private copy
#      that .gitignore keeps out of the repo; use this to run under your own
#      Spark client rather than the desk's
#   3. <repo>/adhoc_scripts/client_credentials.csv -- the copy committed to the
#      repo, so a fresh clone has working Spark access with no setup
#
# Returning None instead falls through to the SPARK_CLIENT_ID /
# SPARK_CLIENT_SECRET environment variables, which is what
# ``retrieve_credentials`` does with no path.
_CREDENTIALS_DIR = REPO_ROOT / "adhoc_scripts"
LOCAL_CREDENTIALS_PATH = _CREDENTIALS_DIR / "client_credentials.local.csv"
SHIPPED_CREDENTIALS_PATH = _CREDENTIALS_DIR / "client_credentials.csv"


def resolve_spark_credentials_path():
    """The credentials CSV to read, or None to fall back to env vars."""
    override = os.getenv("SPARK_CREDENTIALS_PATH")
    if override:
        return override
    for candidate in (LOCAL_CREDENTIALS_PATH, SHIPPED_CREDENTIALS_PATH):
        if Path(candidate).is_file():
            return str(candidate)
    return None


SPARK_CREDENTIALS_PATH = resolve_spark_credentials_path()

# Spark contract ticker for each supported quote type.
_QUOTE_TYPE_CONFIG = {
    "Cargo": ("cargo", "sparknwe"),
    "NWEDiscountsFinancial": ("cargo", "sparknwe-fin-monthly"),
    "FreightSpark30": ("freight", "spark30fo"),
    "FreightSpark30Spot": ("freight", "spark30s"),
    "FreightSpark30FFA": ("freight", "spark30ffa-monthly"),
}


def extract_spark_quotes(quote_type="Cargo", latest_only=True, limit=90,
                         cal_month="Apr-2026", print_available_contacts=False):
    """Fetch a Spark price series for the given quote type.

    Returns the raw DataFrame from the matching Spark fetch helper.
    """
    client_id, client_secret = retrieve_credentials(file_path=SPARK_CREDENTIALS_PATH)
    access_token = get_access_token(client_id, client_secret)
    if print_available_contacts:
        print(list_contracts(access_token))

    if quote_type not in _QUOTE_TYPE_CONFIG:
        raise ValueError(f"Spark type {quote_type} not recognized")

    kind, ticker = _QUOTE_TYPE_CONFIG[quote_type]
    if kind == "cargo":
        return fetch_cargo_prices(access_token, ticker, limit,
                                  latest_only=latest_only, cal_month=cal_month)

    # Freight quotes share the vessel spec; the spot series ignores cal_month.
    freight_cal_month = None if quote_type == "FreightSpark30Spot" else cal_month
    return fetch_freight_prices(access_token, ticker, limit, my_vessel='174-2stroke',
                                latest_only=latest_only, cal_month=freight_cal_month)
