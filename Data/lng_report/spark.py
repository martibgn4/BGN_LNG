"""Spark Commodities API access: cargo, freight and NWE-discount quotes."""

from Utils.utils_spark import (
    retrieve_credentials,
    get_access_token,
    fetch_cargo_prices,
    fetch_freight_prices,
    list_contracts,
)

# Path to the Spark API client credentials CSV.
SPARK_CREDENTIALS_PATH = (
    "C:\\Marti\\python_tests_2\\BGN_LNG\\adhoc_scripts\\client_credentials.csv"
)

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
