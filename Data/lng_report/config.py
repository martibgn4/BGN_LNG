"""Static configuration for the BGN LNG daily report.

Ticker maps, Bloomberg month codes and the FX tenor grid live here so the rest
of the package never has to hard-code a symbol.
"""

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

# --- Shared-drive output ------------------------------------------------------

# Where the desk's daily reports are published. Lives here rather than in
# ``report`` so a module needing only the path does not drag in the whole
# plotting stack to get it.
ONLINE_DIR = (r"C:\Users\marti.fernandezreal\BAYEGAN DIS TIC. A.S"
              r"\LNG Team - 01. Miscellaneous\17. LNG BGN Reports")
