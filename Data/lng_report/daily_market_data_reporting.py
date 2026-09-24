"""Daily settlement snapshot for the LNG desk: one table, emailed each morning.

Rows are the front four years of monthly contracts (delivery months only);
columns are the latest settlement for each benchmark the desk watches:

    TTF, Brent, Henry Hub            -- Bloomberg (``px_settle``)
    JKM                              -- the manually-processed JKM settlement master
    NWE discounts, Spark30, Spark25  -- Spark Commodities API

Each source settles on its own clock, so the "as of" date is shown per column
rather than assumed to be today. The Spark curves only run about 16 months out,
so their cells are blank beyond that -- that is the market, not a data failure.

Run as a script to build and email today's snapshot:
    python -m Data.lng_report.daily_market_data_reporting
Add ``--dry-run`` to build the table without sending anything.
"""

import argparse
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xbbg import blp

from Utils.datetime_utils import date_tenor_from_label
from Utils.utils_spark import (
    retrieve_credentials,
    get_access_token,
    fetch_latest_price_releases,
)

from Data.lng_report.bloomberg import BloombergExtractor
from Data.lng_report.config import ONLINE_DIR
from Data.lng_report.emailer import send_email
from Data.lng_report.spark import SPARK_CREDENTIALS_PATH

# --- What the table covers ---------------------------------------------------

# Front four years of monthly contracts. The first row is the month *after* the
# current one: by the time this goes out in the morning the running month is no
# longer a forward contract on any of these benchmarks.
N_MONTHS = 48

# Bloomberg field carrying the settlement price and the date it settled on.
BBG_SETTLE_FIELD = "px_settle"
BBG_SETTLE_DATE_FIELD = "px_settle_last_dt"

# --- TTF unit conversion -----------------------------------------------------

# 1 MWh = 3.6 GJ, 1 MMBtu = 1.055056 GJ.
MMBTU_PER_MWH = 3.412141633

# EUR/USD spot plus the forward tenor grid, quoted in pips over spot. Bloomberg
# has no "EUR1Y" -- the twelve-month point is "EUR12M" -- and switches from
# month to year form after 18M. The grid reaches 5Y, so every month in a
# four-year strip interpolates rather than extrapolates.
FX_SPOT_TICKER = "EUR BGN Curncy"
FX_FORWARD_TENORS = [
    ("1M", 1), ("2M", 2), ("3M", 3), ("4M", 4), ("5M", 5), ("6M", 6),
    ("9M", 9), ("12M", 12), ("15M", 15), ("18M", 18),
    ("2Y", 24), ("3Y", 36), ("4Y", 48), ("5Y", 60),
]
FX_PIPS_PER_UNIT = 10000
FX_DATE_FIELD = "last_update_dt"

# CSV built up from the manually-uploaded JKM settlement screenshots.
JKM_MASTER_PATH = "C:\\Marti\\ClaudeProjects\\data\\raw\\jkm_settlements\\master.csv"

# Spark contract ids (see ``list_contracts``) and which derived price to read.
SPARK_NWE_DISCOUNT = ("sparknwe-fin-monthly", "usdPerMMBtu")
SPARK30_FREIGHT = ("spark30ffa-monthly", "usdPerDay")
SPARK25_FREIGHT = ("spark25ffa-monthly", "usdPerDay")

# Spot route assessments. They carry no monthly curve, so they contribute no
# column -- they are here only for their publishing date, which runs on a
# different clock to the curve and is worth seeing separately.
SPARK30_SPOT = "spark30s"
SPARK25_SPOT = "spark25s"

# Column order as it appears in the email, with display units and decimals.
# ``TTF`` is the ICE Endex contract in EUR/MWh -- point BBG_COMMODITIES at "TFU"
# instead for the USD/MMBtu quote.
COLUMNS = [
    ("TTF", "EUR/MWh", 3),
    ("TTF_USD", "$/MMBtu", 3),
    ("JKM", "$/MMBtu", 3),
    ("Brent", "$/bbl", 2),
    ("Henry Hub", "$/MMBtu", 3),
    ("NWE Discount", "$/MMBtu", 3),
    ("Spark30", "$/day", 0),
    ("Spark25", "$/day", 0),
    ("BLNG1", "$/day", 0),
    ("BLNG2", "$/day", 0),
    ("BLNG3", "$/day", 0),
]

# The Baltic LNG freight routes settle on Bloomberg like the energy curves, but
# only run to Dec-29 -- the 2030 rows are blank because nothing is quoted there.
BBG_COMMODITIES = {
    "TTF": "TTF", "Brent": "Brent", "Henry Hub": "HH",
    "BLNG1": "BLNG1", "BLNG2": "BLNG2", "BLNG3": "BLNG3",
}

# The "last published" block above the table, one line per group. Spark spot and
# curve assessments release on different schedules, so they are dated apart.
PUBLICATION_ROWS = [
    ("", ["TTF", "JKM", "Brent", "Henry Hub", "Baltic"]),
    ("", ["NWE Discount", "Spark30 spot", "Spark30 curve (FFA)",
               "Spark25 spot", "Spark25 curve (FFA)"]),
]

DEFAULT_RECIPIENTS = "lng@bgn-int.com; vasileios.giannoutsos@bgn-int.com; "

# A dated copy of the rendered table is published to the shared drive next to
# the chart report, so the last send can be eyeballed after the fact without
# re-running the pull.
ONLINE_FILENAME = "market_curves_{pricing_date}.html"

# The same table as a workbook, published alongside the HTML and attached to
# the email so the desk can pull the numbers straight into a model.
ONLINE_XLSX_FILENAME = "market_curves_{pricing_date}.xlsx"


def online_html_path(pricing_date):
    """Shared-drive path for the table published on `pricing_date`."""
    return f"{ONLINE_DIR}\\{ONLINE_FILENAME.format(pricing_date=pricing_date)}"


def online_xlsx_path(pricing_date):
    """Shared-drive path for the workbook published on `pricing_date`."""
    return f"{ONLINE_DIR}\\{ONLINE_XLSX_FILENAME.format(pricing_date=pricing_date)}"

# --- Row axis ----------------------------------------------------------------


def _add_months(month, n):
    """Return the first of the month `n` months after `month`."""
    total = (month.year * 12 + month.month - 1) + n
    return date(total // 12, total % 12 + 1, 1)


def front_months(today=None, n_months=N_MONTHS):
    """The `n_months` delivery months making up the table's rows."""
    today = today or date.today()
    first = _add_months(date(today.year, today.month, 1), 1)
    return [_add_months(first, i) for i in range(n_months)]


# --- Bloomberg ---------------------------------------------------------------


def bloomberg_settlements(commodity, months):
    """Latest settlements for `commodity` over `months`.

    Returns ``(series indexed by delivery month, settlement date)``. The
    settlement date is the one most contracts agree on -- an illiquid far month
    can lag, but the strip as a whole marks to a single day.
    """
    start_year = months[0].year % 100
    years = months[-1].year - months[0].year + 1
    extractor = BloombergExtractor(commodity, start_year=start_year, years=years)
    tickers, labels = extractor.get_monthly_tickers()

    bbg_tickers = [t + " Comdty" for t in tickers]
    label_of = dict(zip(bbg_tickers, labels))

    quotes = blp.bdp(tickers=bbg_tickers,
                     flds=[BBG_SETTLE_FIELD, BBG_SETTLE_DATE_FIELD])

    wanted = set(months)
    prices, settle_dates = {}, []
    for ticker, row in quotes.iterrows():
        month = date_tenor_from_label(label_of[ticker])
        if month not in wanted:
            continue
        price = row.get(BBG_SETTLE_FIELD)
        if pd.notna(price):
            prices[month] = float(price)
        settle_date = row.get(BBG_SETTLE_DATE_FIELD)
        if pd.notna(settle_date):
            settle_dates.append(pd.Timestamp(settle_date).date())

    as_of = pd.Series(settle_dates).mode().iloc[0] if settle_dates else None
    return pd.Series(prices, dtype=float), as_of


def eurusd_forward_curve(months, pricing_date=None):
    """Outright EUR/USD forward for each delivery month.

    Returns ``(series indexed by delivery month, quote date)``. Bloomberg quotes
    the forwards as pips over spot on a coarse tenor grid, so the outrights are
    interpolated linearly in time onto the monthly strip -- straight-line
    between quoted tenors, flat past the last one.
    """
    pricing_date = pricing_date or date.today()

    tenor_tickers = {f"EUR{tenor} BGN Curncy": n_months
                     for tenor, n_months in FX_FORWARD_TENORS}
    quotes = blp.bdp(tickers=[FX_SPOT_TICKER] + list(tenor_tickers),
                     flds=["px_last", FX_DATE_FIELD])

    if FX_SPOT_TICKER not in quotes.index:
        raise ValueError("No EUR/USD spot returned by Bloomberg")
    spot = float(quotes.loc[FX_SPOT_TICKER, "px_last"])

    # Spot is the zero-month point; each forward is spot plus its pips.
    grid_months, grid_rates = [0.0], [spot]
    for ticker, n_months in tenor_tickers.items():
        if ticker not in quotes.index:
            continue
        pips = quotes.loc[ticker, "px_last"]
        if pd.notna(pips):
            grid_months.append(float(n_months))
            grid_rates.append(spot + float(pips) / FX_PIPS_PER_UNIT)

    order = np.argsort(grid_months)
    grid_months = np.asarray(grid_months)[order]
    grid_rates = np.asarray(grid_rates)[order]

    # Each delivery month is priced at its midpoint, matching how the rest of
    # the codebase floats a monthly tenor (see ForwardCurve).
    tenors = [(date(m.year, m.month, 15) - pricing_date).days / 365.25 * 12
              for m in months]
    rates = np.interp(tenors, grid_months, grid_rates)

    quote_dates = quotes[FX_DATE_FIELD].dropna()
    as_of = pd.Timestamp(quote_dates.mode().iloc[0]).date() if len(quote_dates) else None
    return pd.Series(dict(zip(months, rates)), dtype=float), as_of


# --- JKM ---------------------------------------------------------------------


def jkm_settlements(months, master_path=JKM_MASTER_PATH):
    """Latest JKM settlements from the screenshot-derived master CSV.

    Returns ``(series indexed by delivery month, settlement date)``. The master
    also carries balance-of-month rows (``bal:Oct``); this table is monthly
    contracts only, so they are dropped.
    """
    master = pd.read_csv(master_path)
    last_date = master.settlement_date.max()
    latest = master[master.settlement_date == last_date]
    is_calendar_month = latest.month_iso.astype(str).str.fullmatch(r"\d{4}-\d{2}")
    latest = latest[is_calendar_month]

    wanted = set(months)
    prices = {}
    for month_iso, settlement in zip(latest.month_iso, latest.settlement):
        month = datetime.strptime(month_iso, "%Y-%m").date()
        if month in wanted:
            prices[month] = float(settlement)

    as_of = datetime.strptime(last_date, "%Y-%m-%d").date()
    return pd.Series(prices, dtype=float), as_of


# --- Spark -------------------------------------------------------------------


def spark_access_token():
    """Authenticate once and reuse the token across every Spark curve."""
    client_id, client_secret = retrieve_credentials(file_path=SPARK_CREDENTIALS_PATH)
    return get_access_token(client_id, client_secret)


def spark_release_date(access_token, contract_id):
    """Publishing date of the latest release for `contract_id`, nothing else.

    Used for the spot assessments, which have no monthly curve to put in the
    table but whose release cadence the desk still wants to see.
    """
    release = fetch_latest_price_releases(access_token, contract_id)
    return datetime.strptime(release["releaseDate"], "%Y-%m-%d").date()


def spark_settlements(access_token, contract_id, price_key, months):
    """Latest monthly Spark curve for `contract_id`.

    Returns ``(series indexed by delivery month, release date)``. `price_key`
    picks the derived price: ``usdPerMMBtu`` for the NWE discount, ``usdPerDay``
    for the freight FFAs.
    """
    release = fetch_latest_price_releases(access_token, contract_id)
    as_of = datetime.strptime(release["releaseDate"], "%Y-%m-%d").date()

    wanted = set(months)
    prices = {}
    for block in release["data"]:
        for data_point in block["dataPoints"]:
            period = data_point["deliveryPeriod"]
            if period.get("type") != "month":
                continue
            month = date.fromisoformat(period["startAt"]).replace(day=1)
            if month not in wanted:
                continue
            derived = data_point["derivedPrices"].get(price_key) or {}
            if derived.get("spark") is not None:
                prices[month] = float(derived["spark"])

    return pd.Series(prices, dtype=float), as_of


# --- Table assembly ----------------------------------------------------------


def _collect(label, loader, columns, as_of, as_of_label=None):
    """Run one source's loader, recording its column and as-of date.

    A source that is down must not hold up the morning mail, so a failure
    leaves that column blank and is flagged in the email header. `as_of_label`
    lets a column be dated under a different name -- the Spark freight columns
    are curves, and say so, to sit alongside their spot counterparts.
    """
    try:
        series, source_date = loader()
    except Exception as exc:
        print(f"{label} skipped: {exc!r}")
        columns[label] = pd.Series(dtype=float)
        as_of[as_of_label or label] = None
        return
    columns[label] = series
    as_of[as_of_label or label] = source_date


def _collect_release_date(label, loader, as_of):
    """Record one source's publishing date where there is no column to fill."""
    try:
        as_of[label] = loader()
    except Exception as exc:
        print(f"{label} publishing date skipped: {exc!r}")
        as_of[label] = None


def build_market_table(today=None):
    """Build the settlement table.

    Returns ``(DataFrame indexed by delivery month, {column: as-of date})``.
    """
    months = front_months(today)
    columns, as_of = {}, {}

    for label, commodity in BBG_COMMODITIES.items():
        _collect(label, lambda c=commodity: bloomberg_settlements(c, months),
                 columns, as_of)

    # The three Baltic routes mark on one clock, so they share a single entry in
    # the published-dates header rather than repeating the same date three times.
    as_of["Baltic"] = as_of.get("BLNG1")

    _collect("JKM", lambda: jkm_settlements(months), columns, as_of)

    # TTF_USD is derived rather than pulled: the EUR/MWh settlement converted at
    # the EUR/USD forward for that same delivery month, so the USD curve carries
    # the FX term structure rather than a single spot rate.
    try:
        fx, as_of["EUR/USD"] = eurusd_forward_curve(months, pricing_date=today)
        columns["TTF_USD"] = columns["TTF"] / MMBTU_PER_MWH * fx
    except Exception as exc:
        print(f"TTF_USD skipped, EUR/USD curve unavailable: {exc!r}")
        columns["TTF_USD"] = pd.Series(dtype=float)
        as_of["EUR/USD"] = None

    try:
        token = spark_access_token()
    except Exception as exc:
        print(f"Spark authentication failed, all Spark columns skipped: {exc!r}")
        token = None

    spark_curves = [
        ("NWE Discount", "NWE Discount", SPARK_NWE_DISCOUNT),
        ("Spark30", "Spark30 curve (FFA)", SPARK30_FREIGHT),
        ("Spark25", "Spark25 curve (FFA)", SPARK25_FREIGHT),
    ]
    for label, as_of_label, (contract_id, price_key) in spark_curves:
        if token is None:
            columns[label] = pd.Series(dtype=float)
            as_of[as_of_label] = None
            continue
        _collect(
            label,
            lambda c=contract_id, k=price_key: spark_settlements(token, c, k, months),
            columns, as_of, as_of_label=as_of_label,
        )

    spark_spots = [("Spark30 spot", SPARK30_SPOT), ("Spark25 spot", SPARK25_SPOT)]
    for label, contract_id in spark_spots:
        if token is None:
            as_of[label] = None
            continue
        _collect_release_date(
            label, lambda c=contract_id: spark_release_date(token, c), as_of
        )

    table = pd.DataFrame(
        {label: columns[label].reindex(months) for label, _, _ in COLUMNS},
        index=months,
    )
    return table, as_of


# --- Rendering ---------------------------------------------------------------

_BG = "#0D1117"
_FG = "#C9D1D9"
_BORDER = "#30363D"
_ALT_ROW = "#161B22"
_MUTED = "#8B949E"
_ALERT = "#F85149"


def _format_value(value, decimals):
    if pd.isna(value):
        return ""
    return f"{value:,.{decimals}f}"


def _as_of_label(label, source_date):
    if source_date is None:
        return f'<span style="color:{_ALERT};">{label}: unavailable</span>'
    return f'{label}: <b>{source_date.strftime("%d-%b-%Y")}</b>'


def _publication_rows_html(as_of):
    """The "last published" block: one line per publication group."""
    lines = []
    for group, labels in PUBLICATION_ROWS:
        dates = " &nbsp;|&nbsp; ".join(
            _as_of_label(label, as_of.get(label)) for label in labels
        )
        # A group may be unnamed -- then the line is just the dates.
        heading = f"Last published &middot; {group}" if group else "Last published"
        lines.append(
            f'<div style="margin:2px 0;">'
            f'<span style="color:{_MUTED};">{heading}</span> '
            f'&mdash; {dates}</div>'
        )
    return "\n        ".join(lines)


def render_market_table_html(table, as_of, report_date):
    """Render the table as inline-styled HTML that survives Outlook."""
    published = _publication_rows_html(as_of)

    header_cells = "".join(
        f'<th style="border:1px solid {_BORDER}; padding:4px 8px; text-align:right; '
        f'white-space:nowrap;">{label}<br>'
        f'<span style="color:{_MUTED}; font-weight:normal; font-size:11px;">{units}</span></th>'
        for label, units, _ in COLUMNS
    )

    rows = []
    for month in table.index:
        # Shade alternate years so a four-year strip stays readable at a glance.
        background = _ALT_ROW if month.year % 2 else _BG
        cells = "".join(
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px; text-align:right; '
            f'white-space:nowrap;">{_format_value(table.loc[month, label], decimals)}</td>'
            for label, _, decimals in COLUMNS
        )
        rows.append(
            f'<tr style="background-color:{background};">'
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px; white-space:nowrap;">'
            f'{month.strftime("%b-%y")}</td>{cells}</tr>'
        )
    body = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="background-color:{_BG}; color:{_FG}; font-family:Arial, sans-serif; margin:10px;">
    <h1 style="text-align:center; font-size:20px; margin:8px 0;">
        BGN LNG Desk: Settlements as of {report_date.strftime('%d-%b-%Y')}
    </h1>
    <div style="text-align:center; font-size:12px; color:{_FG}; margin:4px 0 12px 0;">
        {published}
    </div>
    <table cellpadding="0" cellspacing="0" border="0"
           style="border-collapse:collapse; font-size:12px; margin:0 auto; color:{_FG};">
        <thead>
            <tr style="background-color:{_ALT_ROW};">
                <th style="border:1px solid {_BORDER}; padding:4px 8px; text-align:left;">Contract</th>
                {header_cells}
            </tr>
        </thead>
        <tbody>
{body}
        </tbody>
    </table>
    <p style="font-size:11px; color:{_MUTED}; margin-top:12px;">
        Monthly contracts only, front {len(table)} months. Blank cells are tenors the
        source does not quote &mdash; the Spark curves run about 16 months out.
    </p>
</body>
</html>"""


# --- Workbook ----------------------------------------------------------------

# Excel number formats matching each column's display decimals. Freight is in
# whole dollars per day, so it gets a thousands separator and no decimals.
_XL_NUMBER_FORMATS = {0: "#,##0", 2: "#,##0.00", 3: "#,##0.000"}

_XL_HEADER_FILL = "FF161B22"
_XL_HEADER_FONT = "FFFFFFFF"


def write_market_table_xlsx(table, as_of, report_date, xlsx_path):
    """Write the settlement table to `xlsx_path` as a formatted worksheet.

    Values are written as real numbers and delivery months as real dates, so
    the sheet can be sorted, charted and referenced rather than re-typed. The
    publishing dates sit above the table, mirroring the email body.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Settlements"

    sheet["A1"] = f"BGN LNG Desk: Settlements as of {report_date.strftime('%d-%b-%Y')}"
    sheet["A1"].font = Font(bold=True, size=13)

    # Publishing dates, one row per group, same grouping as the email.
    row = 3
    for _group, labels in PUBLICATION_ROWS:
        parts = []
        for label in labels:
            source_date = as_of.get(label)
            shown = source_date.strftime("%d-%b-%Y") if source_date else "unavailable"
            parts.append(f"{label}: {shown}")
        sheet.cell(row=row, column=1, value="Last published  |  " + "  |  ".join(parts))
        sheet.cell(row=row, column=1).font = Font(italic=True, size=9)
        row += 1

    header_row = row + 1
    units_row = header_row + 1

    headers = ["Contract"] + [label for label, _, _ in COLUMNS]
    units = [""] + [unit for _, unit, _ in COLUMNS]
    for column, (label, unit) in enumerate(zip(headers, units), start=1):
        head = sheet.cell(row=header_row, column=column, value=label)
        head.font = Font(bold=True, color=_XL_HEADER_FONT)
        head.fill = PatternFill("solid", fgColor=_XL_HEADER_FILL)
        head.alignment = Alignment(horizontal="center")

        unit_cell = sheet.cell(row=units_row, column=column, value=unit)
        unit_cell.font = Font(italic=True, size=9, color=_XL_HEADER_FONT)
        unit_cell.fill = PatternFill("solid", fgColor=_XL_HEADER_FILL)
        unit_cell.alignment = Alignment(horizontal="center")

    for offset, month in enumerate(table.index):
        excel_row = units_row + 1 + offset
        month_cell = sheet.cell(row=excel_row, column=1, value=month)
        month_cell.number_format = "mmm-yy"

        for column, (label, _, decimals) in enumerate(COLUMNS, start=2):
            value = table.loc[month, label]
            if pd.isna(value):
                continue  # leave genuinely unquoted tenors empty, not zero
            cell = sheet.cell(row=excel_row, column=column, value=float(value))
            cell.number_format = _XL_NUMBER_FORMATS[decimals]

    sheet.freeze_panes = sheet.cell(row=units_row + 1, column=2)
    sheet.column_dimensions["A"].width = 11
    for column in range(2, len(COLUMNS) + 2):
        sheet.column_dimensions[get_column_letter(column)].width = 13

    Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)
    workbook.save(xlsx_path)
    print(f"Settlement workbook written to {xlsx_path}")
    return xlsx_path


# --- Delivery ----------------------------------------------------------------


def send_market_table_email(html, report_date, sending_to=DEFAULT_RECIPIENTS,
                            attachment_path=None):
    """Send the snapshot as an Outlook mail, attaching the workbook when built."""
    send_email(
        html,
        subject=f"LNG Daily Settlements - {report_date.isoformat()}",
        sending_to=sending_to,
        attachment_path=attachment_path,
    )


def main(sending_to=DEFAULT_RECIPIENTS, dry_run=False, out_path=None,
         xlsx_path=None):
    """Build today's snapshot and email it (or, with `dry_run`, just build it).

    `out_path` and `xlsx_path` default to the dated shared-drive copies; pass a
    path to write elsewhere, or ``False`` to skip that file entirely.
    """
    report_date = date.today()
    if out_path is None:
        out_path = online_html_path(report_date)
    if xlsx_path is None:
        xlsx_path = online_xlsx_path(report_date)

    table, as_of = build_market_table(report_date)
    html = render_market_table_html(table, as_of, report_date)

    if out_path:
        Path(out_path).write_text(html, encoding="utf-8")
        print(f"Settlement snapshot written to {out_path}")

    attachment = None
    if xlsx_path:
        try:
            attachment = write_market_table_xlsx(table, as_of, report_date, xlsx_path)
        except Exception as exc:
            # A workbook that fails to write must not hold up the mail -- the
            # table is in the body either way.
            print(f"Settlement workbook skipped: {exc!r}")

    if dry_run:
        print(table.to_string())
        return table

    send_market_table_email(html, report_date, sending_to=sending_to,
                            attachment_path=attachment)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", default=DEFAULT_RECIPIENTS,
                        help="semicolon-separated Outlook recipients")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the table and print it without emailing")
    parser.add_argument("--out", default=None,
                        help="write the rendered HTML here instead of the "
                             "dated shared-drive copy")
    parser.add_argument("--xlsx", default=None,
                        help="write the workbook here instead of the dated "
                             "shared-drive copy")
    args = parser.parse_args()
    main(sending_to=args.to, dry_run=args.dry_run, out_path=args.out,
         xlsx_path=args.xlsx)
