"""Daily forward-curve snapshot for the refined products complex, emailed each morning.

Rows are the front two years of monthly contracts (delivery months only);
columns are the latest mark for each product benchmark, straight off Bloomberg.
Everything here is exchange-listed -- ICE Futures Europe, ICE's Platts/Argus-
settled swap futures ("ISF"), and NYMEX -- so a cell can always be tied back to
a ticker you can type into the terminal.

Two things differ from the gas/LNG snapshot next door and drive the design:

*   **Not every contract settles into ``px_settle``.** The ICE swap futures
    (Naphtha, Jet, the fuel oils, NWE Diesel) publish a daily mark into
    ``px_last`` and leave ``px_settle`` empty, while the outright exchange
    contracts (Gasoil, Heating Oil, RBOB, Mont Belvieu) fill both. Each product
    is read with the field it actually carries -- see ``PriceField``.

*   **The two groups run on different clocks.** The exchange contracts mark
    today; the Platts/Argus-settled swaps lag a business day or two behind the
    assessment. The "as of" date is therefore shown *per product* rather than
    assumed to be today, so a stale curve is visible rather than passed off as
    this morning's.

Units are left exactly as Bloomberg quotes them -- $/mt for the European
barrels, US cents per gallon for the NYMEX distillate and gasoline contracts,
and US dollars per gallon for the Mont Belvieu LPG legs. Nothing is converted,
so every number in the mail can be ticked off against the screen without having
to undo a factor first.

Run as a script to build and email today's snapshot:
    python -m Data.lng_report.oil_products_reporting
Add ``--dry-run`` to build the table without sending anything, or ``--tickers``
to print the full ticker map and pull nothing at all.
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from xbbg import blp

from Data.lng_report.config import ONLINE_DIR, bbg_month_codes
from Data.lng_report.emailer import send_email

# --- What the table covers ---------------------------------------------------

# Front two years of monthly contracts, starting with the month after the
# current one: by the time this goes out the running month has stopped being a
# forward contract on any of these benchmarks.
N_MONTHS = 24

# Which Bloomberg field carries the daily mark, and which carries its date.
# ``SETTLE`` is the outright exchange contracts, ``LAST`` the ICE swap futures
# that mark against a Platts or Argus assessment instead of settling on-screen.
PRICE_FIELDS = {
    "SETTLE": ("px_settle", "px_settle_last_dt"),
    "LAST": ("px_last", "last_update_dt"),
}

# --- The products ------------------------------------------------------------
#
# Each entry is:
#   label     column heading in the mail
#   root      Bloomberg futures root; the ticker for a delivery month is
#             root + month code + two-digit year + " Comdty"
#             (e.g. Naphtha Jan-27 -> "BCAF27 Comdty")
#   venue     exchange as Bloomberg labels it, shown in the ticker legend
#   contract  the contract's full name, for the legend
#   units     display units, exactly as Bloomberg quotes the contract
#   decimals  display precision
#   field     which of PRICE_FIELDS above the contract populates
#
# Roots were picked off Bloomberg's instrument search and then checked on the
# live terminal for open interest and curve depth -- where a product has both a
# full-size and a "mini" listing, or a Platts and an Argus listing, the one
# carrying the open interest is the one here. The mini Jet contract (CKP) is
# deliberate: the full-size Jet CIF NWE listing carries no marks on this feed.

PRODUCTS = [
    ("Gasoil", "QS", "ICE", "Low Sulphur Gasoil Futures",
     "$/mt", 2, "SETTLE"),
    ("Diesel NWE", "AXP", "ICE ISF", "ULSD 10ppm CIF NWE Cargoes Future",
     "$/mt", 2, "LAST"),
    ("Heating Oil", "HO", "NYMEX", "NY Harbor ULSD Future",
     "USc/gal", 2, "SETTLE"),
    ("Jet NWE", "CKP", "ICE ISF", "Jet CIF NWE Cargoes (Platts) Mini Future",
     "$/mt", 2, "LAST"),
    ("Naphtha NWE", "BCA", "ICE ISF", "Naphtha CIF NWE Cargoes Outright Future",
     "$/mt", 2, "LAST"),
    # Eurobob has two listings worth knowing about. NYMEX's (IHW) carries the
    # open interest but only marks the months someone has traded, so its curve
    # comes back with holes -- eleven of the front twenty-four were empty when
    # this was written. ICE's (MZS) marks every month off the GX assessment and
    # sat within ~1% of IHW wherever both quoted, so it is the one used here: a
    # complete curve of the same market beats a patchy one.
    ("Gasoline EU", "MZS", "ICE ISF", "Eurobob Oxy Gasoline NWE FOB Barges (GX) Outright Future",
     "$/mt", 2, "LAST"),
    ("Gasoline US", "XB", "NYMEX", "RBOB Gasoline Future",
     "USc/gal", 2, "SETTLE"),
    ("Fuel Oil HS", "AYY", "ICE ISF", "Fuel Oil 3.5% FOB Rotterdam Barges Outright Future",
     "$/mt", 2, "LAST"),
    ("Fuel Oil LS", "JWC", "ICE ISF", "Marine Fuel 0.5% FOB Rotterdam Barges (Platts) Future",
     "$/mt", 2, "LAST"),
    ("Propane EU", "PSA", "NYMEX", "Propane Euro CIF ARA (Argus) Swap",
     "$/mt", 2, "SETTLE"),
    ("Propane US", "BAP", "NYMEX", "Mont Belvieu Propane (OPIS) 5 Decimal Swap",
     "$/gal", 4, "SETTLE"),
    ("Butane US", "DAE", "NYMEX", "Mont Belvieu Normal Butane (OPIS) 5 Decimal Swap",
     "$/gal", 4, "SETTLE"),
]

# Products grouped for the "last marked" block above the table. The split is the
# one that matters operationally: the first group settles on-screen and marks
# today, the second marks off a Platts/Argus assessment and runs a day or two
# behind.
PUBLICATION_GROUPS = [
    ("exchange settled", ["Gasoil", "Heating Oil", "Gasoline US",
                          "Propane EU", "Propane US", "Butane US"]),
    ("assessment settled", ["Diesel NWE", "Jet NWE", "Naphtha NWE", "Gasoline EU",
                            "Fuel Oil HS", "Fuel Oil LS"]),
]

DEFAULT_RECIPIENTS = "marti.fernandezreal@bgn-int.com; "

# A dated copy is published to the shared drive next to the gas report, so the
# last send can be eyeballed after the fact without re-running the pull.
ONLINE_FILENAME = "oil_products_curves_{pricing_date}.html"
ONLINE_XLSX_FILENAME = "oil_products_curves_{pricing_date}.xlsx"


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


def product_ticker(root, month):
    """The Bloomberg ticker for `root`'s contract delivering in `month`.

    Two-digit years throughout. The one-digit form resolves for some of these
    roots but not all, and it collides outright -- "MZSZ6" is a DAX future, not
    a gasoline one -- so the longer form is the only safe rule.
    """
    return f"{root}{bbg_month_codes[month.month - 1]}{month.year % 100:02d} Comdty"


def product_tickers(months=None, today=None):
    """The full ticker map: ``{label: {delivery month: ticker}}``.

    Built without touching Bloomberg, so it can be printed and checked against
    the terminal before anything is pulled.
    """
    months = months if months is not None else front_months(today)
    return {label: {month: product_ticker(root, month) for month in months}
            for label, root, _, _, _, _, _ in PRODUCTS}


def product_curve(root, months, field_kind):
    """Latest marks for `root` over `months`.

    Returns ``(series indexed by delivery month, mark date)``. The mark date is
    the one most contracts agree on -- an illiquid far month can lag, but the
    strip as a whole marks to a single day.
    """
    price_field, date_field = PRICE_FIELDS[field_kind]
    ticker_of = {product_ticker(root, month): month for month in months}

    quotes = blp.bdp(tickers=list(ticker_of), flds=[price_field, date_field])

    prices, mark_dates = {}, []
    for ticker, row in quotes.iterrows():
        month = ticker_of.get(ticker)
        if month is None:
            continue
        price = row.get(price_field)
        if pd.notna(price):
            prices[month] = float(price)
        mark_date = row.get(date_field)
        if pd.notna(mark_date):
            mark_dates.append(pd.Timestamp(mark_date).date())

    as_of = pd.Series(mark_dates).mode().iloc[0] if mark_dates else None
    return pd.Series(prices, dtype=float), as_of


# --- Table assembly ----------------------------------------------------------


def build_products_table(today=None, n_months=N_MONTHS):
    """Build the forward-curve table.

    Returns ``(DataFrame indexed by delivery month, {product: as-of date})``. A
    product whose pull fails leaves its column blank and is flagged in the mail
    header rather than holding up the send.
    """
    months = front_months(today, n_months)
    columns, as_of = {}, {}

    for label, root, _venue, _contract, _units, _decimals, field_kind in PRODUCTS:
        try:
            series, mark_date = product_curve(root, months, field_kind)
        except Exception as exc:
            print(f"{label} skipped: {exc!r}")
            series, mark_date = pd.Series(dtype=float), None
        columns[label] = series
        as_of[label] = mark_date

    table = pd.DataFrame(
        {label: columns[label].reindex(months) for label, *_ in PRODUCTS},
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
    return f'{label}: <b>{source_date.strftime("%d-%b")}</b>'


def _publication_rows_html(as_of):
    """The "last marked" block: one line per settlement group."""
    lines = []
    for group, labels in PUBLICATION_GROUPS:
        dates = " &nbsp;|&nbsp; ".join(
            _as_of_label(label, as_of.get(label)) for label in labels
        )
        lines.append(
            f'<div style="margin:2px 0;">'
            f'<span style="color:{_MUTED};">Last marked &middot; {group}</span> '
            f'&mdash; {dates}</div>'
        )
    return "\n        ".join(lines)


def _ticker_legend_html(months):
    """The ticker table under the curves, so every column can be checked.

    Shows the root alongside a worked example -- the first delivery month in the
    table -- which is what you actually type into the terminal to compare.
    """
    example_month = months[0]
    rows = []
    for label, root, venue, contract, units, _decimals, field_kind in PRODUCTS:
        price_field, _ = PRICE_FIELDS[field_kind]
        rows.append(
            f'<tr>'
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px;">{label}</td>'
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px;"><b>{root}</b></td>'
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px; white-space:nowrap;">'
            f'{product_ticker(root, example_month)}</td>'
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px;">{venue}</td>'
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px;">{units}</td>'
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px;">{price_field}</td>'
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px;">{contract}</td>'
            f'</tr>'
        )
    body = "\n".join(rows)

    heads = ["Product", "Root", f'{example_month.strftime("%b-%y")} ticker',
             "Venue", "Units", "Field", "Contract"]
    header_cells = "".join(
        f'<th style="border:1px solid {_BORDER}; padding:4px 8px; text-align:left;">{h}</th>'
        for h in heads
    )

    return f"""
    <h2 style="font-size:14px; margin:20px 0 6px 0; color:{_FG};">Tickers</h2>
    <p style="font-size:11px; color:{_MUTED}; margin:0 0 8px 0;">
        A delivery month's ticker is root + month code + two-digit year + " Comdty"
        &mdash; month codes F G H J K M N Q U V X Z for Jan through Dec.
    </p>
    <table cellpadding="0" cellspacing="0" border="0"
           style="border-collapse:collapse; font-size:11px; color:{_FG};">
        <thead><tr style="background-color:{_ALT_ROW};">{header_cells}</tr></thead>
        <tbody>
{body}
        </tbody>
    </table>"""


def render_products_table_html(table, as_of, report_date):
    """Render the table as inline-styled HTML that survives Outlook."""
    published = _publication_rows_html(as_of)

    header_cells = "".join(
        f'<th style="border:1px solid {_BORDER}; padding:4px 8px; text-align:right; '
        f'white-space:nowrap;">{label}<br>'
        f'<span style="color:{_MUTED}; font-weight:normal; font-size:11px;">{units}</span></th>'
        for label, _, _, _, units, _, _ in PRODUCTS
    )

    rows = []
    for month in table.index:
        # Shade alternate years so a two-year strip stays readable at a glance.
        background = _ALT_ROW if month.year % 2 else _BG
        cells = "".join(
            f'<td style="border:1px solid {_BORDER}; padding:3px 8px; text-align:right; '
            f'white-space:nowrap;">{_format_value(table.loc[month, label], decimals)}</td>'
            for label, _, _, _, _, decimals, _ in PRODUCTS
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
        BGN: Refined Products Forward Curves &mdash; {report_date.strftime('%d-%b-%Y')}
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
        Monthly contracts only, front {len(table)} months. Units are as Bloomberg
        quotes each contract &mdash; nothing is converted. Blank cells are tenors
        the contract does not quote, not a failed pull.
    </p>
    {_ticker_legend_html(list(table.index))}
</body>
</html>"""


# --- Workbook ----------------------------------------------------------------

_XL_HEADER_FILL = "FF161B22"
_XL_HEADER_FONT = "FFFFFFFF"


def write_products_table_xlsx(table, as_of, report_date, xlsx_path):
    """Write the curve table to `xlsx_path`, with the ticker map on a second sheet.

    Values are written as real numbers and delivery months as real dates, so the
    sheet can be sorted, charted and referenced rather than re-typed.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Curves"

    sheet["A1"] = f"BGN Refined Products Forward Curves - {report_date.strftime('%d-%b-%Y')}"
    sheet["A1"].font = Font(bold=True, size=13)

    row = 3
    for group, labels in PUBLICATION_GROUPS:
        parts = []
        for label in labels:
            mark_date = as_of.get(label)
            shown = mark_date.strftime("%d-%b-%Y") if mark_date else "unavailable"
            parts.append(f"{label}: {shown}")
        sheet.cell(row=row, column=1,
                   value=f"Last marked ({group})  |  " + "  |  ".join(parts))
        sheet.cell(row=row, column=1).font = Font(italic=True, size=9)
        row += 1

    header_row = row + 1
    units_row = header_row + 1

    headers = ["Contract"] + [label for label, *_ in PRODUCTS]
    units = [""] + [entry[4] for entry in PRODUCTS]
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

        for column, entry in enumerate(PRODUCTS, start=2):
            label, decimals = entry[0], entry[5]
            value = table.loc[month, label]
            if pd.isna(value):
                continue  # leave genuinely unquoted tenors empty, not zero
            cell = sheet.cell(row=excel_row, column=column, value=float(value))
            cell.number_format = "#,##0." + "0" * decimals

    sheet.freeze_panes = sheet.cell(row=units_row + 1, column=2)
    sheet.column_dimensions["A"].width = 11
    for column in range(2, len(PRODUCTS) + 2):
        sheet.column_dimensions[get_column_letter(column)].width = 13

    # Second sheet: every ticker behind the table, one row per contract month,
    # so a number in the grid can be traced to the exact security it came from.
    ticker_sheet = workbook.create_sheet("Tickers")
    ticker_headers = ["Product", "Root", "Venue", "Units", "Field",
                      "Contract", "Delivery", "Ticker"]
    for column, head in enumerate(ticker_headers, start=1):
        cell = ticker_sheet.cell(row=1, column=column, value=head)
        cell.font = Font(bold=True, color=_XL_HEADER_FONT)
        cell.fill = PatternFill("solid", fgColor=_XL_HEADER_FILL)

    ticker_row = 2
    for label, root, venue, contract, unit, _decimals, field_kind in PRODUCTS:
        price_field, _ = PRICE_FIELDS[field_kind]
        for month in table.index:
            values = [label, root, venue, unit, price_field, contract,
                      month, product_ticker(root, month)]
            for column, value in enumerate(values, start=1):
                cell = ticker_sheet.cell(row=ticker_row, column=column, value=value)
                if column == 7:
                    cell.number_format = "mmm-yy"
            ticker_row += 1

    for column, width in enumerate([14, 8, 10, 10, 12, 56, 11, 18], start=1):
        ticker_sheet.column_dimensions[get_column_letter(column)].width = width
    ticker_sheet.freeze_panes = "A2"

    Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)
    workbook.save(xlsx_path)
    print(f"Products workbook written to {xlsx_path}")
    return xlsx_path


# --- Delivery ----------------------------------------------------------------


def send_products_email(html, report_date, sending_to=DEFAULT_RECIPIENTS,
                        attachment_path=None):
    """Send the snapshot as an Outlook mail, attaching the workbook when built."""
    send_email(
        html,
        subject=f"Refined Products Forward Curves - {report_date.isoformat()}",
        sending_to=sending_to,
        attachment_path=attachment_path,
    )


def print_ticker_map(today=None, n_months=N_MONTHS):
    """Print every ticker the table would pull, without pulling anything."""
    months = front_months(today, n_months)
    for label, root, venue, contract, unit, _decimals, field_kind in PRODUCTS:
        price_field, _ = PRICE_FIELDS[field_kind]
        print(f"\n{label}  --  {contract} ({venue}), {unit}, read from {price_field}")
        print(f"  root {root}")
        line = "  " + "  ".join(
            f'{month.strftime("%b-%y")}:{product_ticker(root, month).split()[0]}'
            for month in months
        )
        print(line)


def main(sending_to=DEFAULT_RECIPIENTS, dry_run=False, out_path=None,
         xlsx_path=None, n_months=N_MONTHS):
    """Build today's snapshot and email it (or, with `dry_run`, just build it).

    `out_path` and `xlsx_path` default to the dated shared-drive copies; pass a
    path to write elsewhere, or ``False`` to skip that file entirely.
    """
    report_date = date.today()
    if out_path is None:
        out_path = online_html_path(report_date)
    if xlsx_path is None:
        xlsx_path = online_xlsx_path(report_date)

    table, as_of = build_products_table(report_date, n_months)
    html = render_products_table_html(table, as_of, report_date)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(html, encoding="utf-8")
        print(f"Products snapshot written to {out_path}")

    attachment = None
    if xlsx_path:
        try:
            attachment = write_products_table_xlsx(table, as_of, report_date, xlsx_path)
        except Exception as exc:
            # A workbook that fails to write must not hold up the mail -- the
            # table is in the body either way.
            print(f"Products workbook skipped: {exc!r}")

    if dry_run:
        print(table.to_string())
        return table

    send_products_email(html, report_date, sending_to=sending_to,
                        attachment_path=attachment)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", default=DEFAULT_RECIPIENTS,
                        help="semicolon-separated Outlook recipients")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the table and print it without emailing")
    parser.add_argument("--tickers", action="store_true",
                        help="print the ticker map and exit, pulling nothing")
    parser.add_argument("--months", type=int, default=N_MONTHS,
                        help=f"how many delivery months to show (default {N_MONTHS})")
    parser.add_argument("--out", default=None,
                        help="write the rendered HTML here instead of the "
                             "dated shared-drive copy")
    parser.add_argument("--xlsx", default=None,
                        help="write the workbook here instead of the dated "
                             "shared-drive copy")
    args = parser.parse_args()

    if args.tickers:
        print_ticker_map(n_months=args.months)
    else:
        main(sending_to=args.to, dry_run=args.dry_run, out_path=args.out,
             xlsx_path=args.xlsx, n_months=args.months)
