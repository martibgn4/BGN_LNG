"""Shared visual styling: the dark Plotly layout and the report HTML templates.

Keeping the layout and page shells here means every figure and both report
formats (interactive grid + email) stay visually consistent.
"""


def dark_layout(**overrides):
    """Return the common dark Plotly layout, optionally overridden per figure."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(family="sans-serif", color="#C9D1D9", size=14),
        xaxis=dict(gridcolor="#30363D", linecolor="#30363D"),
        yaxis=dict(gridcolor="#30363D", linecolor="#30363D"),
    )
    layout.update(overrides)
    return layout


def build_grid_report_html(report_date, divs, full_width_divs=()):
    """Full interactive HTML page: a 2-column grid of Plotly <div>s.

    `divs` holds the rendered figure/HTML fragments in report order, laid out
    two per row. `full_width_divs` are appended below the grid, one per row —
    for charts too wide to read at half width.
    """
    grid_cells = "\n".join(f"        <div>{d}</div>" for d in divs)
    wide_cells = "\n".join(
        f'    <div class="full-width">{d}</div>' for d in full_width_divs
    )
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>
        body {{ background-color: #0D1117; color: #C9D1D9; font-family: sans-serif; margin: 10px; }}
        h1 {{ text-align: center; font-size: 22px; margin: 8px 0; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
        .grid > div {{ background: #0D1117; border: 1px solid #30363D; border-radius: 6px; }}
        .full-width {{ margin-top: 10px; border: 1px solid #30363D; border-radius: 6px; }}
    </style>
</head>
<body>
    <h1>BGN LNG Desk: {report_date.strftime('%d-%b-%Y')} Daily Report</h1>
    <div class="grid">
{grid_cells}
    </div>
{wide_cells}
</body>
</html>"""


def build_email_html(report_date, img_tags, full_width_img_tags=()):
    """Simplified, table-based HTML for the Outlook email body.

    `img_tags` holds the embedded <img>/HTML fragments in report order, two per
    table row. `full_width_img_tags` are appended as single full-width rows.
    """
    cell = 'style="width:50%; border:1px solid #30363D; padding:5px;"'
    rows = []
    for i in range(0, len(img_tags), 2):
        pair = img_tags[i:i + 2]
        cells = "\n".join(f"                <td {cell}>{t}</td>" for t in pair)
        rows.append(f"            <tr>\n{cells}\n            </tr>")
    for tag in full_width_img_tags:
        rows.append(
            '            <tr>\n'
            '                <td colspan="2" style="border:1px solid #30363D; '
            f'padding:5px;">{tag}</td>\n'
            '            </tr>'
        )
    table_rows = "\n".join(rows)
    return f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
    </head>
    <body style="background-color: #0D1117; color: #C9D1D9; font-family: Arial, sans-serif; margin: 10px;">
        <h1 style="text-align: center; font-size: 22px; margin: 8px 0;">
            BGN LNG Desk: {report_date.strftime('%d-%b-%Y')} Daily Report
        </h1>
        <table cellpadding="3" cellspacing="0" border="0" style="width:100%; border-collapse: collapse;">
{table_rows}
        </table>
    </body>
    </html>"""
