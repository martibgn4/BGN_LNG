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


def build_grid_report_html(report_date, divs):
    """Full interactive HTML page: a 2-column grid of Plotly <div>s.

    `divs` must contain the 11 rendered figure/HTML fragments in report order.
    """
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
        <div>{divs[0]}</div>
        <div>{divs[1]}</div>
        <div>{divs[2]}</div>
        <div>{divs[3]}</div>
        <div>{divs[4]}</div>
        <div>{divs[5]}</div>
        <div>{divs[6]}</div>
        <div>{divs[7]}</div>
        <div>{divs[8]}</div>
        <div>{divs[9]}</div>
         <div>{divs[10]}</div>
    </div>
</body>
</html>"""


def build_email_html(report_date, img_tags):
    """Simplified, table-based HTML for the Outlook email body.

    `img_tags` must contain the 11 embedded <img>/HTML fragments in report order.
    """
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
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[0]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[1]}</td>
            </tr>
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[2]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[3]}</td>
            </tr>
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[4]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[5]}</td>
            </tr>
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[6]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[7]}</td>
            </tr>
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[8]}</td>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[9]}</td>
            </tr>
            <tr>
                <td style="width:50%; border:1px solid #30363D; padding:5px;">{img_tags[10]}</td>
            </tr>
        </table>
    </body>
    </html>"""
