"""Send the LNG report by email through the local Outlook client."""

from datetime import date
from pathlib import Path

import win32com.client


def send_email(html, html_out_path, sending_to="marti.fernandezreal@bgn-int.com"):
    """Send `html` as the body of an Outlook mail, attaching the report file."""
    outlook = win32com.client.Dispatch("Outlook.Application")
    mail = outlook.CreateItem(0)  # 0 = MailItem

    mail.To = sending_to
    mail.Subject = f"LNG Report - {date.today().isoformat()}"
    report_path = Path(html_out_path)
    mail.Attachments.Add(str(report_path.absolute()))
    mail.HTMLBody = html  # for Outlook

    mail.Send()


def send_test_email():
    """Send a minimal placeholder report to verify the Outlook pipeline works."""
    simplified_html = """<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body style="background-color: #0D1117; color: #C9D1D9; font-family: Arial, sans-serif; margin: 10px;">
            <h1 style="text-align: center; font-size: 22px; margin: 8px 0;">
                BGN LNG Desk: Daily Report
            </h1>
            <table cellpadding="3" cellspacing="0" border="0" style="width:100%; border-collapse: collapse;">
                <tr>
                    <td style="width:50%; border:1px solid #30363D; padding:5px;"></td>
                    <td style="width:50%; border:1px solid #30363D; padding:5px;"></td>
                </tr>
            </table>
        </body>
        </html>"""

    send_email(
        simplified_html,
        html_out_path="C:\\Marti\\lng_on_water.html",
        sending_to="marti.fernandezreal@bgn-int.com"
    )
