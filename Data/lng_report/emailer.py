"""Send desk mail through the local Outlook client.

One sender for every report the desk pushes out: the subject and the optional
attachment are the caller's business, the COM plumbing lives here.
"""

from datetime import date
from pathlib import Path

import win32com.client


def send_email(html, subject, sending_to, attachment_path=None):
    """Send `html` as the body of an Outlook mail.

    `sending_to` is a semicolon-separated recipient list. `attachment_path`
    attaches a file when given -- the LNG report ships its interactive HTML
    that way, the settlement snapshot is body-only.
    """

    outlook = win32com.client.Dispatch("Outlook.Application")
    mail = outlook.CreateItem(0)  # 0 = MailItem

    mail.To = sending_to
    mail.Subject = subject
    if attachment_path is not None:
        mail.Attachments.Add(str(Path(attachment_path).absolute()))
    mail.HTMLBody = html  # for Outlook

    mail.Send()
    print(f"Sent '{subject}' to {sending_to}")


def send_test_email(sending_to="marti.fernandezreal@bgn-int.com"):
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
        subject=f"LNG Report - {date.today().isoformat()} ",
        sending_to=sending_to,
    )
