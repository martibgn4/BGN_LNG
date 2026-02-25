import unittest
from datetime import date

from Data.forward_curve_interpolator import ForwardCurve
from Utils.datetime_utils import infer_ratios_from_monthly_quotes


class TestDatetimeUtils(unittest.TestCase):

    def setUp(self):
        # First example from Haug, Option Pricing Formulas
        self.forward_quotes = {
            "Jan_25": 31.0,
            "Feb_25": 31.0,
            "Mar_25": 30.0,
            "Apr_25": 25.0,
            "May_25": 26.0,
            "Jun_25": 27.0,
            "Jul_25": 28.0,
            "Aug_25": 29.0,
            "Sep_25": 29.0,
            "Oct_25": 23.0,
            "Nov_25": 22.0,
            "Dec_25": 21.0
        }

        self.period_to_infer_ratios = ["Q1", "Q2", "Q3", "Q4", "SS", "WS", "Y"]

    def test_inferred_ratios_for_periods(self):
        forward_quotes = self.forward_quotes
        avg_q1 = (forward_quotes["Jan_25"] + forward_quotes["Feb_25"] + forward_quotes["Mar_25"])/3
        avg_q2 = (forward_quotes["Apr_25"] + forward_quotes["May_25"] + forward_quotes["Jun_25"])/3
        avg_q3 = (forward_quotes["Jul_25"] + forward_quotes["Aug_25"] + forward_quotes["Sep_25"])/3
        avg_q4 = (forward_quotes["Oct_25"] + forward_quotes["Nov_25"] + forward_quotes["Dec_25"])/3

        avg_ss = (forward_quotes["Apr_25"] + forward_quotes["May_25"] + forward_quotes["Jun_25"] + forward_quotes["Jul_25"] + forward_quotes["Aug_25"] + forward_quotes["Sep_25"])/6

        avg_y = (forward_quotes["Jan_25"] + forward_quotes["Feb_25"] + forward_quotes["Mar_25"] + forward_quotes["Apr_25"]
                 + forward_quotes["May_25"] + forward_quotes["Jun_25"] + forward_quotes["Jul_25"] + forward_quotes["Aug_25"]
                 + forward_quotes["Sep_25"] + forward_quotes["Oct_25"] + forward_quotes["Nov_25"] + forward_quotes["Dec_25"])/12

        theoretical_ratios = {
            "Q1": {
                "Jan": forward_quotes["Jan_25"]/avg_q1,
                "Feb": forward_quotes["Feb_25"]/avg_q1,
                "Mar": forward_quotes["Mar_25"]/avg_q1,
            },
            "Q2": {
                "Apr": forward_quotes["Apr_25"] / avg_q2,
                "May": forward_quotes["May_25"] / avg_q2,
                "Jun": forward_quotes["Jun_25"] / avg_q2,
            },
            "Q3": {
                "Jul": forward_quotes["Jul_25"] / avg_q3,
                "Aug": forward_quotes["Aug_25"] / avg_q3,
                "Sep": forward_quotes["Sep_25"] / avg_q3,
            },
            "Q4": {
                "Oct": forward_quotes["Oct_25"] / avg_q4,
                "Nov": forward_quotes["Nov_25"] / avg_q4,
                "Dec": forward_quotes["Dec_25"] / avg_q4,
            },
            "SS": {
                "Apr": forward_quotes["Apr_25"] / avg_ss,
                "May": forward_quotes["May_25"] / avg_ss,
                "Jun": forward_quotes["Jun_25"] / avg_ss,
                "Jul": forward_quotes["Jul_25"] / avg_ss,
                "Aug": forward_quotes["Aug_25"] / avg_ss,
                "Sep": forward_quotes["Sep_25"] / avg_ss,
            },
            "WS": {
                "Oct": 1.0,
                "Nov": 1.0,
                "Dec": 1.0,
                "Jan": 1.0,
                "Feb": 1.0,
                "Mar": 1.0,
            },
            "Y": {
                "Jan": forward_quotes["Jan_25"] / avg_y,
                "Feb": forward_quotes["Feb_25"] / avg_y,
                "Mar": forward_quotes["Mar_25"] / avg_y,
                "Apr": forward_quotes["Apr_25"] / avg_y,
                "May": forward_quotes["May_25"] / avg_y,
                "Jun": forward_quotes["Jun_25"] / avg_y,
                "Jul": forward_quotes["Jul_25"] / avg_y,
                "Aug": forward_quotes["Aug_25"] / avg_y,
                "Sep": forward_quotes["Sep_25"] / avg_y,
                "Oct": forward_quotes["Oct_25"] / avg_y,
                "Nov": forward_quotes["Nov_25"] / avg_y,
                "Dec": forward_quotes["Dec_25"] / avg_y,
            }
        }

        inferred_ratios = infer_ratios_from_monthly_quotes(forward_quotes, self.period_to_infer_ratios)

        self.assertEqual(inferred_ratios, theoretical_ratios)

        # Adding WS quotes to be inferred as well:
        remaining_winter_quotes = {
            "Jan_26": 41.0,
            "Feb_26": 42.0,
            "Mar_26": 43.0
        }

        extended_forward_quotes = {**forward_quotes, **remaining_winter_quotes}
        extended_inferred_ratios = infer_ratios_from_monthly_quotes(extended_forward_quotes, self.period_to_infer_ratios)


        avg_ws = (extended_forward_quotes["Jan_26"] + extended_forward_quotes["Feb_26"] + extended_forward_quotes["Mar_26"]
                  + extended_forward_quotes["Oct_25"] + extended_forward_quotes["Nov_25"] + extended_forward_quotes[
                     "Dec_25"]) / 6
        theoretical_ratios["WS"] = {
            "Jan": extended_forward_quotes["Jan_26"] / avg_ws,
            "Feb": extended_forward_quotes["Feb_26"] / avg_ws,
            "Mar": extended_forward_quotes["Mar_26"] / avg_ws,
            "Oct": extended_forward_quotes["Oct_25"] / avg_ws,
            "Nov": extended_forward_quotes["Nov_25"] / avg_ws,
            "Dec": extended_forward_quotes["Dec_25"] / avg_ws
        }

        self.assertEqual(extended_inferred_ratios, theoretical_ratios)

    def test_forwardcurve_from_quotes(self):
        forward_quotes = {
            **self.forward_quotes,
            "Y_26": 50,
        }

        today_date = date(2025, 12, 1)
        fwd_curve = ForwardCurve(forward_quotes, today_date)

        monthly_tenors_from_fwd_curve = fwd_curve.monthly_quotes
        assert "Aug_26" in monthly_tenors_from_fwd_curve, f"Aug_26 should be in quotes as provided shape and yearly"
        a = 1

