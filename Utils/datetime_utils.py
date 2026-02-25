
import calendar
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Any, Optional

from xlwings.conversion import Converter

DAYS_PER_YEAR = 365.0

def time_between(start_date, end_date):
    """Calculates time between two dates as a fraction of a year"""
    return (end_date - start_date).days / DAYS_PER_YEAR

def month_int_from_string(s_month):
    return {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6,
              "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}[s_month]

def month_string_from_int(i_month):
    _dict = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6,
              "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
    return {i: m for m, i in _dict.items()}[i_month]

_period_str_to_months_dict = {
    "M": 1
}

def is_tenor_month(tenor):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return tenor[:3] in months

def _add_year(m, year):
    return m + "_" + year

def get_month_from_tenor(tenor):
    month, _ = tenor.split("_")
    return month

def get_int_year_from_tenor(tenor):
    _, year = tenor.split("_")
    return int(year)

def month_codes_only_for_period(period):
    months = {
        "Q1": ("Jan", "Feb", "Mar"),
        "Q2": ("Apr", "May", "Jun"),
        "Q3": ("Jul", "Aug", "Sep"),
        "Q4": ("Oct", "Nov", "Dec"),
        "SS": ("Apr", "May", "Jun", "Jul", "Aug", "Sep"),
        "WS": ("Oct", "Nov", "Dec", "Jan", "Feb", "Mar"),
        "Y": ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    }[period]
    return months


def date_tenor_from_label(label):
    month, year = label.split("_")
    month_int_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                      "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                  "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    month_int = month_int_dict[month] if month in month_int_dict else -1
    if month_int < 0:
        return label
    return date(int("20" + year), month_int, 1)

def label_from_date_tenor(date_tenor):
    year, month, day = date_tenor.split("-")
    month_int_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                      "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                  "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    int_month_dict = {v:k for k, v in month_int_dict.items()}

    int_month = int_month_dict[int(month)] if int(month) in int_month_dict else -1
    if int_month not in month_int_dict:
        return date_tenor
    return int_month + "_" + str(year)[-2:]


def tenor_to_monthly_strip(long_tenor):
    if is_tenor_month(long_tenor):
        raise ValueError(f"Expected non-month tenor got {long_tenor}")

    _period, _y = long_tenor.split("_")
    months = {
        "Q1": [_add_year(m, _y) for m in month_codes_only_for_period("Q1")],
        "Q2": [_add_year(m, _y) for m in month_codes_only_for_period("Q2")],
        "Q3": [_add_year(m, _y) for m in month_codes_only_for_period("Q3")],
        "Q4": [_add_year(m, _y) for m in month_codes_only_for_period("Q4")],
        "SS": [_add_year(m, _y) for m in month_codes_only_for_period("SS")],
        "WS": [_add_year(m, y) for m, y in zip(month_codes_only_for_period("WS"), (_y, _y, _y, str(int(_y)+1), str(int(_y)+1), str(int(_y)+1)))],
        "Y": [_add_year(m, _y) for m in month_codes_only_for_period("Y")]
    }[_period]
    return months

def full_period_year_in_list(tenor_list, year, tenor_period):
    # Return true for example for Q1, 2028 only if Jan_28, Feb_28 and Mar_28 are present in tenor_list, otherwise False
    year_months_for_period = tenor_to_monthly_strip(tenor_period + "_" + year)
    # tenor_months = month_codes_only_for_period(tenor_period)
    return all([t in tenor_list for t in year_months_for_period])


def infer_ratios_from_monthly_quotes(forward_quotes, period_to_infer_ratios):
    assert all(is_tenor_month(t) for t in forward_quotes), f"Contains non-month tenors, cannot infer shape: {forward_quotes}"

    q_ratios = {}
    for q_str in period_to_infer_ratios: # E.j., period_to_infer_ratios = ["Q1", "Q2", "Q3", "Q4", "SS", "WS", "Y"]:
        months = month_codes_only_for_period(q_str)
        q_quotes = {q:forward_quotes[q] for q in forward_quotes if get_month_from_tenor(q) in months}

        year_to_infer_ratios = "-1"
        for y in range(15, 30): # Very long shot here
            if full_period_year_in_list(q_quotes, str(y), q_str) and year_to_infer_ratios=="-1":
                year_to_infer_ratios = str(y)

                full_quotes = {
                    get_month_from_tenor(tenor): q_quotes[tenor]
                    for tenor in tenor_to_monthly_strip(q_str + "_" + year_to_infer_ratios)
                }

                average_period = sum(full_quotes.values())/len(full_quotes)
                q_ratios[q_str] = {m: full_quotes[m]/average_period for m in months}
        if year_to_infer_ratios == "-1":
            # Period is just not fully represented
            q_ratios[q_str] = {m: 1.0 for m in months}

    return q_ratios



def get_T_and_D_from_tenor(pricing_date, tenor="M1"):
    # Implements the logic of:
    # start of tenor = T = M1.start (date) - pricing_date
    # end of tenor = M1.end (date) - start_date
    str_tenor = tenor[0]
    assert str_tenor == "M", f"Only monthly tenors supported for now"
    n_periods = int(tenor[1:]) # Months for now only

    actual_date = pricing_date + timedelta(days=5) # hardcoded expiry date for now, approximation

    tenor_start_date = get_first_of_mth_next_month(actual_date, n_periods)
    tenor_end_date = get_first_of_mth_next_month(tenor_start_date, 1)
    T = time_between(pricing_date, tenor_start_date)
    D = time_between(tenor_start_date, tenor_end_date)
    return T, D


def parse_month(s):
    dt = parse_date(s, ("%b-%y",))
    return dt.year, dt.month


def _parse_date_or_datetime(_input, formats):
    """Implementation for parse_date() and parse_datetime()"""
    if isinstance(_input, (date, datetime)):
        return force_datetime(_input)
    if _input:
        _input = _input.strip()
    for fmt in formats:
        try:
            return datetime.strptime(_input, fmt)
        except ValueError:
            pass
    raise ValueError("Can't parse '%s' as date/datetime with formats (%s)" % (_input, formats))


date_formats = (
    "%Y-%m-%d", "%Y%m%d", "%Y_%m_%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%y", "%Y-%b-%d", "%d-%b-%Y", "%d-%m-%y",
    "%a %d/%m/%Y", "%Y-%m-%d 00:00", "%Y-%m-%d 00:00:00", "%Y-%m-%dT00:00:00",)


@lru_cache(128)
def parse_date(_input, formats=date_formats):
    """ The list of formats should be unambiguous: If the year comes first, it must be 4 digits.
    The other formats are really to support legacy input formats...
    """
    if isinstance(_input, (float, int)):
        return convert_ole_to_date(round(_input))
    return (_parse_date_or_datetime(_input, formats)).date()


def parse_datetime(_input, formats=(
        "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S.%f"
)):
    """ The list of formats should be unambiguous: If the year comes first, it must be 4 digits.
    The other formats are really to support legacy input formats...
    """
    return _parse_date_or_datetime(_input, formats)


# Hacks that are artefacts from our mixed usage of datetime and date in the power station payoff calculators
def force_datetime(d):
    """Transforms a date into a datetime"""
    if type(d) == type(date(2013, 1, 1)):
        return datetime.combine(d, time())
    return d


def force_date(d):
    """Given a date or a datetime, guarantees to return a date"""
    return date(d.year, d.month, d.day)


# Define the weekday mnemonics to match the date.weekday function
(MON, TUE, WED, THU, FRI, SAT, SUN) = range(7)


def workdays(start_date, end_date, which_days=(MON, TUE, WED, THU, FRI)):
    """Calculate the number of working days between two dates inclusive (start_date <= end_date).

    The actual working days can be set with the optional which_days parameter (default is MON-FRI)
    """
    delta_days = (end_date - start_date).days + 1
    full_weeks, extra_days = divmod(delta_days, 7)
    # num_workdays = how many days/week you work * total # of weeks
    num_workdays = (full_weeks + 1) * len(which_days)
    # subtract out any working days that fall in the 'shortened week'
    end_weekday = end_date.weekday()
    for d in range(1, 8 - extra_days):
        weekday = (end_weekday + d) % 7
        if weekday in which_days:
            num_workdays -= 1
    return num_workdays


def create_date(year, month, day):
    plus_years, remaining_month = divmod((month - 1), 12)
    remaining_month += 1
    return date(year + plus_years, remaining_month, day)


def get_nth_of_next_month(d, n):
    return create_date(d.year, d.month + 1, n)


def get_first_of_next_month(d):
    return get_nth_of_next_month(d, 1)

def get_nth_of_mth_next_month(d, n, m):
    return create_date(d.year, d.month + m, n)

def get_first_of_mth_next_month(d, m):
    return get_nth_of_mth_next_month(d, 1, m)


def is_weekday(d):
    return d.isoweekday() < 6


def is_weekend_day(d):
    return not is_weekday(d)


def previous_weekday(d):
    """Returns the previous weekday to the day"""
    d_minus_1 = d - timedelta(days=1)
    return d_minus_1 - timedelta(days=max(0, d_minus_1.isoweekday() - 5))


def next_weekday(d):
    d_plus_1 = d + timedelta(days=1)
    if d_plus_1.isoweekday() == 6:
        return d_plus_1 + timedelta(2)
    if d_plus_1.isoweekday() == 7:
        return d_plus_1 + timedelta(1)
    return d_plus_1


def this_or_next_weekday(d):
    if is_weekday(d):
        return d
    return next_weekday(d)


def this_or_previous_weekday(d):
    if is_weekday(d):
        return d
    return previous_weekday(d)


def convert_ole_to_date(ole_date):
    date_1_jan_1980 = date(1980, 1, 1)
    xl_date_1_jan_1980 = 29221
    day_delta = ole_date - xl_date_1_jan_1980
    output_datetime = date_1_jan_1980 + timedelta(days=day_delta)
    return output_datetime


def convert_datetime_to_ole(dt):
    jan_1_1970_xl = 25569
    return jan_1_1970_xl + (dt - datetime(1970, 1, 1)).total_seconds() / (24 * 60 * 60)


def last_occurrence_of_day_in_month(year, month, day_index):
    """Given an index of a day, return the day of the month when this last occurs.
    e.g Last Sunday (day_index=6) in August 2021 -> 29th
    """
    return max(week[day_index] for week in calendar.monthcalendar(year, month))


def dst_start(y):
    """Last Sunday in March for a given year"""
    march = 3
    return date(y, march, last_occurrence_of_day_in_month(y, march, SUN))


def dst_end(y):
    """Last Sunday in October for a given year"""
    october = 10
    return date(y, october, last_occurrence_of_day_in_month(y, october, SUN))


def is_dst_start(d: date):
    """BST (DST) starts last Sunday in March. Check if date is this day."""
    if d.month != 3:
        return False
    elif d.weekday() != 6:
        return False
    elif d == dst_start(d.year):
        return True
    else:
        return False


def is_dst_end(d):
    """BST (DST) starts last Sunday in October. Check if date is this day."""
    if d.month != 10:
        return False
    elif d.weekday() != 6:
        return False
    elif d == dst_end(d.year):
        return True
    else:
        return False


class DateConverter(Converter):

    @classmethod
    def read_value(self, value: Any, options: Optional[dict] = None) -> Any:
        return parse_date(value)

    @classmethod
    def write_value(self, value: Any, options: Optional[dict] = None) -> Any:
        return value

DateConverter.register(datetime.date)
