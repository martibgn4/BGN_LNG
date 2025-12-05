
import calendar
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Any, Optional

from xlwings.conversion import Converter

DAYS_PER_YEAR = 365.0

def time_between(start_date, end_date):
    """Calculates time between two dates as a fraction of a year"""
    return (end_date - start_date).days / DAYS_PER_YEAR

_period_str_to_months_dict = {
    "M": 1
}

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
