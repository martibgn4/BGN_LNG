import calendar
from datetime import date, datetime, time, timedelta, tzinfo
from functools import lru_cache

from .memoize_utils import memoize

__all__ = [
    "DAYS_PER_YEAR",
    "MON",
    "TUE",
    "WED",
    "THU",
    "FRI",
    "SAT",
    "SUN",
    "time_between",
    "parse_date",
    "parse_datetime",
    "parse_month",
    "force_date",
    "force_datetime",
    "workdays",
    "days_split",
    "create_date",
    "get_nth_of_next_month",
    "get_first_of_next_month",
    "is_weekday",
    "is_weekend_day",
    "previous_weekday",
    "convert_ole_to_date",
    "convert_datetime_to_ole",
    "is_excel_date",
    "BST",
    "last_occurrence_of_day_in_month",
    "dst_start",
    "dst_end",
    "is_dst_start",
    "is_dst_end",
    "this_or_next_weekday",
    "this_or_previous_weekday",
    "next_weekday",
    "date_formats"
]

DAYS_PER_YEAR = 365.0


def time_between(start_date, end_date):
    """Calculates time between two dates as a fraction of a year"""
    return (end_date - start_date).days / DAYS_PER_YEAR


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


@memoize
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


@memoize
def days_split(start_date, end_date):
    """Calculate the number of mondays, tuesdays, ..  between two dates inclusive (start_date <= end_date)"""
    assert start_date <= end_date, f"Cannot split period because {start_date} is after {end_date}."
    delta_days = (end_date - start_date).days + 1
    full_weeks, extra_days = divmod(delta_days, 7)
    split = [full_weeks + 1] * 7
    # subtract out any working days that fall in the 'shortened week'
    end_weekday = end_date.weekday()
    for d in range(1, 8 - extra_days):
        weekday = (end_weekday + d) % 7
        split[weekday] -= 1
    return split  # list of num_of_days (index 0 is Monday, ... )


def create_date(year, month, day):
    plus_years, remaining_month = divmod((month - 1), 12)
    remaining_month += 1
    return date(year + plus_years, remaining_month, day)


def get_nth_of_next_month(d, n):
    return create_date(d.year, d.month + 1, n)


def get_first_of_next_month(d):
    return get_nth_of_next_month(d, 1)


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


@memoize
def dst_start(y):
    """Last Sunday in March for a given year"""
    march = 3
    return date(y, march, last_occurrence_of_day_in_month(y, march, SUN))


@memoize
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


class BST(tzinfo):

    def utcoffset(self, dt):
        return self.dst(dt)

    def dst(self, dt):
        # DST starts last Sunday in March
        d = datetime(dt.year, 4, 1)  # ends last Sunday in October
        dston = d - timedelta(days=d.weekday() + 1)
        d = datetime(dt.year, 11, 1)
        dstoff = d - timedelta(days=d.weekday() + 1)
        if dston <= dt.replace(tzinfo=None) < dstoff:
            return timedelta(hours=1)
        else:
            return timedelta(0)

    def tzname(self, dt):
        return "BST"


def is_excel_date(cell):
    """Assumes we're not looking at dates before 1970/1/1"""
    if isinstance(cell, (float, int)):
        return float(cell) >= 25569.00
    try:
        _ = parse_date(cell)
    except Exception:
        return False
    return True


