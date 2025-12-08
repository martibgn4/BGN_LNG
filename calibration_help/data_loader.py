import json
from functools import wraps

from official_config import get_full_config_file_path
from thorn.api import load_underlyings as thorn_api_load_underlyings
from xlwings.udfs import xlfunc

CONFIG = None


def null_operation(*args, **kwargs):
    """ Function that does nothing and returns None """
    pass


def load_underlyings(f):
    """Load the underlyings config file."""
    orig_xlfunc = xlfunc()(f)

    @wraps(f)
    def wrapper(*args):
        thorn_api_load_underlyings(null_operation)()
        return_val = f(*args)
        return return_val

    wrapper_xlfunc = xlfunc()(wrapper)
    wrapper_xlfunc.__xlfunc__ = orig_xlfunc.__xlfunc__

    return wrapper_xlfunc


def get_config_value(config_key):
    global CONFIG
    if CONFIG is None:
        try:
            config_path = get_full_config_file_path('thorn_excel/config.json')
            with open(config_path) as fh:
                CONFIG = json.load(fh)
        except Exception as e:
            raise e.__class__("Failed to load thorn_excel config") from e

    return CONFIG[config_key]


