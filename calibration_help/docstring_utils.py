from xlwings.udfs import xlfunc
from functools import wraps

DOCUMENTED = []
TAB = "    "


def get_first_paragraph(doc):
    loc = doc.find("\n\n")
    return doc[:loc]


def docstring_formatter(f):
    orig_xlfunc = xlfunc()(f)
    global DOCUMENTED
    DOCUMENTED.append(f.__name__)

    @wraps(f)
    def wrapper(*args):
        return_val = f(*args)
        return return_val

    wrapper_xlfunc = xlfunc()(wrapper)
    wrapper_xlfunc.__xlfunc__ = orig_xlfunc.__xlfunc__
    docstring = orig_xlfunc.__xlfunc__['ret']['doc']
    wrapper_xlfunc.__xlfunc__['ret']['doc'] = get_first_paragraph(docstring)

    docstring += (f"\n{TAB}"
                  f"\n{TAB}Parameters"
                  f"\n{TAB}----------")

    for arg in orig_xlfunc.__xlfunc__['args']:
        docstring += (f"\n{TAB}{arg['name']}:"
                      f"\n{TAB}{TAB}{arg['doc']}")

    if 'doc' in orig_xlfunc.__xlfunc__['ret']['options']:
        docstring += (f"\n{TAB}"
                      f"\n{TAB}Returns\n{TAB}-------"
                      f"\n{TAB}{TAB}_"
                      f"\n{TAB}{TAB}{TAB}{orig_xlfunc.__xlfunc__['ret']['options']['doc']}"
                      )

    wrapper_xlfunc.__doc__ = docstring

    return wrapper_xlfunc


