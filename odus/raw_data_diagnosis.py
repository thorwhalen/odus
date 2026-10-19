"""Diagnosis of the raw survey spreadsheets.

Reads the source ``.xlsx`` files with ``openpyxl`` (values *and* cell colors)
and reports the data-entry anomalies that must be resolved before the data can
be prepared: cells whose content and color disagree, values that aren't 0 or 1,
ages and years that don't follow each other, repeated variables...

Each check is a ``Rule``: a name, a function marking the problematic cells of a
``SheetView``, and a message explaining the problem. ``DFLT_RULES`` maps names to
the rules used by default; pick, drop or add rules with ``select_rules``, or by
passing any ``{name: Rule}`` mapping as ``rules``:

>>> sorted(select_rules(exclude=['filled_but_uncolored'])) # doctest: +NORMALIZE_WHITESPACE
['age_not_consecutive', 'age_year_mismatch', 'duplicate_variable',
 'empty_but_colored', 'not_binary', 'year_not_consecutive']

``diagnose_file`` and ``diagnose_dir`` yield one ``Problem`` per flagged cell,
``diagnosis_table`` puts them in a DataFrame, and ``print_diagnosis`` prints them.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
import os

import pandas as pd
import numpy as np
from openpyxl import (
    load_workbook,
)  # Get it here: https://pypi.org/project/openpyxl/ (or pip install openpyxl)


def pra(obj):
    print(
        *(x for x in dir(obj) if not x[0].startswith("_") and not x[0].isupper()),
        sep="\t",
    )


def last_true_idx(series):
    return len(series) - series.iloc[::-1].values.argmax()


def mk_dotpath_getter(dotpath):
    attrs = dotpath.split(".")

    def attr_get(x):
        for a in attrs:
            x = getattr(x, a)
        return x

    return attr_get


def dfof(sheet, dotpath="value"):
    attr_get = mk_dotpath_getter(dotpath)
    data = [[attr_get(c) for c in row] for row in sheet]
    return pd.DataFrame(data)


def coord_val_and_bgcolor_of_cell(c):
    return {"coordinate": c.coordinate, "value": c.value, "rgb": c.fill.bgColor.rgb}


def crop(df, last_row_idx, last_col_idx):
    return df.iloc[:last_row_idx, :last_col_idx]


def matmask(df, mask):
    if isinstance(mask, pd.DataFrame):
        mask = mask.values
    d = df.values.copy()
    d[~mask] = ""
    return pd.DataFrame(d, index=df.index, columns=df.columns)


def row_lidx_to_mat_lidx(row_lidx, n_cols):
    return np.tile(row_lidx, (n_cols, 1)).T


white = "00000000"


def valid_vals_int_coords(vals_df):
    first_column = vals_df.iloc[:, 0]
    last_row_idx = last_true_idx(
        (first_column != "List of States") & (~first_column.isna())
    )

    age_row = vals_df.iloc[2, :]
    last_col_idx = last_true_idx(~age_row.isna())

    return last_row_idx, last_col_idx


def sheet_to_format_prepped_df(sheet):
    vals = dfof(sheet, dotpath="value")
    color = dfof(sheet, dotpath="fill.bgColor.rgb")
    coordinate = dfof(sheet, dotpath="coordinate")
    #     print(vals.shape, color.shape, coordinate.shape)

    int_coords = valid_vals_int_coords(vals)
    vals, color, coordinate = list(
        map(lambda x: crop(x, *int_coords), (vals, color, coordinate))
    )
    #     print(vals.shape, color.shape, coordinate.shape)

    df = pd.concat(
        (vals, color, coordinate), axis=1, keys=("vals", "color", "coordinate")
    )
    #     df = df[~df.iloc[:, 0].isna()]

    return df


def lidx_of_empty_non_white_or_non_empty_white(df):
    is_null = df.vals.isnull()
    is_white = df.color == white
    return (is_null & ~is_white) | (~is_null & is_white)


def mk_known_exceptions_lidx(vals_df):
    n_rows, n_cols = vals_df.shape
    mk_mat_lidx = partial(row_lidx_to_mat_lidx, n_cols=n_cols)

    lidx = mk_mat_lidx([False] * 3 + [True] * (n_rows - 3))
    lidx[:, 0] = False
    lidx &= mk_mat_lidx(~vals_df.iloc[:, 0].isna())
    lidx &= mk_mat_lidx(~(vals_df.iloc[:, 0] == "Select States below"))
    return lidx


def diagnosis_lidx(df):
    lidx = lidx_of_empty_non_white_or_non_empty_white(df)
    lidx &= mk_known_exceptions_lidx(df.vals)
    return lidx


def diagnosis_lidx_of_xls_file(xls_filepath, sheetname=None):
    wb = load_workbook(xls_filepath, data_only=True)
    sheetname = sheetname or next(iter(wb.sheetnames))
    sheet = wb[sheetname]

    df = sheet_to_format_prepped_df(sheet)
    lidx = lidx_of_empty_non_white_or_non_empty_white(df)
    lidx &= mk_known_exceptions_lidx(df.vals)
    return df, lidx


def diagnosis_items(source_dir, sheetname=None):
    is_excel = lambda x: x.endswith(".xlsx")
    join_dir = lambda *p: os.path.join(source_dir, *p)
    for xls_filepath in map(join_dir, filter(is_excel, os.listdir(source_dir))):
        filename = os.path.basename(xls_filepath)
        df, lidx = diagnosis_lidx_of_xls_file(xls_filepath, sheetname)
        if lidx.sum().sum() != 0:
            yield filename, df, lidx


# --------------------------------------------------------------------------------------
# Rule-based diagnosis

YEAR_ROW = 1  # row holding the years (0-based, in the sheet)
AGE_ROW = 2  # row holding the ages
DATA_START_ROW = 3  # first row of variables
NON_VARIABLE_LABELS = ("List of States", "Select States below")  # section titles


def _is_colored(cell):
    fill = cell.fill
    return fill is not None and fill.fill_type is not None


def _is_empty(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or x == ""


@dataclass
class SheetView:
    """The parts of a survey sheet that rules look at.

    ``vals``, ``colored`` and ``coordinate`` are aligned DataFrames (one entry per
    cell). ``label_col`` is the column holding the variable names, and
    ``data_mask`` marks the cells holding a variable's value at some age.
    """

    vals: pd.DataFrame
    colored: pd.DataFrame
    coordinate: pd.DataFrame
    label_col: int
    data_mask: np.ndarray
    year_row: int = YEAR_ROW
    age_row: int = AGE_ROW

    @property
    def data_cols(self):
        return list(range(self.label_col + 1, self.vals.shape[1]))

    @property
    def labels(self):
        return self.vals.iloc[:, self.label_col]

    def empty(self):
        """Boolean array: is the cell empty?"""
        return self.vals.map(_is_empty).to_numpy(dtype=bool)

    def numeric(self):
        """Float array of the values: NaN where the cell doesn't hold a number."""
        return self.vals.map(lambda x: float(x) if _is_number(x) else np.nan).to_numpy(
            dtype=float
        )

    def false_mask(self):
        return np.zeros(self.vals.shape, dtype=bool)


def _label_col(vals, *, data_start_row):
    """The first column whose first variable row holds text (the variable names)."""
    row = vals.iloc[data_start_row]
    for j, x in enumerate(row):
        if isinstance(x, str) and x.strip():
            return j
    raise ValueError(
        f"Couldn't find the column of variable names: no text in row "
        f"{data_start_row + 1} of the sheet"
    )


def sheet_view(
    sheet,
    *,
    year_row=YEAR_ROW,
    age_row=AGE_ROW,
    data_start_row=DATA_START_ROW,
    non_variable_labels=NON_VARIABLE_LABELS,
):
    """Make a ``SheetView`` of an ``openpyxl`` worksheet.

    The sheet is cropped to its last labeled row and last column with an age.
    """
    rows = [list(row) for row in sheet]
    vals = pd.DataFrame([[c.value for c in row] for row in rows], dtype=object)
    colored = pd.DataFrame([[_is_colored(c) for c in row] for row in rows])
    coordinate = pd.DataFrame([[c.coordinate for c in row] for row in rows])
    if vals.shape[0] <= data_start_row:
        raise ValueError(f"The sheet has only {vals.shape[0]} rows: Nothing to check")

    label_col = _label_col(vals, data_start_row=data_start_row)
    labels = vals.iloc[:, label_col]
    is_row_label = labels.map(lambda x: not _is_empty(x))
    n_rows = last_true_idx(is_row_label & ~labels.isin(non_variable_labels))
    has_age = vals.iloc[age_row, label_col + 1 :].map(lambda x: not _is_empty(x))
    if not has_age.any():
        raise ValueError(f"No ages found in row {age_row + 1} of the sheet")
    n_cols = label_col + 1 + last_true_idx(has_age)
    vals, colored, coordinate = (
        crop(x, n_rows, n_cols) for x in (vals, colored, coordinate)
    )

    is_variable_row = np.zeros(n_rows, dtype=bool)
    is_variable_row[data_start_row:] = True
    labels = vals.iloc[:, label_col]
    is_variable_row &= labels.map(lambda x: not _is_empty(x)).to_numpy()
    is_variable_row &= ~labels.isin(non_variable_labels).to_numpy()
    data_mask = np.zeros((n_rows, n_cols), dtype=bool)
    data_mask[np.ix_(is_variable_row, range(label_col + 1, n_cols))] = True
    return SheetView(
        vals,
        colored.astype(bool),
        coordinate,
        label_col,
        data_mask,
        year_row=year_row,
        age_row=age_row,
    )


@dataclass(frozen=True)
class Rule:
    """A diagnosis rule.

    Args:
        name: Identifier of the rule (and its key in a rules mapping).
        flag: ``flag(view: SheetView) -> bool array`` marking the problematic cells
            (same shape as ``view.vals``).
        message: Explanation shown for each flagged cell. Can use ``{value}``,
            ``{coordinate}`` and ``{label}`` (the variable of the cell's row).
    """

    name: str
    flag: Callable[[SheetView], np.ndarray]
    message: str = field(default="")


@dataclass(frozen=True)
class Problem:
    """A cell flagged by a rule."""

    filename: str
    rule: str
    coordinate: str
    value: object
    message: str


def _empty_but_colored(view):
    return view.data_mask & view.empty() & view.colored.to_numpy()


def _filled_but_uncolored(view):
    if not view.colored.to_numpy().any():
        return view.false_mask()  # the sheet uses no colors: nothing to compare with
    return view.data_mask & ~view.empty() & ~view.colored.to_numpy()


def _not_binary(view):
    def is_bad(x):
        if _is_empty(x) or isinstance(x, bool):
            return False
        return not (isinstance(x, (int, float)) and x in (0, 1))

    return view.data_mask & view.vals.map(is_bad).to_numpy(dtype=bool)


def _header_row_flags(view, row, is_bad_step):
    """Flag cells of a header row whose step from the previous cell is bad."""
    mask = view.false_mask()
    cols = view.data_cols
    for prev, j in zip(cols, cols[1:]):
        a, b = view.vals.iat[row, prev], view.vals.iat[row, j]
        if _is_number(a) and _is_number(b):
            mask[row, j] = is_bad_step(a, b)
        elif not _is_number(b):
            mask[row, j] = True  # not a number at all
    if cols and not _is_number(view.vals.iat[row, cols[0]]):
        mask[row, cols[0]] = True
    return mask


def _is_number(x):
    return (
        isinstance(x, (int, float, np.number))
        and not isinstance(x, bool)
        and (not np.isnan(x))
    )


def _age_not_consecutive(view):
    return _header_row_flags(view, view.age_row, lambda a, b: b - a != 1)


def _year_not_consecutive(view):
    return _header_row_flags(view, view.year_row, lambda a, b: b - a != 1)


def _age_year_mismatch(view):
    """Flag ages for which ``year - age`` differs from that of the first column."""
    mask = view.false_mask()
    diffs = {
        j: view.vals.iat[view.year_row, j] - view.vals.iat[view.age_row, j]
        for j in view.data_cols
        if _is_number(view.vals.iat[view.year_row, j])
        and _is_number(view.vals.iat[view.age_row, j])
    }
    if diffs:
        expected = next(iter(diffs.values()))
        for j, d in diffs.items():
            mask[view.age_row, j] = d != expected
    return mask


def _duplicate_variable(view):
    mask = view.false_mask()
    rows = np.flatnonzero(view.data_mask.any(axis=1))
    labels = view.labels.iloc[rows]
    mask[rows[labels.duplicated(keep="first").to_numpy()], view.label_col] = True
    return mask


DFLT_RULES = {
    rule.name: rule
    for rule in (
        Rule(
            "empty_but_colored",
            _empty_but_colored,
            "{coordinate} ({label}) is colored but empty",
        ),
        Rule(
            "filled_but_uncolored",
            _filled_but_uncolored,
            "{coordinate} ({label}) has a value ({value!r}) but no color",
        ),
        Rule(
            "not_binary",
            _not_binary,
            "{coordinate} ({label}) should be 0 or 1, but is {value!r}",
        ),
        Rule(
            "age_not_consecutive",
            _age_not_consecutive,
            "age {value!r} in {coordinate} doesn't follow the previous age",
        ),
        Rule(
            "year_not_consecutive",
            _year_not_consecutive,
            "year {value!r} in {coordinate} doesn't follow the previous year",
        ),
        Rule(
            "age_year_mismatch",
            _age_year_mismatch,
            "age {value!r} in {coordinate} doesn't match its year "
            "(year - age differs from the first column's)",
        ),
        Rule(
            "duplicate_variable",
            _duplicate_variable,
            "{coordinate}: variable {value!r} appears more than once",
        ),
    )
}


def select_rules(include=None, exclude=(), *, extra=None, rules=None):
    """Make a ``{name: Rule}`` mapping: turn rules on or off, and add your own.

    Args:
        include: Names of the rules to keep (default: all of ``rules``).
        exclude: Names of the rules to drop.
        extra: More rules to add: a ``{name: Rule}`` mapping or an iterable of
            ``Rule`` instances.
        rules: The rules to start from (default: ``DFLT_RULES``).

    >>> list(select_rules(include=['not_binary']))
    ['not_binary']
    >>> my_rule = Rule('never', lambda view: view.false_mask(), 'never happens')
    >>> 'never' in select_rules(extra=[my_rule])
    True
    """
    rules = dict(DFLT_RULES if rules is None else rules)
    unknown = (set(include or ()) | set(exclude)) - set(rules)
    if unknown:
        raise KeyError(
            f"Unknown rule(s): {sorted(unknown)}. Known rules: {sorted(rules)}"
        )
    if include is not None:
        rules = {name: rules[name] for name in include}
    for name in exclude:
        rules.pop(name, None)
    if extra is not None:
        if isinstance(extra, Mapping):
            extra = extra.values()
        rules.update({rule.name: rule for rule in extra})
    return rules


def _iter_rules(rules):
    if isinstance(rules, Mapping):
        return rules.values()
    return list(rules)


def diagnose_view(view, *, rules=DFLT_RULES, filename=""):
    """Yield the ``Problem``s that ``rules`` find in a ``SheetView``."""
    for rule in _iter_rules(rules):
        mask = np.asarray(rule.flag(view), dtype=bool)
        if mask.shape != view.vals.shape:
            raise ValueError(
                f"Rule {rule.name!r} returned a mask of shape {mask.shape}, but the "
                f"sheet's shape is {view.vals.shape}"
            )
        for i, j in zip(*np.nonzero(mask)):
            value = view.vals.iat[i, j]
            coordinate = view.coordinate.iat[i, j]
            label = view.vals.iat[i, view.label_col]
            message = rule.message.format(
                value=value, coordinate=coordinate, label=label
            )
            yield Problem(filename, rule.name, coordinate, value, message)


def diagnose_file(xls_filepath, sheetname=None, *, rules=DFLT_RULES, **view_kwargs):
    """Yield the ``Problem``s that ``rules`` find in a spreadsheet.

    ``view_kwargs`` (``year_row``, ``age_row``...) go to ``sheet_view``.
    """
    wb = load_workbook(xls_filepath, data_only=True)
    sheet = wb[sheetname] if sheetname is not None else wb[wb.sheetnames[0]]
    view = sheet_view(sheet, **view_kwargs)
    filename = os.path.basename(xls_filepath)
    yield from diagnose_view(view, rules=rules, filename=filename)


def diagnose_dir(source_dir, sheetname=None, *, rules=DFLT_RULES, **view_kwargs):
    """Yield the ``Problem``s found in the ``.xlsx`` files of ``source_dir``."""
    source_dir = os.path.abspath(os.path.expanduser(source_dir))
    for filename in sorted(os.listdir(source_dir)):
        if filename.endswith(".xlsx") and not filename.startswith("~$"):
            yield from diagnose_file(
                os.path.join(source_dir, filename),
                sheetname,
                rules=rules,
                **view_kwargs,
            )


def diagnosis_table(problems: Iterable[Problem]) -> pd.DataFrame:
    """A DataFrame with one row per problem (columns: the ``Problem`` fields)."""
    columns = list(Problem.__dataclass_fields__)
    return pd.DataFrame(
        [[getattr(p, c) for c in columns] for p in problems], columns=columns
    )


def print_diagnosis(source_dir=".", sheetname=None, *, rules=DFLT_RULES):
    """Print the diagnosis of the raw xls files in `source_dir`.

    For every file that has a problem, print its name, then for each rule that
    flagged something, the rule's name and the explanation of each flagged cell.

    The checks are the ``rules`` (default ``DFLT_RULES``); see the module docstring
    and ``select_rules`` to turn some off or add your own. For instance, the
    color rules only make sense for spreadsheets where filled cells are colored.
    """
    source_dir = os.path.abspath(os.path.expanduser(source_dir))
    print(f'Diagnosing xls files of "{source_dir}" ...\n\n')

    problems = diagnose_dir(source_dir, sheetname, rules=rules)
    n_files = 0
    for filename, file_problems in groupby(problems, key=lambda p: p.filename):
        n_files += 1
        print(f"The possibly problematic cells for {filename}:")
        for rule, rule_problems in groupby(file_problems, key=lambda p: p.rule):
            print(f"  {rule}:")
            for p in rule_problems:
                print(f"    - {p.message}")
        print()
    if n_files == 0:
        print("No problems found.")
