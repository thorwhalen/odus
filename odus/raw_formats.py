"""Reading raw survey data, of several formats, into trajectories.

A *trajectory* is a DataFrame with one row per age (the index, named ``age``) and
one column per variable, holding the respondent's value of that variable at that
age (typically 0 or 1). Its ``attrs[YEAR_OF_AGE]`` maps each age to its year.

What varies from one data source to another is configuration, not code:

- **The file format**: ``read_raw_table`` picks a reader in ``TABLE_READERS`` by
  file extension (``.xlsx``, ``.xls``, ``.csv``, ``.tsv``). Pass ``readers=`` with
  more entries to read other formats.
- **The layout of a "wide" table** (variables as rows, ages as columns, as in the
  ODUS spreadsheets): a ``SurveyLayout`` says where the years, the ages, the
  variable names and the values are. ``trajectory_from_table`` reads one.
- **"Long" (tidy) tables**, with one row per (age, variable, value), are read by
  ``trajectory_from_long``.

>>> raw = pd.DataFrame([
...     [None, 'ID', 'X01', None],
...     [0, 'YEAR', 2000, 2001],
...     [1, 'AGE', 20, 21],
...     [3, 'HEROIN', 0, 1],
...     [4, 'WORK', 1, 1],
... ])
>>> t = trajectory_from_table(raw)
>>> t.index.tolist(), t.columns.tolist(), t.values.tolist()
([20, 21], ['HEROIN', 'WORK'], [[0, 1], [1, 1]])
>>> t.attrs[YEAR_OF_AGE]
{20: 2000, 21: 2001}
"""

from dataclasses import dataclass
from functools import partial
from io import BytesIO
import os

import numpy as np
import pandas as pd

YEAR_OF_AGE = "year_of_age"  # attrs key of a trajectory's {age: year} mapping
SECTION_TITLES = (  # labels of rows that title a section, rather than a variable
    "List of States",
    "Select States below",
    "Selects States below",  # a typo found in the surveys
)


@dataclass(frozen=True)
class SurveyLayout:
    """Where things are in a "wide" survey table (0-based row and column numbers).

    Args:
        year_row: The row holding the years (None if there's none).
        age_row: The row holding the ages.
        data_start_row: The first row of variables.
        label_col: The column holding the variable names. None: the column with
            the most text from ``data_start_row`` on. The ages and values are in
            the columns after it.
        section_titles: Labels of rows that title a section rather than name a
            variable (compared ignoring case and extra spaces). Those rows are
            skipped.
        empty_value: The value of an empty data cell. None: an empty cell is an
            error.
    """

    year_row: int | None = 1
    age_row: int = 2
    data_start_row: int = 3
    label_col: int | None = None
    section_titles: tuple = SECTION_TITLES
    empty_value: float | None = None


DFLT_LAYOUT = SurveyLayout()  # the layout of the ODUS spreadsheets


def normalized_label(label):
    """A label with case and extra whitespace ignored, to compare labels.

    >>> normalized_label('  New   York ') == normalized_label('new york')
    True
    """
    return " ".join(str(label).split()).casefold()


def is_section_title(label, *, section_titles=SECTION_TITLES):
    """Whether a row label is a section title (like "Select States below").

    >>> is_section_title(' select states BELOW'), is_section_title('HEROIN')
    (True, False)
    """
    return normalized_label(label) in set(map(normalized_label, section_titles))


def is_empty(x):
    """Whether a cell value is empty (None, NaN or an empty string)."""
    return x is None or (isinstance(x, float) and np.isnan(x)) or x == ""


def find_label_col(vals, *, data_start_row):
    """The column with the most text from ``data_start_row`` on (the variable names)."""
    is_text = vals.iloc[data_start_row:].map(
        lambda x: isinstance(x, str) and bool(x.strip())
    )
    n_texts = is_text.sum(axis=0).to_numpy()
    if not n_texts.any():
        raise ValueError(
            f"Couldn't find the column of variable names: no text from row "
            f"{data_start_row + 1} of the table on"
        )
    return int(np.argmax(n_texts))


def _read_excel(src, **kwargs):
    return pd.read_excel(src, header=None, **kwargs)


TABLE_READERS = {  # file extension -> function(file path or file-like) -> DataFrame
    ".xlsx": _read_excel,
    ".xls": _read_excel,
    ".csv": partial(pd.read_csv, header=None),
    ".tsv": partial(pd.read_csv, header=None, sep="\t"),
}


def read_raw_table(src, *, ext=None, readers=TABLE_READERS):
    """Read a raw table, as is (no header), with the reader of its file extension.

    Args:
        src: A file path, or the file's bytes.
        ext: The file extension (like ``".csv"``). Default: the one of ``src``
            (required when ``src`` is bytes).
        readers: ``{extension: reader}``; a reader takes a path or file-like
            object and returns a DataFrame.
    """
    if ext is None:
        if not isinstance(src, (str, os.PathLike)):
            raise ValueError("Give the file extension (ext=...) to read bytes")
        ext = os.path.splitext(str(src))[1]
    reader = readers.get(ext.lower())
    if reader is None:
        raise ValueError(
            f"No reader for {ext!r} files. Readers exist for: {sorted(readers)}. "
            "Pass readers={..., ext: reader} to read others."
        )
    if isinstance(src, (bytes, bytearray)):
        src = BytesIO(src)
    return reader(src).astype(object)


def _as_number(x, *, where):
    if isinstance(x, (int, float, np.number)) and not isinstance(x, bool):
        return x
    try:
        return float(str(x).strip())
    except ValueError:
        raise ValueError(f"{where} should be a number, but is {x!r}") from None


def _as_int_if_integral(values):
    values = np.asarray(values, dtype=float)
    if np.all(np.isfinite(values)) and np.all(values == np.round(values)):
        return values.astype(int)
    return values


def trajectory_from_table(raw, *, layout=DFLT_LAYOUT):
    """Make a trajectory from a "wide" table, read as is (e.g. by ``read_raw_table``).

    Rows from ``layout.data_start_row`` on with a (non section title) label are
    variables; columns after the label column with an age are ages.
    """
    raw = raw.astype(object)
    n_rows = raw.shape[0]
    if n_rows <= max(layout.age_row, layout.data_start_row - 1):
        raise ValueError(f"The table has only {n_rows} rows: no ages or variables")
    label_col = layout.label_col
    if label_col is None:
        label_col = find_label_col(raw, data_start_row=layout.data_start_row)

    age_cells = raw.iloc[layout.age_row]
    cols = [j for j in range(label_col + 1, raw.shape[1]) if not is_empty(age_cells[j])]
    if not cols:
        raise ValueError(f"No ages found in row {layout.age_row + 1} of the table")
    ages = _as_int_if_integral(
        [_as_number(age_cells[j], where=f"The age in column {j + 1}") for j in cols]
    )

    labels = raw.iloc[:, label_col]
    rows = [
        i
        for i in range(layout.data_start_row, n_rows)
        if not is_empty(labels[i])
        and not is_section_title(labels[i], section_titles=layout.section_titles)
    ]
    if not rows:
        raise ValueError("No variables found in the table")

    def value(i, j):
        x = raw.iat[i, j]
        if is_empty(x):
            if layout.empty_value is None:
                raise ValueError(
                    f"The value of {labels[i]!r} at age {ages[cols.index(j)]} is "
                    "empty. Fill it in, or set the layout's empty_value"
                )
            return layout.empty_value
        return _as_number(x, where=f"The value of {labels[i]!r} at row {i + 1}")

    values = _as_int_if_integral([[value(i, j) for i in rows] for j in cols])
    df = pd.DataFrame(
        values,
        index=pd.Index(ages, name="age"),
        columns=pd.Index([labels[i] for i in rows], name="category"),
    )
    if layout.year_row is not None:
        year_cells = raw.iloc[layout.year_row]
        years = [year_cells[j] for j in cols]
        if not any(map(is_empty, years)):
            years = _as_int_if_integral([_as_number(y, where="A year") for y in years])
            df.attrs[YEAR_OF_AGE] = dict(zip(ages.tolist(), years.tolist()))
    return df


def trajectory_from_long(
    df, *, age="age", variable="variable", value="value", year=None, fill_value=0
):
    """Make a trajectory from a "long" table: one row per (age, variable, value).

    Args:
        df: The long table.
        age, variable, value: The names of its columns holding those.
        year: The name of the column holding the year (optional).
        fill_value: The value of (age, variable) pairs absent from the table.

    >>> long = pd.DataFrame({
    ...     'age': [20, 21, 21], 'year': [2000, 2001, 2001],
    ...     'variable': ['HEROIN', 'HEROIN', 'WORK'], 'value': [0, 1, 1],
    ... })
    >>> t = trajectory_from_long(long, year='year')
    >>> t.values.tolist(), t.attrs[YEAR_OF_AGE]
    ([[0, 0], [1, 1]], {20: 2000, 21: 2001})
    """
    missing = [c for c in (age, variable, value, year) if c is not None and c not in df]
    if missing:
        raise KeyError(f"The table has no columns {missing}: it has {list(df.columns)}")
    repeated = df.duplicated([age, variable])
    if repeated.any():
        raise ValueError(
            f"Some (age, variable) pairs appear more than once: "
            f"{df.loc[repeated, [age, variable]].values.tolist()[:10]}"
        )
    traj = df.pivot(index=age, columns=variable, values=value)
    traj = traj.reindex(sorted(traj.index)).fillna(fill_value)
    traj = traj.astype(int) if (traj == traj.round()).all().all() else traj
    traj.index.name, traj.columns.name = "age", "category"
    if year is not None:
        years = df.groupby(age)[year].unique()
        if inconsistent := [a for a, ys in years.items() if len(ys) > 1]:
            raise ValueError(f"Ages with more than one year: {inconsistent}")
        traj.attrs[YEAR_OF_AGE] = {a: ys[0].item() for a, ys in years.items()}
    return traj
