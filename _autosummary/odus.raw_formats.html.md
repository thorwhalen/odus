# odus.raw_formats

Reading raw survey data, of several formats, into trajectories.

A *trajectory* is a DataFrame with one row per age (the index, named `age`) and
one column per variable, holding the respondent’s value of that variable at that
age (typically 0 or 1). Its `attrs[YEAR_OF_AGE]` maps each age to its year.

What varies from one data source to another is configuration, not code:

- **The file format**: `read_raw_table` picks a reader in `TABLE_READERS` by
  file extension (`.xlsx`, `.xls`, `.csv`, `.tsv`). Pass `readers=` with
  more entries to read other formats.
- **The layout of a “wide” table** (variables as rows, ages as columns, as in the
  ODUS spreadsheets): a `SurveyLayout` says where the years, the ages, the
  variable names and the values are. `trajectory_from_table` reads one.
- **“Long” (tidy) tables**, with one row per (age, variable, value), are read by
  `trajectory_from_long`.

```pycon
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
```

### Functions

| [`find_label_col`](#odus.raw_formats.find_label_col)(vals, \*, data_start_row)           | The column with the most text from `data_start_row` on (the variable names).   |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`is_empty`](#odus.raw_formats.is_empty)(x)                                        | Whether a cell value is empty (None, NaN or an empty string).                  |
| [`is_section_title`](#odus.raw_formats.is_section_title)(label, \*[, section_titles])      | Whether a row label is a section title (like "Select States below").           |
| [`normalized_label`](#odus.raw_formats.normalized_label)(label)                            | A label with case and extra whitespace ignored, to compare labels.             |
| [`read_raw_table`](#odus.raw_formats.read_raw_table)(src, \*[, ext, readers])            | Read a raw table, as is (no header), with the reader of its file extension.    |
| [`trajectory_from_long`](#odus.raw_formats.trajectory_from_long)(df, \*[, age, variable, ...]) | Make a trajectory from a "long" table: one row per (age, variable, value).     |
| [`trajectory_from_table`](#odus.raw_formats.trajectory_from_table)(raw, \*[, layout])           | Make a trajectory from a "wide" table, read as is (e.g. by `read_raw_table`).  |

### Classes

| [`SurveyLayout`](#odus.raw_formats.SurveyLayout)([year_row, age_row, ...])   | Where things are in a "wide" survey table (0-based row and column numbers).   |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|

### *class* odus.raw_formats.SurveyLayout(year_row=1, age_row=2, data_start_row=3, label_col=None, section_titles=('List of States', 'Select States below', 'Selects States below'), empty_value=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Where things are in a “wide” survey table (0-based row and column numbers).

* **Parameters:**
  * **year_row** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The row holding the years (None if there’s none).
  * **age_row** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The row holding the ages.
  * **data_start_row** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The first row of variables.
  * **label_col** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The column holding the variable names. None: the column with
    the most text from `data_start_row` on. The ages and values are in
    the columns after it.
  * **section_titles** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – Labels of rows that title a section rather than name a
    variable (compared ignoring case and extra spaces). Those rows are
    skipped.
  * **empty_value** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The value of an empty data cell. None: an empty cell is an
    error.

### odus.raw_formats.find_label_col(vals, , data_start_row)

The column with the most text from `data_start_row` on (the variable names).

### odus.raw_formats.is_empty(x)

Whether a cell value is empty (None, NaN or an empty string).

### odus.raw_formats.is_section_title(label, , section_titles=('List of States', 'Select States below', 'Selects States below'))

Whether a row label is a section title (like “Select States below”).

```pycon
>>> is_section_title(' select states BELOW'), is_section_title('HEROIN')
(True, False)
```

### odus.raw_formats.normalized_label(label)

A label with case and extra whitespace ignored, to compare labels.

```pycon
>>> normalized_label('  New   York ') == normalized_label('new york')
True
```

### odus.raw_formats.read_raw_table(src, \*, ext=None, readers={'.csv': functools.partial(<function read_csv>, header=None, dtype=<class 'object'>, keep_default_na=False), '.tsv': functools.partial(<function read_csv>, sep='\\t', header=None, dtype=<class 'object'>, keep_default_na=False), '.xls': <function \_read_excel>, '.xlsx': <function \_read_excel>})

Read a raw table, as is (no header), with the reader of its file extension.

* **Parameters:**
  * **src** – A file path, or the file’s bytes.
  * **ext** – The file extension (like `".csv"`). Default: the one of `src`
    (required when `src` is bytes).
  * **readers** – `{extension: reader}`; a reader takes a path or file-like
    object and returns a DataFrame.

### odus.raw_formats.trajectory_from_long(df, , age='age', variable='variable', value='value', year=None, fill_value=0)

Make a trajectory from a “long” table: one row per (age, variable, value).

* **Parameters:**
  * **df** – The long table.
  * **age** – The names of its columns holding those.
  * **variable** – The names of its columns holding those.
  * **value** – The names of its columns holding those.
  * **year** – The name of the column holding the year (optional).
  * **fill_value** – The value of (age, variable) pairs absent from the table.

```pycon
>>> long = pd.DataFrame({
...     'age': [20, 21, 21], 'year': [2000, 2001, 2001],
...     'variable': ['HEROIN', 'HEROIN', 'WORK'], 'value': [0, 1, 1],
... })
>>> t = trajectory_from_long(long, year='year')
>>> t.values.tolist(), t.attrs[YEAR_OF_AGE]
([[0, 0], [1, 1]], {20: 2000, 21: 2001})
```

### odus.raw_formats.trajectory_from_table(raw, , layout=SurveyLayout(year_row=1, age_row=2, data_start_row=3, label_col=None, section_titles=('List of States', 'Select States below', 'Selects States below'), empty_value=None))

Make a trajectory from a “wide” table, read as is (e.g. by `read_raw_table`).

Rows from `layout.data_start_row` on with a (non section title) label are
variables; columns after the label column with an age are ages.
