# odus.raw_data_diagnosis

Diagnosis of the raw survey spreadsheets.

Reads the source `.xlsx` files with `openpyxl` (values *and* cell colors)
and reports the data-entry anomalies that must be resolved before the data can
be prepared: cells whose content and color disagree, values that aren’t 0 or 1,
ages and years that don’t follow each other, repeated variables…

Each check is a `Rule`: a name, a function marking the problematic cells of a
`SheetView`, and a message explaining the problem. `DFLT_RULES` maps names to
the rules used by default; pick, drop or add rules with `select_rules`, or by
passing any `{name: Rule}` mapping as `rules`:

```pycon
>>> sorted(select_rules(exclude=['filled_but_uncolored']))
['age_not_consecutive', 'age_year_mismatch', 'duplicate_variable',
 'empty_but_colored', 'not_binary', 'year_not_consecutive']
```

`diagnose_file` and `diagnose_dir` yield one `Problem` per flagged cell,
`diagnosis_table` puts them in a DataFrame, and `print_diagnosis` prints them.

### Functions

| `coord_val_and_bgcolor_of_cell`(c)                                                               |                                                                        |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `crop`(df, last_row_idx, last_col_idx)                                                           |                                                                        |
| `dfof`(sheet[, dotpath])                                                                         |                                                                        |
| [`diagnose_dir`](#odus.raw_data_diagnosis.diagnose_dir)(source_dir[, sheetname, rules])    | Yield the `Problem``s found in the ``.xlsx` files of `source_dir`.     |
| [`diagnose_file`](#odus.raw_data_diagnosis.diagnose_file)(xls_filepath[, sheetname, rules]) | Yield the `Problem``s that ``rules` find in a spreadsheet.             |
| [`diagnose_view`](#odus.raw_data_diagnosis.diagnose_view)(view, \*[, rules, filename])      | Yield the `Problem``s that ``rules` find in a `SheetView`.             |
| `diagnosis_items`(source_dir[, sheetname])                                                       |                                                                        |
| `diagnosis_lidx`(df)                                                                             |                                                                        |
| `diagnosis_lidx_of_xls_file`(xls_filepath[, ...])                                                |                                                                        |
| [`diagnosis_table`](#odus.raw_data_diagnosis.diagnosis_table)(problems)                       | A DataFrame with one row per problem (columns: the `Problem` fields).  |
| `last_true_idx`(series)                                                                          |                                                                        |
| `lidx_of_empty_non_white_or_non_empty_white`(df)                                                 |                                                                        |
| `matmask`(df, mask)                                                                              |                                                                        |
| `mk_dotpath_getter`(dotpath)                                                                     |                                                                        |
| `mk_known_exceptions_lidx`(vals_df)                                                              |                                                                        |
| `pra`(obj)                                                                                       |                                                                        |
| [`print_diagnosis`](#odus.raw_data_diagnosis.print_diagnosis)([source_dir, sheetname, rules]) | Print the diagnosis of the raw xls files in `source_dir`.              |
| `row_lidx_to_mat_lidx`(row_lidx, n_cols)                                                         |                                                                        |
| [`select_rules`](#odus.raw_data_diagnosis.select_rules)([include, exclude, extra, rules])  | Make a `{name: Rule}` mapping: turn rules on or off, and add your own. |
| `sheet_to_format_prepped_df`(sheet)                                                              |                                                                        |
| [`sheet_view`](#odus.raw_data_diagnosis.sheet_view)(sheet, \*[, year_row, age_row, ...]) | Make a `SheetView` of an `openpyxl` worksheet.                         |
| `valid_vals_int_coords`(vals_df)                                                                 |                                                                        |

### Classes

| [`Problem`](#odus.raw_data_diagnosis.Problem)(filename, rule, coordinate, value, ...)   | A cell flagged by a rule.                       |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------|
| [`Rule`](#odus.raw_data_diagnosis.Rule)(name, flag[, message])                       | A diagnosis rule.                               |
| [`SheetView`](#odus.raw_data_diagnosis.SheetView)(vals, colored, coordinate, ...[, ...])  | The parts of a survey sheet that rules look at. |

### *class* odus.raw_data_diagnosis.Problem(filename, rule, coordinate, value, message)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A cell flagged by a rule.

### *class* odus.raw_data_diagnosis.Rule(name, flag, message='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A diagnosis rule.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Identifier of the rule (and its key in a rules mapping).
  * **flag** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`SheetView`](#odus.raw_data_diagnosis.SheetView)], `ndarray`]) – `flag(view: SheetView) -> bool array` marking the problematic cells
    (same shape as `view.vals`).
  * **message** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Explanation shown for each flagged cell. Can use `{value}`,
    `{coordinate}` and `{label}` (the variable of the cell’s row).

### *class* odus.raw_data_diagnosis.SheetView(vals, colored, coordinate, label_col, data_mask, year_row=1, age_row=2)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The parts of a survey sheet that rules look at.

`vals`, `colored` and `coordinate` are aligned DataFrames (one entry per
cell). `label_col` is the column holding the variable names, and
`data_mask` marks the cells holding a variable’s value at some age.

#### empty()

Boolean array: is the cell empty?

#### numeric()

Float array of the values: NaN where the cell doesn’t hold a number.

### odus.raw_data_diagnosis.diagnose_dir(source_dir, sheetname=None, \*, rules={'age_not_consecutive': Rule(name='age_not_consecutive', flag=<function \_age_not_consecutive>, message="age {value!r} in {coordinate} doesn't follow the previous age"), 'age_year_mismatch': Rule(name = 'age_year_mismatch', flag=<function \_age_year_mismatch>, message="age {value!r} in {coordinate} doesn't match its year (year - age differs from the first column's)"), 'duplicate_variable': Rule(name = 'duplicate_variable', flag=<function \_duplicate_variable>, message='{coordinate}: variable {value!r} appears more than once'), 'empty_but_colored': Rule(name = 'empty_but_colored', flag=<function \_empty_but_colored>, message='{coordinate} ({label}) is colored but empty'), 'filled_but_uncolored': Rule(name = 'filled_but_uncolored', flag=<function \_filled_but_uncolored>, message='{coordinate} ({label}) has a value ({value!r}) but no color'), 'not_binary': Rule(name = 'not_binary', flag=<function \_not_binary>, message='{coordinate} ({label}) should be 0 or 1, but is {value!r}'), 'year_not_consecutive': Rule(name = 'year_not_consecutive', flag=<function \_year_not_consecutive>, message="year {value!r} in {coordinate} doesn't follow the previous year")}, \*\*view_kwargs)

Yield the `Problem``s found in the ``.xlsx` files of `source_dir`.

### odus.raw_data_diagnosis.diagnose_file(xls_filepath, sheetname=None, \*, rules={'age_not_consecutive': Rule(name='age_not_consecutive', flag=<function \_age_not_consecutive>, message="age {value!r} in {coordinate} doesn't follow the previous age"), 'age_year_mismatch': Rule(name = 'age_year_mismatch', flag=<function \_age_year_mismatch>, message="age {value!r} in {coordinate} doesn't match its year (year - age differs from the first column's)"), 'duplicate_variable': Rule(name = 'duplicate_variable', flag=<function \_duplicate_variable>, message='{coordinate}: variable {value!r} appears more than once'), 'empty_but_colored': Rule(name = 'empty_but_colored', flag=<function \_empty_but_colored>, message='{coordinate} ({label}) is colored but empty'), 'filled_but_uncolored': Rule(name = 'filled_but_uncolored', flag=<function \_filled_but_uncolored>, message='{coordinate} ({label}) has a value ({value!r}) but no color'), 'not_binary': Rule(name = 'not_binary', flag=<function \_not_binary>, message='{coordinate} ({label}) should be 0 or 1, but is {value!r}'), 'year_not_consecutive': Rule(name = 'year_not_consecutive', flag=<function \_year_not_consecutive>, message="year {value!r} in {coordinate} doesn't follow the previous year")}, \*\*view_kwargs)

Yield the `Problem``s that ``rules` find in a spreadsheet.

`view_kwargs` (`year_row`, `age_row`…) go to `sheet_view`. A sheet
that doesn’t have the expected layout gives one `Problem` whose rule is
`UNREADABLE_SHEET`.

### odus.raw_data_diagnosis.diagnose_view(view, \*, rules={'age_not_consecutive': Rule(name='age_not_consecutive', flag=<function \_age_not_consecutive>, message="age {value!r} in {coordinate} doesn't follow the previous age"), 'age_year_mismatch': Rule(name = 'age_year_mismatch', flag=<function \_age_year_mismatch>, message="age {value!r} in {coordinate} doesn't match its year (year - age differs from the first column's)"), 'duplicate_variable': Rule(name = 'duplicate_variable', flag=<function \_duplicate_variable>, message='{coordinate}: variable {value!r} appears more than once'), 'empty_but_colored': Rule(name = 'empty_but_colored', flag=<function \_empty_but_colored>, message='{coordinate} ({label}) is colored but empty'), 'filled_but_uncolored': Rule(name = 'filled_but_uncolored', flag=<function \_filled_but_uncolored>, message='{coordinate} ({label}) has a value ({value!r}) but no color'), 'not_binary': Rule(name = 'not_binary', flag=<function \_not_binary>, message='{coordinate} ({label}) should be 0 or 1, but is {value!r}'), 'year_not_consecutive': Rule(name = 'year_not_consecutive', flag=<function \_year_not_consecutive>, message="year {value!r} in {coordinate} doesn't follow the previous year")}, filename='')

Yield the `Problem``s that ``rules` find in a `SheetView`.

### odus.raw_data_diagnosis.diagnosis_table(problems)

A DataFrame with one row per problem (columns: the `Problem` fields).

* **Return type:**
  `DataFrame`

### odus.raw_data_diagnosis.print_diagnosis(source_dir='.', sheetname=None, \*, rules={'age_not_consecutive': Rule(name='age_not_consecutive', flag=<function \_age_not_consecutive>, message="age {value!r} in {coordinate} doesn't follow the previous age"), 'age_year_mismatch': Rule(name = 'age_year_mismatch', flag=<function \_age_year_mismatch>, message="age {value!r} in {coordinate} doesn't match its year (year - age differs from the first column's)"), 'duplicate_variable': Rule(name = 'duplicate_variable', flag=<function \_duplicate_variable>, message='{coordinate}: variable {value!r} appears more than once'), 'empty_but_colored': Rule(name = 'empty_but_colored', flag=<function \_empty_but_colored>, message='{coordinate} ({label}) is colored but empty'), 'filled_but_uncolored': Rule(name = 'filled_but_uncolored', flag=<function \_filled_but_uncolored>, message='{coordinate} ({label}) has a value ({value!r}) but no color'), 'not_binary': Rule(name = 'not_binary', flag=<function \_not_binary>, message='{coordinate} ({label}) should be 0 or 1, but is {value!r}'), 'year_not_consecutive': Rule(name = 'year_not_consecutive', flag=<function \_year_not_consecutive>, message="year {value!r} in {coordinate} doesn't follow the previous year")}, \*\*view_kwargs)

Print the diagnosis of the raw xls files in `source_dir`.

For every file that has a problem, print its name, then for each rule that
flagged something, the rule’s name and the explanation of each flagged cell.

The checks are the `rules` (default `DFLT_RULES`); see the module docstring
and `select_rules` to turn some off or add your own. For instance, the
color rules only make sense for spreadsheets where filled cells are colored.

### odus.raw_data_diagnosis.select_rules(include=None, exclude=(), , extra=None, rules=None)

Make a `{name: Rule}` mapping: turn rules on or off, and add your own.

* **Parameters:**
  * **include** – Names of the rules to keep (default: all of `rules`).
  * **exclude** – Names of the rules to drop.
  * **extra** – More rules to add: a `{name: Rule}` mapping or an iterable of
    `Rule` instances.
  * **rules** – The rules to start from (default: `DFLT_RULES`).

```pycon
>>> list(select_rules(include=['not_binary']))
['not_binary']
>>> my_rule = Rule('never', lambda view: view.false_mask(), 'never happens')
>>> 'never' in select_rules(extra=[my_rule])
True
```

### odus.raw_data_diagnosis.sheet_view(sheet, , year_row=1, age_row=2, data_start_row=3, non_variable_labels=('List of States', 'Select States below'))

Make a `SheetView` of an `openpyxl` worksheet.

The sheet is cropped to its last labeled row and last column with an age.
