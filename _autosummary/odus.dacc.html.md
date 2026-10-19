# odus.dacc

Data access: stores and accessors over the survey data.

Defines the `py2store`-based stores (`DfStore`, `VarSetCountsStore`,
`PotStore`) and the `Dacc` facade that give easy access to the prepared
trajectory dataframes, their count tables, and the corresponding potentials.

### Functions

| `counts_of_kps`(store, categories, kps_list)                                               |                                                                 |
|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `ensure_slash_suffix`(x)                                                                   |                                                                 |
| `mk_pvar_str_struct`(df_store[, ...])                                                      |                                                                 |
| `mk_pvar_struct`(df_store[, ...])                                                          |                                                                 |
| [`prep_trajectory_df`](#odus.dacc.prep_trajectory_df)(df, \*[, source, ...]) | Clean a trajectory DataFrame (index: ages, columns: variables). |
| `two_way_map`(keys, val_of_key_func)                                                       |                                                                 |

### Classes

| `Dacc`(xls_rootdir[, categories])                                                     |                                                         |
|---------------------------------------------------------------------------------------|---------------------------------------------------------|
| `DelegMap`(dfunc[, dmap])                                                             |                                                         |
| [`DfStore`](#odus.dacc.DfStore)(path_format[, max_levels])   | The trajectory DataFrames of a folder of survey tables. |
| `HashableMixin`()                                                                     |                                                         |
| [`PotStore`](#odus.dacc.PotStore)(df_store[, store])          |                                                         |
| `Struct`(\*\*kwargs)                                                                  |                                                         |
| [`VarSetCountsStore`](#odus.dacc.VarSetCountsStore)(df_store[, store]) |                                                         |
| [`VarStr`](#odus.dacc.VarStr)                               |                                                         |

### *class* odus.dacc.DfStore(path_format, max_levels=None)

Bases: `HashableMixin`, `ReadOnlyMixin`, `LocalBinaryStore`

The trajectory DataFrames of a folder of survey tables.

`DfStore(folder + "{}.xlsx")` (or `"{}.csv"`, `"{}.tsv"`…) gives the
trajectory of each file, read by `odus.raw_formats` with the reader of its
extension (`readers`) and the `layout` of the tables. Values are then
cleaned by `prep_trajectory_df` with the `non_binary` and `duplicates`
policies. All four are attributes: set them on a store, or in a subclass.

### *class* odus.dacc.PotStore(df_store, store=None)

Bases: [`VarSetCountsStore`](#odus.dacc.VarSetCountsStore)

### *class* odus.dacc.VarSetCountsStore(df_store, store=None)

Bases: `HashableMixin`, `ReadOnlyMixin`, `Store`

### *class* odus.dacc.VarStr

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### odus.dacc.prep_trajectory_df(df, , source=None, non_binary='warn', duplicates='merge')

Clean a trajectory DataFrame (index: ages, columns: variables).

- Columns named like a section title of the survey sheet (`SECTION_TITLES`
  of `odus.raw_formats`, e.g. “Select States below”) are dropped.
- `duplicates`: what to do with variables appearing more than once (e.g. a
  state lived in during two periods, entered as two rows), names compared
  ignoring case and extra spaces. `"merge"`: one column (with the first
  name), holding at each age the first non-zero value of the duplicates;
  `"keep"`: leave them; `"raise"`: raise a `ValueError`.
- `non_binary`: what to do with values other than 0 and 1. `"warn"`: set
  them to 1 and say which (variable, ages, values); `"binarize"`: set them to
  1 silently; `"keep"`: leave them; `"raise"`: raise a `ValueError`.

`source` (e.g. the file name) is only used in messages.

```pycon
>>> df = pd.DataFrame(
...     [[1, 0, 0, 0], [0, 0, 1, 2]],
...     columns=['GA', 'NJ', 'NJ', 'Select States below'],
...     index=pd.Index([20, 21], name='age'),
... )
>>> out = prep_trajectory_df(df)
>>> out.columns.tolist(), out.values.tolist()
(['GA', 'NJ'], [[1, 0], [0, 1]])
```
