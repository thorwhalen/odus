# odus.dacc

Data access: stores and accessors over the survey data.

Defines the `py2store`-based stores (`DfStore`, `VarSetCountsStore`,
`PotStore`) and the `Dacc` facade that give easy access to the prepared
trajectory dataframes, their count tables, and the corresponding potentials.

### Functions

| `counts_of_kps`(store, categories, kps_list)   |    |
|------------------------------------------------|----|
| `ensure_slash_suffix`(x)                       |    |
| `mk_pvar_str_struct`(df_store[, ...])          |    |
| `mk_pvar_struct`(df_store[, ...])              |    |
| `two_way_map`(keys, val_of_key_func)           |    |

### Classes

| `Dacc`(xls_rootdir[, categories])                                                     |    |
|---------------------------------------------------------------------------------------|----|
| `DelegMap`(dfunc[, dmap])                                                             |    |
| [`DfStore`](#odus.dacc.DfStore)(path_format[, max_levels])   |    |
| `HashableMixin`()                                                                     |    |
| [`PotStore`](#odus.dacc.PotStore)(df_store[, store])          |    |
| `Struct`(\*\*kwargs)                                                                  |    |
| [`VarSetCountsStore`](#odus.dacc.VarSetCountsStore)(df_store[, store]) |    |
| [`VarStr`](#odus.dacc.VarStr)                               |    |

### *class* odus.dacc.DfStore(path_format, max_levels=None)

Bases: `HashableMixin`, `ReadOnlyMixin`, `LocalBinaryStore`

### *class* odus.dacc.PotStore(df_store, store=None)

Bases: [`VarSetCountsStore`](#odus.dacc.VarSetCountsStore)

### *class* odus.dacc.VarSetCountsStore(df_store, store=None)

Bases: `HashableMixin`, `ReadOnlyMixin`, `Store`

### *class* odus.dacc.VarStr

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
