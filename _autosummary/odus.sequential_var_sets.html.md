# odus.sequential_var_sets

Variables and variable sets over sequential (time-indexed) data.

`PVar` describes a variable at a time offset, `VarSet` a set of such
variables, and `DfData` extracts the corresponding key-point sequences from a
trajectory dataframe.

### Functions

| `extract_kps`(df, kps)   |    |
|--------------------------|----|

### Classes

| `DfData`(df)       |    |
|--------------------|----|
| `PVar`(var[, i])   |    |
| `VarSet`(\*varset) |    |
| `VarSetFactory`()  |    |
