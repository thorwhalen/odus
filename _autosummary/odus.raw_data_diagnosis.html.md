# odus.raw_data_diagnosis

Diagnosis of the raw survey spreadsheets.

Reads the source `.xlsx` files with `openpyxl` (values *and* cell colors)
and reports cells whose content and formatting disagree – the data-entry
anomalies that must be resolved before the data can be prepared.

### Functions

| `coord_val_and_bgcolor_of_cell`(c)                                                        |                                                           |
|-------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `crop`(df, last_row_idx, last_col_idx)                                                    |                                                           |
| `dfof`(sheet[, dotpath])                                                                  |                                                           |
| `diagnosis_items`(source_dir[, sheetname])                                                |                                                           |
| `diagnosis_lidx`(df)                                                                      |                                                           |
| `diagnosis_lidx_of_xls_file`(xls_filepath[, ...])                                         |                                                           |
| `last_true_idx`(series)                                                                   |                                                           |
| `lidx_of_empty_non_white_or_non_empty_white`(df)                                          |                                                           |
| `matmask`(df, mask)                                                                       |                                                           |
| `mk_dotpath_getter`(dotpath)                                                              |                                                           |
| `mk_known_exceptions_lidx`(vals_df)                                                       |                                                           |
| `pra`(obj)                                                                                |                                                           |
| [`print_diagnosis`](#odus.raw_data_diagnosis.print_diagnosis)([source_dir, sheetname]) | Print the diagnosis of the raw xls files in `source_dir`. |
| `row_lidx_to_mat_lidx`(row_lidx, n_cols)                                                  |                                                           |
| `sheet_to_format_prepped_df`(sheet)                                                       |                                                           |
| `valid_vals_int_coords`(vals_df)                                                          |                                                           |

### odus.raw_data_diagnosis.print_diagnosis(source_dir='.', sheetname=None)

Print the diagnosis of the raw xls files in `source_dir`.
Namely, for every file that has a problem, the script will print the name of the problematic files,
and print the coordinates of the cells that should be checked for problems.

The problems that are checked for:

- Cells (in the “data” range) that are (a) empty, but have color or (b) NOT empty, but without color.

See functions `lidx_of_empty_non_white_or_non_empty_white` and `mk_known_exceptions_lidx` functions for details.
