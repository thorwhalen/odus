# odus

ODUS (Older Drug User Study): data and tools to study drug-use trajectories.

Exposes the package data locations (`root_dir`, `data_dir`, `data_path_of`)
used by the rest of the package to reach the bundled survey files.

### Functions

| `data_path_of`(path)   |    |
|------------------------|----|

### Modules

| [`analysis_utils`](odus.analysis_utils.html.md#module-odus.analysis_utils)           | High-level analyses of drug-use trajectories.                     |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`dacc`](odus.dacc.html.md#module-odus.dacc)                               | Data access: stores and accessors over the survey data.           |
| [`nothing`](odus.nothing.html.md#module-odus.nothing)                         | Universal neutral (identity) elements for accumulation.           |
| [`plot_utils`](odus.plot_utils.html.md#module-odus.plot_utils)                   | Plotting helpers for trajectories and matrices.                   |
| [`pobj`](odus.pobj.html.md#module-odus.pobj)                               | Dynamic method injection on objects and classes.                  |
| [`raw_data_diagnosis`](odus.raw_data_diagnosis.html.md#module-odus.raw_data_diagnosis)   | Diagnosis of the raw survey spreadsheets.                         |
| [`sequential_var_sets`](odus.sequential_var_sets.html.md#module-odus.sequential_var_sets) | Variables and variable sets over sequential (time-indexed) data.  |
| [`util`](odus.util.html.md#module-odus.util)                               | General utilities: image conversion and multi-image file writing. |
