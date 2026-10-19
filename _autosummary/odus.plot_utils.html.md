# odus.plot_utils

Plotting helpers for trajectories and matrices.

Heatmaps, life-course plots, iteration over a store’s trajectory plots, and
writing a selection of trajectories out to a (multi-page) file.

`stacked_trajectories` puts several of them one above the other, aligned on
age, year, or the years since some variable was first non-zero.

`plot_trajectory` is the readable rendering of a single trajectory: a gap
between cells (wider every `major_every` years), alternating grey rows so the
eye can follow a row back to its variable, optional light shading of variable
groups, and per-variable colors (for instance the ones read from the source
spreadsheet with `row_colors_of_xls`).

### Functions

| `heatmap`(X[, y, col_labels, figsize, cmap, ...])                                              |                                                                             |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `ihead`(it)                                                                                    |                                                                             |
| [`life_plots`](#odus.plot_utils.life_plots)(df_store[, fields, keys, ...])     | Yield the axes of the trajectory plot of each (selected) key of `df_store`. |
| [`plot_life`](#odus.plot_utils.plot_life)(df[, fields, title, ax])            | Plot a trajectory; `plot_kwargs` go to `plot_trajectory`.                   |
| `plot_life_course`(df[, grid, figsize])                                                        |                                                                             |
| [`plot_trajectory`](#odus.plot_utils.plot_trajectory)(df[, fields, title, ax, ...]) | Plot one trajectory (index: ages or years, columns: variables) readably.    |
| [`row_colors_of_xls`](#odus.plot_utils.row_colors_of_xls)(xls_filepath, \*[, ...])    | Get a `{variable: color}` mapping from the cell colors of a spreadsheet.    |
| [`stacked_trajectories`](#odus.plot_utils.stacked_trajectories)(df_store[, fields, ...]) | Plot several trajectories one above the other, aligned on a common x axis.  |
| [`trajectory_colors`](#odus.plot_utils.trajectory_colors)(df, \*[, row_colors, ...])  | The colors of the cells of a trajectory: the styling every renderer shares. |
| [`write_trajectories_to_file`](#odus.plot_utils.write_trajectories_to_file)(df_store[, ...])   | Write the trajectory plots of (selected) keys of `df_store`, one per page.  |

### odus.plot_utils.life_plots(df_store, fields=None, keys=None, k_df_to_title=<function \_title_of_key>, \*\*plot_kwargs)

Yield the axes of the trajectory plot of each (selected) key of `df_store`.

### odus.plot_utils.plot_life(df, fields=None, title=None, ax=None, \*\*plot_kwargs)

Plot a trajectory; `plot_kwargs` go to `plot_trajectory`.

### odus.plot_utils.plot_trajectory(df, fields=None, , title=None, ax=None, row_colors=None, default_color='#1b2a49', row_backgrounds=('#ffffff', '#ebebeb'), row_groups=None, group_colors=None, group_tint=0.35, gap_color='white', minor_gap=0.6, major_gap=2.5, major_every=5, xtick_every=None, cell_size=0.22, fontsize=8, ylabel_right=False, x=None)

Plot one trajectory (index: ages or years, columns: variables) readably.

Each variable is a row and each age/year a column. A cell takes its row’s
color in proportion to its value, over the row’s background.

* **Parameters:**
  * **df** – A trajectory DataFrame: one row per age/year, one column per variable.
  * **fields** – The variables (columns) to show, in order. Defaults to all.
  * **title** – Optional title.
  * **ax** – The axes to draw on. If None, a new figure is made, sized so that
    cells are `cell_size` inches and every label fits.
  * **row_colors** – Color of the “on” cells: a single color, a list (one per
    variable), or a `{variable: color}` mapping (missing variables get
    `default_color`). See `row_colors_of_xls` to reuse the colors of
    the source spreadsheet.
  * **default_color** – Color of variables that `row_colors` doesn’t specify.
  * **row_backgrounds** – Background colors, cycled row by row (the alternating
    greys that make it easy to follow a row). Use `None` for white.
  * **row_groups** – A `{group_name: variables}` mapping. Rows of a group get a
    light tint of its color, and a wider gap separates rows of different
    groups. Keep it to a few (2 to 4) groups.
  * **group_colors** – Colors of the groups: a list (cycled) or a
    `{group_name: color}` mapping. Defaults to `DFLT_GROUP_COLORS`.
  * **group_tint** – How strongly (0 to 1) a group color tints the background.
  * **gap_color** – Color of the gaps between cells.
  * **minor_gap** – Width (points) of the gap between two cells.
  * **major_gap** – Width (points) of the wider gaps: before every age/year
    multiple of `major_every`, and between row groups.
  * **major_every** – See `major_gap`. If the index isn’t integer-valued, every
    `major_every`-th column. Falsy: no wider gaps.
  * **xtick_every** – Label every `xtick_every`-th column. Defaults to every
    column when there’s room for it, else to `major_every`.
  * **cell_size** – Size (inches) of a cell, used when `ax` is None.
  * **fontsize** – Font size of the tick labels.
  * **ylabel_right** – Whether to also label the variables on the right.
  * **x** – Where to draw the columns on the x axis: one number per age, each one
    more than the previous (e.g. the years, or the years since some event).
    Default: column positions, labeled with the index (the ages). Plots
    drawn with comparable `x` line up (see `stacked_trajectories`).
* **Returns:**
  The matplotlib axes drawn on.

### odus.plot_utils.row_colors_of_xls(xls_filepath, , sheetname=None, label_col=2, data_col=3)

Get a `{variable: color}` mapping from the cell colors of a spreadsheet.

Use it as the `row_colors` of `plot_trajectory` to color variables as the
source spreadsheet does. A variable’s color is the fill of its label cell, or
if that has none, the most common fill of its data cells. Variables with no
solid RGB or indexed (palette) fill are left out, as are theme colors.

* **Parameters:**
  * **xls_filepath** – Path to the `.xlsx` file.
  * **sheetname** – The sheet to read (default: the active one).
  * **label_col** – The (1-based) column holding the variable names.
  * **data_col** – The (1-based) first column holding the data.

### odus.plot_utils.stacked_trajectories(df_store, fields=None, keys=None, \*, x_align='age', x_label=None, k_df_to_title=<function \_title_of_key>, cell_size=0.22, fontsize=8, major_every=5, xtick_every=None, \*\*plot_kwargs)

Plot several trajectories one above the other, aligned on a common x axis.

* **Parameters:**
  * **df_store** – A mapping of trajectories (like a `DfStore`).
  * **fields** – The variables to show (default: all, per trajectory).
  * **keys** – The keys of `df_store` to plot, or a filter function of the keys.
  * **x_align** – 

    What to align the trajectories on:
    - `"age"` (default) or `"year"` (needs `df.attrs["year_of_age"]`,
      which `DfStore` provides);
    - a variable name: the years since that variable was first non-zero
      (e.g. `"HEROIN"`: years since first heroin use). Trajectories
      where it never is are left out, with a warning;
    - a function `x_align(df)` returning the x of each age (one more from
      each age to the next), or None to leave the trajectory out.
  * **x_label** – Label of the x axis (default: from `x_align`).
  * **k_df_to_title** – `k_df_to_title(key, df)` gives the title of each plot.
  * **cell_size** – Passed on to
    `plot_trajectory` (see there).
  * **fontsize** – Passed on to
    `plot_trajectory` (see there).
  * **major_every** – Passed on to
    `plot_trajectory` (see there).
  * **xtick_every** – Passed on to
    `plot_trajectory` (see there).
  * **plot_kwargs** – Passed on to
    `plot_trajectory` (see there).
* **Returns:**
  The matplotlib figure.

### odus.plot_utils.trajectory_colors(df, , row_colors=None, default_color='#1b2a49', row_backgrounds=('#ffffff', '#ebebeb'), row_groups=None, group_colors=None, group_tint=0.35)

The colors of the cells of a trajectory: the styling every renderer shares.

Returns `(img, group_of_row)`: an `(n_variables, n_ages, 3)` RGB array (in
[0, 1]), and the group (or None) of each variable. See `plot_trajectory`
for the arguments.

### odus.plot_utils.write_trajectories_to_file(df_store, fields=None, keys=None, fp='test.pdf', pil_write_format=None, to_pil_image_kwargs=None, \*, k_df_to_title=<function \_title_of_key>, plot_kwargs=None, \*\*pil_save_kwargs)

Write the trajectory plots of (selected) keys of `df_store`, one per page.

`plot_kwargs` are passed on to `plot_trajectory`.
