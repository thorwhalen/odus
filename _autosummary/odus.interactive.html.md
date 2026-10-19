# odus.interactive

Interactive trajectory plots: hover a cell to see its variable, age, year and value.

`plot_trajectory_interactive` draws a trajectory with plotly, styled like
`odus.plot_utils.plot_trajectory`: the cell colors come from the same
`trajectory_colors` (row colors, alternating backgrounds, group tints), with
gaps between cells, wider every `major_every` years and between groups.

plotly is optional: `pip install "odus[interactive]"`. Importing this module
doesn’t need it; calling the function does.

### Functions

| [`plot_trajectory_interactive`](#odus.interactive.plot_trajectory_interactive)(df[, fields, ...])    | Plot a trajectory with plotly; hovering a cell shows what it is.     |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`stacked_trajectories_interactive`](#odus.interactive.stacked_trajectories_interactive)(df_store[, ...]) | Plot several trajectories one above the other, aligned, with plotly. |

### odus.interactive.plot_trajectory_interactive(df, fields=None, , title=None, x=None, key=None, gap_color='white', minor_gap=1, major_gap=4, major_every=5, cell_px=18, \*\*style)

Plot a trajectory with plotly; hovering a cell shows what it is.

* **Parameters:**
  * **df** – A trajectory DataFrame: one row per age, one column per variable.
  * **fields** – The variables to show, in order (default: all).
  * **title** – Optional title.
  * **x** – Where to put the ages on the x axis (one number per age, each one more
    than the previous), as in `plot_trajectory`. Default: the ages.
  * **key** – The respondent’s key, put with the variable and the age in each
    cell’s `customdata`, for click handlers (`plotly_click`).
  * **gap_color** – Color of the gaps between cells.
  * **minor_gap** – Gap between cells, in pixels.
  * **major_gap** – Wider gap (pixels) before every age multiple of
    `major_every`, and between row groups.
  * **major_every** – See `major_gap`.
  * **cell_px** – Size of a cell, in pixels.
  * **style** – The cell colors, as in `plot_trajectory` (and only those):
    `row_colors`, `default_color`, `row_backgrounds`,
    `row_groups`, `group_colors`, `group_tint`.
* **Returns:**
  A `plotly.graph_objects.Figure` (show it with `fig.show()`, or save
  it with `fig.write_html(path)`).

### odus.interactive.stacked_trajectories_interactive(df_store, fields=None, keys=None, , x_align='age', x_label=None, title=None, k_df_to_title=None, gap_color='white', minor_gap=1, major_gap=4, major_every=5, cell_px=18, \*\*style)

Plot several trajectories one above the other, aligned, with plotly.

The interactive version of `odus.plot_utils.stacked_trajectories` (see it
for `fields`, `keys`, `x_align`, `x_label` and `k_df_to_title`): one
block of rows per trajectory, labeled with its title, separated by a wider
gap. The cells are those of `plot_trajectory_interactive`: hover text, and a
`customdata` of `[respondent key, variable, age]` for click handlers.

* **Returns:**
  the plotly figure, and the trajectories left out
  (as `odus.plot_utils.aligned_trajectories` gives them).
* **Return type:**
  `(figure, left_out)`
