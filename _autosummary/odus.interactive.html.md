# odus.interactive

Interactive trajectory plots: hover a cell to see its variable, age, year and value.

`plot_trajectory_interactive` draws a trajectory with plotly, styled like
`odus.plot_utils.plot_trajectory`: the cell colors come from the same
`trajectory_colors` (row colors, alternating backgrounds, group tints), with
gaps between cells, wider every `major_every` years and between groups.

plotly is optional: `pip install "odus[interactive]"`. Importing this module
doesn’t need it; calling the function does.

### Functions

| [`plot_trajectory_interactive`](#odus.interactive.plot_trajectory_interactive)(df[, fields, ...])   | Plot a trajectory with plotly; hovering a cell shows what it is.   |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|

### odus.interactive.plot_trajectory_interactive(df, fields=None, , title=None, x=None, gap_color='white', minor_gap=1, major_gap=4, major_every=5, cell_px=18, \*\*style)

Plot a trajectory with plotly; hovering a cell shows what it is.

* **Parameters:**
  * **df** – A trajectory DataFrame: one row per age, one column per variable.
  * **fields** – The variables to show, in order (default: all).
  * **title** – Optional title.
  * **x** – Where to put the ages on the x axis (one number per age, each one more
    than the previous), as in `plot_trajectory`. Default: the ages.
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
