"""Interactive trajectory plots: hover a cell to see its variable, age, year and value.

``plot_trajectory_interactive`` draws a trajectory with plotly, styled like
``odus.plot_utils.plot_trajectory``: the cell colors come from the same
``trajectory_colors`` (row colors, alternating backgrounds, group tints), with
gaps between cells, wider every ``major_every`` years and between groups.

plotly is optional: ``pip install "odus[interactive]"``. Importing this module
doesn't need it; calling the function does.
"""

import numpy as np

from odus.plot_utils import (
    DFLT_MAJOR_EVERY,
    _consecutive_numbers,
    _major_boundaries,
    _select_fields,
    trajectory_colors,
)

YEAR_OF_AGE = "year_of_age"  # same attrs key as odus.dacc.YEAR_OF_AGE
DFLT_CELL_PX = 18  # size of a cell, in pixels
DFLT_LABEL_PX_PER_CHAR = 7  # room for the variable names, per character


def _graph_objects():
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "Interactive plots need plotly. Install it with: "
            'pip install "odus[interactive]"   (or: pip install plotly)'
        ) from e
    return go


def _hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(int(round(255 * c)) for c in rgb))


def _discrete_colorscale(img):
    """Map each cell to the index of its color, with a colorscale of those colors."""
    hexes = np.array([[_hex(c) for c in row] for row in img])
    colors, z = np.unique(hexes, return_inverse=True)
    n = len(colors)
    colorscale = []
    for i, c in enumerate(colors):  # constant over [i/n, (i+1)/n]: z=i is centered
        colorscale += [[i / n, c], [(i + 1) / n, c]]
    return z.reshape(hexes.shape), colorscale, n


def _hover_texts(df, x):
    year_of_age = df.attrs.get(YEAR_OF_AGE) or {}
    texts = []
    for var in df.columns:
        row = []
        for j, (age, value) in enumerate(zip(df.index, df[var].to_numpy())):
            lines = [f"<b>{var}</b>", f"age {age}"]
            if age in year_of_age:
                lines.append(f"year {year_of_age[age]}")
            if x is not None:
                lines.append(f"x {x[j]}")
            lines.append(f"value {value}")
            row.append("<br>".join(lines))
        texts.append(row)
    return texts


def plot_trajectory_interactive(
    df,
    fields=None,
    *,
    title=None,
    x=None,
    gap_color="white",
    minor_gap=1,
    major_gap=4,
    major_every=DFLT_MAJOR_EVERY,
    cell_px=DFLT_CELL_PX,
    **style,
):
    """Plot a trajectory with plotly; hovering a cell shows what it is.

    Args:
        df: A trajectory DataFrame: one row per age, one column per variable.
        fields: The variables to show, in order (default: all).
        title: Optional title.
        x: Where to put the ages on the x axis (one number per age, each one more
            than the previous), as in ``plot_trajectory``. Default: the ages.
        gap_color: Color of the gaps between cells.
        minor_gap: Gap between cells, in pixels.
        major_gap: Wider gap (pixels) before every age multiple of
            ``major_every``, and between row groups.
        major_every: See ``major_gap``.
        cell_px: Size of a cell, in pixels.
        style: The cell colors, as in ``plot_trajectory``: ``row_colors``,
            ``default_color``, ``row_backgrounds``, ``row_groups``,
            ``group_colors``, ``group_tint``.

    Returns:
        A ``plotly.graph_objects.Figure`` (show it with ``fig.show()``, or save
        it with ``fig.write_html(path)``).
    """
    go = _graph_objects()
    df = _select_fields(df, fields)
    img, group_of_row = trajectory_colors(df, **style)
    varnames, ages = list(df.columns), list(df.index)
    n_rows, n_cols = len(varnames), len(ages)

    if x is not None:
        xs = _consecutive_numbers(x, n_cols)
        tick_values = list(xs)
    else:
        try:
            xs = _consecutive_numbers(ages, n_cols)
            tick_values = list(xs)
        except (TypeError, ValueError):  # ages that aren't consecutive numbers
            xs, tick_values = np.arange(n_cols), ages
    xs = [x_.item() if hasattr(x_, "item") else x_ for x_ in xs]

    z, colorscale, n_colors = _discrete_colorscale(img)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=xs,
            y=list(range(n_rows)),
            zmin=-0.5,
            zmax=n_colors - 0.5,
            colorscale=colorscale,
            showscale=False,
            xgap=minor_gap,
            ygap=minor_gap,
            text=_hover_texts(df, None if x is None else xs),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    def line(**coordinates):
        gap = dict(color=gap_color, width=major_gap)
        return dict(type="line", line=gap, layer="above", **coordinates)

    shapes = [
        line(x0=xs[j] - 0.5, x1=xs[j] - 0.5, y0=-0.5, y1=n_rows - 0.5)
        for j in _major_boundaries(tick_values, major_every)
    ]
    shapes += [
        line(x0=xs[0] - 0.5, x1=xs[-1] + 0.5, y0=i - 0.5, y1=i - 0.5)
        for i in range(1, n_rows)
        if group_of_row[i] != group_of_row[i - 1]
    ]

    label_px = max(len(str(v)) for v in varnames) * DFLT_LABEL_PX_PER_CHAR
    fig.update_layout(
        title=title,
        shapes=shapes,
        width=n_cols * cell_px + label_px + 60,
        height=n_rows * cell_px + 120,
        margin=dict(l=label_px + 10, r=20, t=60 if title else 30, b=50),
        plot_bgcolor=gap_color,
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        tickvals=xs,
        ticktext=[str(t) for t in tick_values],
        showgrid=False,
        zeroline=False,
        range=[xs[0] - 0.5, xs[-1] + 0.5],
    )
    fig.update_yaxes(
        tickvals=list(range(n_rows)),
        ticktext=[str(v) for v in varnames],
        autorange="reversed",
        showgrid=False,
        zeroline=False,
    )
    return fig
