"""Interactive trajectory plots: hover a cell to see its variable, age, year and value.

``plot_trajectory_interactive`` draws a trajectory with plotly, styled like
``odus.plot_utils.plot_trajectory``: the cell colors come from the same
``trajectory_colors`` (row colors, alternating backgrounds, group tints), with
gaps between cells, wider every ``major_every`` years and between groups.

plotly is optional: ``pip install "odus[interactive]"``. Importing this module
doesn't need it; calling the function does.
"""

import html

import numpy as np

from odus.raw_formats import YEAR_OF_AGE
from odus.plot_utils import (
    DFLT_MAJOR_EVERY,
    _consecutive_numbers,
    _major_boundaries,
    _select_fields,
    trajectory_colors,
)

_STYLE_ARGS = (  # the keyword arguments of trajectory_colors
    "row_colors",
    "default_color",
    "row_backgrounds",
    "row_groups",
    "group_colors",
    "group_tint",
)
DFLT_CELL_PX = 18  # size of a cell, in pixels
DFLT_LABEL_PX_PER_CHAR = 8  # room for the variable names, per character
MIN_PLOT_WIDTH_PX = 200  # so that a short trajectory still has room for its title


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
    year_of_age = df.attrs.get(YEAR_OF_AGE)
    year_of_age = {} if year_of_age is None else dict(year_of_age)
    values = df.to_numpy()
    texts = []
    for i, var in enumerate(df.columns):  # by position: names may repeat
        row = []
        for j, (age, value) in enumerate(zip(df.index, values[:, i])):
            lines = [f"<b>{html.escape(str(var))}</b>", f"age {age}"]
            if age in year_of_age:
                lines.append(f"year {year_of_age[age]}")
            if x is not None:
                lines.append(f"x {x[j]}")
            lines.append(f"value {value}")
            row.append("<br>".join(lines))
        texts.append(row)
    return texts


def _check_style(style, func_name):
    if not_colors := set(style) - set(_STYLE_ARGS):
        raise TypeError(
            f"{func_name} doesn't take {sorted(not_colors)}: of "
            f"plot_trajectory's styling, it takes the colors {list(_STYLE_ARGS)}"
        )


def _heatmap_parts(df, *, x, key, gap_color, minor_gap, major_gap, major_every, style):
    """What a trajectory's heatmap is made of: the trace's arguments, gap lines..."""
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
    ages_json = [a.item() if hasattr(a, "item") else a for a in ages]
    trace = dict(
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
        # what a click on a cell is about: [respondent key, variable, age]
        customdata=[[[key, str(v), a] for a in ages_json] for v in varnames],
    )

    def line(**coordinates):
        gap = dict(color=gap_color, width=major_gap)
        return dict(type="line", line=gap, layer="above", **coordinates)

    lines = [
        line(x0=xs[j] - 0.5, x1=xs[j] - 0.5, y0=-0.5, y1=n_rows - 0.5)
        for j in _major_boundaries(tick_values, major_every)
    ]
    lines += [
        line(x0=xs[0] - 0.5, x1=xs[-1] + 0.5, y0=i - 0.5, y1=i - 0.5)
        for i in range(1, n_rows)
        if group_of_row[i] != group_of_row[i - 1]
    ]
    return dict(
        trace=trace, lines=lines, xs=xs, tick_values=tick_values, varnames=varnames
    )


def _y_axis(varnames):
    return dict(
        tickvals=list(range(len(varnames))),
        ticktext=[str(v) for v in varnames],
        autorange="reversed",
        showgrid=False,
        zeroline=False,
    )


def plot_trajectory_interactive(
    df,
    fields=None,
    *,
    title=None,
    x=None,
    key=None,
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
        key: The respondent's key, put with the variable and the age in each
            cell's ``customdata``, for click handlers (``plotly_click``).
        gap_color: Color of the gaps between cells.
        minor_gap: Gap between cells, in pixels.
        major_gap: Wider gap (pixels) before every age multiple of
            ``major_every``, and between row groups.
        major_every: See ``major_gap``.
        cell_px: Size of a cell, in pixels.
        style: The cell colors, as in ``plot_trajectory`` (and only those):
            ``row_colors``, ``default_color``, ``row_backgrounds``,
            ``row_groups``, ``group_colors``, ``group_tint``.

    Returns:
        A ``plotly.graph_objects.Figure`` (show it with ``fig.show()``, or save
        it with ``fig.write_html(path)``).
    """
    go = _graph_objects()
    _check_style(style, "plot_trajectory_interactive")
    df = _select_fields(df, fields)
    parts = _heatmap_parts(
        df,
        x=x,
        key=key,
        gap_color=gap_color,
        minor_gap=minor_gap,
        major_gap=major_gap,
        major_every=major_every,
        style=style,
    )
    xs, varnames = parts["xs"], parts["varnames"]
    fig = go.Figure(go.Heatmap(**parts["trace"]))

    label_px = max(len(str(v)) for v in varnames) * DFLT_LABEL_PX_PER_CHAR
    top, bottom = (60 if title else 30), 50
    fig.update_layout(
        title=title,
        shapes=parts["lines"],
        width=max(len(xs) * cell_px, MIN_PLOT_WIDTH_PX) + label_px + 40,
        height=len(varnames) * cell_px + top + bottom,
        margin=dict(l=label_px + 20, r=20, t=top, b=bottom),
        plot_bgcolor=gap_color,
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        tickvals=xs,
        ticktext=[str(t) for t in parts["tick_values"]],
        showgrid=False,
        zeroline=False,
        range=[xs[0] - 0.5, xs[-1] + 0.5],
    )
    fig.update_yaxes(**_y_axis(varnames))
    return fig


def stacked_trajectories_interactive(
    df_store,
    fields=None,
    keys=None,
    *,
    x_align="age",
    x_label=None,
    title=None,
    k_df_to_title=None,
    gap_color="white",
    minor_gap=1,
    major_gap=4,
    major_every=DFLT_MAJOR_EVERY,
    cell_px=DFLT_CELL_PX,
    **style,
):
    """Plot several trajectories one above the other, aligned, with plotly.

    The interactive version of ``odus.plot_utils.stacked_trajectories`` (see it
    for ``fields``, ``keys``, ``x_align``, ``x_label`` and ``k_df_to_title``): one
    block of rows per trajectory, labeled with its title, separated by a wider
    gap. The cells are those of ``plot_trajectory_interactive``: hover text, and a
    ``customdata`` of ``[respondent key, variable, age]`` for click handlers.

    Returns:
        ``(figure, left_out)``: the plotly figure, and the trajectories left out
        (as ``odus.plot_utils.aligned_trajectories`` gives them).
    """
    from odus.plot_utils import _title_of_key, _x_label, aligned_trajectories

    go = _graph_objects()
    _check_style(style, "stacked_trajectories_interactive")
    k_df_to_title = k_df_to_title or _title_of_key
    aligned, left_out = aligned_trajectories(df_store, fields, keys, x_align=x_align)
    if not aligned:
        raise ValueError(f"No trajectory to plot (aligned on {x_align!r})")

    lo = min(x[0] for _, _, x in aligned)
    hi = max(x[-1] for _, _, x in aligned)
    full_x = list(range(int(lo), int(hi) + 1))
    hexes, texts, customdata, ticktext, row_breaks = [], [], [], [], []
    for k, df, x in aligned:
        img, group_of_row = trajectory_colors(df, **style)
        offset = int(x[0]) - int(lo)
        pad = [None] * offset, [None] * (len(full_x) - offset - len(x))
        block_texts = _hover_texts(df, list(x))
        ages = [a.item() if hasattr(a, "item") else a for a in df.index]
        if hexes:
            row_breaks.append(len(hexes))  # a wider gap between trajectories
        for i, var in enumerate(df.columns):
            if i and group_of_row[i] != group_of_row[i - 1]:
                row_breaks.append(len(hexes))
            hexes.append([*pad[0], *(_hex(c) for c in img[i]), *pad[1]])
            title_line = f"<b>{html.escape(str(k_df_to_title(k, df)))}</b><br>"
            texts.append([*pad[0], *(title_line + t for t in block_texts[i]), *pad[1]])
            cells = [[k, str(var), a] for a in ages]
            customdata.append([*pad[0], *cells, *pad[1]])
            label = str(var) if i else f"{k_df_to_title(k, df)} · {var}"
            ticktext.append(label)

    colors = sorted({h for row in hexes for h in row if h is not None})
    index_of = {c: i for i, c in enumerate(colors)}
    z = [[None if h is None else index_of[h] for h in row] for row in hexes]
    n = len(colors)
    colorscale = [
        stop for i, c in enumerate(colors) for stop in ([i / n, c], [(i + 1) / n, c])
    ]
    n_rows = len(z)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=full_x,
            y=list(range(n_rows)),
            zmin=-0.5,
            zmax=n - 0.5,
            colorscale=colorscale,
            showscale=False,
            xgap=minor_gap,
            ygap=minor_gap,
            text=texts,
            customdata=customdata,
            hovertemplate="%{text}<extra></extra>",
            hoverongaps=False,
        )
    )

    def line(**coordinates):
        gap = dict(color=gap_color, width=major_gap)
        return dict(type="line", line=gap, layer="above", **coordinates)

    shapes = [
        line(x0=full_x[j] - 0.5, x1=full_x[j] - 0.5, y0=-0.5, y1=n_rows - 0.5)
        for j in _major_boundaries(full_x, major_every)
    ]
    shapes += [
        line(x0=lo - 0.5, x1=hi + 0.5, y0=i - 0.5, y1=i - 0.5) for i in row_breaks
    ]
    label_px = max(map(len, ticktext)) * DFLT_LABEL_PX_PER_CHAR
    top, bottom = (60 if title else 30), 60
    fig.update_layout(
        title=title,
        shapes=shapes,
        width=max(len(full_x) * cell_px, MIN_PLOT_WIDTH_PX) + label_px + 40,
        height=n_rows * cell_px + top + bottom,
        margin=dict(l=label_px + 20, r=20, t=top, b=bottom),
        plot_bgcolor=gap_color,
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        title_text=x_label or _x_label(x_align),
        showgrid=False,
        zeroline=False,
        range=[lo - 0.5, hi + 0.5],
        side="bottom",
        mirror="ticks",
        showticklabels=True,
    )
    fig.update_yaxes(**_y_axis(ticktext))
    return fig, left_out
