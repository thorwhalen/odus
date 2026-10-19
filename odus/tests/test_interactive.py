"""Tests for odus.interactive (plotly trajectory plots)."""

import sys

import numpy as np
import pandas as pd
import pytest

from odus.plot_utils import trajectory_colors

pytest.importorskip("plotly")

from odus.interactive import _hex, plot_trajectory_interactive  # noqa: E402


@pytest.fixture
def traj():
    df = pd.DataFrame(
        {"HEROIN": [0] * 3 + [1] * 4, "WORK": [1] * 7, "SPOUSE": [0] * 7},
        index=pd.Index(range(18, 25), name="age"),
    )
    df.attrs["year_of_age"] = {a: a + 1980 for a in df.index}
    return df


def _cell_colors(fig):
    """The hex color plotly shows for each cell (from z and the discrete colorscale)."""
    (heatmap,) = fig.data
    scale = heatmap.colorscale
    return [[scale[2 * int(z)][1] for z in row] for row in heatmap.z]


def test_cells_are_colored_like_the_matplotlib_plot(traj):
    style = dict(row_colors={"HEROIN": "#c0392b"}, row_groups={"g": ["SPOUSE"]})
    fig = plot_trajectory_interactive(traj, **style)
    img, _ = trajectory_colors(traj, **style)
    assert _cell_colors(fig) == [[_hex(c) for c in row] for row in img]


def test_hover_says_what_a_cell_is(traj):
    fig = plot_trajectory_interactive(traj, ["WORK", "HEROIN"], title="t")
    (heatmap,) = fig.data
    assert heatmap.text[1][3] == "<b>HEROIN</b><br>age 21<br>year 2001<br>value 1"
    assert "%{text}" in heatmap.hovertemplate
    assert list(fig.layout.yaxis.ticktext) == ["WORK", "HEROIN"]
    assert fig.layout.title.text == "t"


def test_axes_and_gaps(traj):
    fig = plot_trajectory_interactive(traj, row_groups={"g": ["HEROIN", "WORK"]})
    assert list(fig.data[0].x) == list(range(18, 25))
    vertical = sorted(s.x0 for s in fig.layout.shapes if s.x0 == s.x1)
    horizontal = sorted(s.y0 for s in fig.layout.shapes if s.y0 == s.y1)
    assert vertical == [19.5]  # before age 20
    assert horizontal == [1.5]  # between the group and SPOUSE

    shifted = plot_trajectory_interactive(traj, x=np.arange(-3, 4))
    assert list(shifted.data[0].x) == list(range(-3, 4))
    assert "x -3" in shifted.data[0].text[0][0]


def test_missing_plotly_gives_an_install_hint(traj, monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(ImportError, match=r"odus\[interactive\]"):
        plot_trajectory_interactive(traj)


def test_hover_with_repeated_names_odd_names_and_a_series_of_years(traj):
    traj = traj.rename(columns={"SPOUSE": "<5 yrs & co"})
    traj.attrs["year_of_age"] = pd.Series(traj.attrs["year_of_age"])
    fig = plot_trajectory_interactive(traj, ["WORK", "WORK", "<5 yrs & co"])
    text = fig.data[0].text
    assert text[1][0] == "<b>WORK</b><br>age 18<br>year 1998<br>value 1"
    assert text[2][0].startswith("<b>&lt;5 yrs &amp; co</b>")


def test_non_color_styling_is_refused_clearly(traj):
    with pytest.raises(TypeError, match=r"doesn't take \['fontsize'\]"):
        plot_trajectory_interactive(traj, fontsize=8)


def test_cells_carry_what_a_click_is_about(traj):
    fig = plot_trajectory_interactive(traj, ["WORK", "HEROIN"], key="R1")
    assert fig.data[0].customdata[1][2] == ["R1", "HEROIN", 20]


def test_stacked_interactive_aligns_blocks_of_rows(traj):
    from odus.interactive import stacked_trajectories_interactive

    late = traj.copy()
    late.index = late.index + 10  # ages 28..34, first HEROIN at 31 (21 in traj)
    store = {"s/A.xlsx": traj, "s/B.xlsx": late, "s/C.xlsx": traj[["WORK"]]}
    fig, left_out = stacked_trajectories_interactive(
        store, ["HEROIN", "WORK"], x_align="HEROIN"
    )
    assert left_out == {"not_aligned": ["s/C.xlsx"], "missing_fields": {}}
    (heatmap,) = fig.data
    assert list(heatmap.x) == list(range(-3, 4))  # both span -3..3 around first use
    labels = ["A.xlsx · HEROIN", "WORK", "B.xlsx · HEROIN", "WORK"]
    assert list(fig.layout.yaxis.ticktext) == labels
    assert heatmap.customdata[2][3] == ["s/B.xlsx", "HEROIN", 31]
    assert "<b>B.xlsx</b>" in heatmap.text[2][3]
    assert fig.layout.xaxis.title.text == "years since first HEROIN"
    breaks = [s.y0 for s in fig.layout.shapes if s.y0 == s.y1]
    assert breaks == [1.5]  # a wider gap between the two respondents


def test_stacked_interactive_needs_whole_number_x(traj):
    from odus.interactive import stacked_trajectories_interactive

    with pytest.raises(ValueError, match="whole-number x"):
        stacked_trajectories_interactive({"k": traj}, x_align=lambda df: df.index - 0.5)
    fig, _ = stacked_trajectories_interactive(
        {"a": traj, "b": traj}, x_align=lambda df: df.index - 30
    )
    assert list(fig.data[0].x) == list(range(-12, -5))  # negative x line up too
