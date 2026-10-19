"""Tests for odus.interactive (plotly trajectory plots)."""

import subprocess
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


def test_importing_odus_doesnt_import_plotly():
    code = "import sys, odus, odus.plot_utils, odus.interactive; print('plotly' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr


def test_missing_plotly_gives_an_install_hint(traj, monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(ImportError, match=r"odus\[interactive\]"):
        plot_trajectory_interactive(traj)
