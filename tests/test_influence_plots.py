"""Influence plots plot log2 of the relative risks and accept str/VarStr/PVar variables."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from odus.analysis_utils import (
    plot_influenced,
    plot_influencers,
    plot_remission_influenced,
    plot_remission_influencers,
)
from odus.sequential_var_sets import PVar


@pytest.fixture
def rr():
    events = ["A", "B"]
    return pd.DataFrame(
        [[4.0, 0.5], [2.0, 8.0]], index=["A-1", "B-1"], columns=events
    )


def _heights():
    return [p.get_height() for p in plt.gca().patches]


@pytest.mark.parametrize("mk", [str, PVar])
def test_influencers_plot_log2_for_str_and_pvar(rr, mk):
    plt.figure()
    plot_influencers(rr, mk("A"))
    assert np.allclose(_heights(), [2.0, 1.0])
    plt.close("all")


@pytest.mark.parametrize("mk", [str, PVar])
def test_influenced_plot_log2_for_str_and_pvar(rr, mk):
    plt.figure()
    plot_influenced(rr, mk("B"))
    assert np.allclose(_heights(), [1.0, 3.0])
    plt.close("all")


def test_remission_variants_accept_pvar(rr):
    plt.figure()
    plot_remission_influencers(rr, PVar("A"))
    assert np.allclose(_heights(), [2.0, 1.0])
    plt.close("all")
    plt.figure()
    plot_remission_influenced(rr.rename(index={"A-1": "A", "B-1": "B"}), PVar("B"))
    assert np.allclose(_heights(), [1.0, 3.0])
    plt.close("all")
