"""Tests that the store-building entry points of odus work on the bundled surveys."""

import os
import warnings

import pytest

from odus import data_dir
from odus.sequential_var_sets import PVar, VarSet

SURVEYS = os.path.join(data_dir, "surveys")
N_SURVEYS = len([f for f in os.listdir(SURVEYS) if f.endswith(".xlsx")])


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the bundled surveys' known anomalies
        yield


def test_dacc_on_the_bundled_surveys():
    from odus.dacc import Dacc

    dacc = Dacc(SURVEYS)
    assert len(list(dacc.s)) == N_SURVEYS
    assert "HEROIN" in dacc.categories


def test_get_stores_v_and_s_default_to_the_bundled_surveys():
    from odus.analysis_utils import get_cstores_v_and_s, get_stores_v_and_s

    df_store, pstore, v, s = get_stores_v_and_s()
    assert len(df_store) == N_SURVEYS
    assert "A01.xlsx" in df_store  # keys relative to the bundled surveys folder
    assert hasattr(v, "heroin") and s.heroin == "HEROIN"

    pstore, v, s = get_cstores_v_and_s()
    pot = pstore[VarSet(PVar("HEROIN", -1), PVar("HEROIN", 0))]
    counts = pot.tb.set_index(["HEROIN-1", "HEROIN"]).pval.to_dict()
    assert counts == {(0, 0): 1176, (0, 1): 205, (1, 0): 114, (1, 1): 1051}


def test_dacc_counts_dont_accumulate_across_calls():
    from odus.dacc import Dacc

    dacc = Dacc(SURVEYS)
    k = VarSet(PVar("HEROIN", -1), PVar("HEROIN", 0))
    first = dict(dacc.mk_counts_of_kps([k])[k])
    assert dict(dacc.mk_counts_of_kps([k])[k]) == first
    assert sum(first.values()) == 1176 + 205 + 114 + 1051


def test_influence_plots_take_plain_variable_names():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from odus.analysis_utils import (
        get_markov_rel_risk,
        get_stores_v_and_s,
        plot_influenced,
        plot_influencers,
    )

    _, pstore, _, _ = get_stores_v_and_s()
    lrr = get_markov_rel_risk(pstore, fields=["HEROIN", "INJECTED"])
    for var in ("HEROIN", "HEROIN-1"):  # failed: unsupported operand 'str' - 'int'
        plot_influenced(lrr, var)
        ax = plt.gca()
        assert ax.get_title() == f"Log2 relative risks when exposed to {var}"
        heights = [bar.get_height() for bar in ax.patches]
        assert heights == pytest.approx(lrr.loc["HEROIN-1"].tolist())
        plt.close("all")
    plot_influencers(lrr, "HEROIN")
    plt.close("all")
