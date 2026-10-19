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
