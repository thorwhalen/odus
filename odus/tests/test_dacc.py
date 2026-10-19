"""Tests for the cleaning of trajectories in odus.dacc."""

import os
import shutil
import warnings

import pandas as pd
import pytest

from odus import data_dir
from odus.dacc import DfStore, prep_trajectory_df

SURVEYS = os.path.join(data_dir, "surveys")


@pytest.fixture
def raw():
    return pd.DataFrame(
        [[1, 1, 0, 1990], [0, 0, 16, 1991], [0, 0, 1, 1992]],
        columns=pd.Index(["NJ", "GA", "NJ", "Selects States below"], name="category"),
        index=pd.Index([20, 21, 22], name="age"),
    )


def test_defaults_merge_drop_titles_and_warn(raw):
    with pytest.warns(
        UserWarning, match=r"in A\.xlsx were set to 1: 'NJ' at ages \[21\]"
    ):
        df = prep_trajectory_df(raw, source="A.xlsx")
    assert list(df.columns) == ["NJ", "GA"]
    assert df["NJ"].tolist() == [1, 1, 1]  # merged, and 16 set to 1
    assert df.columns.name == "category" and df.index.name == "age"


def test_policies(raw):
    kept = prep_trajectory_df(raw, non_binary="keep", duplicates="keep")
    assert list(kept.columns) == ["NJ", "GA", "NJ"]
    assert kept.iloc[1, 2] == 16

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = prep_trajectory_df(raw, non_binary="binarize")
    assert quiet["NJ"].tolist() == [1, 1, 1]

    with pytest.raises(ValueError, match="other than 0 and 1"):
        prep_trajectory_df(raw, non_binary="raise")
    with pytest.raises(ValueError, match=r"more than once.*\['NJ'\]"):
        prep_trajectory_df(raw, duplicates="raise", non_binary="keep")
    with pytest.raises(ValueError, match="non_binary should be one of"):
        prep_trajectory_df(raw, non_binary="nope")


def test_df_store_cleans_the_bundled_surveys():
    store = DfStore(SURVEYS + os.sep)
    with pytest.warns(UserWarning, match="A13.xlsx"):
        a13 = store["A13.xlsx"]
    assert set(a13.stack().unique()) <= {0, 1}
    assert not store["A03.xlsx"].columns.duplicated().any()
    assert "Selects States below" not in store["A37.xlsx"].columns

    store.non_binary = "raise"
    with pytest.raises(ValueError, match="A13.xlsx"):
        store["A13.xlsx"]


def test_df_stores_dont_share_their_cache(tmp_path):
    for name in ("A", "B"):
        os.mkdir(tmp_path / name)
    shutil.copy(os.path.join(SURVEYS, "A01.xlsx"), tmp_path / "A" / "X.xlsx")
    shutil.copy(os.path.join(SURVEYS, "A60.xlsx"), tmp_path / "B" / "X.xlsx")
    a = DfStore(str(tmp_path / "A") + os.sep)["X.xlsx"]
    b = DfStore(str(tmp_path / "B") + os.sep)["X.xlsx"]
    assert a.shape != b.shape


def test_non_binary_values_in_duplicated_columns():
    df = pd.DataFrame([[0, 1], [0, 5]], columns=["NJ", "NJ"], index=[20, 21])
    with pytest.warns(UserWarning, match=r"'NJ' at ages \[21\] \(\[5\]\)"):
        kept = prep_trajectory_df(df, duplicates="keep")
    assert kept.values.tolist() == [[0, 1], [0, 1]]
    with pytest.raises(ValueError, match="other than 0 and 1"):
        prep_trajectory_df(df, duplicates="keep", non_binary="raise")


def test_merge_keeps_odd_values_and_names_visible():
    df = pd.DataFrame(
        [[-1, 0, 0, 1], [0, 0, 1, 0]],
        columns=["NJ", "nj ", None, "GA"],
        index=[20, 21],
    )
    merged = prep_trajectory_df(df, non_binary="keep")
    names = list(merged.columns)  # merged ignoring case/space; None kept, not dropped
    assert names[0] == "NJ" and pd.isna(names[1]) and names[2] == "GA"
    assert merged["NJ"].tolist() == [-1, 0]  # not hidden by a max
    with pytest.warns(UserWarning, match=r"'NJ' at ages \[20\] \(\[-1\]\)"):
        prep_trajectory_df(df)


def test_count_stores_dont_share_their_cache():
    from odus.dacc import VarSetCountsStore
    from odus.sequential_var_sets import VarSet, PVar

    binarized, kept = DfStore(SURVEYS + os.sep), DfStore(SURVEYS + os.sep)
    kept.non_binary = "keep"
    k = VarSet(PVar("INCARCERATION"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts_binarized = VarSetCountsStore(binarized)[k]
    counts_kept = VarSetCountsStore(kept)[k]
    assert 16 not in {key[0] for key in counts_binarized}
    assert 16 in {key[0] for key in counts_kept}
