"""Data access: stores and accessors over the survey data.

Defines the ``py2store``-based stores (``DfStore``, ``VarSetCountsStore``,
``PotStore``) and the ``Dacc`` facade that give easy access to the prepared
trajectory dataframes, their count tables, and the corresponding potentials.
"""

from collections import defaultdict, Counter
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os
import re
import warnings
from functools import wraps
from io import BytesIO

# get py2store here: https://github.com/i2mint/py2store
from py2store.base import Store
from py2store.stores.local_store import RelativePathFormatStore, LocalBinaryStore
from py2store.mixins import ReadOnlyMixin

from odus.nothing import nothing
from spyn.ppi.pot import Pot

from odus.sequential_var_sets import PVar, VarSet, DfData, extract_kps
from odus.plot_utils import heatmap

# from ut.pplot.matrix import heatmap

path_sep = os.path.sep


def ensure_slash_suffix(x):
    if not x.endswith(path_sep):
        return x + path_sep
    return x


def _commun_columns_of_dfs(dfs):
    categories = nothing
    df = None
    for df in dfs:
        categories = categories.intersection(set(df.columns))
    if df is not None:
        categories = [ss for ss in df.columns if ss in categories]
        return categories
    else:
        return []


def _all_columns_of_dfs(dfs):
    categories = []
    for df in dfs:
        for c in df.columns:
            if c not in categories:
                categories.append(c)
    return categories


simple_cat_p = re.compile(r"\W")


class Struct:
    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)


class VarStr(str):
    def __add__(self, other):
        if isinstance(other, int):
            other = "+" + str(other)
        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            other = "-" + str(other)
        return super().__add__(other)


def two_way_map(keys, val_of_key_func):
    val_of_key = dict()
    key_of_val = dict()
    for k in keys:
        v = val_of_key_func(k)
        val_of_key[k] = v
        key_of_val[v] = k
    return val_of_key, key_of_val


def mk_pvar_struct(df_store, only_for_cols_in_all_dfs=False):
    if only_for_cols_in_all_dfs:
        categories = _commun_columns_of_dfs(df_store.values())
    else:
        categories = _all_columns_of_dfs(df_store.values())
    val_of_attr = {
        simple_cat_p.sub("_", c.lower().strip()): PVar(c) for c in categories
    }
    return Struct(**val_of_attr)


def mk_pvar_str_struct(df_store, only_for_cols_in_all_dfs=False):
    if isinstance(df_store, Struct):
        return Struct(**{k: VarStr(v) for k, v in df_store.__dict__.items()})
    else:
        return mk_pvar_str_struct(
            mk_pvar_str_struct(df_store, only_for_cols_in_all_dfs)
        )


class HashableMixin:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return hash(self) == hash(other)


NON_BINARY_POLICIES = ("warn", "binarize", "keep", "raise")


def _first_non_zero(block):
    """Merge same-variable columns: per row, the first non-zero value (else 0)."""
    merged = block.iloc[:, 0].copy()
    for j in range(1, block.shape[1]):
        merged = merged.where(merged.ne(0), block.iloc[:, j])
    return merged


DUPLICATES_POLICIES = ("merge", "keep", "raise")


def prep_trajectory_df(df, *, source=None, non_binary="warn", duplicates="merge"):
    """Clean a trajectory DataFrame (index: ages, columns: variables).

    - Columns named like a section title of the survey sheet (``NON_VARIABLE_LABELS``
      of ``odus.raw_data_diagnosis``, e.g. "Select States below") are dropped.
    - ``duplicates``: what to do with variables appearing more than once (e.g. a
      state lived in during two periods, entered as two rows), names compared
      ignoring case and extra spaces. ``"merge"``: one column (with the first
      name), holding at each age the first non-zero value of the duplicates;
      ``"keep"``: leave them; ``"raise"``: raise a ``ValueError``.
    - ``non_binary``: what to do with values other than 0 and 1. ``"warn"``: set
      them to 1 and say which (variable, ages, values); ``"binarize"``: set them to
      1 silently; ``"keep"``: leave them; ``"raise"``: raise a ``ValueError``.

    ``source`` (e.g. the file name) is only used in messages.

    >>> df = pd.DataFrame(
    ...     [[1, 0, 0, 0], [0, 0, 1, 2]],
    ...     columns=['GA', 'NJ', 'NJ', 'Select States below'],
    ...     index=pd.Index([20, 21], name='age'),
    ... )
    >>> prep_trajectory_df(df)
        GA  NJ
    age
    20   1   0
    21   0   1
    """
    from odus.raw_data_diagnosis import is_section_title, normalized_label

    if non_binary not in NON_BINARY_POLICIES:
        raise ValueError(f"non_binary should be one of {NON_BINARY_POLICIES}")
    if duplicates not in DUPLICATES_POLICIES:
        raise ValueError(f"duplicates should be one of {DUPLICATES_POLICIES}")
    where = f" in {source}" if source is not None else ""

    df = df.loc[:, [not is_section_title(c) for c in df.columns]]

    positions_of = {}  # normalized label -> positions of the columns with that label
    for j, c in enumerate(df.columns):
        positions_of.setdefault(normalized_label(c), []).append(j)
    dups = [ps for ps in positions_of.values() if len(ps) > 1]
    if dups and duplicates == "raise":
        names = [df.columns[ps[0]] for ps in dups]
        raise ValueError(f"Variables appearing more than once{where}: {names}")
    if dups and duplicates == "merge":
        groups = list(positions_of.values())
        merged = df.iloc[:, [ps[0] for ps in groups]].copy()
        for i, ps in enumerate(groups):
            if len(ps) > 1:
                merged.iloc[:, i] = _first_non_zero(df.iloc[:, ps]).to_numpy()
        df = merged

    is_non_binary = ~df.isin([0, 1]).to_numpy()
    if non_binary != "keep" and is_non_binary.any():
        problems = "; ".join(
            f"{df.columns[j]!r} at ages {list(df.index[is_non_binary[:, j]])} "
            f"({sorted(set(df.iloc[:, j][is_non_binary[:, j]]))})"
            for j in np.flatnonzero(is_non_binary.any(axis=0))
        )
        if non_binary == "raise":
            raise ValueError(f"Values other than 0 and 1{where}: {problems}")
        if non_binary == "warn":
            warnings.warn(
                f"Values other than 0 and 1{where} were set to 1: {problems}",
                stacklevel=2,
            )
        df = df.where(df.eq(0), 1)
    return df


class DfStore(HashableMixin, ReadOnlyMixin, LocalBinaryStore):
    """The trajectory DataFrames of a folder of survey spreadsheets.

    Values are cleaned by ``prep_trajectory_df``, with the ``non_binary`` and
    ``duplicates`` policies set as attributes (``store.non_binary = "raise"``, or
    in a subclass).
    """

    non_binary = "warn"
    duplicates = "merge"

    def __getitem__(self, k):
        cache = self.__dict__.setdefault("_df_cache", {})
        policies = (self.non_binary, self.duplicates)
        if (k, policies) not in cache:
            cache[k, policies] = prep_trajectory_df(
                super().__getitem__(k),
                source=k,
                non_binary=self.non_binary,
                duplicates=self.duplicates,
            )
        return cache[k, policies]

    def _obj_of_data(self, data):
        df = pd.read_excel(BytesIO(data), header=0)
        df = df.iloc[:, 1:]
        df.iloc[0, 0] = "year"
        df.iloc[1, 0] = "age"
        df = df.rename(columns={"INTERVIEW ID NUMBER (enter in red)": "category"})
        df = df.T
        df.columns = df.iloc[0, :]
        df = df.iloc[1:, 1:]  # TODO: would like to keep year, but not in column
        df = df.set_index("age")
        return df.astype(int)


def _cached_per_instance(getter):
    """Cache ``getter(self, k)`` in ``self``, so that stores don't share a cache."""

    @wraps(getter)
    def cached_getter(self, k):
        cache = self.__dict__.setdefault("_cache_" + getter.__qualname__, {})
        if k not in cache:
            cache[k] = getter(self, k)
        return cache[k]

    return cached_getter


class VarSetCountsStore(HashableMixin, ReadOnlyMixin, Store):
    def __init__(self, df_store, store=None):
        self.df_store = df_store
        super().__init__(store)

    def mk_pvar_attrs(self, only_for_cols_in_all_dfs=False):
        self.v = mk_pvar_struct(self.df_store, only_for_cols_in_all_dfs)

    @_cached_per_instance
    def __getitem__(self, k):
        if isinstance(k, (tuple, PVar)):
            return self.__getitem__(VarSet(k))
        else:
            c = Counter()
            for df in self.df_store.values():
                # counts = DfData(df).extract_kps(k)
                counts = extract_kps(df, k)
                c.update(counts)
            return c


class PotStore(VarSetCountsStore):
    @_cached_per_instance
    def __getitem__(self, k):
        if isinstance(k, (tuple, PVar)):
            return self.__getitem__(VarSet(k))
        else:
            counter = super().__getitem__(k)
            d = list()
            for key, val in counter.items():
                dd = {str(k.varset[i]): var_val for i, var_val in enumerate(key)}
                dd["pval"] = val
                d.append(dd)
            # return pd.DataFrame(d)
            return Pot.from_count_df_to_count(
                pd.DataFrame(d)[k.varset_strs + ["pval"]], count_col="pval"
            )


class DelegMap:
    def __init__(self, dfunc, dmap=None):
        if dmap is None:
            dmap = dict()
        self.dmap = dmap
        self.dfunc = dfunc

    def __getitem__(self, k):
        if k not in self.dmap:
            self.dmap[k] = self.dfunc(k)
        return self.dmap[k]


class Dacc:
    def __init__(self, xls_rootdir, categories=None):
        self.s = DfStore(ensure_slash_suffix(xls_rootdir) + "{}.xlsx", mode="b")
        if categories is None:
            categories = _commun_columns_of_dfs(self.s.values())
        self.categories = categories
        self.counts_of = defaultdict(Counter)

    def mk_counts_of_kps(self, kps_list):
        for f in self.s.keys():
            df = self.s[f][self.categories]
            dd = DfData(df)
            for k, v in dd.extract_with_key_pattern_sets(kps_list):
                self.counts_of[k].update([v])

    # def counts_of(self):


def counts_of_kps(store, categories, kps_list):
    counts_of = defaultdict(Counter)
    for f in store.keys():
        df = store[f][categories]
        dd = DfData(df)
        for k, v in dd.extract_with_key_pattern_sets(kps_list):
            counts_of[k].update([v])
    return counts_of


# vs = tuple(map(str, VarSet.from_str(s)))
# d = list()
# for k, v in t.items():
#     dd = {vs[i]: kk for i, kk in enumerate(k)}
#     dd['pval'] = v
#     d.append(dd)
# u = ProbPot.from_count_df_to_count(pd.DataFrame(d), count_col='pval')
# u

if __name__ == "__main__":
    import os

    proj_root = os.path.dirname(__file__)
    pjoin = lambda f: os.path.join(proj_root, f)
    xls_rootdir = pjoin("data/surveys/")

    # vs = tuple(map(str, VarSet.from_str(s)))

    # d = list()
    # for k, v in counts_of.items():
    #     dd = {vs[i]: kk for i, kk in enumerate(k)}
    #     dd['pval'] = v
    #     d.append(dd)
    # u = ProbPot.from_count_df_to_count(pd.DataFrame(d), count_col='pval')
    # u
