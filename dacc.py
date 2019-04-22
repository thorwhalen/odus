from collections import defaultdict, Counter
import matplotlib.pylab as plt
import pandas as pd
from cachetools import cached
import os
import re
from io import BytesIO

# get py2store here: https://github.com/i2mint/py2store
from py2store.base import Store
from py2store.stores.local_store import RelativePathFormatStore
from py2store.mixins import ReadOnlyMixin

from odus.nothing import nothing
from hyp.ppi.pot import Pot

from ut.ml.feature_extraction.sequential_var_sets import PVar, VarSet, DfData, extract_kps
from ut.pplot.matrix import heatmap

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


simple_cat_p = re.compile('\W')


class Struct:
    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)


class VarStr(str):
    def __add__(self, other):
        if isinstance(other, int):
            other = '+' + str(other)
        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            other = '-' + str(other)
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
    val_of_attr = {simple_cat_p.sub('_', c.lower().strip()): PVar(c)
                   for c in categories}
    return Struct(**val_of_attr)


def mk_pvar_str_struct(df_store, only_for_cols_in_all_dfs=False):
    if isinstance(df_store, Struct):
        return Struct(**{k: VarStr(v) for k, v in df_store.__dict__.items()})
    else:
        return mk_pvar_str_struct(mk_pvar_str_struct(df_store, only_for_cols_in_all_dfs))


class HashableMixin:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return hash(self) == hash(other)


class DfStore(HashableMixin, ReadOnlyMixin, RelativePathFormatStore):

    @cached(cache={})
    def __getitem__(self, k):
        return super().__getitem__(k)

    def _obj_of_data(self, data):
        df = pd.read_excel(BytesIO(data), header=0)
        df = df.iloc[:, 1:]
        df.iloc[0, 0] = 'year'
        df.iloc[1, 0] = 'age'
        df = df.rename(columns={'INTERVIEW ID NUMBER (enter in red)': 'category'})
        df = df.T
        df.columns = df.iloc[0, :]
        df = df.iloc[1:, 1:]  # TODO: would like to keep year, but not in column
        df = df.set_index('age')
        df = df.astype(int)
        df[df != 0] = 1  # TODO: May want to remove this to expose mistakes instead of repairing them
        return df


class VarSetCountsStore(HashableMixin, ReadOnlyMixin, Store):
    def __init__(self, df_store, store=None):
        self.df_store = df_store
        super().__init__(store)

    def mk_pvar_attrs(self, only_for_cols_in_all_dfs=False):
        self.v = mk_pvar_struct(self.df_store, only_for_cols_in_all_dfs)

    @cached(cache={})
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
    @cached(cache={})
    def __getitem__(self, k):
        if isinstance(k, (tuple, PVar)):
            return self.__getitem__(VarSet(k))
        else:
            counter = super().__getitem__(k)
            d = list()
            for key, val in counter.items():
                dd = {str(k.varset[i]): var_val for i, var_val in enumerate(key)}
                dd['pval'] = val
                d.append(dd)
            # return pd.DataFrame(d)
            return Pot.from_count_df_to_count(pd.DataFrame(d)[k.varset_strs + ['pval']], count_col='pval')


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
        self.s = DfStore(ensure_slash_suffix(xls_rootdir) + '{}.xlsx', read='b', write='b')
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


def plot_life_course(df):
    heatmap(df.T);
    plt.grid(False);


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

if __name__ == '__main__':
    import os

    proj_root = os.path.dirname(__file__)
    pjoin = lambda f: os.path.join(proj_root, f)
    xls_rootdir = pjoin('data/surveys/')

    # vs = tuple(map(str, VarSet.from_str(s)))

    # d = list()
    # for k, v in counts_of.items():
    #     dd = {vs[i]: kk for i, kk in enumerate(k)}
    #     dd['pval'] = v
    #     d.append(dd)
    # u = ProbPot.from_count_df_to_count(pd.DataFrame(d), count_col='pval')
    # u
