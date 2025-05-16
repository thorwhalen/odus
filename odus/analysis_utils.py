from os.path import dirname, join, sep
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from itertools import combinations, chain

from odus.sequential_var_sets import PVar, VarSet, DfData, VarSetFactory
from spyn.ppi.pot import Pot
from odus.dacc import (
    DfStore,
    counts_of_kps,
    Dacc,
    VarSetCountsStore,
    mk_pvar_struct,
    PotStore,
    _commun_columns_of_dfs,
    Struct,
    mk_pvar_str_struct,
    VarStr,
)
from odus.plot_utils import (
    plot_life_course,
    plot_life,
    life_plots,
    write_trajectories_to_file,
)

# __all__ = ['plot_life_course', 'plot_life', 'life_plots', 'write_trajectories_to_file',
#            'counts_of_kps', 'Dacc', 'VarSetCountsStore', 'PVar', 'VarSet', 'DfData', 'VarSetFactory']

package_dir = dirname(__file__)
DFLT_SURVEY_DIR = join(
    package_dir,
)
dflt_figsize = (16, 5)


def get_stores_v_and_s(survey_dir=DFLT_SURVEY_DIR):
    df_store = DfStore(survey_dir + sep + '{}.xlsx', mode='b')
    #     cstore = VarSetCountsStore(df_store)
    pstore = PotStore(df_store)
    v = mk_pvar_struct(df_store, only_for_cols_in_all_dfs=True)
    s = mk_pvar_str_struct(v)
    return df_store, pstore, v, s


def get_cstores_v_and_s(survey_dir=DFLT_SURVEY_DIR):
    df_store = DfStore(survey_dir + sep + '{}.xlsx', mode='b')
    pstore = PotStore(df_store)
    v = mk_pvar_struct(df_store, only_for_cols_in_all_dfs=True)
    s = mk_pvar_str_struct(v)
    return pstore, v, s


def get_markov_rel_risk(pstore, fields=None):
    if fields is None:
        fields = _commun_columns_of_dfs(pstore.df_store.values())
    c = list()
    for vs in VarSetFactory.markov_pairs(fields):
        p = pstore[vs]
        if vs.varset[0].i == -1:
            exposure = str(vs.varset[0])
            event = str(vs.varset[1])
        else:
            event = str(vs.varset[0])
            exposure = str(vs.varset[1])
        rel_risk = p.relative_risk(
            event_var=event, exposure_var=exposure, event_val=1, exposure_val=1
        )

        c.append({'exposure': exposure, 'event': event, 'rel_risk': rel_risk})

    markov_rel_risk = pd.DataFrame(c)
    return markov_rel_risk.pivot(index='exposure', columns='event', values='rel_risk')


def remission_relative_risk(pstore, event, exposure):
    """
    Returns the remission influence of exposure on event.
    By "remission influence" we mean the relative risk of the event, given the exposure to an influencer
    during the "previous" time tick, conditioned on the existence of the event.
    That is, we're looking at:
        P(event=true | exposure-1=true, event-1=true)
    divided by
        P(event=true | exposure-1=false, event-1=true)
    """
    if isinstance(event, str):
        event = VarStr(event)
    if isinstance(exposure, str):
        exposure = VarStr(exposure)
    p = pstore[exposure - 1, event - 1, event]
    event_last_year = p * Pot.from_hard_evidence(**{event - 1: 1}) >> [
        exposure - 1,
        event,
    ]
    return event_last_year.relative_risk(event, exposure - 1, smooth_count=1)


def get_markov_remission_rel_risk(pstore, fields=None):
    if fields is None:
        fields = _commun_columns_of_dfs(pstore.df_store.values())
    c = list()
    for event, exposure in chain(
        *([(x, y), (y, x)] for x, y in combinations(fields, 2))
    ):
        rel_risk = remission_relative_risk(pstore, event, exposure)
        c.append({'exposure': exposure, 'event': event, 'remission_rel_risk': rel_risk})

    markov_rel_risk = pd.DataFrame(c)
    return markov_rel_risk.pivot(
        index='exposure', columns='event', values='remission_rel_risk'
    )


# Print ################################################################################################################
def print_counts(pot):
    df = pot.tb.copy()
    df['count'] = df['pval']
    del df['pval']
    print(df)


def print_relrisk_and_table(pstore, event, exposure):
    pot = pstore[exposure, event]
    rr = pot.relative_risk(event, exposure, smooth_count=1)
    print(f"Relative risk of {event} when {exposure}:\n\t{rr:0.2f}")
    print("Based on the count table:")
    print_counts(pot)


def print_relrisk_and_table_from_pot(pot, event, exposure):
    rr = pot.relative_risk(event, exposure, smooth_count=1)
    print(f"Relative risk of {event} when {exposure}:\n\t{rr:0.2f}")
    print("Based on the count table:")
    print_counts(pot)


def print_remission_influence_supporting_info(pstore, event, exposure):
    """
    Prints remission influence supporting information.
    By "remission influence" we mean the relative risk of the event, given the exposure to an influencer
    during the "previous" time tick, conditioned on the existence of the event.
    That is, we're looking at:
        P(event=true | exposure-1=true, event-1=true)
    divided by
        P(event=true | exposure-1=false, event-1=true)
    """
    if isinstance(event, str):
        event = VarStr(event)
    if isinstance(exposure, str):
        exposure = VarStr(exposure)
    p = pstore[exposure - 1, event - 1, event]
    print(p)
    event_last_year = p * Pot.from_hard_evidence(**{event - 1: 1}) >> [
        exposure - 1,
        event,
    ]
    print_relrisk_and_table_from_pot(event_last_year, event, exposure - 1)


# Plot #################################################################################################################


def get_tick_and_labels(y_ticks, y_tick_labels=None):
    y_tick = []
    y_tick_label = []
    for y in y_ticks:
        if int(y) == y:
            y_tick.append(y)
            if y == 0:
                y_tick_label.append("")
            else:
                y_tick_label.append(str(np.sign(y) * int(2 ** abs(y))) + 'x')
    return y_tick, y_tick_label


def format_for_influencer_plot():
    t, _ = plt.yticks()
    if max(abs(t)) > 1:
        plt.yticks(*get_tick_and_labels(t))
    else:
        plt.yticks(t, list(map(lambda x: "{:0.2f}".format(2**x), t)))
    plt.grid(axis='y')


def diagonal_rr(lrr):
    c = list()
    for i, j in zip(lrr.index, lrr.columns):
        assert i.startswith(j)
        c.append({'x': i, 'var': j, 'val': lrr.loc[i, j]})
    return pd.DataFrame(c)[['var', 'val']].set_index('var')['val']


def plot_diagonal(lrr, figsize=dflt_figsize, **kwargs):
    t = diagonal_rr(lrr)
    kwargs = dict(title='Self (log2) relative risk', figsize=figsize, **kwargs)
    t.plot(kind='bar', **kwargs)
    format_for_influencer_plot()


def plot_influencers(lrr, var, figsize=dflt_figsize, **kwargs):
    kwargs = dict(
        title='Log2 relative risks for event: {}'.format(var), figsize=figsize, **kwargs
    )
    lrr.loc[:, var].plot(kind='bar', **kwargs)
    format_for_influencer_plot()


def plot_influenced(lrr, var, figsize=dflt_figsize, **kwargs):
    kwargs = dict(
        title='Log2 relative risks when exposed to {}'.format(var),
        figsize=figsize,
        **kwargs,
    )
    lrr.loc[var - 1, :].plot(kind='bar', **kwargs)
    format_for_influencer_plot()


def plot_remission_influencers(lrr, var, figsize=dflt_figsize, **kwargs):
    kwargs = dict(
        title='Log2 remission relative risks for event: {}'.format(var),
        figsize=figsize,
        **kwargs,
    )
    lrr.loc[:, var].plot(kind='bar', **kwargs)
    format_for_influencer_plot()


def plot_remission_influenced(lrr, var, figsize=dflt_figsize, **kwargs):
    kwargs = dict(
        title='Log2 remission relative risks when exposed to {}'.format(var),
        figsize=figsize,
        **kwargs,
    )
    lrr.loc[var, :].plot(kind='bar', **kwargs)
    format_for_influencer_plot()
