import matplotlib.pylab as plt
from itertools import islice
from typing import Iterable
import numpy as np
import pandas as pd

from odus.util import write_images

ihead = lambda it: islice(it, 0, 5)  # just a little useful util. Not used within module


def heatmap(X, y=None, col_labels=None, figsize=None, cmap=None, return_gcf=False, ax=None,
            xlabel_top=True, ylabel_left=True, xlabel_bottom=True, ylabel_right=True, **kwargs):
    n_items, n_cols = X.shape
    if col_labels is not None:
        if col_labels is not False:
            assert len(col_labels) == n_cols, \
                "col_labels length should be the same as the number of columns in the matrix"
    elif isinstance(X, pd.DataFrame):
        col_labels = list(X.columns)

    if figsize is None:
        x_size, y_size = X.shape
        if x_size >= y_size:
            figsize = (6, min(18, 6 * x_size / y_size))
        else:
            figsize = (min(18, 6 * y_size / x_size), 6)

    if cmap is None:
        if X.min(axis=0).min(axis=0) < 0:
            cmap = 'RdBu_r'
        else:
            cmap = 'hot_r'

    kwargs['cmap'] = cmap
    kwargs = dict(kwargs, interpolation='nearest', aspect='auto')

    if figsize is not False:
        plt.figure(figsize=figsize)

    if ax is None:
        plt.imshow(X, **kwargs)
    else:
        ax.imshow(X, **kwargs)
    plt.grid(None)

    if y is not None:
        y = np.array(y)
        assert all(sorted(y) == y), "This will only work if your row_labels are sorted"

        unik_ys, unik_ys_idx = np.unique(y, return_index=True)
        for u, i in zip(unik_ys, unik_ys_idx):
            plt.hlines(i - 0.5, 0 - 0.5, n_cols - 0.5, colors='b', linestyles='dotted', alpha=0.5)
        plt.hlines(n_items - 0.5, 0 - 0.5, n_cols - 0.5, colors='b', linestyles='dotted', alpha=0.5)
        plt.yticks(unik_ys_idx + np.diff(np.hstack((unik_ys_idx, n_items))) / 2, unik_ys)
    elif isinstance(X, pd.DataFrame):
        y_tick_labels = list(X.index)
        plt.yticks(list(range(len(y_tick_labels))), y_tick_labels);

    if col_labels is not None:
        plt.xticks(list(range(len(col_labels))), col_labels)
    else:
        plt.xticks([])

    plt.gca().xaxis.set_tick_params(labeltop=xlabel_top, labelbottom=xlabel_bottom)
    plt.gca().yaxis.set_tick_params(labelleft=ylabel_left, labelright=ylabel_right)

    if return_gcf:
        return plt.gcf()


def plot_life_course(df, grid=False, figsize=3.5, **kwargs):
    if isinstance(figsize, (int, float)):  # if figsize is a number, it's a factor of the df size (shape)
        figsize = np.array(df.shape) / figsize
    kwargs['figsize'] = figsize
    heatmap(df.T, **kwargs);
    plt.grid(grid);


def plot_life(df, fields=None, title=None, ax=None):
    if fields is None:
        fields = slice(None, None)
    plot_life_course(df[fields], ax=ax)
    plt.grid(which='both', axis='x')
    if title is not None:
        plt.title(title)
        plt.gca().xaxis.set_tick_params(labeltop=False, labelbottom=True)


def _get_keys(df_store, keys) -> Iterable:
    if keys is None:
        keys = df_store.keys()
    elif callable(keys):
        keys_filt = keys
        keys = filter(keys_filt, df_store.keys())
    assert isinstance(keys, Iterable), "keys should be iterable at this point"
    return keys


def life_plots(df_store, fields=None, keys=None, k_df_to_title=lambda k, df: k.split('/')[1]):
    keys = _get_keys(df_store, keys)
    for k in keys:
        df = df_store[k]
        plot_life(df, fields, title=k_df_to_title(k, df))
        yield plt.gca()


def write_trajectories_to_file(df_store, fields=None, keys=None, fp='test.pdf', pil_write_format=None,
                               to_pil_image_kwargs=None, **pil_save_kwargs):
    keys = _get_keys(df_store, keys)

    def figs():
        fig_gen = map(lambda k, df: plot_life(df, fields, title=k.split('/')[1]),
                      *zip(*((k, df_store[k]) for k in keys)))
        for _ in fig_gen:
            fig = plt.gcf()
            yield fig
            fig.clear()

    write_images(figs(), fp, pil_write_format, to_pil_image_kwargs, **pil_save_kwargs)
