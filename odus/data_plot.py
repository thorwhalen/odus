import matplotlib.pylab as plt
from itertools import islice
from typing import Iterable

from odus.dacc import plot_life_course
from odus.util import write_images

ihead = lambda it: islice(it, 0, 5)  # just a little useful util. Not used within module


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
