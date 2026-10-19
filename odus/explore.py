"""Explore the survey trajectories: plain functions returning JSON-ready dicts.

This is the single source of truth of the web explorer (``odus.webui``), and
knows nothing about HTTP: ``odus.webui`` exposes exactly the functions of
``EXPLORE_FUNCS``, so any other surface (a CLI, an MCP server...) can reuse them.

Every function takes a ``store`` (a mapping of respondent keys to trajectories,
like a ``DfStore``), defaulting to the surveys bundled with odus.

>>> 'A01.xlsx' in respondents()['respondents']
True
"""

from collections import Counter
from functools import lru_cache
import json
import os
import warnings


@lru_cache(maxsize=1)
def bundled_store():
    """A ``DfStore`` of the surveys bundled with odus."""
    from odus import data_path_of
    from odus.dacc import DfStore

    return DfStore(data_path_of("surveys") + os.sep)


def _store(store):
    return bundled_store() if store is None else store


def _trajectory(store, key):
    with warnings.catch_warnings():  # data anomalies are the diagnosis' business
        warnings.simplefilter("ignore")
        return store[key]


class NotFound(KeyError):
    """What was asked for doesn't exist (a surface may map it to "404")."""

    def __str__(self):
        return str(self.args[0]) if self.args else ""


def _as_list(fields):
    """``fields`` as a list: a single name (names may contain commas) or names."""
    if fields is None or fields == "" or fields == []:
        return None
    if isinstance(fields, str):
        return [fields]
    return [f for f in fields if f != ""]


def respondents(*, store=None):
    """The keys of the respondents, sorted."""
    return {"respondents": sorted(_store(store))}


def variables(*, store=None):
    """The variables, with the number of respondents having each (most common first).

    Respondents whose data can't be read are listed in ``unreadable``, with why.
    """
    store = _store(store)
    counts, unreadable = Counter(), {}
    for k in store:
        try:
            counts.update(set(_trajectory(store, k).columns))
        except Exception as e:  # one bad file shouldn't hide all the others
            unreadable[str(k)] = f"{type(e).__name__}: {e}"
    return {
        "variables": [
            {"name": str(name), "n_respondents": n}
            for name, n in sorted(counts.items(), key=lambda x: (-x[1], str(x[0])))
        ],
        "unreadable": unreadable,
    }


def trajectory(respondent, fields=None, *, store=None):
    """The interactive (plotly) figure of a respondent's trajectory, as JSON data.

    Args:
        respondent: The key of the respondent.
        fields: The variables to show: a list of names, or one name. Default:
            all. Those the respondent doesn't have are listed in
            ``missing_fields`` rather than plotted.
        store: The trajectories (default: the bundled surveys).

    Returns:
        ``{"respondent", "fields", "missing_fields", "figure"}``, where ``figure``
        is a plotly figure as a dict (``Plotly.react(div, figure.data,
        figure.layout)`` draws it).
    """
    from odus.interactive import plot_trajectory_interactive

    store = _store(store)
    if respondent not in set(store):  # only listed keys: never a path to elsewhere
        raise NotFound(f"No respondent {respondent!r}")
    df = _trajectory(store, respondent)
    wanted = _as_list(fields)
    present = None if wanted is None else [f for f in wanted if f in df.columns]
    missing = [] if wanted is None else [f for f in wanted if f not in df.columns]
    if present == []:
        raise ValueError(f"{respondent!r} has none of the variables {wanted}")
    fig = plot_trajectory_interactive(df, present, title=respondent)
    return {
        "respondent": respondent,
        "fields": [str(c) for c in (df.columns if present is None else present)],
        "missing_fields": missing,
        "figure": json.loads(fig.to_json()),
    }


EXPLORE_FUNCS = (respondents, variables, trajectory)  # what the surfaces expose
