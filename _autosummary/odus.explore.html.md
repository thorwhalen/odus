# odus.explore

Explore the survey trajectories: plain functions returning JSON-ready dicts.

This is the single source of truth of the web explorer (`odus.webui`), and
knows nothing about HTTP: `odus.webui` exposes exactly the functions of
`EXPLORE_FUNCS`, so any other surface (a CLI, an MCP server…) can reuse them.

Every function takes a `store` (a mapping of respondent keys to trajectories,
like a `DfStore`), defaulting to the surveys bundled with odus.

```pycon
>>> 'A01.xlsx' in respondents()['respondents']
True
```

### Functions

| [`bundled_store`](#odus.explore.bundled_store)()                         | A `DfStore` of the surveys bundled with odus.                                  |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`respondents`](#odus.explore.respondents)(\*[, store])                | The keys of the respondents, sorted.                                           |
| [`trajectory`](#odus.explore.trajectory)(respondent[, fields, store]) | The interactive (plotly) figure of a respondent's trajectory, as JSON data.    |
| [`variables`](#odus.explore.variables)(\*[, store])                  | The variables, with the number of respondents having each (most common first). |

### Exceptions

| [`NotFound`](#odus.explore.NotFound)   | What was asked for doesn't exist (a surface may map it to "404").   |
|-------------------------------------------------------------|---------------------------------------------------------------------|

### *exception* odus.explore.NotFound

Bases: [`KeyError`](https://docs.python.org/3/builtins/exceptions.html#KeyError)

What was asked for doesn’t exist (a surface may map it to “404”).

### odus.explore.bundled_store()

A `DfStore` of the surveys bundled with odus.

### odus.explore.respondents(, store=None)

The keys of the respondents, sorted.

### odus.explore.trajectory(respondent, fields=None, , store=None)

The interactive (plotly) figure of a respondent’s trajectory, as JSON data.

* **Parameters:**
  * **respondent** – The key of the respondent.
  * **fields** – The variables to show: a list of names, or one name. Default:
    all. Those the respondent doesn’t have are listed in
    `missing_fields` rather than plotted.
  * **store** – The trajectories (default: the bundled surveys).
* **Returns:**
  `{"respondent", "fields", "missing_fields", "figure"}`, where `figure`
  is a plotly figure as a dict (`Plotly.react(div, figure.data,
  figure.layout)` draws it).

### odus.explore.variables(, store=None)

The variables, with the number of respondents having each (most common first).

Respondents whose data can’t be read are listed in `unreadable`, with why.
