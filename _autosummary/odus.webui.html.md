# odus.webui

A web UI to explore the survey trajectories: pick a respondent, and some variables.

`make_app()` is a WSGI application serving a page (`odus/webui_static`) and a
JSON API that is exactly `odus.explore.EXPLORE_FUNCS`: `/api/<function name>`
with the function’s arguments as query parameters. `serve()` runs it with the
standard library’s server; any WSGI server can run `make_app()` too.

From a shell (needs plotly: `pip install "odus[interactive]"`):

```default
python -m odus.webui                      # the surveys bundled with odus
python -m odus.webui --data FOLDER --ext .csv --port 8050 --open
```

### Functions

| `main`([argv])                                                                                |                                                             |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [`make_app`](#odus.webui.make_app)(\*[, store, funcs])                 | Make the WSGI app of the explorer.                          |
| [`serve`](#odus.webui.serve)(\*[, store, host, port, open_browser]) | Serve the explorer at `http://host:port` until interrupted. |

### odus.webui.make_app(\*, store=None, funcs=(<function respondents>, <function variables>, <function trajectory>))

Make the WSGI app of the explorer.

* **Parameters:**
  * **store** – The trajectories to explore (a mapping of respondent keys to
    trajectories, like a `DfStore`). Default: the bundled surveys.
  * **funcs** – The functions of the JSON API, served at `/api/<name>`.

### odus.webui.serve(, store=None, host='127.0.0.1', port=8050, open_browser=False)

Serve the explorer at `http://host:port` until interrupted.
