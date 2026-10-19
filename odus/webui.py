"""A web UI to explore the survey trajectories: pick a respondent, and some variables.

``make_app()`` is a WSGI application serving a page (``odus/webui_static``) and a
JSON API that is exactly ``odus.explore.EXPLORE_FUNCS``: ``/api/<function name>``
with the function's arguments as query parameters. ``serve()`` runs it with the
standard library's server; any WSGI server can run ``make_app()`` too.

From a shell (needs plotly: ``pip install "odus[interactive]"``)::

    python -m odus.webui                      # the surveys bundled with odus
    python -m odus.webui --data FOLDER --ext .csv --port 8050 --open
"""

import argparse
from functools import lru_cache
import inspect
import json
import mimetypes
import os
from urllib.parse import parse_qs

from odus.explore import EXPLORE_FUNCS

STATIC_DIR = os.path.join(os.path.dirname(__file__), "webui_static")
STATIC_FILES = ("index.html", "app.js", "style.css")  # what the page is made of
DFLT_HOST = "127.0.0.1"  # this machine only: the data may be confidential
DFLT_PORT = 8050


@lru_cache(maxsize=1)
def _plotly_js():
    from plotly.offline import get_plotlyjs  # the version matching plotly's JSON

    return get_plotlyjs().encode()


def _http_status(error):
    if isinstance(error, KeyError):
        return "404 Not Found"
    if isinstance(error, ImportError):
        return "501 Not Implemented"
    return "400 Bad Request"


def make_app(*, store=None, funcs=EXPLORE_FUNCS):
    """Make the WSGI app of the explorer.

    Args:
        store: The trajectories to explore (a mapping of respondent keys to
            trajectories, like a ``DfStore``). Default: the bundled surveys.
        funcs: The functions of the JSON API, served at ``/api/<name>``.
    """
    api = {f.__name__: f for f in funcs}

    def respond(start_response, status, body, content_type):
        start_response(
            status,
            [("Content-Type", content_type), ("Content-Length", str(len(body)))],
        )
        return [body]

    def call_api(name, query):
        func = api[name]
        params = inspect.signature(func).parameters
        kwargs = {
            k: v[-1]
            for k, v in parse_qs(query, keep_blank_values=True).items()
            if k in params and k != "store"
        }
        return func(**kwargs, store=store)

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        if path.startswith("/api/"):
            name = path[len("/api/") :]
            if name not in api:
                body = json.dumps({"error": f"No API function {name!r}"}).encode()
                return respond(
                    start_response, "404 Not Found", body, "application/json"
                )
            try:
                result = call_api(name, environ.get("QUERY_STRING", ""))
                status = "200 OK"
            except (KeyError, ValueError, TypeError, ImportError) as e:
                result, status = {"error": str(e).strip("'\"")}, _http_status(e)
            return respond(
                start_response, status, json.dumps(result).encode(), "application/json"
            )
        if path == "/plotly.min.js":
            try:
                return respond(
                    start_response, "200 OK", _plotly_js(), "text/javascript"
                )
            except ImportError:
                body = b'console.error("odus explorer: pip install odus[interactive]")'
                return respond(
                    start_response, "501 Not Implemented", body, "text/javascript"
                )
        filename = "index.html" if path == "/" else path.lstrip("/")
        if filename not in STATIC_FILES:
            return respond(start_response, "404 Not Found", b"Not found", "text/plain")
        with open(os.path.join(STATIC_DIR, filename), "rb") as f:
            body = f.read()
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return respond(start_response, "200 OK", body, content_type)

    return app


def serve(*, store=None, host=DFLT_HOST, port=DFLT_PORT, open_browser=False):
    """Serve the explorer at ``http://host:port`` until interrupted."""
    from wsgiref.simple_server import make_server

    with make_server(host, port, make_app(store=store)) as server:
        url = f"http://{host}:{server.server_port}"
        print(f"odus explorer at {url}  (Ctrl+C to stop)")
        if open_browser:
            import webbrowser

            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="folder of survey files (default: bundled)")
    parser.add_argument("--ext", default=".xlsx", help="extension of those files")
    parser.add_argument("--host", default=DFLT_HOST)
    parser.add_argument("--port", type=int, default=DFLT_PORT)
    parser.add_argument("--open", action="store_true", help="open a browser")
    args = parser.parse_args(argv)
    store = None
    if args.data:
        from odus.dacc import DfStore

        folder = os.path.abspath(os.path.expanduser(args.data))
        store = DfStore(os.path.join(folder, "") + "{}" + args.ext)
    serve(store=store, host=args.host, port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
