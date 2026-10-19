"""Tests for the web explorer: odus.explore (the functions) and odus.webui (the app)."""

import json
from io import BytesIO

import pandas as pd
import pytest

pytest.importorskip("plotly")

from odus.explore import respondents, trajectory, variables  # noqa: E402
from odus.webui import make_app  # noqa: E402


@pytest.fixture
def store():
    def traj(heroin):
        return pd.DataFrame(
            {"HEROIN": heroin, "WORK": [1, 1, 0]},
            index=pd.Index([20, 21, 22], name="age"),
        )

    only_work = traj([0, 0, 0])[["WORK"]]
    return {"B.xlsx": traj([0, 1, 1]), "A.xlsx": traj([1, 1, 1]), "C.xlsx": only_work}


def get(app, path, query=""):
    status = {}

    def start_response(s, headers):
        status["code"], status["headers"] = int(s.split()[0]), dict(headers)

    environ = {"PATH_INFO": path, "QUERY_STRING": query, "wsgi.input": BytesIO()}
    body = b"".join(app(environ, start_response))
    return status["code"], status["headers"]["Content-Type"], body


def test_explore_functions(store):
    assert respondents(store=store) == {"respondents": ["A.xlsx", "B.xlsx", "C.xlsx"]}
    assert variables(store=store)["variables"] == [
        {"name": "WORK", "n_respondents": 3},
        {"name": "HEROIN", "n_respondents": 2},
    ]
    result = trajectory("B.xlsx", "HEROIN,NOPE", store=store)
    assert result["fields"] == ["HEROIN"] and result["missing_fields"] == ["NOPE"]
    (heatmap,) = result["figure"]["data"]
    assert heatmap["type"] == "heatmap" and len(heatmap["text"]) == 1
    json.dumps(result)  # JSON-ready
    with pytest.raises(KeyError, match="No respondent"):
        trajectory("Z.xlsx", store=store)
    with pytest.raises(ValueError, match="none of the variables"):
        trajectory("C.xlsx", ["HEROIN"], store=store)


def test_api(store):
    app = make_app(store=store)
    code, ctype, body = get(app, "/api/trajectory", "respondent=A.xlsx&fields=WORK")
    assert (code, ctype) == (200, "application/json")
    assert json.loads(body)["fields"] == ["WORK"]
    assert get(app, "/api/trajectory", "respondent=Z.xlsx")[0] == 404
    assert get(app, "/api/trajectory", "respondent=C.xlsx&fields=HEROIN")[0] == 400
    assert get(app, "/api/trajectory", "")[0] == 400  # no respondent
    assert get(app, "/api/nope")[0] == 404
    code, _, body = get(app, "/api/respondents", "store=x")  # store can't be set
    assert code == 200 and json.loads(body)["respondents"][0] == "A.xlsx"


def test_page_and_assets(store):
    app = make_app(store=store)
    code, ctype, body = get(app, "/")
    assert code == 200 and ctype == "text/html" and b"odus explorer" in body
    for path in ("/app.js", "/style.css", "/plotly.min.js"):
        assert get(app, path)[0] == 200, path
    for path in ("/../explore.py", "/webui.py", "/x/../../setup.py"):
        assert get(app, path)[0] == 404, path


def test_bundled_surveys_by_default():
    app = make_app()
    code, _, body = get(app, "/api/trajectory", "respondent=A01.xlsx&fields=HEROIN")
    assert code == 200 and json.loads(body)["respondent"] == "A01.xlsx"
