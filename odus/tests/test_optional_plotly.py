"""plotly is optional: odus imports without it, and says how to get it."""

import subprocess
import sys

BLOCK_PLOTLY = """
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "plotly" or name.startswith("plotly."):
            raise ImportError("plotly is blocked for this test")
sys.meta_path.insert(0, _Block())
"""


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", BLOCK_PLOTLY + code], capture_output=True, text=True
    )


def test_odus_imports_without_plotly():
    out = _run("import odus, odus.plot_utils, odus.interactive; print('ok')")
    assert out.stdout.strip() == "ok", out.stderr


def test_calling_without_plotly_gives_an_install_hint():
    code = """
import pandas as pd
from odus.interactive import plot_trajectory_interactive
try:
    plot_trajectory_interactive(pd.DataFrame({"A": [0, 1]}))
except ImportError as e:
    print(e)
"""
    out = _run(code)
    assert 'pip install "odus[interactive]"' in out.stdout, out.stderr
