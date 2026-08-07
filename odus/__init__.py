"""ODUS (Older Drug User Study): data and tools to study drug-use trajectories.

Exposes the package data locations (``root_dir``, ``data_dir``, ``data_path_of``)
used by the rest of the package to reach the bundled survey files.
"""

import os

root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, "data")
data_path_of = lambda path: os.path.join(data_dir, path)
