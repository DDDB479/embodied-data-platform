"""Test package for embodied-core.

Run with either runner from the repository root:

    python -m unittest discover -s tests -t .
    python -m pytest tests

The source tree is put on ``sys.path`` below so that no ``pip install -e .``
step is needed before running the suite. Optional dependencies (pyarrow, h5py,
mcap) are only required by the tests that exercise them; those tests skip when
the import fails.
"""
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
