"""Pytest configuration: import path, and keeping TensorFlow away from SciPy.

Puts ``src/`` on the path so the suite runs from a bare checkout, without
requiring an editable install.  An installed copy takes precedence.

TensorFlow and SciPy's optimisers do not coexist reliably in one interpreter
on Windows.  In bulk it segfaults outright, which is why
``scripts/compare_organic.py`` fits in a subprocess; at test scale it usually
survives but intermittently dumps core *during interpreter shutdown*, after
every test has passed.  The result is a non-zero exit status from a green run:
a flaky CI failure with nothing to point at.

Test modules touching TensorFlow are therefore marked automatically, so the
suite can run as two processes -- ``pytest -m "not tensorflow"`` and
``pytest -m tensorflow``.  ``make test`` and CI both do this.  Plain ``pytest``
still passes; only its exit status is unreliable.

Marking is by content rather than a list of filenames, so a new TensorFlow
test is covered when it is written rather than when someone remembers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Cache of "does this file mention TensorFlow", keyed by path.  Collection
#: asks once per test, and there are hundreds of tests across a handful of
#: files.
_USES_TENSORFLOW: dict[Path, bool] = {}


def _uses_tensorflow(path: Path) -> bool:
    """Whether a test module references TensorFlow anywhere in its source."""
    cached = _USES_TENSORFLOW.get(path)
    if cached is None:
        try:
            cached = "tensorflow" in path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file
            cached = False
        _USES_TENSORFLOW[path] = cached
    return cached


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test in a TensorFlow-using module with ``tensorflow``."""
    for item in items:
        path = getattr(item, "path", None)
        if path is not None and _uses_tensorflow(Path(path)):
            item.add_marker(pytest.mark.tensorflow)
