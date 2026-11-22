"""Pytest configuration for boxed_piston tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import pytest


@pytest.fixture(autouse=True)
def _stub_matplotlib_show(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


def pytest_configure(config: object) -> None:
    getattr(config, "addinivalue_line")(
        "markers",
        "slow: long-running diagnostic tests (deselect with '-m \"not slow\"')",
    )