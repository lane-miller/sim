"""Pytest configuration for boxed_piston tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config: object) -> None:
    getattr(config, "addinivalue_line")(
        "markers",
        "slow: long-running diagnostic tests (deselect with '-m \"not slow\"')",
    )
