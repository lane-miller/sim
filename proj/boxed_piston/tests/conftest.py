"""Pytest configuration for boxed_piston tests."""

from __future__ import annotations


def pytest_configure(config: object) -> None:
    getattr(config, "addinivalue_line")(
        "markers",
        "slow: long-running diagnostic tests (deselect with '-m \"not slow\"')",
    )
