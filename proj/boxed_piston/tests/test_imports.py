"""Each boxrad submodule must import in isolation (no circular imports)."""

from __future__ import annotations

import subprocess
import sys

import pytest

_MODULES = (
    "boxrad.config",
    "boxrad.cpml",
    "boxrad.grid",
    "boxrad.kernels",
    "boxrad.post",
    "boxrad.post.kirchhoff",
    "boxrad.post.transfer",
    "boxrad.post.zrad",
    "boxrad.recorder",
    "boxrad.result",
    "boxrad.runner",
    "boxrad.source",
    "boxrad.viz",
    "boxrad.info",
)


@pytest.mark.parametrize("module", _MODULES)
def test_boxrad_module_imports_clean(module: str) -> None:
    cmd = [sys.executable, "-c", f"import {module}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_run_export_and_runner_module() -> None:
    import importlib

    import boxrad

    runner = importlib.import_module("boxrad.runner")
    assert hasattr(runner, "__file__")
    assert callable(boxrad.run)