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
    "boxrad.post.zrad",
    "boxrad.recorder",
    "boxrad.run",
    "boxrad.source",
)


@pytest.mark.parametrize("module", _MODULES)
def test_boxrad_module_imports_clean(module: str) -> None:
    cmd = [sys.executable, "-c", f"import {module}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout