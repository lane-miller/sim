"""Each fdtd submodule must import in isolation (no circular imports)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_MODULES = (
    "fdtd.config",
    "fdtd.cpml",
    "fdtd.grid",
    "fdtd.kernels",
    "fdtd.post",
    "fdtd.post.kirchhoff",
    "fdtd.post.zrad",
    "fdtd.recorder",
    "fdtd.run",
    "fdtd.source",
)


@pytest.mark.parametrize("module", _MODULES)
def test_fdtd_module_imports_clean(module: str) -> None:
    cmd = [
        sys.executable,
        "-I",
        "-c",
        f"import sys; sys.path.insert(0,{str(ROOT)!r}); import {module}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr or proc.stdout