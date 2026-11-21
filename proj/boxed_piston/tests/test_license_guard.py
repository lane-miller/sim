"""Guardrails: optional extras must not pull forbidden or unused dependencies."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DEFAULT_CONFIG = ROOT / "configs" / "default.json"

FORBIDDEN = {
    "gmsh",
    "pygmsh",
    "pymeshlab",
    "triangle",
    "meshpy",
    "pygalmesh",
    "tetgen",
    "cadquery",
}
NOT_USED = {"shapely", "trimesh", "vtk"}
BLOCKED = FORBIDDEN | NOT_USED

_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dependency_name(requirement: str) -> str:
    req = requirement.strip().split(";")[0].strip()
    match = _REQ_NAME_RE.match(req)
    if not match:
        raise ValueError(f"could not parse dependency: {requirement!r}")
    return match.group(1).split("[", 1)[0]


def _iter_pyproject_requirements() -> list[tuple[str, str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    out: list[tuple[str, str]] = []
    for req in project.get("dependencies", []):
        out.append(("base", req))
    for extra, reqs in (project.get("optional-dependencies") or {}).items():
        for req in reqs:
            out.append((extra, req))
    return out


def _assert_modules_absent(*, forbid_ocp: bool) -> None:
    for mod in list(sys.modules):
        root = _normalize_name(mod.split(".", 1)[0])
        if root in BLOCKED:
            pytest.fail(f"unexpected module loaded: {mod!r}")
    if forbid_ocp and "OCP" in sys.modules:
        pytest.fail("OCP must not be loaded on this code path")


def test_pyproject_extras_clean() -> None:
    for extra, req in _iter_pyproject_requirements():
        name = _normalize_name(_dependency_name(req))
        assert name not in BLOCKED, f"extra {extra!r} lists blocked dependency {req!r}"


def test_box_path_imports_clean(tmp_path: Path) -> None:
    import boxrad  # noqa: F401
    from boxrad.runner import run

    out = tmp_path / "dry.h5"
    summary = run(DEFAULT_CONFIG, out_path=out, dry_run=True, progress=False)
    assert isinstance(summary, dict)
    _assert_modules_absent(forbid_ocp=True)


def test_geom_imports_clean() -> None:
    pytest.importorskip("manifold3d")
    import manifold3d  # noqa: F401

    _assert_modules_absent(forbid_ocp=True)


def test_cad_imports_clean() -> None:
    pytest.importorskip("OCP")
    import OCP  # noqa: F401

    _assert_modules_absent(forbid_ocp=False)