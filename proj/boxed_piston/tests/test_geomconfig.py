"""Tests for boxrad.geomconfig and geometry_spec wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config
from boxrad.frames import Pose
from boxrad.geomconfig import PrimitiveSpec, parse_geometry
from boxrad.runner import run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHOWCASE = CONFIGS / "showcase"
_LEGACY_TO_JSON_KEYS = frozenset(
    {"geometry", "air", "box", "radiator", "band", "numerics", "output"}
)


def _showcase_json_paths() -> list[Path]:
    return sorted(SHOWCASE.glob("*.json"))


def _assert_geometry_spec_parity(a, b) -> None:
    assert a.tag == b.tag
    assert len(a.parts) == len(b.parts)
    assert len(a.radiators) == len(b.radiators)
    assert len(a.environment.planes) == len(b.environment.planes)
    for r_a, r_b in zip(a.radiators, b.radiators, strict=True):
        assert r_a.name == r_b.name
        if r_a.pose is not None and r_b.pose is not None:
            assert np.allclose(r_a.pose.position, r_b.pose.position, atol=1e-12)
            assert np.allclose(r_a.pose.normal, r_b.pose.normal, atol=1e-12)
            assert np.allclose(r_a.pose.x_axis, r_b.pose.x_axis, atol=1e-12)


def _legacy_config_paths() -> list[Path]:
    paths: list[Path] = []
    for path in sorted(CONFIGS.rglob("*.json")):
        if "showcase" in path.parts:
            continue
        paths.append(path)
    return paths


@pytest.mark.parametrize("path", _legacy_config_paths(), ids=lambda p: p.name)
def test_legacy_json_geometry_spec_and_dry_run(path: Path, tmp_path: Path) -> None:
    cfg = Config.from_json(path.read_text(encoding="utf-8"))
    spec = cfg.geometry_spec
    assert spec is not None
    expected_tag = "baffle" if cfg.geometry == "baffle" else "box"
    assert spec.tag == expected_tag
    assert len(spec.radiators) == 1
    assert spec.radiators[0].pose == Pose.identity()
    assert cfg.geometry in ("box", "baffle")
    run(cfg, out_path=tmp_path / "dry.h5", dry_run=True, progress=False)


def test_s1_cylinder_parse_and_guard() -> None:
    data = json.loads((SHOWCASE / "s1_cylinder.json").read_text(encoding="utf-8"))
    spec = parse_geometry(data)
    assert spec.tag == "parametric"
    assert len(spec.parts) == 1
    cyl = spec.parts[0]
    assert isinstance(cyl, PrimitiveSpec)
    assert cyl.kind == "cylinder"
    assert cyl.params["d"] == pytest.approx(0.2)
    assert cyl.params["length"] == pytest.approx(0.3)
    assert np.allclose(cyl.pose.position, [0.0, 0.0, -0.15])
    assert len(spec.radiators) == 1
    assert spec.radiators[0].name == "driver"
    cfg = Config.from_json(json.dumps(data))
    summary = run(cfg, dry_run=True, progress=False)
    assert isinstance(summary, dict)
    assert summary.get("geometry") == "parametric"
    assert "grid_shape" in summary


def test_s2_bookshelf_parse_and_guard() -> None:
    data = json.loads((SHOWCASE / "s2_bookshelf.json").read_text(encoding="utf-8"))
    spec = parse_geometry(data)
    assert spec.tag == "parametric"
    assert len(spec.parts) == 1
    box = spec.parts[0]
    assert isinstance(box, PrimitiveSpec)
    assert box.kind == "box"
    assert np.allclose(
        [box.params["sx"], box.params["sy"], box.params["sz"]],
        [0.20, 0.25, 0.30],
    )
    assert np.allclose(box.pose.position, [0.0, 0.0, 0.15])
    assert len(spec.radiators) == 2
    woofer, tweeter = spec.radiators
    assert woofer.name == "woofer"
    assert tweeter.name == "tweeter"
    assert np.allclose(woofer.pose.position, [0.0, 0.125, 0.10])
    assert np.allclose(tweeter.pose.position, [0.0, 0.125, 0.25])
    assert np.allclose(woofer.pose.normal, [0.0, 1.0, 0.0])
    assert np.allclose(tweeter.pose.normal, [0.0, 1.0, 0.0])
    assert len(spec.environment.planes) == 2
    floor, wall = spec.environment.planes
    assert floor.axis == "z" and floor.side == "-" and floor.offset == pytest.approx(0.0)
    assert wall.axis == "y" and wall.side == "-" and wall.offset == pytest.approx(-0.425)
    with pytest.raises(NotImplementedError, match="step 7"):
        run(Config.from_json(json.dumps(data)), dry_run=True, progress=False)


@pytest.mark.parametrize(
    "cfg_dict,match",
    [
        (
            {
                "geometry": "baffle",
                "radiator": {"shape": "circle", "dims": [0.1]},
                "radiators": [
                    {
                        "name": "extra",
                        "circle": {"d": 0.05},
                        "position": [0, 0, 0],
                        "normal": [0, 0, 1],
                    },
                ],
            },
            "infinite baffle",
        ),
        (
            {
                "geometry": "baffle",
                "radiator": {"shape": "circle", "dims": [0.1]},
                "environment": {"planes": [{"axis": "z", "side": "-", "offset": 0}]},
            },
            "infinite baffle",
        ),
        (
            {"geometry": "box", "environment": {"planes": [
                {"axis": "z", "side": "-", "offset": 0},
                {"axis": "z", "side": "+", "offset": 1},
            ]}, "radiator": {"shape": "circle", "dims": [0.1]}},
            "duplicate plane axis",
        ),
        (
            {"geometry": "box", "environment": {"planes": [
                {"axis": "w", "side": "-", "offset": 0},
            ]}, "radiator": {"shape": "circle", "dims": [0.1]}},
            "axis must be",
        ),
        (
            {"geometry": "parametric", "parts": [
                {"cylinder": {"d": 0.2, "length": 0.2}, "position": [0, 0, 0]},
            ], "radiators": [{
                "name": "r", "racetrack": {"w": 0.1, "h": 0.2},
                "position": [0, 0, 0], "normal": [0, 0, 1],
            }], "radiator": {"shape": "circle", "dims": [0.1]}},
            "w >= h",
        ),
        (
            {"geometry": "parametric", "parts": [{
                "cylinder": {"d": -0.1, "length": 0.2},
                "position": [0, 0, 0],
            }], "radiators": [{
                "name": "r", "circle": {"d": 0.1},
                "position": [0, 0, 0], "normal": [0, 0, 1],
            }], "radiator": {"shape": "circle", "dims": [0.1]}},
            "must be finite",
        ),
        (
            {"geometry": "parametric", "parts": [{"cylinder": {"d": 0.2, "length": 0.2}, "position": [0, 0, 0]}],
             "radiators": [{
                 "name": "r", "circle": {"d": 0.1}, "rect": {"w": 0.1, "h": 0.1},
                 "position": [0, 0, 0], "normal": [0, 0, 1],
             }], "radiator": {"shape": "circle", "dims": [0.1]}},
            "exactly one shape",
        ),
        (
            {"geometry": "parametric", "parts": [{"cylinder": {"d": 0.2, "length": 0.2}, "position": [0, 0, 0]}],
             "radiators": [{
                 "name": "r", "circle": {"d": 0.1},
                 "hole": {"circle": {"d": 0.05}, "hole": {"circle": {"d": 0.02}}},
                 "position": [0, 0, 0], "normal": [0, 0, 1],
             }], "radiator": {"shape": "circle", "dims": [0.1]}},
            "nested holes",
        ),
        (
            {"geometry": "parametric", "parts": [{"cylinder": {"d": 0.2, "length": 0.2}, "position": [0, 0, 0]}],
             "radiators": [
                 {"name": "a", "circle": {"d": 0.1}, "position": [0, 0, 0], "normal": [0, 0, 1]},
                 {"name": "a", "circle": {"d": 0.1}, "position": [0, 0, 0], "normal": [0, 0, 1]},
             ], "radiator": {"shape": "circle", "dims": [0.1]}},
            "duplicate radiator",
        ),
        (
            {"geometry": "parametric", "parts": [{"step": {"file": "x.step"}, "position": [0, 0, 0]}],
             "radiators": [{"name": "r", "circle": {"d": 0.1}, "position": [0, 0, 0], "normal": [0, 0, 1]}],
             "radiator": {"shape": "circle", "dims": [0.1]}},
            "mixing STEP",
        ),
        (
            {"geometry": "step", "parts": [
                {"step": {"file": "a.step"}},
                {"step": {"file": "b.step"}},
            ], "radiators": [{"name": "r", "part": "face"}],
             "radiator": {"shape": "circle", "dims": [0.1]}},
            "exactly one step",
        ),
        (
            {"geometry": "mesh", "radiator": {"shape": "circle", "dims": [0.1]}},
            "geometry tag",
        ),
        (
            {"geometry": {"tag": "box"}, "radiator": {"shape": "circle", "dims": [0.1]}},
            "geometry tag",
        ),
        (
            {"geometry": "parametric", "parts": [{"cylinder": {"d": 0.2, "length": 0.2}, "position": [0, 0, 0]}],
             "radiators": [{
                 "name": "r", "circle": {"d": 0.1}, "motion": "uniform_normal",
                 "position": [0, 0, 0], "normal": [0, 0, 1],
             }], "radiator": {"shape": "circle", "dims": [0.1]}},
            "motion",
        ),
    ],
)
def test_parse_geometry_errors(cfg_dict: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_geometry(cfg_dict)


@pytest.mark.parametrize("path", _showcase_json_paths(), ids=lambda p: p.name)
def test_showcase_to_json_roundtrip(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    c = Config.from_json(text)
    j0 = c.to_json()
    c2 = Config.from_json(j0)
    if c.geometry_spec != c2.geometry_spec:
        _assert_geometry_spec_parity(c.geometry_spec, c2.geometry_spec)
    assert c2.to_json() == j0


@pytest.mark.parametrize(
    "path",
    [CONFIGS / "c09_box.json", CONFIGS / "c06_baffle.json"],
    ids=lambda p: p.name,
)
def test_legacy_box_baffle_to_json_keys_unchanged(path: Path) -> None:
    cfg = Config.from_json(path.read_text(encoding="utf-8"))
    keys = frozenset(json.loads(cfg.to_json()))
    assert "parts" not in keys
    assert keys <= _LEGACY_TO_JSON_KEYS


def test_box_environment_dry_run_not_implemented(tmp_path: Path) -> None:
    text = (CONFIGS / "smoke.json").read_text(encoding="utf-8")
    data = json.loads(text)
    data["environment"] = {"planes": [{"axis": "z", "side": "-", "offset": 0.0}]}
    cfg = Config.from_dict(data)
    with pytest.raises(NotImplementedError, match="step 7"):
        run(cfg, out_path=tmp_path / "dry.h5", dry_run=True, progress=False)