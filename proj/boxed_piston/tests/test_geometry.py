"""Tests for boxrad.geom.geometry."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config, derive
from boxrad.frames import Pose, orthonormal_basis
from boxrad.geom import geometry, mesh, poly2d
from boxrad.grid import radiator_area_exact

PKG = Path(__file__).resolve().parents[1]
CONFIGS = PKG / "configs"
SHOWCASE = CONFIGS / "showcase"
RTOL = 1e-12


def _box_config_paths():
    out = []
    for p in sorted(CONFIGS.rglob("*.json")):
        if "showcase" in p.parts:
            continue
        data = json.loads(p.read_text())
        if data.get("geometry", "box") != "box":
            continue
        out.append(p)
    return out


@pytest.mark.parametrize("path", _box_config_paths(), ids=lambda p: p.name)
def test_from_box_legacy_configs(path: Path) -> None:
    cfg = Config.from_json(path.read_text(encoding="utf-8"))
    derived = derive(cfg)
    geom = geometry.from_box(cfg)
    assert geom.solid.is_closed()
    W, H, D = derived.realized_W, derived.realized_H, derived.realized_D
    assert geom.solid.volume == pytest.approx(W * H * D, rel=RTOL)
    lo = geom.solid.bbox
    assert lo[2] == pytest.approx(-D, rel=RTOL)
    assert lo[5] == pytest.approx(0.0, rel=RTOL)
    xc, yc = cfg.radiator.x_c, cfg.radiator.y_c
    assert lo[0] == pytest.approx(-xc - 0.5 * W, rel=RTOL)
    assert lo[3] == pytest.approx(-xc + 0.5 * W, rel=RTOL)
    rad0 = geom.radiators[0]
    assert rad0.area == pytest.approx(radiator_area_exact(cfg.radiator), rel=1e-4)
    geometry.seating_check(geom, rad0)


def test_box_path_no_manifold_import() -> None:
    code = (
        "import sys; from pathlib import Path; "
        "from boxrad.config import Config; "
        "from boxrad.geom.geometry import from_box; "
        f"p=Path({repr(str(CONFIGS / 'c09_box.json'))}); "
        "cfg=Config.from_json(p.read_text()); "
        "from_box(cfg); "
        "assert 'manifold3d' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(PKG))


_manifold = pytest.importorskip("manifold3d", reason="geom extra")


def test_s1_cylinder() -> None:
    cfg = Config.from_json((SHOWCASE / "s1_cylinder.json").read_text())
    geom = geometry.build_geometry(cfg)
    assert geom is not None
    assert geom.solid.is_closed()
    part = cfg.geometry_spec.parts[0]
    d, L = part.params["d"], part.params["length"]
    r = 0.5 * d
    n = max(16, int(math.ceil(math.pi / math.acos(max(-1.0, min(1.0, 1.0 - 1e-5 / r))))))
    assert geom.solid.volume == pytest.approx(mesh.polygonal_cylinder_volume(n, r, L), rel=RTOL)
    driver = geom.radiators[0]
    shape = cfg.geometry_spec.radiators[0].shape
    assert driver.area == pytest.approx(poly2d.from_shape_spec(shape).area, rel=RTOL)


def test_s2_bookshelf() -> None:
    cfg = Config.from_json((SHOWCASE / "s2_bookshelf.json").read_text())
    geom = geometry.build_geometry(cfg)
    assert geom.solid.is_closed()
    v_exact = 0.20 * 0.25 * 0.30
    assert geom.solid.volume == pytest.approx(v_exact, rel=RTOL)
    for rd in geom.radiators:
        geometry.seating_check(geom, rd)


def test_s5_platform() -> None:
    cfg = Config.from_json((SHOWCASE / "s5_platform.json").read_text())
    geom = geometry.build_geometry(cfg)
    assert geom.solid.is_closed()
    v_exact = 0.20 * 0.25 * 0.30 + 0.40 * 0.35 * 0.03
    assert geom.solid.volume == pytest.approx(v_exact, rel=RTOL)
    for rd in geom.radiators:
        geometry.seating_check(geom, rd)
    woofer = next(rd for rd in geom.radiators if rd.name == "woofer")
    cmin, _ = geometry.radiator_clearance(geom, woofer)
    assert math.isinf(cmin)


def test_s6_downfire() -> None:
    cfg = Config.from_json((SHOWCASE / "s6_downfire.json").read_text())
    geom = geometry.build_geometry(cfg)
    assert geom.solid.is_closed()
    part0 = cfg.geometry_spec.parts[0]
    d, L = part0.params["d"], part0.params["length"]
    r = 0.5 * d
    tol = 1e-5
    n = max(16, int(math.ceil(math.pi / math.acos(max(-1.0, min(1.0, 1.0 - tol / r))))))
    v_cyl = mesh.polygonal_cylinder_volume(n, r, L)
    v_slab = 0.30 * 0.30 * 0.025
    assert geom.solid.volume == pytest.approx(v_cyl + v_slab, rel=RTOL)
    woofer = geom.radiators[0]
    geometry.seating_check(geom, woofer)
    cmin, _ = geometry.radiator_clearance(geom, woofer)
    assert cmin == pytest.approx(0.030, abs=1e-9)


def test_radiator_clearance_environment_plane() -> None:
    data = json.loads((SHOWCASE / "s2_bookshelf.json").read_text())
    data["environment"] = {
        "planes": [{"axis": "y", "side": "+", "offset": 0.325}]
    }
    cfg = Config.from_dict(data)
    geom = geometry.build_geometry(cfg)
    woofer = next(rd for rd in geom.radiators if rd.name == "woofer")
    cmin, _ = geometry.radiator_clearance(geom, woofer)
    assert cmin == pytest.approx(0.2, rel=1e-9)


def test_clearance_warnings_s6() -> None:
    cfg = Config.from_json((SHOWCASE / "s6_downfire.json").read_text())
    geom = geometry.build_geometry(cfg)
    assert geometry.clearance_warnings(geom, h=0.005) == []
    w = geometry.clearance_warnings(geom, h=0.008)
    assert len(w) == 1
    assert "woofer" in w[0]


def test_s4_tilted() -> None:
    cfg = Config.from_json((SHOWCASE / "s4_tilted.json").read_text())
    geom = geometry.build_geometry(cfg)
    W, H, D = 0.2, 0.15, 0.25
    assert geom.solid.volume == pytest.approx(W * H * D, rel=RTOL)
    geometry.seating_check(geom, geom.radiators[0])
    rd = cfg.geometry_spec.radiators[0]
    n_world = rd.pose.R[:, 2]
    assert np.allclose(n_world / np.linalg.norm(n_world), rd.pose.normal / np.linalg.norm(rd.pose.normal), rtol=RTOL)


def test_overlap_union() -> None:
    data = {
        "geometry": "parametric",
        "band": {"f_min": 50.0, "f_max": 3000.0},
        "parts": [
            {"box": {"size": [0.2, 0.2, 0.2]}, "position": [0, 0, 0]},
            {"box": {"size": [0.2, 0.2, 0.2]}, "position": [0.1, 0, 0]},
        ],
        "radiators": [
            {"name": "r", "rect": {"w": 0.05, "h": 0.05}, "position": [0, 0, 0.1], "normal": [0, 0, 1]},
        ],
    }
    cfg = Config.from_dict(data)
    geom = geometry.build_geometry(cfg)
    v1 = 0.2**3
    overlap = 0.1 * 0.2 * 0.2
    assert geom.solid.volume == pytest.approx(2 * v1 - overlap, rel=RTOL)


@pytest.mark.parametrize(
    "mutator,match",
    [
        ("s1_z", "surface"),
        ("s1_overhang", "surface"),
        ("s1_curved", "surface"),
        ("s2_woofer", "surface"),
    ],
)
def test_seating_failures(mutator: str, match: str) -> None:
    if mutator == "s1_z":
        data = json.loads((SHOWCASE / "s1_cylinder.json").read_text())
        data["radiators"][0]["position"] = [0.0, 0.0, 0.01]
    elif mutator == "s1_overhang":
        data = json.loads((SHOWCASE / "s1_cylinder.json").read_text())
        data["radiators"][0]["circle"]["d"] = 0.25
    elif mutator == "s1_curved":
        data = json.loads((SHOWCASE / "s1_cylinder.json").read_text())
        data["radiators"][0]["position"] = [0.1, 0.0, 0.0]
        data["radiators"][0]["normal"] = [1.0, 0.0, 0.0]
    else:
        data = json.loads((SHOWCASE / "s2_bookshelf.json").read_text())
        data["radiators"][0]["position"] = [0.0, 0.10, 0.25]
    cfg = Config.from_dict(data)
    with pytest.raises(ValueError, match=match):
        geometry.build_geometry(cfg)


def test_build_geometry_baffle_and_step() -> None:
    baffle = Config.from_dict(
        {"geometry": "baffle", "radiator": {"shape": "circle", "dims": [0.05]}},
    )
    assert geometry.build_geometry(baffle) is None
    step = Config.from_dict(
        {
            "geometry": "step",
            "parts": [{"step": {"file": "x.step"}}],
            "radiators": [{"name": "r", "part": "face", "position": [0, 0, 0]}],
        }
    )
    with pytest.raises(NotImplementedError):
        geometry.build_geometry(step)