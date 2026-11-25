"""Small-cell floor, thin-wall detector, and reference stepper (v0.3 step 5)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import CFL_CUTCELL, Config, derive
from boxrad.cutcell import Grid, compute_fractions, grid_for_geometry
from boxrad.geom import geometry, mesh
from boxrad.geom.geometry import build_geometry
from boxrad.refstep import closed_alpha, init_velocity, lam_max, step
from boxrad.smallcell import apply_floor, courant_lambda, stability_ok, thin_wall_cells

PKG = Path(__file__).resolve().parents[1]
SHOWCASE = PKG / "configs" / "showcase"
H = 0.005
LAM_SHOW = 0.99 / math.sqrt(3.0)


def _c9_lambda() -> float:
    cfg = Config.from_json((PKG / "configs" / "c09_box.json").read_text(encoding="utf-8"))
    return courant_lambda(cfg, derive(cfg))


def _showcase_geom(name: str):
    cfg = Config.from_json((SHOWCASE / name).read_text(encoding="utf-8"))
    geom = build_geometry(cfg)
    assert geom is not None
    grid = grid_for_geometry(geom, H)
    fr = compute_fractions(geom, grid)
    lam = LAM_SHOW
    return cfg, geom, grid, fr, lam


@pytest.mark.parametrize(
    "jname",
    ["s1_cylinder.json", "s4_tilted.json", "s6_downfire.json"],
)
def test_floor_stability_showcase(jname: str) -> None:
    _, _, _, fr, lam = _showcase_geom(jname)
    beta_tilde, _ = apply_floor(fr, lam, h=H)
    ok, _ = stability_ok(beta_tilde, fr.alpha, lam)
    assert ok


def test_full_cell_no_floor_needed() -> None:
    lam = 1.0 / math.sqrt(3.0)
    beta = np.ones((1, 1, 1), dtype=np.float64)
    alpha = (
        np.ones((2, 1, 1)),
        np.ones((1, 2, 1)),
        np.ones((1, 1, 2)),
    )

    class _Fr:
        pass

    fr = _Fr()
    fr.beta = beta
    fr.alpha = alpha
    fr.patch_A = np.zeros(0, dtype=np.float64)
    beta_tilde, st = apply_floor(fr, lam)  # type: ignore[arg-type]
    assert st["n_floored"] == 0
    ok, _ = stability_ok(beta_tilde, alpha, lam)
    assert ok


def test_aligned_box_c9() -> None:
    cfg = Config.from_json((PKG / "configs" / "c09_box.json").read_text(encoding="utf-8"))
    geom = build_geometry(cfg)
    assert geom is not None
    grid = grid_for_geometry(geom, H)
    fr = compute_fractions(geom, grid)
    lam = courant_lambda(cfg, derive(cfg))
    beta_tilde, st = apply_floor(fr, lam, h=H)
    assert st["n_floored"] == 0
    assert st["delta_V"] == pytest.approx(0.0, abs=0.0)
    assert np.allclose(beta_tilde, fr.beta)


def _rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _energy_fractions():
    R = _rot_z(37.0) @ _rot_x(20.0)
    solid = mesh.box_mesh([6.0, 6.0, 6.0]).transformed(R, np.zeros(3))
    grid = Grid(
        x0=np.array([-7.0, -7.0, -7.0], dtype=np.float64),
        h=1.0,
        shape=(14, 14, 14),
        rigid_sides={},
    )
    geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
    fr = compute_fractions(geom, grid)
    return geom, grid, fr


def test_energy_floored_conserved() -> None:
    _, _, fr = _energy_fractions()
    lam = LAM_SHOW
    beta_tilde, _ = apply_floor(fr, lam, h=1.0)
    alpha_c = closed_alpha(fr.alpha)
    rng = np.random.default_rng(0)
    p = np.zeros_like(fr.beta)
    p[fr.beta > 0.0] = rng.standard_normal(np.sum(fr.beta > 0.0))
    u = init_velocity(alpha_c)
    _, _, hist = step(p, u, beta_tilde, alpha_c, lam, 4000)
    h0 = hist[0]
    drift = float(np.max(np.abs(hist - h0))) / max(abs(h0), 1e-30)
    assert drift <= 1e-12


def test_energy_unfloored_blows_up() -> None:
    _, _, fr = _energy_fractions()
    lam = LAM_SHOW
    beta_tilde = fr.beta.copy()
    alpha_c = closed_alpha(fr.alpha)
    rng = np.random.default_rng(0)
    p = np.zeros_like(fr.beta)
    p[fr.beta > 0.0] = rng.standard_normal(np.sum(fr.beta > 0.0))
    p0 = p.copy()
    pmax0 = float(np.max(np.abs(p0)))
    u = init_velocity(alpha_c)
    blow_step = None
    for n in range(1, 501):
        p, u, _ = step(p, u, beta_tilde, alpha_c, lam, 1)
        if not np.all(np.isfinite(p)):
            blow_step = n
            break
        if float(np.max(np.abs(p))) > 1e6 * pmax0:
            blow_step = n
            break
    assert blow_step is not None


def test_s1_true_stability_lam_max() -> None:
    _, _, _, fr, lam = _showcase_geom("s1_cylinder.json")
    beta_tilde, _ = apply_floor(fr, lam, h=H)
    alpha_c = closed_alpha(fr.alpha)
    lm = lam_max(beta_tilde, alpha_c)
    assert lam**2 * lm <= 4.0


def _thin_wall_setup(thickness_h: float):
    h = 1.0
    grid = Grid(
        x0=np.array([-8.0, -8.0, -8.0], dtype=np.float64),
        h=h,
        shape=(16, 16, 16),
        rigid_sides={},
    )
    if thickness_h <= 1.0:
        solid = mesh.box_mesh([10.0, 10.0, thickness_h * h]).transformed(
            np.eye(3), np.array([0.0, 0.0, 0.5 * h], dtype=np.float64)
        )
    else:
        solid = mesh.box_mesh([10.0, 10.0, thickness_h * h]).transformed(
            np.eye(3), np.array([0.0, 0.0, 1.55], dtype=np.float64)
        )
    geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
    fr = compute_fractions(geom, grid)
    return geom, grid, fr


def test_thin_wall_thin_plate_raises() -> None:
    geom, grid, fr = _thin_wall_setup(0.4)
    _, st = thin_wall_cells(fr, geom, grid, max_thin=10**9)
    assert st["n_confirmed"] == 100
    with pytest.raises(ValueError, match="thin-wall"):
        thin_wall_cells(fr, geom, grid, max_thin=0)


def test_thin_wall_thick_plate_none() -> None:
    geom, grid, fr = _thin_wall_setup(2.5)
    cells, st = thin_wall_cells(fr, geom, grid, max_thin=0)
    assert st["n_confirmed"] == 0
    assert cells == []


def test_delta_eq_s1() -> None:
    _, _, grid, fr, lam = _showcase_geom("s1_cylinder.json")
    _, st = apply_floor(fr, lam, h=H)
    assert st["delta_eq_over_h"] == pytest.approx(0.047, abs=0.003)
    assert st["floor_status"] == "check"
    print(f"S1 delta_eq/h = {st['delta_eq_over_h']:.6g} ({st['floor_status']})")


def test_delta_eq_s1_auto() -> None:
    _, _, grid, fr, _ = _showcase_geom("s1_cylinder.json")
    lam = CFL_CUTCELL / math.sqrt(3.0)
    _, st = apply_floor(fr, lam, h=H)
    assert st["delta_eq_over_h"] < 0.01
    assert st["floor_status"] == "ok"
    print(f"S1 delta_eq/h = {st['delta_eq_over_h']:.6g} ({st['floor_status']})")