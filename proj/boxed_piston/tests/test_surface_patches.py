"""Step 9: wetted-wall surface patch geometry and analytic moments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config, derive
from boxrad.cutcell import Grid, compute_fractions, wetted_wall_arrays
from boxrad.geom import geometry, mesh
from boxrad.geomrun import dry_run_geom, prepare_geom_run
from boxrad.validation.analytic import sphere_point_source, sphere_surface_moments

PKG = Path(__file__).resolve().parents[1]
CONFIGS = PKG / "configs"
SHOWCASE = CONFIGS / "showcase"
RTOL = 1e-12

pytest_plugins = ["test_result_api"]


def _floor_fr(cfg: Config):
    run = prepare_geom_run(cfg)
    return run.info["fractions"], run.info["beta_tilde"], float(run.derived.h)


def test_oracle_wall_arrays_match_reference():
    from ref_cutcell import box_mesh as ref_box_mesh, cylinder_mesh, fractions as ref_fractions

    def _check(V, F, x0, h, shape):
        ref = ref_fractions(V, F, np.asarray(x0, float), h, shape)
        solid = mesh.TriMesh(np.asarray(V, float), np.asarray(F, int))
        geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
        grid = Grid(x0=np.asarray(x0, float), h=h, shape=tuple(shape), rigid_sides={})
        fr = compute_fractions(geom, grid)
        np.testing.assert_allclose(fr.patch_cell, ref["wall_cell"], rtol=0, atol=0)
        np.testing.assert_allclose(fr.patch_A, ref["wall_area"], rtol=RTOL, atol=RTOL)
        np.testing.assert_allclose(fr.patch_centroid, ref["wall_centroid"], rtol=RTOL, atol=RTOL)
        n_w = fr.patch_normal * fr.patch_A[:, np.newaxis]
        np.testing.assert_allclose(n_w, ref["wall_area_vec"], rtol=RTOL, atol=RTOL)
        for a in range(3):
            d = np.max(np.abs(fr.alpha[a] - ref["alpha"][a]))
            assert d <= RTOL
        assert np.max(np.abs(fr.beta - ref["beta"])) <= RTOL

    V, F = ref_box_mesh([0.37, 0.37, 0.37], [4.37, 4.37, 4.37])
    _check(V, F, [-2.0, -2.0, -2.0], 1.0, (9, 9, 9))
    V, F = cylinder_mesh(1.0, 0.0, 1.0, 32)
    _check(V, F, [-2.0, -2.0, -2.0], 0.5, (12, 12, 8))


def test_sphere_wall_patches_c13():
    cfg = Config.from_json((CONFIGS / "c13_sphere_ps.json").read_text())
    raw = json.loads(cfg.to_json())
    raw["numerics"]["h_override"] = 0.05 / 9.0
    raw["numerics"].pop("cells_across_radiator", None)
    cfg = Config.from_dict(raw)
    fr, beta_tilde, h = _floor_fr(cfg)
    _, areas, cents, avec = wetted_wall_arrays(fr, beta_tilde, h)
    mesh_a = float(np.sum(fr.patch_A))
    assert float(np.sum(areas)) == pytest.approx(mesh_a, rel=1e-9)
    assert np.linalg.norm(np.sum(avec, axis=0)) <= 1e-9 * float(np.sum(areas))
    assert np.all(np.linalg.norm(avec, axis=1) <= areas * (1.0 + 1e-12))
    center = np.array([0.0, 0.0, 0.0])
    a_sph = 0.05
    radii = np.linalg.norm(cents - center, axis=1)
    assert np.max(np.abs(radii - a_sph)) <= h + 1e-9


def test_aligned_box_c19_sym_full():
    cfg = Config.from_json((SHOWCASE / "c19_sym_full.json").read_text())
    fr, beta_tilde, h = _floor_fr(cfg)
    cells, areas, _, avec = wetted_wall_arrays(fr, beta_tilde, h)
    raw = json.loads((SHOWCASE / "c19_sym_full.json").read_text())
    W, H, D = raw["parts"][0]["box"]["size"]
    box_a = 2.0 * (W * H + W * D + H * D)
    assert float(np.sum(areas)) == pytest.approx(box_a, rel=RTOL)
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([0.0, 0.0, 1.0])
    for n_w, a_w in zip(avec, areas):
        ax = np.argmax(np.abs(n_w))
        u = n_w / max(a_w, 1e-30)
        ref = (ex, ey, ez)[ax]
        if np.dot(u, ref) < 0:
            ref = -ref
        if abs(abs(u[ax]) - 1.0) > 1e-6 and np.linalg.norm(n_w) < a_w * (1.0 - 1e-9):
            continue
        assert abs(abs(u[ax]) - 1.0) <= 1e-9
    bt = beta_tilde.reshape(-1)
    for nid in cells:
        assert bt[int(nid)] == pytest.approx(1.0, abs=1e-9)


def test_sphere_surface_moments_vs_quadrature():
    rho, c = 1.204, 343.0
    a, r0 = 0.05, 0.1
    thetas = np.linspace(0.0, np.pi, 4001)
    pts = np.column_stack(
        [
            a * np.sin(thetas),
            np.zeros_like(thetas),
            a * np.cos(thetas),
        ]
    )
    src = np.array([0.0, 0.0, r0])
    for f_hz in (300.0, 3000.0):
        c0, f_ax = sphere_surface_moments(f_hz, a, r0, rho, c)
        p_q = sphere_point_source(pts, f_hz, np.zeros(3), a, src, rho, c)
        sin_t = np.sin(thetas)
        mean_q = 0.5 * np.trapezoid(p_q * sin_t, thetas)
        c1_q = 1.5 * np.trapezoid(p_q * np.cos(thetas) * sin_t, thetas)
        f_q = -(4.0 * np.pi * a**2 / 3.0) * c1_q
        assert abs(mean_q - c0) / max(abs(c0), 1e-30) <= 1e-6
        assert abs(f_q - f_ax) / max(abs(f_ax), 1e-30) <= 1e-6


def test_box_legacy_surface_kind(tiny_h5_run):
    path, _ = tiny_h5_run
    import h5py

    with h5py.File(path, "r") as f:
        assert f["meta"].attrs.get("surface_kind") == "box_faces"
        assert "box_surface/p" in f
        assert "surface" not in f


def test_surface_p_false_dry_run_no_surface_bytes():
    cfg = Config.from_json((CONFIGS / "c13_sphere_ps.json").read_text())
    d_off = json.loads(cfg.to_json())
    d_off.setdefault("output", {})["surface_p"] = False
    cfg_off = Config.from_dict(d_off)
    off = dry_run_geom(cfg_off)
    on = dry_run_geom(cfg)
    assert off["surface_extra_bytes_est"] == 0
    assert on["surface_extra_bytes_est"] > 0