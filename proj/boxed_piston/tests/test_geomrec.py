"""Parametric HDF5 recording (step 6b2a)."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from boxrad.config import Config, derive
from boxrad.geom.geometry import build_geometry
from boxrad.geomderive import derive_geom
from boxrad.geomop import build_run_operator
from boxrad.geomrec import (
    GeomRecorder,
    _build_huygens_gathers,
    validate_huygens_patches_geom,
)
from boxrad.geomrun import prepare_geom_run, step_geom
from boxrad.grid import Grid, allocate_fields
from boxrad.cpml import make_cpml_state
from boxrad.kernels import make_update_coeffs, step
from boxrad.post.kirchhoff import huygens_box_patches
from boxrad.recorder import (
    Recorder,
    _build_huygens_gathers as _leg_huygens_gathers,
    _geometry_origin,
    resolve_huygens_offset,
)
from boxrad.result import Result
from boxrad.source import BoxBoundary

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHOWCASE = CONFIGS / "showcase"
_F64 = np.float64
_N_FAST = 300


def _load_showcase(name: str) -> Config:
    return Config.from_json((SHOWCASE / name).read_text(encoding="utf-8"))


def _coarse(cfg: Config, *, h: float = 0.012, n_transits: float = 0.35) -> Config:
    c = copy.deepcopy(cfg)
    c.numerics.h_override = h
    c.numerics.n_transits = n_transits
    return c


def _fast_transits(cfg: Config, *, n_transits: float = 0.35) -> Config:
    c = copy.deepcopy(cfg)
    c.numerics.n_transits = n_transits
    return c


def _assert_grid_parity_precondition(leg, par) -> None:
    msg = "grids differ: parity undefined at this h"
    assert (leg.nx, leg.ny, leg.nz) == (par.nx, par.ny, par.nz), msg
    assert abs(leg.realized_W - par.realized_W) <= 1e-12, msg
    assert abs(leg.realized_H - par.realized_H) <= 1e-12, msg
    assert abs(leg.realized_D - par.realized_D) <= 1e-12, msg


def _flat_centers(gg, flats: np.ndarray) -> np.ndarray:
    ny, nz = gg.ny, gg.nz
    h = gg.h
    x0 = gg.x0.astype(_F64)
    out = np.empty((flats.size, 3), dtype=_F64)
    for q, flat in enumerate(flats):
        i = int(flat) // (ny * nz)
        rem = int(flat) % (ny * nz)
        j = rem // nz
        k = rem % nz
        out[q] = x0 + (np.array([i, j, k], dtype=_F64) + 0.5) * h
    return out


def _legacy_huygens_bundle(c09: Config):
    leg = derive(c09)
    grid = Grid(c09, leg)
    offset = resolve_huygens_offset(c09)
    origin = _geometry_origin(grid)
    patches = huygens_box_patches(
        leg.realized_W,
        leg.realized_H,
        leg.realized_D,
        leg.h,
        origin,
        offset_cells=offset,
    )
    ib, jb, kb = grid.i_box, grid.j_box, grid.k_box
    idx4, iv, sgn, vaxis = _leg_huygens_gathers(grid, patches, ib=ib, jb=jb, kb=kb)
    return leg, grid, patches, idx4, iv, sgn, vaxis


def _geom_huygens_bundle(s3p: Config):
    geom = build_geometry(s3p)
    assert geom is not None
    derived, gg = derive_geom(s3p, geom)
    offset = resolve_huygens_offset(s3p)
    h = derived.h
    lo_s = gg.x0.astype(_F64) + (derived.gap_cells + derived.cpml_cells) * h
    patches = huygens_box_patches(
        derived.realized_W,
        derived.realized_H,
        derived.realized_D,
        h,
        lo_s,
        offset_cells=offset,
    )
    validate_huygens_patches_geom(gg, patches)
    idx4, iv, sgn, vaxis = _build_huygens_gathers(gg, patches)
    return derived, gg, geom, patches, idx4, iv, sgn, vaxis


@pytest.mark.parametrize(
    "name",
    ["s1_cylinder.json", "s3p_c09_parametric.json", "s4_tilted.json"],
)
def test_radiator_centroids_on_plane(name: str) -> None:
    cfg = _coarse(_load_showcase(name))
    geom = build_geometry(cfg)
    assert geom is not None
    derived, gg = derive_geom(cfg, geom)
    _, op_info = build_run_operator(cfg, derived, gg, geom)
    fr = op_info["fractions"]
    rd = geom.radiators[0]
    n_hat = rd.pose.R[:, 2].astype(_F64)
    n_hat = n_hat / np.linalg.norm(n_hat)
    p0 = rd.pose.position.astype(_F64)
    cells = fr.radiator_cell[0]
    g_c = fr.radiator_centroid[0]
    h = derived.h
    for q in range(cells.size):
        assert abs(float(np.dot(g_c[q] - p0, n_hat))) <= 1.0e-9
    if name in ("s1_cylinder.json", "s3p_c09_parametric.json"):
        xc = _flat_centers(gg, cells)
        d_c = np.sum((xc - g_c) * n_hat, axis=1)
        assert np.all(np.abs(d_c - 0.5 * h) <= 1.0e-9 * h)


def test_huygens_parity_legacy_s3p() -> None:
    c09 = Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8"))
    s3p = _load_showcase("s3p_c09_parametric.json")
    leg, _, leg_p, leg_i4, leg_iv, leg_sgn, leg_ax = _legacy_huygens_bundle(c09)
    par, _, _, geo_p, geo_i4, geo_iv, geo_sgn, geo_ax = _geom_huygens_bundle(s3p)
    _assert_grid_parity_precondition(leg, par)
    assert np.allclose(leg_p.centers, geo_p.centers, atol=1e-12)
    assert np.allclose(leg_p.normals, geo_p.normals, atol=1e-12)
    assert np.allclose(leg_p.areas, geo_p.areas, atol=1e-12)
    assert np.array_equal(leg_i4, geo_i4)
    assert np.array_equal(leg_iv, geo_iv)
    assert np.allclose(leg_sgn, geo_sgn, atol=1e-12)
    assert np.array_equal(leg_ax, geo_ax)


def test_radiator_pickup_parity_legacy_s3p() -> None:
    c09 = Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8"))
    c09 = _fast_transits(c09, n_transits=0.35)
    leg = derive(c09)
    grid = Grid(c09, leg)
    boundary = BoxBoundary.from_grid(grid, leg)
    vz_shape = (grid.nx, grid.ny, grid.nz + 1)
    ir, jr, kr = np.unravel_index(boundary.idx_radiator, vz_shape)
    kf = int(grid.k_face_z)
    leg_idx = np.ravel_multi_index((ir, jr, np.full_like(ir, kf)), (grid.nx, grid.ny, grid.nz))
    w_leg = boundary.w_radiator.astype(_F64)

    s3p = _fast_transits(_load_showcase("s3p_c09_parametric.json"), n_transits=0.35)
    run = prepare_geom_run(s3p)
    _assert_grid_parity_precondition(leg, run.derived)
    fr = run.info["fractions"]
    cells = fr.radiator_cell[0]
    w_geo = fr.radiator_a[0] / (run.derived.h**2)

    sig_leg = set(leg_idx[w_leg > 1e-5].tolist())
    sig_geo = set(cells[w_geo > 1e-5].tolist())
    assert sig_leg == sig_geo
    w_by_leg = dict(zip(leg_idx.tolist(), w_leg.tolist()))
    w_by_geo = dict(zip(cells.tolist(), w_geo.tolist()))
    for fi in sig_leg:
        assert abs(w_by_geo[fi] - w_by_leg[fi]) <= 1e-6


def _run_legacy_geom_recorders(
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c09 = Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8"))
    c09 = _fast_transits(c09, n_transits=0.35)
    leg = derive(c09)
    grid = Grid(c09, leg)
    fields = allocate_fields(grid)
    coeffs = make_update_coeffs(c09.air.rho, c09.air.c, leg.h, leg.dt)
    cpml = make_cpml_state(grid, c09, leg, leg.dt)
    boundary = BoxBoundary.from_grid(grid, leg)

    s3p = _fast_transits(_load_showcase("s3p_c09_parametric.json"), n_transits=0.35)
    run = prepare_geom_run(s3p)
    _assert_grid_parity_precondition(leg, run.derived)

    with tempfile.TemporaryDirectory() as tmp:
        p_leg = Path(tmp) / "leg.h5"
        p_geo = Path(tmp) / "geo.h5"
        rec_leg = Recorder(c09, leg, grid, boundary, p_leg)
        rec_geo = GeomRecorder(s3p, run.derived, run.gg, run.geometry, run.info, p_geo)
        for n in range(n_steps):
            step(fields, coeffs, grid.cpml_slabs, cpml, boundary, n)
            step_geom(run, n)
            rec_leg.sample(fields, n)
            rec_geo.sample(run.fields, n)
        rec_leg.close()
        rec_geo.close()
        with h5py.File(p_leg, "r") as f1, h5py.File(p_geo, "r") as f2:
            p1 = np.asarray(f1["radiator/p_avg"][()], dtype=_F64)
            p2 = np.asarray(f2["radiator/p_avg"][()], dtype=_F64)
            h1 = np.asarray(f1["huygens/p_face"][()], dtype=_F64)
            h2 = np.asarray(f2["huygens/p_face"][()], dtype=_F64)
            v1 = np.asarray(f1["huygens/vn"][()], dtype=_F64)
            v2 = np.asarray(f2["huygens/vn"][()], dtype=_F64)
    return p1, p2, h1, h2, v1, v2


def test_short_run_parity_legacy_s3p() -> None:
    p1, p2, h1, h2, v1, v2 = _run_legacy_geom_recorders(_N_FAST)
    scale = max(float(np.max(np.abs(p1))), 1e-30)
    assert float(np.max(np.abs(p2 - p1))) <= 1e-5 * scale
    scale_h = max(float(np.max(np.abs(h1))), 1e-30)
    assert float(np.max(np.abs(h2 - h1))) <= 1e-5 * scale_h
    scale_v = max(float(np.max(np.abs(v1))), 1e-30)
    assert float(np.max(np.abs(v2 - v1))) <= 1e-5 * scale_v


def test_hdf5_layout_short_s3p() -> None:
    s3p = _coarse(_load_showcase("s3p_c09_parametric.json"))
    run = prepare_geom_run(s3p)
    n_steps = 80
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.h5"
        rec = GeomRecorder(s3p, run.derived, run.gg, run.geometry, run.info, path)
        for n in range(n_steps):
            step_geom(run, n)
            rec.sample(run.fields, n)
        rec.close()
        with h5py.File(path, "r") as f:
            assert "box_surface" not in f
            assert "results/zrad" in f
            assert f["meta"].attrs["geometry"] == "parametric"
            assert "S_grid" in f["meta"].attrs
            for key in (
                "radiator/p_avg",
                "radiator/v_pulse_integer",
                "huygens/p_face",
                "huygens/vn",
            ):
                assert key in f
        with Result(path) as res:
            assert res.S_grid == pytest.approx(float(res.meta["S_grid"]))
            res.summary()


def test_probe_in_solid_raises() -> None:
    cfg = _coarse(_load_showcase("s4_tilted.json"))
    solid = build_geometry(cfg)
    assert solid is not None
    run = prepare_geom_run(cfg)
    xlo, ylo, zlo, xhi, yhi, zhi = solid.solid.bbox
    cfg.output.probes = [
        [
            0.5 * (xlo + xhi),
            0.5 * (ylo + yhi),
            0.5 * (zlo + zhi),
        ]
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="solid cell"):
            GeomRecorder(
                cfg,
                run.derived,
                run.gg,
                run.geometry,
                run.info,
                Path(tmp) / "x.h5",
            )