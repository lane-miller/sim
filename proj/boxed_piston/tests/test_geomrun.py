"""Parametric derive, operator, and stepping (step 6b1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config, derive
from boxrad.cpml import make_cpml_state
from boxrad.cutkernel import FastRadiator, build_operator, step_cut
from boxrad.geom.geometry import build_geometry
from boxrad.geomderive import derive_geom
from boxrad.geomop import build_run_operator
from boxrad.geomrun import dry_run_geom, prepare_geom_run, simulate_geom, step_geom
from boxrad.grid import Grid, allocate_fields
from boxrad.kernels import make_update_coeffs, step
from boxrad.runner import run
from boxrad.source import BoxBoundary

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHOWCASE = CONFIGS / "showcase"
_F32 = np.float32


def _load_showcase(name: str) -> Config:
    return Config.from_json((SHOWCASE / name).read_text(encoding="utf-8"))


def _copy_fields(run):
    f = run.fields
    from boxrad.grid import Fields

    return Fields(
        p=f.p.copy(),
        vx=f.vx.copy(),
        vy=f.vy.copy(),
        vz=f.vz.copy(),
        psi_p_x_neg=f.psi_p_x_neg.copy(),
        psi_p_x_pos=f.psi_p_x_pos.copy(),
        psi_p_y_neg=f.psi_p_y_neg.copy(),
        psi_p_y_pos=f.psi_p_y_pos.copy(),
        psi_p_z_neg=f.psi_p_z_neg.copy(),
        psi_p_z_pos=f.psi_p_z_pos.copy(),
        psi_vx_neg=f.psi_vx_neg.copy(),
        psi_vx_pos=f.psi_vx_pos.copy(),
        psi_vy_neg=f.psi_vy_neg.copy(),
        psi_vy_pos=f.psi_vy_pos.copy(),
        psi_vz_neg=f.psi_vz_neg.copy(),
        psi_vz_pos=f.psi_vz_pos.copy(),
    )


def test_derive_parity_s3p_c09() -> None:
    c09 = Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8"))
    leg = derive(c09)
    s3p = _load_showcase("s3p_c09_parametric.json")
    geom = build_geometry(s3p)
    assert geom is not None
    par, gg = derive_geom(s3p, geom)
    assert par.h == leg.h
    assert par.dt == leg.dt
    assert (par.nx, par.ny, par.nz) == (leg.nx, leg.ny, leg.nz)
    assert par.n_steps == leg.n_steps
    assert par.decimation_m == leg.decimation_m
    assert par.pulse_sigma == leg.pulse_sigma
    assert par.pulse_t0 == leg.pulse_t0
    assert (par.realized_W, par.realized_H, par.realized_D) == (
        leg.realized_W,
        leg.realized_H,
        leg.realized_D,
    )
    grid = Grid(c09, leg)
    ib, jb, kb = grid.i_box, grid.j_box, grid.k_box
    xc, yc = c09.radiator.x_c, c09.radiator.y_c
    W, H, D = leg.realized_W, leg.realized_H, leg.realized_D
    h = leg.h
    x0_legacy = np.array(
        [-xc - 0.5 * W - ib * h, -yc - 0.5 * H - jb * h, -D - kb * h],
        dtype=np.float64,
    )
    assert np.allclose(gg.x0, x0_legacy, atol=1e-12)


def test_operator_parity_s3p() -> None:
    c09 = Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8"))
    leg = derive(c09)
    grid = Grid(c09, leg)
    s3p = _load_showcase("s3p_c09_parametric.json")
    geom = build_geometry(s3p)
    assert geom is not None
    par, gg = derive_geom(s3p, geom)
    op, info = build_run_operator(s3p, par, gg, geom)
    assert info["path"] == "fast"
    assert op.radiators_fast
    ir, jr, kr = grid.radiator_face_indices
    vz_shape = (grid.nx, grid.ny, grid.nz + 1)
    leg_idx = np.ravel_multi_index((ir, jr, kr), vz_shape).astype(np.int64)
    fast = op.radiators_fast[0]
    assert fast.axis == 2
    wleg = grid.radiator_weights.astype(np.float64)
    # v0.2 float32 cell centers leave ~1e-7 "dust" on faces where a rect edge
    # lies exactly on a grid line; exact legacy clipping gives w=0 there.
    sig = wleg > 1e-5
    l_sig = set(leg_idx[sig].tolist())
    ours = set(fast.idx.tolist())
    leg_all = set(leg_idx.tolist())
    assert l_sig <= ours
    assert ours <= leg_all
    w_ours = dict(zip(fast.idx.tolist(), fast.sw.astype(np.float64).tolist()))
    w_by_leg = dict(zip(leg_idx.tolist(), wleg.tolist()))
    for fi in l_sig:
        assert abs(w_ours[fi] - w_by_leg[fi]) <= 2e-6
    for fi in leg_all - ours:
        assert w_by_leg[fi] <= 1e-5
    s_rel = abs(info["S_grid"] - grid.radiator_area_grid) / max(
        grid.radiator_area_grid, 1e-30
    )
    assert s_rel <= 1e-6


def test_stepping_parity_s3p_400() -> None:
    c09 = Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8"))
    leg_d = derive(c09)
    grid = Grid(c09, leg_d)
    leg_fields = allocate_fields(grid)
    coeffs = make_update_coeffs(c09.air.rho, c09.air.c, leg_d.h, leg_d.dt)
    cpml = make_cpml_state(grid, c09, leg_d, leg_d.dt)
    boundary = BoxBoundary.from_grid(grid, leg_d)

    s3p = _load_showcase("s3p_c09_parametric.json")
    run_p = prepare_geom_run(s3p)
    cut_fields = _copy_fields(run_p)
    fr = run_p.info["fractions"]
    mask_p = fr.beta > 0
    for n in range(400):
        v = boundary.v_at_step(n)
        step(leg_fields, coeffs, grid.cpml_slabs, cpml, boundary, n)
        step_geom(run_p, n)
    diff = run_p.fields.p[mask_p].astype(np.float64) - leg_fields.p[mask_p].astype(np.float64)
    den = float(np.linalg.norm(leg_fields.p[mask_p].astype(np.float64)))
    p_rel = float(np.linalg.norm(diff)) / max(den, 1e-30)
    assert p_rel <= 5e-5

    run_p2 = prepare_geom_run(s3p)
    cut2 = _copy_fields(run_p2)
    ir, jr, kr = grid.radiator_face_indices
    vz_shape = (grid.nx, grid.ny, grid.nz + 1)
    leg_idx = np.ravel_multi_index((ir, jr, kr), vz_shape).astype(np.int64)
    sw_leg = grid.radiator_weights.astype(np.float32)
    fast = run_p2.op.radiators_fast[0]
    fr2 = run_p2.info["fractions"]
    beta_t = run_p2.info["beta_tilde"].astype(np.float32)
    # Bit-identity: legacy idx+weights so operator matches grid clipping exactly.
    op_leg_w = build_operator(
        fr2,
        beta_t,
        [FastRadiator(axis=fast.axis, idx=leg_idx, sw=sw_leg)],
        [],
        leg_d.h,
    )
    leg2 = allocate_fields(grid)
    for n in range(400):
        v = boundary.v_at_step(n)
        step(leg2, coeffs, grid.cpml_slabs, cpml, boundary, n)
        step_cut(cut2, coeffs, op_leg_w, run_p2.cpml, [v])
    assert np.array_equal(leg2.p[mask_p], cut2.p[mask_p])


@pytest.mark.parametrize(
    "name,path_kind,sign",
    [
        ("s1_cylinder.json", "fast", 1),
        ("s4_tilted.json", "patch", 1),
        ("s6_downfire.json", "fast", -1),
    ],
)
def test_simulate_300_steps(name: str, path_kind: str, sign: int) -> None:
    cfg = _load_showcase(name)
    run = simulate_geom(cfg, n_steps=300)
    assert run.info["path"] == path_kind
    if path_kind == "fast":
        sw = run.op.radiators_fast[0].sw.astype(np.float64)
        assert np.all(sw <= 0) if sign < 0 else np.all(sw >= 0)


def test_dry_run_showcases() -> None:
    s1 = dry_run_geom(_load_showcase("s1_cylinder.json"))
    assert s1["radiator_path"] == "fast"
    s4 = dry_run_geom(_load_showcase("s4_tilted.json"))
    assert s4["radiator_path"] == "patch"
    dry_run_geom(_load_showcase("s5_platform.json"))
    s6 = dry_run_geom(_load_showcase("s6_downfire.json"))
    assert s6["clearances_m"].get("woofer", 0.0) == pytest.approx(0.030, rel=0.02)
    s2 = dry_run_geom(_load_showcase("s2_bookshelf.json"))
    assert isinstance(s2, dict)
    assert s2.get("geometry") == "parametric"


def test_guard_s1_dry_run_dict() -> None:
    summary = run(_load_showcase("s1_cylinder.json"), dry_run=True, progress=False)
    assert isinstance(summary, dict)
    assert summary["geometry"] == "parametric"