"""Baffle CPML: +z absorber active, −z (rigid baffle) off."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fdtd.config import (  # noqa: E402
    Band,
    Config,
    CPML_SIDES_BAFFLE,
    Numerics,
    Output,
    Radiator,
    SQRT3,
    derive,
)
from fdtd.cpml import apply_p, make_cpml_state  # noqa: E402
from fdtd.grid import Grid, allocate_fields  # noqa: E402
from fdtd.kernels import make_update_coeffs, step  # noqa: E402
from fdtd.source import BoxBoundary  # noqa: E402

_F64 = np.float64


def _sync_dt(cfg: Config, d: object) -> None:
    d.dt = cfg.numerics.cfl_safety * d.h / (cfg.air.c * SQRT3)  # type: ignore[attr-defined]


def _tiny_baffle():
    cfg = Config(
        geometry="baffle",
        box=None,
        radiator=Radiator("circle", dims=[0.03], x_c=0.0, y_c=0.0),
        band=Band(f_min=200.0, f_max=3_000.0),
        numerics=Numerics(
            gap_cells=8,
            cpml_cells=6,
            ppw=8.0,
            cells_across_radiator=6,
            cell_budget=10**9,
            cfl_safety=0.99,
            n_transits=0.4,
        ),
        output=Output(path="/tmp/baffle_cpml_test.h5"),
    )
    d = derive(cfg)
    _sync_dt(cfg, d)
    g = Grid(cfg, d, radiator_supersample=8)
    return cfg, d, g


def _interior_energy_half_step(p_n, p_np1, vx, vy, vz, grid, rho, c) -> float:
    s = grid.cpml_slabs
    i0, i1 = s.i_inner.lo, s.i_inner.hi
    j0, j1 = s.j_inner.lo, s.j_inner.hi
    k0, k1 = s.k_inner.lo, s.k_inner.hi
    h = grid.h
    vol = h**3
    pn = p_n[i0:i1, j0:j1, k0:k1]
    pp = p_np1[i0:i1, j0:j1, k0:k1]
    pe = 0.5 * vol / (rho * c**2) * np.sum(pn.astype(_F64) * pp.astype(_F64))
    vxi = vx[i0 : i1 + 1, j0:j1, k0:k1]
    vyi = vy[i0:i1, j0 : j1 + 1, k0:k1]
    vzi = vz[i0:i1, j0:j1, k0 : k1 + 1]
    ke = 0.5 * rho * vol * (
        np.sum(vxi.astype(_F64) ** 2)
        + np.sum(vyi.astype(_F64) ** 2)
        + np.sum(vzi.astype(_F64) ** 2)
    )
    return float(pe + ke)


def test_baffle_derive_cpml_sides():
    _, d, _ = _tiny_baffle()
    assert d.cpml_sides == CPML_SIDES_BAFFLE


def test_baffle_zpos_cpml_graded_zneg_inactive():
    cfg, d, g = _tiny_baffle()
    cpml = make_cpml_state(g, cfg, d, d.dt)
    assert cpml.sides == CPML_SIDES_BAFFLE
    assert cpml.coeffs.sigma_max > 0.0
    assert float(np.ptp(cpml.rows.b_p_pos)) > 0.0
    assert float(np.ptp(cpml.rows.a_p_pos)) > 0.0

    fields = allocate_fields(g)
    fields.p[...] = 0.0
    fields.vx[...] = 0.0
    fields.vy[...] = 0.0
    fields.vz[...] = 0.0
    mid_i, mid_j = g.nx // 2, g.ny // 2
    c = d.cpml_cells
    base = g.nz - c
    # CPML p_z_pos updates cells k ∈ [base, nz); needs Δv_z in that slab, not interior-only vz.
    fields.vz[mid_i, mid_j, base : g.nz + 1] = np.linspace(
        0.0, 0.4, c + 1, dtype=np.float32
    )

    psi_pos0 = float(np.sum(np.abs(fields.psi_p_z_pos)))
    psi_neg0 = float(np.sum(np.abs(fields.psi_p_z_neg)))
    assert psi_neg0 == 0.0

    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    apply_p(fields, cpml, coeffs)

    assert float(np.sum(np.abs(fields.psi_p_z_pos))) > psi_pos0
    assert float(np.sum(np.abs(fields.psi_p_z_neg))) == pytest.approx(0.0, abs=1e-30)


def test_baffle_short_run_interior_energy_tail():
    cfg, d, g = _tiny_baffle()
    d.n_steps = 1200
    fields = allocate_fields(g)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml = make_cpml_state(g, cfg, d, d.dt)
    boundary = BoxBoundary.from_grid(g, d)
    slabs = g.cpml_slabs
    rho, c = cfg.air.rho, cfg.air.c

    peak = 0.0
    e_last = 0.0
    for n in range(d.n_steps):
        p_n = fields.p.copy()
        step(fields, coeffs, slabs, cpml, boundary, n)
        e = _interior_energy_half_step(
            p_n, fields.p, fields.vx, fields.vy, fields.vz, g, rho, c
        )
        peak = max(peak, e)
        e_last = e

    assert peak > 0.0
    tail_db = 20.0 * math.log10(max(e_last, 1e-30) / peak)
    assert tail_db <= -60.0, f"interior energy tail {tail_db:.1f} dB (need ≤ −60)"