"""Unit tests for boxrad.source (pulse, box boundary, radiator injection)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import (  # noqa: E402
    Air,
    Band,
    Box,
    Config,
    Numerics,
    Output,
    Radiator,
    SQRT3,
    derive,
)
from boxrad.cpml import build_cpml_coeffs, make_cpml_state  # noqa: E402
from boxrad.grid import Grid, allocate_fields  # noqa: E402
from boxrad.kernels import make_update_coeffs, step  # noqa: E402
from boxrad.source import BoxBoundary, pulse_spectrum, velocity_pulse  # noqa: E402

_FLOAT32 = np.dtype(np.float32)


def _sync_dt(cfg: Config, d: object, cfl_safety: float | None = None) -> float:
    cs = cfg.numerics.cfl_safety if cfl_safety is None else cfl_safety
    dt = cs * d.h / (cfg.air.c * SQRT3)
    d.dt = dt
    return dt


def _box_problem(
    *,
    box: Box = Box(W=0.08, H=0.08, D=0.06),
    radiator: Radiator | None = None,
    f_max: float = 5_000.0,
    cpml_cells: int = 0,
    gap_cells: int = 8,
    ppw: float = 10.0,
    drive_enabled: bool = True,
) -> tuple[Config, object, Grid, object, BoxBoundary]:
    if radiator is None:
        radiator = Radiator("circle", dims=[0.04], x_c=0.0, y_c=0.0)
    cfg = Config(
        box=box,
        radiator=radiator,
        band=Band(f_min=20.0, f_max=f_max),
        numerics=Numerics(
            cpml_cells=cpml_cells,
            gap_cells=gap_cells,
            ppw=ppw,
            cells_across_radiator=8,
            cell_budget=10**9,
            cfl_safety=0.99,
        ),
        output=Output(path="/tmp/source_test.h5"),
    )
    d = derive(cfg)
    _sync_dt(cfg, d)
    g = Grid(cfg, d, radiator_supersample=24)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    boundary = BoxBoundary.from_grid(g, d, drive_enabled=drive_enabled)
    return cfg, d, g, coeffs, boundary


def _yee_energy_half_step(
    p_n: np.ndarray,
    p_np1: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    rho: float,
    c: float,
    h: float,
) -> float:
    vol = h**3
    ke = 0.5 * rho * vol * (
        np.sum(vx.astype(np.float64) ** 2)
        + np.sum(vy.astype(np.float64) ** 2)
        + np.sum(vz.astype(np.float64) ** 2)
    )
    pe = 0.5 * vol / (rho * c**2) * np.sum(
        p_n.astype(np.float64) * p_np1.astype(np.float64)
    )
    return float(pe + ke)


def test_velocity_pulse_peak_and_spectrum():
    cfg = Config(
        box=Box(0.1, 0.1, 0.08),
        radiator=Radiator("circle", [0.03]),
        band=Band(f_min=20.0, f_max=12_000.0),
        numerics=Numerics(),
        output=Output(path="/tmp/pulse.h5"),
    )
    d = derive(cfg)
    sigma, t0 = d.pulse_sigma, d.pulse_t0
    assert velocity_pulse(t0, sigma, t0) == pytest.approx(1.0, rel=0, abs=1e-12)

    f_ref = cfg.numerics.pulse_f_factor * d.f_max
    v_dc = abs(pulse_spectrum(0.0, sigma, t0))
    v_ref = abs(pulse_spectrum(f_ref, sigma, t0))
    db = 20.0 * math.log10(float(np.squeeze(v_ref / v_dc)))
    assert db == pytest.approx(cfg.numerics.pulse_db, abs=2.0)


def test_volume_velocity_and_rigid_faces():
    cfg, d, g, coeffs, boundary = _box_problem(
        box=Box(W=0.06, H=0.06, D=0.05),
        radiator=Radiator("circle", [0.03], x_c=0.0, y_c=0.0),
        f_max=4_000.0,
        cpml_cells=0,
        gap_cells=6,
    )
    fields = allocate_fields(g)
    slabs = g.cpml_slabs
    n_steps = min(80, d.n_steps)
    w = boundary.w_radiator.astype(np.float64)
    h = float(boundary.h)
    area_w = float(np.sum(w) * h * h)
    print(
        "volume_velocity area (once): "
        f"n_faces={boundary.idx_radiator.size} len(w)={w.size} "
        f"sum(w)*h^2={area_w:.6e} S_grid={float(boundary.s_grid):.6e} "
        f"radiator_area_grid={float(g.radiator_area_grid):.6e}"
    )
    for n in range(n_steps):
        step(fields, coeffs, slabs, boundary=boundary, n=n)
        qv = boundary.volume_velocity(fields.vz)
        qe = boundary.expected_volume_velocity(n)
        np.testing.assert_allclose(
            qv, qe, rtol=0.0, atol=2.0 * float(np.finfo(_FLOAT32).eps) * max(abs(qe), 1.0)
        )
        vx_r = fields.vx.ravel()[boundary.idx_vx]
        vy_r = fields.vy.ravel()[boundary.idx_vy]
        vz_r = fields.vz.ravel()[boundary.idx_vz]
        assert np.all(vx_r == 0.0)
        assert np.all(vy_r == 0.0)
        assert np.all(vz_r == 0.0)


def _place_exterior_blob(
    g: Grid, f: object, *, margin: int = 2, sigma_cells: float = 2.0
) -> None:
    """Gaussian pressure blob in the air gap outside the box (not in box_cells)."""
    assert g.box_cells is not None and g.i_box is not None
    mask = np.zeros(g.box_cells.shape, dtype=bool)
    ib, jb, kb = g.i_box, g.j_box, g.k_box
    nx, ny, nz = g.nx, g.ny, g.nz
    mask[margin : nx - margin, margin : ny - margin, margin : nz - margin] = True
    mask &= ~g.box_cells
    ii, jj, kk = np.nonzero(mask)
    if ii.size == 0:
        raise RuntimeError("no exterior gap cells for blob")
    ic, jc, kc = int(ii[0]), int(jj[0]), int(kk[0])
    blob = np.zeros_like(f.p)
    i_grid, j_grid, k_grid = np.ogrid[: g.nx, : g.ny, : g.nz]
    blob[...] = np.exp(
        -(
            (i_grid - ic) ** 2
            + (j_grid - jc) ** 2
            + (k_grid - kc) ** 2
        )
        / (2.0 * sigma_cells**2)
    ).astype(_FLOAT32)
    blob[g.box_cells] = 0.0
    f.p[...] = blob


def test_rigid_box_obstacle_energy_and_interior():
    cfg, d, g, coeffs, boundary = _box_problem(
        f_max=3_500.0,
        cpml_cells=0,
        gap_cells=10,
        ppw=8.0,
        drive_enabled=False,
    )
    fields = allocate_fields(g)
    fields.vx[...] = 0.0
    fields.vy[...] = 0.0
    fields.vz[...] = 0.0
    _place_exterior_blob(g, fields, margin=2, sigma_cells=2.5)
    slabs = g.cpml_slabs
    assert g.box_cells is not None
    interior = g.box_cells
    e0 = None
    e_last = None
    for n in range(500):
        p_n = fields.p.copy()
        step(fields, coeffs, slabs, boundary=boundary, n=n)
        assert np.all(fields.p[interior] == 0.0)
        ib, jb, kb = g.i_box, g.j_box, g.k_box
        nbx, nby, nbz = g.nbx, g.nby, g.nbz
        i0, i1 = ib + 1, ib + nbx - 1
        j0, j1 = jb + 1, jb + nby - 1
        k0, k1 = kb + 1, kb + nbz - 1
        if i0 < i1 and j0 < j1 and k0 < k1:
            assert np.all(fields.vx[i0:i1, j0:j1, k0:k1] == 0.0)
            assert np.all(fields.vy[i0:i1, j0:j1, k0:k1] == 0.0)
            assert np.all(fields.vz[i0:i1, j0:j1, k0:k1] == 0.0)
        e = _yee_energy_half_step(
            p_n, fields.p, fields.vx, fields.vy, fields.vz,
            cfg.air.rho, cfg.air.c, d.h,
        )
        if e0 is None:
            e0 = e
        e_last = e
    rel = abs(e_last - e0) / max(e0, 1e-30)
    assert rel < 1.0e-4


def _stagger_exterior_masks(g: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ib, jb, kb = g.i_box, g.j_box, g.k_box
    nbx, nby, nbz = g.nbx, g.nby, g.nbz
    mask_vx = np.ones((g.nx + 1, g.ny, g.nz), dtype=bool)
    mask_vy = np.ones((g.nx, g.ny + 1, g.nz), dtype=bool)
    mask_vz = np.ones((g.nx, g.ny, g.nz + 1), dtype=bool)
    mask_vx[ib : ib + nbx + 1, jb : jb + nby, kb : kb + nbz] = False
    mask_vy[ib : ib + nbx, jb : jb + nby + 1, kb : kb + nbz] = False
    mask_vz[ib : ib + nbx, jb : jb + nby, kb : kb + nbz + 1] = False
    return mask_vx, mask_vy, mask_vz


def _assert_exterior_fields_match(
    g: Grid,
    a_p: np.ndarray,
    a_vx: np.ndarray,
    a_vy: np.ndarray,
    a_vz: np.ndarray,
    b_p: np.ndarray,
    b_vx: np.ndarray,
    b_vy: np.ndarray,
    b_vz: np.ndarray,
    *,
    rtol: float = 1e-7,
) -> None:
    assert g.box_cells is not None
    exterior_p = ~g.box_cells
    scale = max(
        float(np.max(np.abs(a_p[exterior_p]))),
        float(np.max(np.abs(b_p[exterior_p]))),
        1e-30,
    )
    atol = rtol * scale
    np.testing.assert_allclose(b_p[exterior_p], a_p[exterior_p], rtol=rtol, atol=atol)
    mx, my, mz = _stagger_exterior_masks(g)
    np.testing.assert_allclose(b_vx[mx], a_vx[mx], rtol=rtol, atol=atol)
    np.testing.assert_allclose(b_vy[my], a_vy[my], rtol=rtol, atol=atol)
    np.testing.assert_allclose(b_vz[mz], a_vz[mz], rtol=rtol, atol=atol)


def test_driven_interior_decoupled():
    """Prescribed radiator: exterior unchanged if interior p is zeroed each step."""
    cfg, d, g, coeffs, boundary = _box_problem(
        box=Box(W=0.08, H=0.08, D=0.06),
        radiator=Radiator("circle", [0.04], x_c=0.0, y_c=0.0),
        f_max=4_000.0,
        cpml_cells=0,
        gap_cells=8,
        drive_enabled=True,
    )
    assert g.box_cells is not None
    interior = g.box_cells
    slabs = g.cpml_slabs
    n_run = 100

    f_ref = allocate_fields(g)
    for n in range(n_run):
        step(f_ref, coeffs, slabs, boundary=boundary, n=n)

    f_zero = allocate_fields(g)
    for n in range(n_run):
        step(f_zero, coeffs, slabs, boundary=boundary, n=n)
        f_zero.p[interior] = np.float32(0.0)

    _assert_exterior_fields_match(
        g,
        f_ref.p,
        f_ref.vx,
        f_ref.vy,
        f_ref.vz,
        f_zero.p,
        f_zero.vx,
        f_zero.vy,
        f_zero.vz,
    )
    assert np.any(np.abs(f_ref.p[interior]) > 0.0)


def _mirror_ij(g: Grid) -> tuple[np.ndarray, np.ndarray]:
    ib, jb = g.i_box, g.j_box
    nbx, nby = g.nbx, g.nby
    nx, ny = g.nx, g.ny
    i_ax = np.arange(nx, dtype=np.int64)
    j_ax = np.arange(ny, dtype=np.int64)
    i_m = 2 * ib + nbx - 1 - i_ax
    j_m = 2 * jb + nby - 1 - j_ax
    return i_m, j_m


def test_centered_radiator_pressure_symmetry():
    box = Box(W=0.1, H=0.1, D=0.08)
    rad = Radiator("circle", dims=[0.05], x_c=0.0, y_c=0.0)
    cfg, d, g, coeffs, boundary = _box_problem(
        box=box, radiator=rad, f_max=4_000.0, cpml_cells=0, gap_cells=8
    )
    fields = allocate_fields(g)
    slabs = g.cpml_slabs
    n_run = 60
    i_m, j_m = _mirror_ij(g)
    for n in range(n_run):
        step(fields, coeffs, slabs, boundary=boundary, n=n)
    p = fields.p
    p_sym = p[i_m[:, None], j_m[None, :], :]
    np.testing.assert_allclose(p, p_sym, rtol=1e-5, atol=1e-5)


def test_low_frequency_monopole_ratio_info(capsys):
    """Print |P_meas| / |P_monopole| at 200–500 Hz (informational; no tight assert)."""
    box = Box(W=0.04, H=0.04, D=0.04)
    rad = Radiator("circle", dims=[0.02], x_c=0.0, y_c=0.0)
    cfg = Config(
        box=box,
        radiator=rad,
        air=Air(),
        band=Band(f_min=20.0, f_max=6_000.0),
        numerics=Numerics(
            cpml_cells=12,
            gap_cells=10,
            ppw=12.0,
            cells_across_radiator=8,
            cell_budget=10**9,
            n_transits=1.5,
        ),
        output=Output(path="/tmp/lowf.h5"),
    )
    d = derive(cfg)
    _sync_dt(cfg, d)
    g = Grid(cfg, d, radiator_supersample=24)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    boundary = BoxBoundary.from_grid(g, d)
    cpml_c = build_cpml_coeffs(g, cfg, d, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    fields = allocate_fields(g)
    slabs = g.cpml_slabs

    _, _, k_probe = g.index_of(0.0, 0.0, 0.2)
    n_steps = min(d.n_steps, 2500)
    p_hist = np.zeros(n_steps, dtype=np.float64)
    v_drv = np.zeros(n_steps, dtype=np.float64)
    rho, c = cfg.air.rho, cfg.air.c
    s_grid = float(boundary.s_grid)

    for n in range(n_steps):
        step(fields, coeffs, slabs, cpml=cpml_st, boundary=boundary, n=n)
        p_hist[n] = float(fields.p[g.nx // 2, g.ny // 2, k_probe])
        v_drv[n] = float(boundary.v_at_step(n))

    dt = float(d.dt)
    t_samp = (np.arange(n_steps, dtype=np.float64) + 0.5) * dt
    win = np.hanning(n_steps)
    p_w = (p_hist - p_hist.mean()) * win
    v_w = v_drv * win
    P = np.fft.rfft(p_w)
    V = np.fft.rfft(v_w)
    freqs = np.fft.rfftfreq(n_steps, dt)
    r_probe = 0.2

    lines = ["low-frequency monopole ratio (probe z=+0.2 m):"]
    for f_hz in (200.0, 300.0, 400.0, 500.0):
        ib = int(np.argmin(np.abs(freqs - f_hz)))
        om = 2.0 * math.pi * freqs[ib]
        p_meas = abs(P[ib])
        v_spec = V[ib] if abs(V[ib]) > 0 else pulse_spectrum(freqs[ib], d.pulse_sigma, d.pulse_t0)
        p_mono = abs(1j * om * rho * s_grid * v_spec) / (4.0 * math.pi * r_probe)
        ratio = p_meas / p_mono if p_mono > 0 else float("nan")
        lines.append(f"  {freqs[ib]:.1f} Hz: |P|/|P_mono| = {ratio:.3f}")
    text = "\n".join(lines)
    print(text)
    out = capsys.readouterr().out
    assert "Hz" in out or "Hz" in text
    assert np.all(np.isfinite(p_hist))