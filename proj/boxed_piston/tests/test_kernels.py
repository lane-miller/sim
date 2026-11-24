"""Unit tests for boxrad.kernels (Stage A interior leapfrog)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import (  # noqa: E402
    Air,
    Band,
    Config,
    Numerics,
    Output,
    Radiator,
    SQRT3,
    cfl_for_legacy_derive,
    derive,
)
from boxrad.grid import Grid, allocate_fields, build_cpml_slabs  # noqa: E402
from boxrad.kernels import courant, make_update_coeffs, step, step_numpy  # noqa: E402

_FLOAT32 = np.dtype(np.float32)


def _sync_dt(cfg: Config, d: object, cfl_safety: float | None = None) -> float:
    """Recompute ``d.dt`` from ``d.h``, air speed, and CFL safety (keeps coeffs in sync)."""
    cs = cfl_for_legacy_derive(cfg.numerics) if cfl_safety is None else cfl_safety
    dt = cs * d.h / (cfg.air.c * SQRT3)
    d.dt = dt
    return dt


def _coeffs_for(cfg: Config, d: object, cfl_safety: float | None = None):
    dt = _sync_dt(cfg, d, cfl_safety)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, dt)
    return coeffs


def _assert_courant_ok(coeffs) -> None:
    assert courant(coeffs) <= 1.0 + 1e-5


def _coeffs_grid(
    *,
    cpml_cells: int = 2,
    ppw: float = 12.0,
    f_max: float = 4_000.0,
    cfl_safety: float = 0.99,
) -> tuple[Grid, object, object]:
    num = Numerics(
        cpml_cells=cpml_cells,
        gap_cells=2,
        ppw=ppw,
        cfl_safety=cfl_safety,
        cell_budget=10**9,
    )
    cfg = Config(
        box=None,
        air=Air(c=343.0, rho=1.204),
        radiator=Radiator("circle", dims=[0.01]),
        band=Band(f_min=20.0, f_max=f_max),
        numerics=num,
        output=Output(path="/tmp/kernels_test.h5"),
    )
    d = derive(cfg)
    _sync_dt(cfg, d, cfl_safety)
    g = Grid(cfg, d)
    coeffs = _coeffs_for(cfg, d, cfl_safety)
    return g, d, coeffs


def _grid_n(n: int, *, cpml: int = 0, cfl_safety: float = 0.99):
    g, d, _ = _coeffs_grid(cpml_cells=cpml, cfl_safety=cfl_safety)
    d.nx = d.ny = d.nz = n
    d.cpml_cells = cpml
    _sync_dt(g.cfg, d, cfl_safety)
    g = Grid(g.cfg, d)
    g.cpml_slabs = build_cpml_slabs(n, n, n, cpml)
    coeffs = _coeffs_for(g.cfg, d, cfl_safety)
    return g, d, coeffs


def _slabs_from_grid(g: Grid):
    return g.cpml_slabs


def _field_diff_report(a_p, a_vx, a_vy, a_vz, b_p, b_vx, b_vy, b_vz) -> str:
    lines = []
    for name, a, b in (
        ("p", a_p, b_p),
        ("vx", a_vx, b_vx),
        ("vy", a_vy, b_vy),
        ("vz", a_vz, b_vz),
    ):
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        if d.size == 0:
            continue
        idx = np.unravel_index(int(np.argmax(d)), d.shape)
        lines.append(
            f"{name}: max|Δ|={d.max():.3e} at {idx} "
            f"(numba={float(a[idx]):.9g}, numpy={float(b[idx]):.9g})"
        )
    return "; ".join(lines)


def _assert_fields_match_ulp(
    a_p, a_vx, a_vy, a_vz, b_p, b_vx, b_vy, b_vz, *, maxulp: int
) -> None:
    try:
        for a, b in ((a_p, b_p), (a_vx, b_vx), (a_vy, b_vy), (a_vz, b_vz)):
            np.testing.assert_array_max_ulp(a, b, maxulp=maxulp)
    except AssertionError as exc:
        report = _field_diff_report(a_p, a_vx, a_vy, a_vz, b_p, b_vx, b_vy, b_vz)
        raise AssertionError(f"{exc}\nNumba vs NumPy: {report}") from exc


def _assert_fields_allclose(
    a_p, a_vx, a_vy, a_vz, b_p, b_vx, b_vy, b_vz, *, rtol: float, atol_scale: float
) -> None:
    ref_scale = max(
        float(np.max(np.abs(b_p))),
        float(np.max(np.abs(b_vx))),
        float(np.max(np.abs(b_vy))),
        float(np.max(np.abs(b_vz))),
        1e-30,
    )
    atol = atol_scale * ref_scale
    try:
        np.testing.assert_allclose(a_p, b_p, rtol=rtol, atol=atol)
        np.testing.assert_allclose(a_vx, b_vx, rtol=rtol, atol=atol)
        np.testing.assert_allclose(a_vy, b_vy, rtol=rtol, atol=atol)
        np.testing.assert_allclose(a_vz, b_vz, rtol=rtol, atol=atol)
    except AssertionError as exc:
        report = _field_diff_report(a_p, a_vx, a_vy, a_vz, b_p, b_vx, b_vy, b_vz)
        raise AssertionError(f"{exc}\nNumba vs NumPy: {report}") from exc


def test_numba_matches_numpy_one_step():
    g, d, coeffs = _coeffs_grid()
    _assert_courant_ok(coeffs)
    f_nb = allocate_fields(g)
    f_np = allocate_fields(g)
    rng = np.random.default_rng(42)
    for arr in (f_nb.p, f_nb.vx, f_nb.vy, f_nb.vz):
        arr[...] = rng.uniform(-0.1, 0.1, size=arr.shape).astype(_FLOAT32)
    f_np.p[...], f_np.vx[...], f_np.vy[...], f_np.vz[...] = (
        f_nb.p,
        f_nb.vx,
        f_nb.vy,
        f_nb.vz,
    )
    slabs = _slabs_from_grid(g)
    step(f_nb, coeffs, slabs)  # warm JIT
    f_nb.p[...], f_nb.vx[...], f_nb.vy[...], f_nb.vz[...] = (
        f_np.p,
        f_np.vx,
        f_np.vy,
        f_np.vz,
    )
    step(f_nb, coeffs, slabs)
    step_numpy(f_np.p, f_np.vx, f_np.vy, f_np.vz, coeffs, slabs)
    _assert_fields_match_ulp(
        f_nb.p, f_nb.vx, f_nb.vy, f_nb.vz,
        f_np.p, f_np.vx, f_np.vy, f_np.vz,
        maxulp=2,
    )


def test_numba_matches_numpy_ten_steps():
    g, d, coeffs = _coeffs_grid()
    _assert_courant_ok(coeffs)
    f_nb = allocate_fields(g)
    f_np = allocate_fields(g)
    rng = np.random.default_rng(7)
    for arr in (f_nb.p, f_nb.vx, f_nb.vy, f_nb.vz):
        arr[...] = rng.uniform(-0.05, 0.05, size=arr.shape).astype(_FLOAT32)
    f_np.p[...], f_np.vx[...], f_np.vy[...], f_np.vz[...] = (
        f_nb.p,
        f_nb.vx,
        f_nb.vy,
        f_nb.vz,
    )
    slabs = _slabs_from_grid(g)
    step(f_nb, coeffs, slabs)
    f_nb.p[...], f_nb.vx[...], f_nb.vy[...], f_nb.vz[...] = (
        f_np.p,
        f_np.vx,
        f_np.vy,
        f_np.vz,
    )
    for _ in range(10):
        step(f_nb, coeffs, slabs)
        step_numpy(f_np.p, f_np.vx, f_np.vy, f_np.vz, coeffs, slabs)
    _assert_fields_allclose(
        f_nb.p, f_nb.vx, f_nb.vy, f_nb.vz,
        f_np.p, f_np.vx, f_np.vy, f_np.vz,
        rtol=1e-5,
        atol_scale=1e-6,
    )


def test_dtypes_remain_float32():
    g, d, coeffs = _coeffs_grid()
    _assert_courant_ok(coeffs)
    f = allocate_fields(g)
    f.p[...] = _FLOAT32.type(1.0)
    slabs = _slabs_from_grid(g)
    step(f, coeffs, slabs)
    assert f.p.dtype == _FLOAT32
    assert f.vx.dtype == _FLOAT32
    assert f.vy.dtype == _FLOAT32
    assert f.vz.dtype == _FLOAT32


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
    """Conserved Yee energy E^{n+½} = ½ρΣ|v^{n+½}|² + ½Σ p^n·p^{n+1}/(ρc²), all × h³."""
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


def test_energy_conservation_closed_rigid():
    n = 18
    cpml = 0
    g, d, coeffs = _grid_n(n, cpml=cpml)
    _assert_courant_ok(coeffs)
    slabs = g.cpml_slabs
    cfg = g.cfg
    f = allocate_fields(g)
    f.p[...] = 0.0
    f.vx[...] = 0.0
    f.vy[...] = 0.0
    f.vz[...] = 0.0
    ci = n // 2
    sigma = 2.0
    ii, jj, kk = np.ogrid[:n, :n, :n]
    blob = np.exp(
        -((ii - ci) ** 2 + (jj - ci) ** 2 + (kk - ci) ** 2) / (2.0 * sigma**2)
    ).astype(_FLOAT32)
    f.p[...] = blob
    e0 = None
    e_last = None
    for _ in range(500):
        p_n = f.p.copy()
        step(f, coeffs, slabs)
        e = _yee_energy_half_step(
            p_n, f.p, f.vx, f.vy, f.vz, cfg.air.rho, cfg.air.c, d.h
        )
        if e0 is None:
            e0 = e
        e_last = e
    rel = abs(e_last - e0) / max(e0, 1e-30)
    assert rel < 1.0e-4


def test_centered_gaussian_pressure_symmetry():
    n = 17
    cpml = 0
    g, d, coeffs = _grid_n(n, cpml=cpml)
    _assert_courant_ok(coeffs)
    slabs = g.cpml_slabs
    f = allocate_fields(g)
    f.p[...] = 0.0
    f.vx[...] = 0.0
    f.vy[...] = 0.0
    f.vz[...] = 0.0
    c = n // 2
    sigma = 2.5
    ii, jj, kk = np.ogrid[:n, :n, :n]
    f.p[...] = np.exp(
        -((ii - c) ** 2 + (jj - c) ** 2 + (kk - c) ** 2) / (2.0 * sigma**2)
    ).astype(_FLOAT32)
    n_steps = 40
    for _ in range(n_steps):
        step(f, coeffs, slabs)
    p_mirror = f.p[::-1, ::-1, ::-1]
    np.testing.assert_allclose(f.p, p_mirror, rtol=1e-5, atol=1e-5)


def test_cfl_stability_random_initial_field():
    g, d, coeffs = _coeffs_grid(cfl_safety=0.99, f_max=8_000.0)
    _assert_courant_ok(coeffs)
    assert courant(coeffs) == pytest.approx(0.99, abs=1e-3)
    f = allocate_fields(g)
    rng = np.random.default_rng(99)
    f.p[...] = rng.uniform(-0.2, 0.2, size=f.p.shape).astype(_FLOAT32)
    z0 = 1.204 * 343.0  # rho*c: scale v so its energy matches p's
    f.vx[...] = (rng.uniform(-0.2, 0.2, size=f.vx.shape) / z0).astype(_FLOAT32)
    f.vy[...] = (rng.uniform(-0.2, 0.2, size=f.vy.shape) / z0).astype(_FLOAT32)
    f.vz[...] = (rng.uniform(-0.2, 0.2, size=f.vz.shape) / z0).astype(_FLOAT32)
    p0_max = float(np.max(np.abs(f.p)))
    slabs = _slabs_from_grid(g)
    for _ in range(1000):
        step(f, coeffs, slabs)
    assert np.all(np.isfinite(f.p))
    assert np.all(np.isfinite(f.vx))
    p_max = float(np.max(np.abs(f.p)))
    assert p_max < 50.0 * max(p0_max, 1e-6)
