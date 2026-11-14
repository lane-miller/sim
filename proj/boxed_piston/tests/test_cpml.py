"""Unit tests for fdtd.cpml (CFS-PML slab updates)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fdtd.config import (  # noqa: E402
    Air,
    Band,
    Config,
    Numerics,
    Output,
    Radiator,
    SQRT3,
    derive,
)
import math

from fdtd.cpml import (  # noqa: E402
    CpmlCoeffs,
    CpmlFaceRows,
    CpmlState,
    _profile_row,
    _profile_row_v,
    apply_p,
    apply_p_numpy,
    apply_v,
    apply_v_numpy,
    build_cpml_coeffs,
    face_rows,
    make_cpml_state,
)
from fdtd.grid import Fields, Grid, allocate_fields, build_cpml_slabs  # noqa: E402
from fdtd.kernels import (  # noqa: E402
    make_update_coeffs,
    step,
    step_numpy,
    update_p,
    update_v,
    zero_rigid_face_velocities,
)

_FLOAT32 = np.dtype(np.float32)
_CPML_AXIS_X_ONLY = (True, False, False)
_PROFILE_F32_ATOL = 3e-6


def _indices_interface_to_wall(n: int, neg: bool) -> np.ndarray:
    """Row indices in order from CPML/interior interface toward the outer wall."""
    if neg:
        return np.arange(n - 1, -1, -1, dtype=int)
    return np.arange(n, dtype=int)


def _profile_b_a_at_d(
    d_over_L: float,
    *,
    sigma_max: float,
    kappa_max: float,
    alpha_max: float,
    m: float,
    dt: float,
) -> tuple[float, float]:
    """``(b, a)`` at normalized depth *d_over_L* (matches :func:`fdtd.cpml._profile_row`)."""
    d_over_L = min(max(d_over_L, 0.0), 1.0)
    power = d_over_L**m
    sigma = sigma_max * power
    kappa = 1.0 + (kappa_max - 1.0) * power
    alpha = alpha_max * (1.0 - d_over_L)
    decay = sigma / kappa + alpha
    b = math.exp(-decay * dt)
    if abs(sigma) < 1e-30:
        a = 0.0
    else:
        denom = sigma * kappa + kappa * kappa * alpha
        a = sigma / denom * (b - 1.0)
    return b, a


def _assert_profile_b_a_monotone(
    b: np.ndarray,
    a: np.ndarray,
    *,
    neg: bool,
    interface_q: int,
    interface_b: float,
    interface_a: float,
) -> None:
    """Interface ``(b, a)`` as given; *b* falls and ``|a|`` rises toward the wall."""
    n = len(b)
    assert len(a) == n
    order = _indices_interface_to_wall(n, neg)
    b_ord = b[order].astype(np.float64)
    a_ord = a[order].astype(np.float64)
    tol = _PROFILE_F32_ATOL

    assert b[interface_q] == pytest.approx(interface_b, abs=tol)
    assert a[interface_q] == pytest.approx(interface_a, abs=tol)
    assert np.all(a_ord <= tol)
    assert np.all(np.diff(-b_ord) >= -tol), "b must be non-increasing toward the wall"
    assert np.all(np.diff(np.abs(a_ord)) >= -tol), "|a| must be non-decreasing toward the wall"


def _sigma_v_profile(
    n_cells: int, neg: bool, sigma_max: float, m: float
) -> np.ndarray:
    """σ(d) on velocity faces (matches :func:`fdtd.cpml._profile_row_v` depths)."""
    denom = max(n_cells, 1)
    out = np.empty(n_cells + 1, dtype=np.float64)
    for q in range(n_cells + 1):
        d = (n_cells - q) / denom if neg else q / denom
        d = min(max(d, 0.0), 1.0)
        out[q] = sigma_max * d**m
    return out


def _depths_p(n_cells: int, neg: bool) -> np.ndarray:
    if neg:
        q = np.arange(n_cells, dtype=np.float64)
        return (n_cells - q - 0.5) / n_cells
    q = np.arange(n_cells, dtype=np.float64)
    return (q + 0.5) / n_cells


def _depths_v(n_cells: int, neg: bool) -> np.ndarray:
    q = np.arange(n_cells + 1, dtype=np.float64)
    if neg:
        return (n_cells - q) / n_cells
    return q / n_cells


def _copy_fields(dst: Fields, src: Fields) -> None:
    """Copy pressure, velocity, and all CPML ψ buffers."""
    dst.p[...] = src.p
    dst.vx[...] = src.vx
    dst.vy[...] = src.vy
    dst.vz[...] = src.vz
    for name in Fields.__dataclass_fields__:
        if name.startswith("psi_"):
            getattr(dst, name)[...] = getattr(src, name)


def _sync_dt(cfg: Config, d: object, cfl_safety: float = 0.99) -> float:
    dt = cfl_safety * d.h / (cfg.air.c * SQRT3)
    d.dt = dt
    return dt


def _grid(
    n: int,
    *,
    cpml: int,
    cpml_R: float = 1e-6,
    cpml_m: float = 3.0,
    cpml_alpha_factor: float = 1.0,
    cpml_kappa_max: float = 1.0,
) -> tuple[Grid, object, Config, object]:
    num = Numerics(
        cpml_cells=cpml,
        gap_cells=0,
        ppw=12.0,
        cfl_safety=0.99,
        cell_budget=10**9,
        cpml_R=cpml_R,
        cpml_m=cpml_m,
        cpml_alpha_factor=cpml_alpha_factor,
        cpml_kappa_max=cpml_kappa_max,
    )
    cfg = Config(
        box=None,
        air=Air(c=343.0, rho=1.204),
        radiator=Radiator("circle", dims=[0.01]),
        band=Band(f_min=20.0, f_max=4_000.0),
        numerics=num,
        output=Output(path="/tmp/cpml_test.h5"),
    )
    d = derive(cfg)
    d.nx = d.ny = d.nz = n
    d.cpml_cells = cpml
    _sync_dt(cfg, d)
    g = Grid(cfg, d)
    g.cpml_slabs = build_cpml_slabs(n, n, n, cpml)
    return g, d, cfg, d


def _grid_aniso(
    nx: int,
    ny: int,
    nz: int,
    *,
    cpml: int,
    cpml_R: float = 1e-6,
    cpml_m: float = 3.0,
    cpml_alpha_factor: float = 1.0,
    cpml_kappa_max: float = 1.0,
) -> tuple[Grid, object, Config, object]:
    num = Numerics(
        cpml_cells=cpml,
        gap_cells=0,
        ppw=12.0,
        cfl_safety=0.99,
        cell_budget=10**9,
        cpml_R=cpml_R,
        cpml_m=cpml_m,
        cpml_alpha_factor=cpml_alpha_factor,
        cpml_kappa_max=cpml_kappa_max,
    )
    cfg = Config(
        box=None,
        air=Air(c=343.0, rho=1.204),
        radiator=Radiator("circle", dims=[0.01]),
        band=Band(f_min=20.0, f_max=4_000.0),
        numerics=num,
        output=Output(path="/tmp/cpml_duct_test.h5"),
    )
    d = derive(cfg)
    d.nx, d.ny, d.nz = nx, ny, nz
    d.cpml_cells = cpml
    _sync_dt(cfg, d)
    g = Grid(cfg, d)
    g.cpml_slabs = build_cpml_slabs(nx, ny, nz, cpml)
    return g, d, cfg, d


def _step_cpml_axes(
    fields: Fields,
    coeffs: object,
    slabs: object,
    cpml: CpmlState,
    axes: tuple[bool, bool, bool],
) -> None:
    nx, ny, nz = fields.p.shape
    zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
    update_v(
        fields.p,
        fields.vx,
        fields.vy,
        fields.vz,
        coeffs.cv,
        0,
        nx,
        0,
        ny,
        0,
        nz,
    )
    apply_v(fields, cpml, coeffs, axes=axes)
    zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
    update_p(
        fields.p,
        fields.vx,
        fields.vy,
        fields.vz,
        coeffs.cp,
        0,
        nx,
        0,
        ny,
        0,
        nz,
    )
    apply_p(fields, cpml, coeffs, axes=axes)


def _inactive_cpml_state(c: int) -> CpmlState:
    """Profiles with σ_max = 0, κ = 1, α = 0 (null absorber)."""
    z1 = np.zeros(c, dtype=_FLOAT32)
    b1 = np.ones(c, dtype=_FLOAT32)
    zv = np.zeros(c + 1, dtype=_FLOAT32)
    bv = np.ones(c + 1, dtype=_FLOAT32)
    coeffs = CpmlCoeffs(c, 0.0, b1, z1, z1, bv, zv, zv)
    rows = CpmlFaceRows(
        b1, z1, z1, b1, z1, z1,
        bv, zv, zv, bv, zv, zv,
    )
    return CpmlState(coeffs=coeffs, rows=rows)


def _assert_fields_match_ulp(a, b, *, maxulp: int = 2) -> None:
    for x, y in zip(a, b):
        np.testing.assert_array_max_ulp(x, y, maxulp=maxulp)


def test_null_cpml_matches_plain_step():
    n = 22
    cpml = 4
    g, d, cfg, _ = _grid(n, cpml=cpml, cpml_R=1.0, cpml_alpha_factor=0.0)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    slabs = g.cpml_slabs
    state = _inactive_cpml_state(cpml)

    f_plain = allocate_fields(g)
    f_cpml = allocate_fields(g)
    rng = np.random.default_rng(11)
    for arr in (f_plain.p, f_plain.vx, f_plain.vy, f_plain.vz):
        arr[...] = rng.uniform(-0.08, 0.08, size=arr.shape).astype(_FLOAT32)
    f_cpml.p[...] = f_plain.p
    f_cpml.vx[...] = f_plain.vx
    f_cpml.vy[...] = f_plain.vy
    f_cpml.vz[...] = f_plain.vz

    step(f_plain, coeffs, slabs)
    step(f_cpml, coeffs, slabs, cpml=state)
    _assert_fields_match_ulp(
        (f_plain.p, f_plain.vx, f_plain.vy, f_plain.vz),
        (f_cpml.p, f_cpml.vx, f_cpml.vy, f_cpml.vz),
    )


def test_slab_numba_matches_numpy_one_step():
    n = 20
    cpml = 5
    g, d, cfg, _ = _grid(n, cpml=cpml)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    slabs = g.cpml_slabs

    f_nb = allocate_fields(g)
    f_np = allocate_fields(g)
    rng = np.random.default_rng(3)
    for arr in (f_nb.p, f_nb.vx, f_nb.vy, f_nb.vz):
        arr[...] = rng.uniform(-0.05, 0.05, size=arr.shape).astype(_FLOAT32)
    f_np.p[...], f_np.vx[...], f_np.vy[...], f_np.vz[...] = (
        f_nb.p, f_nb.vx, f_nb.vy, f_nb.vz,
    )

    f_warm = allocate_fields(g)
    step(f_warm, coeffs, slabs, cpml=cpml_st)  # warm JIT (discard)
    step(f_nb, coeffs, slabs, cpml=cpml_st)
    step_numpy(
        f_np.p, f_np.vx, f_np.vy, f_np.vz, coeffs, slabs,
        fields=f_np, cpml=cpml_st,
    )
    _assert_fields_match_ulp(
        (f_nb.p, f_nb.vx, f_nb.vy, f_nb.vz),
        (f_np.p, f_np.vx, f_np.vy, f_np.vz),
    )


def test_slab_numba_matches_numpy_ten_steps():
    n = 18
    cpml = 4
    g, d, cfg, _ = _grid(n, cpml=cpml)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    slabs = g.cpml_slabs

    f_nb = allocate_fields(g)
    f_np = allocate_fields(g)
    rng = np.random.default_rng(5)
    for arr in (f_nb.p, f_nb.vx, f_nb.vy, f_nb.vz):
        arr[...] = rng.uniform(-0.04, 0.04, size=arr.shape).astype(_FLOAT32)
    f_np.p[...], f_np.vx[...], f_np.vy[...], f_np.vz[...] = (
        f_nb.p, f_nb.vx, f_nb.vy, f_nb.vz,
    )

    for _ in range(10):
        step(f_nb, coeffs, slabs, cpml=cpml_st)
        step_numpy(
            f_np.p, f_np.vx, f_np.vy, f_np.vz, coeffs, slabs,
            fields=f_np, cpml=cpml_st,
        )

    ref_scale = max(
        float(np.max(np.abs(f_np.p))),
        float(np.max(np.abs(f_np.vx))),
        1e-30,
    )
    atol = 1e-6 * ref_scale
    np.testing.assert_allclose(f_nb.p, f_np.p, rtol=1e-5, atol=atol)
    np.testing.assert_allclose(f_nb.vx, f_np.vx, rtol=1e-5, atol=atol)
    np.testing.assert_allclose(f_nb.vy, f_np.vy, rtol=1e-5, atol=atol)
    np.testing.assert_allclose(f_nb.vz, f_np.vz, rtol=1e-5, atol=atol)


def _gaussian_pulse_p(
    f: Fields,
    n: int,
    *,
    ic: int,
    jc: int,
    kc: int,
    sigma: float,
) -> float:
    ii, jj, kk = np.ogrid[:n, :n, :n]
    blob = np.exp(
        -((ii - ic) ** 2 + (jj - jc) ** 2 + (kk - kc) ** 2) / (2.0 * sigma**2)
    ).astype(_FLOAT32)
    f.p[...] = blob
    f.vx[...] = 0.0
    f.vy[...] = 0.0
    f.vz[...] = 0.0
    return float(np.max(blob))


def _x_cell_center_m(i: int, n: int, h: float) -> float:
    return (i + 0.5 - 0.5 * n) * h


def _first_wall_reflection_time_s(
    *,
    n: int,
    h: float,
    c_air: float,
    i_src: int,
    i_rec: int,
    margin: float = 0.9,
) -> float:
    """Earliest round-trip from source to any outer x face and back to receiver."""
    x_src = _x_cell_center_m(i_src, n, h)
    x_rec = _x_cell_center_m(i_rec, n, h)
    x_neg = -0.5 * n * h
    x_pos = 0.5 * n * h
    t_neg = ((x_src - x_neg) + (x_rec - x_neg)) / c_air
    t_pos = ((x_pos - x_src) + (x_pos - x_rec)) / c_air
    return margin * min(t_neg, t_pos)


def _reflection_db(ts: np.ndarray, tl: np.ndarray) -> float:
    err = ts - tl
    e_ref = float(np.sum(tl**2) + 1e-30)
    e_err = float(np.sum(err**2))
    return float(10.0 * np.log10(e_err / e_ref))


def _record_trace(p_hist: list[np.ndarray], f, ic: int, jc: int, kc: int) -> None:
    p_hist.append(np.float32(f.p[ic, jc, kc]))


def reflection_db(
    cpml_cells: int,
    R: float,
    m: float,
    kappa_max: float,
    *,
    sigma: float = 3.0,
    recv_inset: int = 2,
) -> tuple[float, float]:
    """Small-domain CPML vs large reference; return (db_cpml, db_rigid) over [0, T_wall)."""
    n_small = 2 * cpml_cells + 24
    n_large = 2 * cpml_cells + 120
    g_s, d_s, cfg_s, _ = _grid(
        n_small, cpml=cpml_cells, cpml_R=R, cpml_m=m, cpml_kappa_max=kappa_max,
    )
    g_l, d_l, cfg_l, _ = _grid(
        n_large, cpml=cpml_cells, cpml_R=R, cpml_m=m, cpml_kappa_max=kappa_max,
    )
    coeffs_s = make_update_coeffs(cfg_s.air.rho, cfg_s.air.c, d_s.h, d_s.dt)
    coeffs_l = make_update_coeffs(cfg_l.air.rho, cfg_l.air.c, d_l.h, d_l.dt)
    cpml_s = make_cpml_state(g_s, cfg_s, d_s, d_s.dt)
    cpml_l = make_cpml_state(g_l, cfg_l, d_l, d_l.dt)
    slabs_s = g_s.cpml_slabs
    slabs_l = g_l.cpml_slabs

    i_src_s = n_small // 2
    j_src_s = n_small // 2
    k_src_s = n_small // 2
    i_rec_s = cpml_cells + recv_inset
    j_rec_s = n_small // 2
    k_rec_s = n_small // 2
    di = i_rec_s - i_src_s
    dj = j_rec_s - j_src_s
    dk = k_rec_s - k_src_s

    i_src_l = n_large // 2
    j_src_l = n_large // 2
    k_src_l = n_large // 2
    i_rec_l = i_src_l + di
    j_rec_l = j_src_l + dj
    k_rec_l = k_src_l + dk

    t_wall_s = _first_wall_reflection_time_s(
        n=n_large,
        h=d_l.h,
        c_air=cfg_l.air.c,
        i_src=i_src_l,
        i_rec=i_rec_l,
    )
    n_steps = max(32, int(t_wall_s / d_l.dt))

    fs_cpml = allocate_fields(g_s)
    fs_rigid = allocate_fields(g_s)
    fl = allocate_fields(g_l)
    _gaussian_pulse_p(
        fs_cpml, n_small, ic=i_src_s, jc=j_src_s, kc=k_src_s, sigma=sigma,
    )
    _copy_fields(fs_rigid, fs_cpml)
    _gaussian_pulse_p(
        fl, n_large, ic=i_src_l, jc=j_src_l, kc=k_src_l, sigma=sigma,
    )

    hist_cpml: list[np.ndarray] = []
    hist_rigid: list[np.ndarray] = []
    hist_l: list[np.ndarray] = []
    for _ in range(n_steps):
        step(fs_cpml, coeffs_s, slabs_s, cpml=cpml_s)
        step(fs_rigid, coeffs_s, slabs_s, cpml=None)
        step(fl, coeffs_l, slabs_l, cpml=cpml_l)
        _record_trace(hist_cpml, fs_cpml, i_rec_s, j_rec_s, k_rec_s)
        _record_trace(hist_rigid, fs_rigid, i_rec_s, j_rec_s, k_rec_s)
        _record_trace(hist_l, fl, i_rec_l, j_rec_l, k_rec_l)

    ts_cpml = np.asarray(hist_cpml, dtype=np.float64)
    ts_rigid = np.asarray(hist_rigid, dtype=np.float64)
    tl = np.asarray(hist_l, dtype=np.float64)
    return _reflection_db(ts_cpml, tl), _reflection_db(ts_rigid, tl)


def test_reflection_regression_guard():
    db_cpml, db_rigid = reflection_db(8, 1e-6, 3.0, 1.0)
    msg = (
        f"reflection error vs large reference over [0, T_wall): "
        f"CPML {db_cpml:.2f} dB, rigid {db_rigid:.2f} dB"
    )
    assert db_cpml < -18.0, msg  # regression guard; harsh grazing geometry. Real characterization in C3-cpml.
    assert db_rigid > -10.0, msg


@pytest.mark.slow
def test_cpml_reflection_parameter_sweep():
    """Diagnostic sweep; run with ``pytest -m slow -s`` to print the full table."""
    cpml_list = (8, 12, 16)
    r_list = (1e-3, 1e-4, 1e-6)
    m_list = (2, 3)
    table_rows: list[str] = []
    db_by_key: dict[tuple[int, float, int], float] = {}

    for cpml_cells in cpml_list:
        for R in r_list:
            for m in m_list:
                db_cpml, _ = reflection_db(cpml_cells, R, float(m), 1.0)
                db_by_key[(cpml_cells, R, m)] = db_cpml
                row = f"cpml={cpml_cells:2d}  R={R:.0e}  m={m}  db_cpml={db_cpml:8.2f} dB"
                table_rows.append(row)
                print(row)

    db_8 = db_by_key[(8, 1e-4, 3)]
    db_16 = db_by_key[(16, 1e-4, 3)]
    table = "\n".join(table_rows)
    summary = (
        f"at R=1e-4, m=3: db_cpml(8)={db_8:.2f} dB, db_cpml(16)={db_16:.2f} dB "
        f"(expect 16-cell < 8-cell)\n--- sweep ---\n{table}"
    )
    assert db_16 < db_8, summary


def test_long_run_decay():
    n = 28
    cpml = 6
    g, d, cfg, _ = _grid(n, cpml=cpml)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    slabs = g.cpml_slabs
    f = allocate_fields(g)
    c = n // 2
    peak = _gaussian_pulse_p(
        f, n, ic=c, jc=c, kc=c, sigma=3.0,
    )

    n_total = 5000
    tail = 1000
    tail_max = 0.0
    for t in range(n_total):
        step(f, coeffs, slabs, cpml=cpml_st)
        if t >= n_total - tail:
            tail_max = max(tail_max, float(np.max(np.abs(f.p))))

    assert tail_max < 1e-3 * peak


def test_sigma_max_dt_at_outer_wall_in_range():
    """σ_max uses L = cpml_cells·h (m); σ·Δt at d/L=1 lies in a sensible CFS band."""
    cpml = 8
    n = 40
    g, d, cfg, _ = _grid(n, cpml=cpml)
    coeffs = build_cpml_coeffs(g, cfg, d, d.dt)
    L_m = cpml * d.h
    m = cfg.numerics.cpml_m
    expected = -(m + 1.0) * cfg.air.c * math.log(cfg.numerics.cpml_R) / (2.0 * L_m)
    assert coeffs.sigma_max == pytest.approx(expected, rel=1e-6)
    sigma_dt = coeffs.sigma_max * d.dt
    assert 0.3 <= sigma_dt <= 10.0


def test_profile_orientation_monotone_to_wall():
    """Profile scalars vanish at d/L=0; *v* has b=1,a=0 there; *p* at d/L=1/(2n) does not."""
    cpml = 6
    n = 24
    g, d, cfg, _ = _grid(n, cpml=cpml)
    m = cfg.numerics.cpml_m
    sigma_max = build_cpml_coeffs(g, cfg, d, d.dt).sigma_max
    kappa_max = cfg.numerics.cpml_kappa_max
    alpha_max = d.cpml_alpha

    def _raw(d_over_L: float) -> tuple[float, float, float]:
        power = d_over_L**m
        sigma = sigma_max * power
        kappa_m1 = (kappa_max - 1.0) * power
        one_minus_alpha_norm = 1.0 - d_over_L
        return sigma, kappa_m1, one_minus_alpha_norm

    for neg in (True, False):
        b_p, a_p, ik_p = _profile_row(
            cpml, neg=neg, sigma_max=sigma_max, kappa_max=kappa_max,
            alpha_max=alpha_max, m=m, dt=d.dt,
        )
        b_v, a_v, ik_v = _profile_row_v(
            cpml, neg=neg, sigma_max=sigma_max, kappa_max=kappa_max,
            alpha_max=alpha_max, m=m, dt=d.dt,
        )
        assert _raw(0.0)[0] == pytest.approx(0.0)
        assert _raw(0.0)[1] == pytest.approx(0.0)
        assert _raw(0.0)[2] == pytest.approx(1.0)
        assert _raw(1.0)[0] == pytest.approx(sigma_max, rel=1e-5)
        if kappa_max <= 1.0:
            assert np.all(ik_p == 0.0)
            assert np.all(ik_v == 0.0)
        elif neg:
            assert float(ik_p[0]) > float(ik_p[-1])
            assert float(ik_v[0]) > float(ik_v[-1])
        else:
            assert float(ik_p[-1]) > float(ik_p[0])
            assert float(ik_v[-1]) > float(ik_v[0])

        q_if_p = cpml - 1 if neg else 0
        q_if_v = cpml if neg else 0
        d_p_if = 1.0 / (2.0 * cpml)
        b_p_if, a_p_if = _profile_b_a_at_d(
            d_p_if,
            sigma_max=sigma_max,
            kappa_max=kappa_max,
            alpha_max=alpha_max,
            m=m,
            dt=d.dt,
        )
        assert b_p_if < 1.0
        assert a_p_if < 0.0
        _assert_profile_b_a_monotone(
            b_p,
            a_p,
            neg=neg,
            interface_q=q_if_p,
            interface_b=b_p_if,
            interface_a=a_p_if,
        )
        
        # v interface face: σ = 0 but α = α_max there, so b = exp(-α_max·dt), a = 0
        b_v_if, a_v_if = _profile_b_a_at_d(
            0.0, sigma_max=sigma_max, kappa_max=kappa_max,
            alpha_max=alpha_max, m=m, dt=d.dt,
        )
        _assert_profile_b_a_monotone(
            b_v,
            a_v,
            neg=neg,
            interface_q=q_if_v,
            interface_b=b_v_if,
            interface_a=a_v_if,
        )


def test_sigma_v_zero_at_interface_on_all_faces():
    """Velocity σ vanishes at the CPML/interior interface and peaks at the wall."""
    cpml = 16
    n = 48
    g, d, cfg, _ = _grid(n, cpml=cpml, cpml_alpha_factor=0.0)
    sigma_max = build_cpml_coeffs(g, cfg, d, d.dt).sigma_max
    m = cfg.numerics.cpml_m
    faces = (
        ("x_neg", True),
        ("x_pos", False),
        ("y_neg", True),
        ("y_pos", False),
        ("z_neg", True),
        ("z_pos", False),
    )
    for _name, neg in faces:
        sig = _sigma_v_profile(cpml, neg, sigma_max, m)
        q_if = cpml if neg else 0
        q_wall = 0 if neg else cpml
        assert sig[q_if] == pytest.approx(0.0, abs=1e-30)
        assert sig[q_wall] == pytest.approx(sigma_max, rel=1e-6)


def test_p_v_depths_interleave_uniformly():
    cpml = 16
    spacing = 1.0 / (2 * cpml)
    for neg in (True, False):
        merged = np.sort(
            np.concatenate([_depths_p(cpml, neg), _depths_v(cpml, neg)])
        )
        assert merged[0] == pytest.approx(0.0, abs=1e-12)
        assert merged[-1] == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(np.diff(merged), spacing, rtol=0.0, atol=1e-12)


def _init_x_plane_wave(
    f: Fields,
    *,
    nx: int,
    ny: int,
    nz: int,
    i_src: int,
    sigma_cells: float,
    rho: float,
    c_air: float,
) -> None:
    """Gaussian envelope in x, uniform over y–z, travelling +x (p and vx matched)."""
    ii = np.arange(nx, dtype=np.float64)
    env = np.exp(-0.5 * ((ii - i_src) / sigma_cells) ** 2).astype(_FLOAT32)
    z = np.float32(1.0 / (rho * c_air))
    for j in range(ny):
        for k in range(nz):
            f.p[:, j, k] = env
            f.vy[:, j, k] = 0.0
            f.vz[:, j, k] = 0.0
            f.vx[0, j, k] = 0.0
            for i in range(nx):
                f.vx[i + 1, j, k] = z * f.p[i, j, k]
            f.vx[nx, j, k] = z * f.p[nx - 1, j, k]


_DUCT_WIN_TAIL_CELLS = 10.0
_DUCT_END_TAPER_FRAC = 0.25
_DUCT_FFT_PAD_FACTOR = 8


def _causal_end_taper(n: int, *, taper_frac: float = _DUCT_END_TAPER_FRAC) -> np.ndarray:
    """Unit gain at the start; raised-cosine taper on the last samples only (C3-style)."""
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    nt = int(max(1, math.floor(taper_frac * n)))
    nt = min(nt, n - 1)
    if nt <= 0:
        return w
    x = np.linspace(0.0, math.pi, nt)
    w[-nt:] = 0.5 * (1.0 + np.cos(x))
    return w


def _pulse_arrival_window_s(
    t_arrival: float, sigma_t: float, *, margin_s: float
) -> tuple[float, float]:
    """``[arrival − 3σ_t, arrival + 3σ_t + margin]``."""
    return t_arrival - 3.0 * sigma_t, t_arrival + 3.0 * sigma_t + margin_s


def _windows_overlap(win_a: tuple[float, float], win_b: tuple[float, float]) -> bool:
    return win_a[1] > win_b[0] and win_b[1] > win_a[0]


def _duct_place_probe_clear_gates(
    *,
    nx: int,
    cpml: int,
    i_src: int,
    i_rec: int,
    h: float,
    c_air: float,
    sigma_t: float,
    margin_s: float,
    min_src_rec_cells: int = 32,
) -> tuple[int, int, float, float, tuple[float, float], tuple[float, float]]:
    """Return ``(nx, i_rec)`` so incident and reflection gate windows do not overlap."""
    min_sep = 6.0 * sigma_t + margin_s
    t_inc0 = (i_rec - i_src) * h / c_air
    t_refl0 = t_inc0 + 2.0 * (nx - cpml - i_rec) * h / c_air
    win_i0 = _pulse_arrival_window_s(t_inc0, sigma_t, margin_s=margin_s)
    win_r0 = _pulse_arrival_window_s(t_refl0, sigma_t, margin_s=margin_s)

    nx_out, i_rec_out = nx, i_rec
    adjusted = False
    for _ in range(10_000):
        t_inc = (i_rec_out - i_src) * h / c_air
        dist_pos_if = (nx_out - cpml - i_rec_out) * h
        t_refl = t_inc + 2.0 * dist_pos_if / c_air
        win_i = _pulse_arrival_window_s(t_inc, sigma_t, margin_s=margin_s)
        win_r = _pulse_arrival_window_s(t_refl, sigma_t, margin_s=margin_s)
        if not _windows_overlap(win_i, win_r):
            if adjusted:
                print(
                    "duct CPML: cleared overlapping gates — "
                    f"nx={nx_out}, i_rec={i_rec_out}, "
                    f"t_inc={t_inc * 1e3:.3f} ms, t_refl={t_refl * 1e3:.3f} ms"
                )
            return nx_out, i_rec_out, t_inc, t_refl, win_i, win_r
        moved = False
        min_i_rec = i_src + min_src_rec_cells
        if i_rec_out > min_i_rec:
            i_rec_out -= 1
            moved = True
            adjusted = True
            print(
                "duct CPML: gate overlap — moving probe toward −x "
                f"(i_rec → {i_rec_out}, larger +x round-trip)"
            )
        else:
            nx_out += max(32, int(math.ceil(min_sep * c_air / (2.0 * h))))
            if i_rec_out < nx_out - cpml - 16:
                i_rec_out = nx_out - cpml - 16
            moved = True
            adjusted = True
            print(
                "duct CPML: gate overlap — lengthening duct "
                f"(nx → {nx_out}, i_rec → {i_rec_out})"
            )
        if not moved:
            break

    raise AssertionError(
        "duct gate windows overlap after geometry adjustment; "
        f"inc {[win_i0[0], win_i0[1]]} vs refl {[win_r0[0], win_r0[1]]}"
    )


def _gated_segment_causal(
    trace: np.ndarray, dt: float, t_lo: float, t_hi: float
) -> np.ndarray:
    """Sample ``[t_lo, t_hi)`` with causal end taper only (window length not rounded)."""
    n = len(trace)
    i0 = max(0, int(math.floor(t_lo / dt)))
    i1 = min(n, int(math.ceil(t_hi / dt)))
    if i1 <= i0:
        i1 = min(n, i0 + 1)
    seg = trace[i0:i1].astype(np.float64)
    return seg * _causal_end_taper(len(seg))


def _rfft_pair_zero_padded(
    seg_inc: np.ndarray, seg_refl: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-pad both segments to a common ``n_fft`` (≥ pad factor × longest window)."""
    n_long = max(len(seg_inc), len(seg_refl), 1)
    n_fft = max(64, _DUCT_FFT_PAD_FACTOR * n_long)
    pad_inc = np.zeros(n_fft, dtype=np.float64)
    pad_refl = np.zeros(n_fft, dtype=np.float64)
    pad_inc[: len(seg_inc)] = seg_inc
    pad_refl[: len(seg_refl)] = seg_refl
    freqs = np.fft.rfftfreq(n_fft, dt)
    return freqs, np.fft.rfft(pad_inc), np.fft.rfft(pad_refl)


def _duct_reflection_spectrum(
    cpml_R: float,
    *,
    f_lo: float = 200.0,
    f_hi: float = 2_500.0,
) -> tuple[np.ndarray, np.ndarray]:
    cpml = 16
    nx_init, ny, nz = 400, 4, 4
    sigma_cells = 8.0
    i_src = cpml + 48

    g, d, cfg, _ = _grid_aniso(
        nx_init,
        ny,
        nz,
        cpml=cpml,
        cpml_R=cpml_R,
        cpml_m=3.0,
        cpml_kappa_max=1.0,
        cpml_alpha_factor=0.0,
    )
    h = d.h
    c_air = cfg.air.c
    dt = d.dt
    sigma_t = sigma_cells * h / c_air
    margin_s = _DUCT_WIN_TAIL_CELLS * h / c_air
    i_rec_init = nx_init - cpml - 20
    nx, i_rec, t_inc, t_refl, win_inc, win_refl = _duct_place_probe_clear_gates(
        nx=nx_init,
        cpml=cpml,
        i_src=i_src,
        i_rec=i_rec_init,
        h=h,
        c_air=c_air,
        sigma_t=sigma_t,
        margin_s=margin_s,
    )
    if nx != nx_init:
        g, d, cfg, _ = _grid_aniso(
            nx,
            ny,
            nz,
            cpml=cpml,
            cpml_R=cpml_R,
            cpml_m=3.0,
            cpml_kappa_max=1.0,
            cpml_alpha_factor=0.0,
        )
        h = d.h
        dt = d.dt
        sigma_t = sigma_cells * h / c_air
        margin_s = _DUCT_WIN_TAIL_CELLS * h / c_air
        nx, i_rec, t_inc, t_refl, win_inc, win_refl = _duct_place_probe_clear_gates(
            nx=nx,
            cpml=cpml,
            i_src=i_src,
            i_rec=i_rec,
            h=h,
            c_air=c_air,
            sigma_t=sigma_t,
            margin_s=margin_s,
        )

    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    slabs = g.cpml_slabs
    f = allocate_fields(g)
    _init_x_plane_wave(
        f,
        nx=nx,
        ny=ny,
        nz=nz,
        i_src=i_src,
        sigma_cells=sigma_cells,
        rho=cfg.air.rho,
        c_air=cfg.air.c,
    )

    t_record_end = win_refl[1] + margin_s
    n_steps = max(256, int(math.ceil(t_record_end / dt)) + 8)
    assert n_steps * dt >= win_refl[1], "simulation ends before reflected gate window"

    hist: list[float] = []
    for _ in range(n_steps):
        _step_cpml_axes(f, coeffs, slabs, cpml_st, _CPML_AXIS_X_ONLY)
        hist.append(float(f.p[i_rec, ny // 2, nz // 2]))

    trace = np.asarray(hist, dtype=np.float64)
    seg_inc = _gated_segment_causal(trace, dt, win_inc[0], win_inc[1])
    seg_refl = _gated_segment_causal(trace, dt, win_refl[0], win_refl[1])
    freqs, spec_i, spec_r = _rfft_pair_zero_padded(seg_inc, seg_refl, dt)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    ratio = np.abs(spec_r[band]) / (np.abs(spec_i[band]) + 1e-30)
    r_db = 20.0 * np.log10(ratio)
    return freqs[band], r_db


@pytest.mark.slow
def test_duct_plane_wave_x_cpml_reflection():
    """1-D duct with CPML on ±x only; spectral reflection ≤ −50 dB at R=1e-4."""
    results: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for cpml_R in (1e-4, 1e-3, 1e-6):
        freqs, r_db = _duct_reflection_spectrum(cpml_R)
        results[cpml_R] = (freqs, r_db)
        lines = [
            f"  f={f:.0f} Hz  R(f)={db:6.2f} dB"
            for f, db in zip(freqs, r_db)
        ]
        print(f"duct CPML R={cpml_R:.0e} (max {float(np.max(r_db)):.2f} dB)\n"
              + "\n".join(lines[:12])
              + (f"\n  ... ({len(lines)} freqs total)" if len(lines) > 12 else ""))

    _f, r_main = results[1e-4]
    assert float(np.max(r_main)) <= -50.0, (
        f"duct +x CPML reflection {float(np.max(r_main)):.2f} dB exceeds −50 dB"
    )
    max_3 = float(np.max(results[1e-3][1]))
    max_6 = float(np.max(results[1e-6][1]))
    assert max_6 <= max_3 + 2.0, (
        f"σ_max sweep worsens with smaller R: 1e-6 → {max_6:.2f} dB, "
        f"1e-3 → {max_3:.2f} dB"
    )


def test_vx_neg_stencil_index_mapping_delta():
    """A single ∂p stencil drives ψ and vx correction at the matching global face."""
    n = 14
    cpml = 4
    g, d, cfg, _ = _grid(n, cpml=cpml)
    coeffs_u = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    f = allocate_fields(g)
    j0 = n // 2
    k0 = n // 2
    i_face = 2
    i_hi = i_face
    p_lo = np.float32(-1.0)
    p_hi = np.float32(2.0)
    f.p[i_hi, j0, k0] = p_hi
    f.p[i_hi - 1, j0, k0] = p_lo
    dp = float(p_hi - p_lo)
    q = i_face
    r = cpml_st.rows
    ps0 = float(f.psi_vx_neg[q, j0, k0])
    apply_v(f, cpml_st, coeffs_u)
    expected_ps = float(r.b_v_neg[q] * ps0 + r.a_v_neg[q] * dp)
    expected_dvx = -float(coeffs_u.cv) * (float(r.ik_v_neg[q]) * dp + expected_ps)
    assert f.psi_vx_neg[q, j0, k0] == pytest.approx(np.float32(expected_ps))
    assert f.vx[i_face, j0, k0] == pytest.approx(np.float32(expected_dvx))
    assert f.vx[i_face, j0, k0] != 0.0


def test_step_cpml_differs_from_plain_in_all_six_slabs():
    n = 22
    cpml = 4
    g, d, cfg, _ = _grid(n, cpml=cpml)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    slabs = g.cpml_slabs
    mid = n // 2

    slab_cells = [
        ("p_x_neg", lambda a, b: (a.p[cpml // 2, mid, mid], b.p[cpml // 2, mid, mid])),
        ("p_x_pos", lambda a, b: (
            a.p[n - cpml + cpml // 2, mid, mid],
            b.p[n - cpml + cpml // 2, mid, mid],
        )),
        ("p_y_neg", lambda a, b: (a.p[mid, cpml // 2, mid], b.p[mid, cpml // 2, mid])),
        ("p_y_pos", lambda a, b: (
            a.p[mid, n - cpml + cpml // 2, mid],
            b.p[mid, n - cpml + cpml // 2, mid],
        )),
        ("p_z_neg", lambda a, b: (a.p[mid, mid, cpml // 2], b.p[mid, mid, cpml // 2])),
        ("p_z_pos", lambda a, b: (
            a.p[mid, mid, n - cpml + cpml // 2],
            b.p[mid, mid, n - cpml + cpml // 2],
        )),
    ]

    f_on = allocate_fields(g)
    f_off = allocate_fields(g)
    rng = np.random.default_rng(21)
    for arr in (f_on.p, f_on.vx, f_on.vy, f_on.vz):
        arr[...] = rng.uniform(-0.12, 0.12, size=arr.shape).astype(_FLOAT32)
    _copy_fields(f_off, f_on)

    for _ in range(4):
        step(f_on, coeffs, slabs, cpml=cpml_st)
        step(f_off, coeffs, slabs, cpml=None)

    diffs = []
    for name, pick in slab_cells:
        va, vb = pick(f_on, f_off)
        diffs.append((name, float(va - vb)))
    assert any(abs(dv) > 1e-9 for _, dv in diffs), f"no slab difference: {diffs}"


def test_psi_shapes_and_float32():
    n = 16
    cpml = 4
    g, d, cfg, _ = _grid(n, cpml=cpml)
    f = allocate_fields(g)
    assert f.psi_p_x_neg.shape == (cpml, n, n)
    assert f.psi_vx_neg.shape == (cpml + 1, n, n)
    assert f.psi_p_x_neg.dtype == _FLOAT32
    coeffs = build_cpml_coeffs(g, cfg, d, d.dt)
    rows = face_rows(coeffs, cfg, d, d.dt)
    assert coeffs.b_p.dtype == _FLOAT32
    assert rows.b_p_pos.dtype == _FLOAT32
