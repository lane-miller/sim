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
from fdtd.kernels import make_update_coeffs, step, step_numpy  # noqa: E402

_FLOAT32 = np.dtype(np.float32)


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


def _x_cell_centre_m(i: int, n: int, h: float) -> float:
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
    x_src = _x_cell_centre_m(i_src, n, h)
    x_rec = _x_cell_centre_m(i_rec, n, h)
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
    assert db_cpml < -18.0, msg  # regression guard; harsh grazing geometry. Real characterisation in C3-cpml.
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
    """σ, κ−1, (1−α/α_max) vanish at the interior interface and peak at the wall."""
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
            cpml + 1, neg=neg, sigma_max=sigma_max, kappa_max=kappa_max,
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
        if neg:
            assert float(a_p[0]) < float(a_p[-1])
            assert float(b_p[0]) < float(b_p[-1])
            assert float(a_v[0]) < float(a_v[-1])
        else:
            assert float(a_p[-1]) < float(a_p[0])
            assert float(b_p[-1]) < float(b_p[0])
            assert float(a_v[-1]) < float(a_v[0])


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
