"""Cut-cell kernel gate tests (step 6a)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config, derive
from boxrad.cpml import make_cpml_state
from boxrad.cutcell import Fractions, Grid as CutGrid
from boxrad.cutcell import compute_fractions
from boxrad.cutkernel import (
    FastRadiator,
    PatchRadiator,
    apply_open_mask,
    assert_geometry_clear_of_cpml,
    CutOperator,
    build_operator,
    step_cut,
    step_cut_dense,
    update_p_cut,
)
from boxrad.geom import geometry
from boxrad.grid import Fields, Grid, allocate_fields
from boxrad.kernels import UpdateCoeffs, make_update_coeffs, step, update_v, zero_rigid_face_velocities
from boxrad.smallcell import apply_floor
from boxrad.source import BoxBoundary

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_F32 = np.float32


def _copy_fields(fields: Fields) -> Fields:
    return Fields(
        p=fields.p.copy(),
        vx=fields.vx.copy(),
        vy=fields.vy.copy(),
        vz=fields.vz.copy(),
        psi_p_x_neg=fields.psi_p_x_neg.copy(),
        psi_p_x_pos=fields.psi_p_x_pos.copy(),
        psi_p_y_neg=fields.psi_p_y_neg.copy(),
        psi_p_y_pos=fields.psi_p_y_pos.copy(),
        psi_p_z_neg=fields.psi_p_z_neg.copy(),
        psi_p_z_pos=fields.psi_p_z_pos.copy(),
        psi_vx_neg=fields.psi_vx_neg.copy(),
        psi_vx_pos=fields.psi_vx_pos.copy(),
        psi_vy_neg=fields.psi_vy_neg.copy(),
        psi_vy_pos=fields.psi_vy_pos.copy(),
        psi_vz_neg=fields.psi_vz_neg.copy(),
        psi_vz_pos=fields.psi_vz_pos.copy(),
    )


def _zero_psi(nx: int, ny: int, nz: int) -> dict[str, np.ndarray]:
    """Minimal CPML-sized ψ buffers (unused when cpml is None)."""
    from boxrad.grid import build_cpml_slabs

    slabs = build_cpml_slabs(nx, ny, nz, 1)
    s = slabs.p_x_neg.hi
    return {
        "psi_p_x_neg": np.zeros((s, ny, nz), dtype=_F32),
        "psi_p_x_pos": np.zeros((s, ny, nz), dtype=_F32),
        "psi_p_y_neg": np.zeros((nx, s, nz), dtype=_F32),
        "psi_p_y_pos": np.zeros((nx, s, nz), dtype=_F32),
        "psi_p_z_neg": np.zeros((nx, ny, s), dtype=_F32),
        "psi_p_z_pos": np.zeros((nx, ny, s), dtype=_F32),
        "psi_vx_neg": np.zeros((s + 1, ny, nz), dtype=_F32),
        "psi_vx_pos": np.zeros((s + 1, ny, nz), dtype=_F32),
        "psi_vy_neg": np.zeros((nx, s + 1, nz), dtype=_F32),
        "psi_vy_pos": np.zeros((nx, s + 1, nz), dtype=_F32),
        "psi_vz_neg": np.zeros((nx, ny, s + 1), dtype=_F32),
        "psi_vz_pos": np.zeros((nx, ny, s + 1), dtype=_F32),
    }


def _c9_cut_grid(grid: Grid, derived, cfg: Config) -> CutGrid:
    xc, yc = cfg.radiator.x_c, cfg.radiator.y_c
    W, H, D = derived.realized_W, derived.realized_H, derived.realized_D
    ib, jb, kb = grid.i_box, grid.j_box, grid.k_box
    h = derived.h
    x0 = np.array(
        [-xc - 0.5 * W - ib * h, -yc - 0.5 * H - jb * h, -D - kb * h],
        dtype=np.float64,
    )
    return CutGrid(
        x0=x0,
        h=h,
        shape=(grid.nx, grid.ny, grid.nz),
        rigid_sides={},
    )


def _c9_setup():
    cfg = Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8"))
    derived = derive(cfg)
    grid = Grid(cfg, derived)
    fields = allocate_fields(grid)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, derived.h, derived.dt)
    slabs = grid.cpml_slabs
    cpml = make_cpml_state(grid, cfg, derived, derived.dt)
    boundary = BoxBoundary.from_grid(grid, derived)
    geom = geometry.from_box(cfg)
    cgrid = _c9_cut_grid(grid, derived, cfg)
    fr = compute_fractions(geom, cgrid)
    beta_t = fr.beta.astype(np.float32)
    ir, jr, kr = grid.radiator_face_indices
    vz_shape = (grid.nx, grid.ny, grid.nz + 1)
    rad_idx = np.ravel_multi_index((ir, jr, kr), vz_shape).astype(np.int64)
    sw = grid.radiator_weights.astype(np.float32)
    fast = [FastRadiator(axis=2, idx=rad_idx, sw=sw)]
    op = build_operator(fr, beta_t, fast, [], derived.h)
    assert_geometry_clear_of_cpml(op, slabs)
    return cfg, derived, grid, fields, coeffs, slabs, cpml, boundary, op, fr


def _exterior_p_mask(grid: Grid, fr: Fractions) -> np.ndarray:
    ib, jb, kb = grid.i_box, grid.j_box, grid.k_box
    interior = np.zeros_like(fr.beta, dtype=bool)
    interior[ib : ib + grid.nbx, jb : jb + grid.nby, kb : kb + grid.nbz] = True
    return (fr.beta > 0) & ~interior


def test_c12a_bit_identity():
    _, _, grid, fields, coeffs, slabs, cpml, boundary, op, fr = _c9_setup()
    leg = _copy_fields(fields)
    cut = _copy_fields(fields)
    for n in range(400):
        v_now = boundary.v_at_step(n)
        step(leg, coeffs, slabs, cpml, boundary, n)
        step_cut(cut, coeffs, op, cpml, [v_now])
    mask_p = _exterior_p_mask(grid, fr)
    assert np.array_equal(leg.p[mask_p], cut.p[mask_p])
    for a, b, o in (
        (leg.vx, cut.vx, op.open_x),
        (leg.vy, cut.vy, op.open_y),
        (leg.vz, cut.vz, op.open_z),
    ):
        assert np.array_equal(a[o > 0], b[o > 0])


def _fast_from_cutcell(
    fr: Fractions, grid: Grid, derived, legacy: FastRadiator
) -> FastRadiator:
    """Legacy face indices with cutcell exact area weights ``a/h²``."""
    h2 = derived.h * derived.h
    n1 = grid.nz + 1
    ny, nz = grid.ny, grid.nz
    by_ij: dict[tuple[int, int], float] = {}
    for c, a in zip(fr.radiator_cell[0], fr.radiator_a[0]):
        i = int(c) // (ny * nz)
        j = (int(c) // nz) % ny
        by_ij[(i, j)] = float(a)
    sw = np.empty(legacy.idx.shape[0], dtype=np.float32)
    for q, idx in enumerate(legacy.idx):
        rem = int(idx) % (ny * n1)
        i = int(idx) // (ny * n1)
        j = rem // n1
        sw[q] = _F32(_F32(by_ij.get((i, j), 0.0)) / _F32(h2))
    return FastRadiator(axis=legacy.axis, idx=legacy.idx, sw=sw)


def test_c12a_exact_weights():
    cfg, derived, grid, fields, coeffs, slabs, cpml, boundary, op_legacy, fr = _c9_setup()
    fast_exact = _fast_from_cutcell(fr, grid, derived, op_legacy.radiators_fast[0])
    sw_legacy = op_legacy.radiators_fast[0].sw.astype(np.float64)
    sw_exact = fast_exact.sw.astype(np.float64)
    max_dsw = float(np.max(np.abs(sw_exact - sw_legacy)))
    s_legacy = float(np.sum(sw_legacy))
    dss = float(np.abs(np.sum(sw_exact) - s_legacy)) / max(s_legacy, 1e-30)
    assert max_dsw <= 2e-6
    assert dss <= 1e-6
    op_exact = build_operator(
        fr, fr.beta.astype(np.float32), [fast_exact], [], derived.h
    )
    cut_legacy_w = _copy_fields(fields)
    cut_exact_w = _copy_fields(fields)
    for n in range(400):
        v_now = boundary.v_at_step(n)
        step_cut(cut_legacy_w, coeffs, op_legacy, cpml, [v_now])
        step_cut(cut_exact_w, coeffs, op_exact, cpml, [v_now])
    mask_p = _exterior_p_mask(grid, fr)
    diff = cut_exact_w.p[mask_p].astype(np.float64) - cut_legacy_w.p[mask_p].astype(np.float64)
    den = float(np.linalg.norm(cut_legacy_w.p[mask_p].astype(np.float64)))
    p_rel = float(np.linalg.norm(diff)) / max(den, 1e-30)
    print(f"max|dsw|={max_dsw:.6e} dS/S={dss:.6e} p_rel={p_rel:.6e}")
    assert p_rel <= 5e-5


def _assert_sparse_dense_equal(dense: Fields, sparse: Fields, op: CutOperator, mask_p: np.ndarray):
    assert np.array_equal(dense.p[mask_p], sparse.p[mask_p])
    for a, b, o in (
        (dense.vx, sparse.vx, op.open_x),
        (dense.vy, sparse.vy, op.open_y),
        (dense.vz, sparse.vz, op.open_z),
    ):
        assert np.array_equal(a[o > 0], b[o > 0])


def test_sparse_vs_dense_c9():
    _, _, grid, fields, coeffs, _, cpml, boundary, op, fr = _c9_setup()
    dense = _copy_fields(fields)
    sparse = _copy_fields(fields)
    mask_p = _exterior_p_mask(grid, fr)
    for n in range(400):
        v_now = boundary.v_at_step(n)
        step_cut_dense(dense, coeffs, op, cpml, [v_now])
        step_cut(sparse, coeffs, op, cpml, [v_now])
    _assert_sparse_dense_equal(dense, sparse, op, mask_p)


def test_sparse_vs_dense_t3():
    fr = _energy_fractions()
    lam = 0.99 / math.sqrt(3.0)
    beta_t, _ = apply_floor(fr, lam, h=1.0)
    op = build_operator(fr, beta_t.astype(np.float32), [], [], 1.0)
    coeffs = UpdateCoeffs(cv=_F32(lam), cp=_F32(lam))
    psi = _zero_psi(14, 14, 14)
    shape = fr.beta.shape
    fluid = fr.beta > 0
    rng = np.random.default_rng(0)
    p0 = np.zeros(shape, dtype=np.float32)
    p0[fluid] = rng.standard_normal(int(np.sum(fluid))).astype(np.float32)
    fields = Fields(
        p=p0,
        vx=np.zeros((15, 14, 14), dtype=np.float32),
        vy=np.zeros((14, 15, 14), dtype=np.float32),
        vz=np.zeros((14, 14, 15), dtype=np.float32),
        **psi,
    )
    dense = _copy_fields(fields)
    sparse = _copy_fields(fields)
    mask_p = fr.beta > 0
    for _ in range(500):
        step_cut_dense(dense, coeffs, op, None, [])
        step_cut(sparse, coeffs, op, None, [])
    _assert_sparse_dense_equal(dense, sparse, op, mask_p)


def test_c9_step_timing_report():
    """Report legacy vs sparse vs dense ms/step on C9 (no assert)."""
    import time

    _, _, _, fields, coeffs, slabs, cpml, boundary, op, _ = _c9_setup()
    n_steps = 50
    t0 = time.perf_counter()
    f = _copy_fields(fields)
    for n in range(n_steps):
        step(f, coeffs, slabs, cpml, boundary, n)
    ms_legacy = (time.perf_counter() - t0) / n_steps * 1e3

    def _bench(step_fn) -> float:
        ff = _copy_fields(fields)
        t1 = time.perf_counter()
        for n in range(n_steps):
            step_fn(ff, coeffs, op, cpml, [boundary.v_at_step(n)])
        return (time.perf_counter() - t1) / n_steps * 1e3

    ms_sparse = _bench(step_cut)
    ms_dense = _bench(step_cut_dense)
    print(f"ms/step legacy={ms_legacy:.3f} sparse={ms_sparse:.3f} dense={ms_dense:.3f}")


def _t2_fractions(n: int = 20) -> Fractions:
    shape = (n, n, n)
    beta = np.ones(shape, dtype=np.float64)
    ax = np.ones((n + 1, n, n), dtype=np.float64)
    ay = np.ones((n, n + 1, n), dtype=np.float64)
    az = np.ones((n, n, n + 1), dtype=np.float64)
    ax[0] = ax[-1] = 0.0
    ay[:, 0] = ay[:, -1] = 0.0
    az[:, :, 0] = az[:, :, -1] = 0.0
    i0, i1, j0, j1, k0, k1 = 6, 14, 7, 13, 4, 12
    beta[i0:i1, j0:j1, k0:k1] = 0.0
    ax[i0, j0:j1, k0:k1] = 0.0
    ax[i1, j0:j1, k0:k1] = 0.0
    ay[i0:i1, j0, k0:k1] = 0.0
    ay[i0:i1, j1, k0:k1] = 0.0
    az[i0:i1, j0:j1, k0] = 0.0
    az[i0:i1, j0:j1, k1] = 0.0
    return Fractions(
        alpha=(ax, ay, az),
        beta=beta,
        state=np.zeros(shape, dtype=np.uint8),
        patch_cell=np.array([], dtype=np.int64),
        patch_A=np.array([], dtype=np.float64),
        patch_centroid=np.zeros((0, 3)),
        patch_normal=np.zeros((0, 3)),
        radiator_cell=[],
        radiator_a=[],
        radiator_S=[],
        stats={},
    )


def _pulse(n: int, dt: float = 1.0, sigma: float = 8.0, t0: float = 40.0) -> np.float32:
    t = (n + 0.5) * dt
    return _F32(math.exp(-0.5 * ((t - t0) / sigma) ** 2))


def _empty_fields_20() -> Fields:
    psi = _zero_psi(20, 20, 20)
    return Fields(
        p=np.zeros((20, 20, 20), dtype=np.float32),
        vx=np.zeros((21, 20, 20), dtype=np.float32),
        vy=np.zeros((20, 21, 20), dtype=np.float32),
        vz=np.zeros((20, 20, 21), dtype=np.float32),
        **psi,
    )


def _t2_radiator_top() -> tuple[FastRadiator, PatchRadiator]:
    ii = jj = np.arange(8, 13)
    kf = 12
    vz_shape = (20, 20, 21)
    p_shape = (20, 20, 20)
    idx, sw, cells, coef = [], [], [], []
    w = _F32(1.0)
    for i in ii:
        for j in jj:
            idx.append(np.ravel_multi_index((i, j, kf), vz_shape))
            sw.append(w)
            cells.append(np.ravel_multi_index((i, j, kf), p_shape))
            coef.append(w)
    return (
        FastRadiator(axis=2, idx=np.asarray(idx, dtype=np.int64), sw=np.asarray(sw, dtype=np.float32)),
        PatchRadiator(
            cells=np.asarray(cells, dtype=np.int64),
            coef=np.asarray(coef, dtype=np.float32),
        ),
    )


def _t2_radiator_bottom() -> tuple[FastRadiator, PatchRadiator]:
    ii = jj = np.arange(8, 13)
    kf = 4
    vz_shape = (20, 20, 21)
    p_shape = (20, 20, 20)
    idx, sw, cells, coef = [], [], [], []
    w = _F32(1.0)
    for i in ii:
        for j in jj:
            idx.append(np.ravel_multi_index((i, j, kf), vz_shape))
            sw.append(-w)
            cells.append(np.ravel_multi_index((i, j, kf - 1), p_shape))
            coef.append(w)
    return (
        FastRadiator(axis=2, idx=np.asarray(idx, dtype=np.int64), sw=np.asarray(sw, dtype=np.float32)),
        PatchRadiator(
            cells=np.asarray(cells, dtype=np.int64),
            coef=np.asarray(coef, dtype=np.float32),
        ),
    )


def _run_t2_pair(fast: FastRadiator, patch: PatchRadiator) -> tuple[Fields, Fields]:
    fr = _t2_fractions()
    beta_t = fr.beta.astype(np.float32)
    op_f = build_operator(fr, beta_t, [fast], [], 1.0)
    op_p = build_operator(fr, beta_t, [], [patch], 1.0)
    fields_f = _empty_fields_20()
    fields_p = _empty_fields_20()
    coeffs = UpdateCoeffs(cv=_F32(0.5), cp=_F32(0.99 / 1.5))
    for n in range(300):
        v = _pulse(n)
        step_cut(fields_f, coeffs, op_f, None, [v])
        step_cut(fields_p, coeffs, op_p, None, [v])
    return fields_f, fields_p


def _exterior_t2(fr: Fractions) -> np.ndarray:
    return fr.beta > 0


def test_t2_patch_vs_fast():
    fast, patch = _t2_radiator_top()
    ff, fp = _run_t2_pair(fast, patch)
    fr = _t2_fractions()
    mask = _exterior_t2(fr)
    rel = float(np.max(np.abs(ff.p[mask] - fp.p[mask]))) / max(
        float(np.max(np.abs(fp.p[mask]))), 1e-30
    )
    assert rel <= 2e-6


def test_t2_down_sign():
    fast, patch = _t2_radiator_bottom()
    ff, fp = _run_t2_pair(fast, patch)
    fr = _t2_fractions()
    mask = _exterior_t2(fr)
    rel = float(np.max(np.abs(ff.p[mask] - fp.p[mask]))) / max(
        float(np.max(np.abs(fp.p[mask]))), 1e-30
    )
    assert rel <= 2e-6
    # positive pressure below radiator center at pulse peak (step ~40)
    peak = 0.0
    fields = _empty_fields_20()
    beta_t = fr.beta.astype(np.float32)
    op_f = build_operator(fr, beta_t, [fast], [], 1.0)
    coeffs = UpdateCoeffs(cv=_F32(0.5), cp=_F32(0.99 / 1.5))
    for n in range(45):
        step_cut(fields, coeffs, op_f, None, [_pulse(n)])
        if n == 40:
            peak = float(fields.p[10, 10, 3])
    assert peak > 0.0


def _energy_fractions() -> Fractions:
    from boxrad.geom import mesh

    def rot_z(d):
        t = math.radians(d)
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    def rot_x(d):
        t = math.radians(d)
        c, s = math.cos(t), math.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    R = rot_z(37) @ rot_x(20)
    solid = mesh.box_mesh([6.0, 6.0, 6.0]).transformed(R, np.zeros(3))
    grid = CutGrid(
        x0=np.array([-7.0, -7.0, -7.0], dtype=np.float64),
        h=1.0,
        shape=(14, 14, 14),
        rigid_sides={},
    )
    geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
    fr = compute_fractions(geom, grid)
    ax, ay, az = (a.copy() for a in fr.alpha)
    ax[0], ax[-1] = 0.0, 0.0
    ay[:, 0], ay[:, -1] = 0.0, 0.0
    az[:, :, 0], az[:, :, -1] = 0.0, 0.0
    return Fractions(
        alpha=(ax, ay, az),
        beta=fr.beta,
        state=fr.state,
        patch_cell=fr.patch_cell,
        patch_A=fr.patch_A,
        patch_centroid=fr.patch_centroid,
        patch_normal=fr.patch_normal,
        radiator_cell=fr.radiator_cell,
        radiator_a=fr.radiator_a,
        radiator_S=fr.radiator_S,
        stats=fr.stats,
    )


def _cut_energy(
    p: np.ndarray,
    beta_tilde: np.ndarray,
    v_old: tuple[np.ndarray, np.ndarray, np.ndarray],
    v_new: tuple[np.ndarray, np.ndarray, np.ndarray],
    op,
) -> float:
    p64 = p.astype(np.float64)
    bt = beta_tilde.astype(np.float64)
    e_p = 0.5 * np.sum(bt * p64 * p64)
    vx0, vy0, vz0 = (x.astype(np.float64) for x in v_old)
    vx1, vy1, vz1 = (x.astype(np.float64) for x in v_new)
    ax, ay, az = (a.astype(np.float64) for a in (op.afl_x, op.afl_y, op.afl_z))
    e_v = 0.5 * (
        np.sum(ax * vx0 * vx1) + np.sum(ay * vy0 * vy1) + np.sum(az * vz0 * vz1)
    )
    return e_p + e_v


def _step_cut_energy_half(fields: Fields, coeffs: UpdateCoeffs, op) -> tuple:
    """Return (v_old, v_new) around update_v + mask for energy accounting."""
    nx, ny, nz = fields.p.shape
    v_old = (fields.vx.copy(), fields.vy.copy(), fields.vz.copy())
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
    apply_open_mask(fields.vx, fields.vy, fields.vz, op.open_x, op.open_y, op.open_z)
    v_new = (fields.vx.copy(), fields.vy.copy(), fields.vz.copy())
    return v_old, v_new


def test_t3_energy_drift():
    fr = _energy_fractions()
    lam = 0.99 / math.sqrt(3.0)
    beta_t, _ = apply_floor(fr, lam, h=1.0)
    op = build_operator(fr, beta_t.astype(np.float32), [], [], 1.0)
    coeffs = UpdateCoeffs(cv=_F32(lam), cp=_F32(lam))
    rng = np.random.default_rng(0)
    fluid = fr.beta > 0
    p = np.zeros(fr.beta.shape, dtype=np.float32)
    p[fluid] = rng.standard_normal(int(np.sum(fluid))).astype(np.float32)
    psi = _zero_psi(14, 14, 14)
    fields = Fields(
        p=p,
        vx=np.zeros((15, 14, 14), dtype=np.float32),
        vy=np.zeros((14, 15, 14), dtype=np.float32),
        vz=np.zeros((14, 14, 15), dtype=np.float32),
        **psi,
    )
    hist = []
    nx, ny, nz = fields.p.shape
    for _ in range(4000):
        p_before = fields.p.copy()
        v_old, v_new = _step_cut_energy_half(fields, coeffs, op)
        hist.append(_cut_energy(p_before, beta_t, v_old, v_new, op))
        zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
        update_p_cut(
            fields.p,
            fields.vx,
            fields.vy,
            fields.vz,
            op.afl_x,
            op.afl_y,
            op.afl_z,
            coeffs.cp,
            op.inv_beta,
        )
        _ = nx, ny, nz
    h0 = hist[0]
    drift = float(np.max(np.abs(np.asarray(hist) - h0))) / max(abs(h0), 1e-30)
    assert drift <= 1e-6


def test_cpml_guard_raises():
    _, derived, grid, _, _, slabs, _, _, _, _ = _c9_setup()
    cfg = Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8"))
    geom = geometry.from_box(cfg)
    h = derived.h
    x0 = np.array(
        [
            -cfg.radiator.x_c - 0.5 * derived.realized_W - (grid.i_box - 10) * h,
            -cfg.radiator.y_c - 0.5 * derived.realized_H - grid.j_box * h,
            -derived.realized_D - grid.k_box * h,
        ],
        dtype=np.float64,
    )
    cgrid = CutGrid(x0=x0, h=h, shape=(grid.nx, grid.ny, grid.nz), rigid_sides={})
    fr = compute_fractions(geom, cgrid)
    op = build_operator(fr, fr.beta.astype(np.float32), [], [], h)
    with pytest.raises(ValueError, match="CPML"):
        assert_geometry_clear_of_cpml(op, slabs)