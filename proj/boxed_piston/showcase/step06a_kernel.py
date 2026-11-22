#!/usr/bin/env python3
"""Step 6a showcase: cut-cell kernel gate vs legacy (C9)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boxrad.config import Config, derive
from boxrad.cpml import make_cpml_state
from boxrad.cutcell import Fractions, Grid as CutGrid
from boxrad.cutcell import compute_fractions
from boxrad.cutkernel import (
    FastRadiator,
    PatchRadiator,
    apply_open_mask,
    build_operator,
    step_cut,
    step_cut_dense,
    update_p_cut,
)
from boxrad.geom import geometry, mesh
from boxrad.grid import Fields, Grid, allocate_fields
from boxrad.kernels import UpdateCoeffs, make_update_coeffs, step, update_v, zero_rigid_face_velocities
from boxrad.smallcell import apply_floor
from boxrad.source import BoxBoundary

_PKG = Path(__file__).resolve().parents[1]
_OUT = Path(__file__).resolve().parent / "out" / "step06a_kernel.png"
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


def _c9_setup():
    cfg = Config.from_json((_PKG / "configs" / "c09_box.json").read_text(encoding="utf-8"))
    derived = derive(cfg)
    grid = Grid(cfg, derived)
    fields = allocate_fields(grid)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, derived.h, derived.dt)
    slabs = grid.cpml_slabs
    cpml = make_cpml_state(grid, cfg, derived, derived.dt)
    boundary = BoxBoundary.from_grid(grid, derived)
    geom = geometry.from_box(cfg)
    xc, yc = cfg.radiator.x_c, cfg.radiator.y_c
    W, H, D = derived.realized_W, derived.realized_H, derived.realized_D
    ib, jb, kb = grid.i_box, grid.j_box, grid.k_box
    h = derived.h
    x0 = np.array(
        [-xc - 0.5 * W - ib * h, -yc - 0.5 * H - jb * h, -D - kb * h],
        dtype=np.float64,
    )
    cgrid = CutGrid(x0=x0, h=h, shape=(grid.nx, grid.ny, grid.nz), rigid_sides={})
    fr = compute_fractions(geom, cgrid)
    ir, jr, kr = grid.radiator_face_indices
    rad_idx = np.ravel_multi_index((ir, jr, kr), (grid.nx, grid.ny, grid.nz + 1)).astype(
        np.int64
    )
    sw = grid.radiator_weights.astype(np.float32)
    op = build_operator(fr, fr.beta.astype(np.float32), [FastRadiator(2, rad_idx, sw)], [], h)
    return derived, grid, fields, coeffs, slabs, cpml, boundary, op, fr


def _exterior_mask(grid: Grid, fr: Fractions) -> np.ndarray:
    ib, jb, kb = grid.i_box, grid.j_box, grid.k_box
    interior = np.zeros(fr.beta.shape, dtype=bool)
    interior[ib : ib + grid.nbx, jb : jb + grid.nby, kb : kb + grid.nbz] = True
    return (fr.beta > 0) & ~interior


def _t2_rel():
    shape = (20, 20, 20)
    beta = np.ones(shape)
    ax = np.ones((21, 20, 20))
    ay = np.ones((20, 21, 20))
    az = np.ones((20, 20, 21))
    ax[0] = ax[-1] = 0.0
    ay[:, 0] = ay[:, -1] = 0.0
    az[:, :, 0] = az[:, :, -1] = 0.0
    i0, i1, j0, j1, k0, k1 = 6, 14, 7, 13, 4, 12
    beta[i0:i1, j0:j1, k0:k1] = 0.0
    ax[i0, j0:j1, k0:k1] = ax[i1, j0:j1, k0:k1] = 0.0
    ay[i0:i1, j0, k0:k1] = ay[i0:i1, j1, k0:k1] = 0.0
    az[i0:i1, j0:j1, k0] = az[i0:i1, j0:j1, k1] = 0.0
    fr = Fractions(
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

    def run_pair(kf: int, sign: float, pcell_k: int) -> float:
        idx, sw, cells, coef = [], [], [], []
        for i in range(8, 13):
            for j in range(8, 13):
                idx.append(np.ravel_multi_index((i, j, kf), (20, 20, 21)))
                sw.append(sign)
                cells.append(np.ravel_multi_index((i, j, pcell_k), shape))
                coef.append(1.0)
        fast = FastRadiator(2, np.asarray(idx, np.int64), np.asarray(sw, np.float32))
        patch = PatchRadiator(np.asarray(cells, np.int64), np.asarray(coef, np.float32))
        op_f = build_operator(fr, beta.astype(np.float32), [fast], [], 1.0)
        op_p = build_operator(fr, beta.astype(np.float32), [], [patch], 1.0)
        ff = Fields(
            np.zeros(shape, np.float32),
            np.zeros((21, 20, 20), np.float32),
            np.zeros((20, 21, 20), np.float32),
            np.zeros((20, 20, 21), np.float32),
            *(np.zeros(1, np.float32) for _ in range(12)),
        )
        fp = _copy_fields(ff)
        coeffs = UpdateCoeffs(cv=_F32(0.5), cp=_F32(0.99 / 1.5))
        for n in range(300):
            v = _F32(math.exp(-0.5 * (((n + 0.5) - 40.0) / 8.0) ** 2))
            step_cut(ff, coeffs, op_f, None, [v])
            step_cut(fp, coeffs, op_p, None, [v])
        mask = beta > 0
        return float(np.max(np.abs(ff.p[mask] - fp.p[mask]))) / max(
            float(np.max(np.abs(fp.p[mask]))), 1e-30
        )

    return run_pair(12, 1.0, 12), run_pair(4, -1.0, 3)


def _t3_drift() -> tuple[float, list[float]]:
    def rot_z(d):
        t = math.radians(d)
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rot_x(d):
        t = math.radians(d)
        c, s = math.cos(t), math.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    R = rot_z(37) @ rot_x(20)
    solid = mesh.box_mesh([6.0, 6.0, 6.0]).transformed(R, np.zeros(3))
    cgrid = CutGrid(
        x0=np.array([-7.0, -7.0, -7.0], dtype=np.float64),
        h=1.0,
        shape=(14, 14, 14),
        rigid_sides={},
    )
    geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
    fr = compute_fractions(geom, cgrid)
    ax, ay, az = (a.copy() for a in fr.alpha)
    ax[0] = ax[-1] = 0.0
    ay[:, 0] = ay[:, -1] = 0.0
    az[:, :, 0] = az[:, :, -1] = 0.0
    fr_c = Fractions(
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
    lam = 0.99 / math.sqrt(3.0)
    beta_t, _ = apply_floor(fr_c, lam, h=1.0)
    op = build_operator(fr_c, beta_t.astype(np.float32), [], [], 1.0)
    coeffs = UpdateCoeffs(cv=_F32(lam), cp=_F32(lam))
    rng = np.random.default_rng(0)
    fluid = fr.beta > 0
    p = np.zeros(fr.beta.shape, np.float32)
    p[fluid] = rng.standard_normal(int(np.sum(fluid))).astype(np.float32)
    fields = Fields(
        p=p,
        vx=np.zeros((15, 14, 14), np.float32),
        vy=np.zeros((14, 15, 14), np.float32),
        vz=np.zeros((14, 14, 15), np.float32),
        **{f"psi_{n}": np.zeros(1, np.float32) for n in (
            "p_x_neg", "p_x_pos", "p_y_neg", "p_y_pos", "p_z_neg", "p_z_pos",
            "vx_neg", "vx_pos", "vy_neg", "vy_pos", "vz_neg", "vz_pos",
        )},
    )
    hist = []
    nx, ny, nz = 14, 14, 14
    for _ in range(4000):
        p0 = fields.p.copy()
        v0 = (fields.vx.copy(), fields.vy.copy(), fields.vz.copy())
        zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
        update_v(fields.p, fields.vx, fields.vy, fields.vz, coeffs.cv, 0, nx, 0, ny, 0, nz)
        apply_open_mask(fields.vx, fields.vy, fields.vz, op.open_x, op.open_y, op.open_z)
        v1 = (fields.vx.copy(), fields.vy.copy(), fields.vz.copy())
        bt = beta_t.astype(np.float64)
        hist.append(
            0.5 * np.sum(bt * p0.astype(np.float64) ** 2)
            + 0.5
            * (
                np.sum(op.afl_x.astype(np.float64) * v0[0] * v1[0])
                + np.sum(op.afl_y.astype(np.float64) * v0[1] * v1[1])
                + np.sum(op.afl_z.astype(np.float64) * v0[2] * v1[2])
            )
        )
        update_p_cut(
            fields.p, fields.vx, fields.vy, fields.vz,
            op.afl_x, op.afl_y, op.afl_z, coeffs.cp, op.inv_beta,
        )
    h0 = hist[0]
    return float(np.max(np.abs(np.asarray(hist) - h0))) / max(abs(h0), 1e-30), hist


def main() -> None:
    derived, grid, fields, coeffs, slabs, cpml, boundary, op, fr = _c9_setup()
    leg = _copy_fields(fields)
    cut = _copy_fields(fields)
    pi = grid.i_box + grid.nbx // 2
    pj = grid.j_box + grid.nby // 2
    pk = grid.k_face_z + 3
    n_steps = 400
    t0 = time.perf_counter()
    for n in range(n_steps):
        step(leg, coeffs, slabs, cpml, boundary, n)
    t_leg = (time.perf_counter() - t0) / n_steps
    t1 = time.perf_counter()
    for n in range(n_steps):
        step_cut(cut, coeffs, op, cpml, [boundary.v_at_step(n)])
    t_sparse = (time.perf_counter() - t1) / n_steps
    cut_d = _copy_fields(fields)
    t2 = time.perf_counter()
    for n in range(n_steps):
        step_cut_dense(cut_d, coeffs, op, cpml, [boundary.v_at_step(n)])
    t_dense = (time.perf_counter() - t2) / n_steps

    mask_p = _exterior_mask(grid, fr)
    bit_id = bool(np.array_equal(leg.p[mask_p], cut.p[mask_p]))
    p_leg, p_cut, diff = [], [], []
    leg2 = _copy_fields(fields)
    cut2 = _copy_fields(fields)
    for n in range(n_steps):
        step(leg2, coeffs, slabs, cpml, boundary, n)
        step_cut(cut2, coeffs, op, cpml, [boundary.v_at_step(n)])
        p_leg.append(float(leg2.p[pi, pj, pk]))
        p_cut.append(float(cut2.p[pi, pj, pk]))
        diff.append(abs(p_leg[-1] - p_cut[-1]))

    ny, nz = grid.ny, grid.nz
    by_ij = {
        (int(c) // (ny * nz), (int(c) // nz) % ny): float(a)
        for c, a in zip(fr.radiator_cell[0], fr.radiator_a[0])
    }
    n1 = nz + 1
    fast0 = op.radiators_fast[0]
    sw_ex = np.empty(fast0.idx.shape[0], dtype=np.float32)
    for q, idx in enumerate(fast0.idx):
        rem = int(idx) % (ny * n1)
        i = int(idx) // (ny * n1)
        j = rem // n1
        sw_ex[q] = np.float32(by_ij.get((i, j), 0.0) / (derived.h * derived.h))
    op_ex = build_operator(fr, fr.beta.astype(np.float32), [FastRadiator(2, fast0.idx, sw_ex)], [], derived.h)
    wl = _copy_fields(fields)
    we = _copy_fields(fields)
    for n in range(n_steps):
        step_cut(wl, coeffs, op, cpml, [boundary.v_at_step(n)])
        step_cut(we, coeffs, op_ex, cpml, [boundary.v_at_step(n)])
    d = wl.p[mask_p].astype(np.float64) - we.p[mask_p].astype(np.float64)
    exact_rel = float(np.linalg.norm(d)) / max(float(np.linalg.norm(wl.p[mask_p].astype(np.float64))), 1e-30)

    t2, t2d = _t2_rel()
    drift, hist = _t3_drift()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].semilogy(np.arange(n_steps), np.maximum(np.asarray(diff), 1e-12))
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("|Δp|")
    axes[0].set_title("C9 probe 3h above radiator center")
    axes[1].semilogy(np.maximum(np.abs(np.asarray(hist) - hist[0]), 1e-30))
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("|H − H0|")
    axes[1].set_title("T3 energy drift (rotated cube)")
    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=120)

    print(f"bit-identity: {bit_id}")
    print(f"exact-weight rel diff: {exact_rel:.6e}")
    print(f"T2 rel: {t2:.6e}")
    print(f"T2-down rel: {t2d:.6e}")
    print(f"T3 drift: {drift:.6e}")
    print(f"time/step legacy: {t_leg * 1e3:.3f} ms")
    print(f"time/step cut sparse: {t_sparse * 1e3:.3f} ms")
    print(f"time/step cut dense: {t_dense * 1e3:.3f} ms")


if __name__ == "__main__":
    main()