"""Test-only reference: v0.3 cut-cell kernel semantics (float32 numpy), verified by Claude 2026-11-22.

Verified: T1 exterior bit-identical to a v0.2 replica (300 steps, aligned box + rect radiator);
T2 patch-flux vs fast path <= 9.9e-7; T3 float32 energy drift 2.5e-7 over 4000 steps (rotated cube).
NOTE: the production numba kernel must evaluate div in kernels.update_p's LEFT-TO-RIGHT order
(a-b+c-d+e-f), not this file's grouped numpy form, to stay bit-identical with the real v0.2 kernel.


Legacy (v0.2) step order (kernels.step): zero outer faces; update_v (all faces); zero outer faces;
BoxBoundary.apply (rigid faces := 0, radiator faces := w*v(t)); update_p (all cells).

General (v0.3) step:  zero outer faces; update_v; v *= open_mask (alpha>0 or radiator face);
radiator fast-path faces := w*v(t); p[c] -= (cp*inv_beta[c]) * (sum_f s*afl_f*v_f - sum_r a_rc*v_r/h^2)
where afl = alpha with radiator fast-path faces set to 1 (so afl*v = w*v exactly as v0.2).
"""
import numpy as np
F32 = np.float32


def legacy_step(p, vx, vy, vz, cv, cp, rig, rad_idx, rad_w, v_now):
    vx[0] = 0; vx[-1] = 0; vy[:, 0] = 0; vy[:, -1] = 0; vz[:, :, 0] = 0; vz[:, :, -1] = 0
    vx[1:-1] -= cv * (p[1:] - p[:-1])
    vy[:, 1:-1] -= cv * (p[:, 1:] - p[:, :-1])
    vz[:, :, 1:-1] -= cv * (p[:, :, 1:] - p[:, :, :-1])
    vx[0] = 0; vx[-1] = 0; vy[:, 0] = 0; vy[:, -1] = 0; vz[:, :, 0] = 0; vz[:, :, -1] = 0
    rx, ry, rz = rig
    vx[rx] = 0; vy[ry] = 0; vz[rz] = 0
    vz.reshape(-1)[rad_idx] = rad_w * v_now
    div = (vx[1:] - vx[:-1]) + (vy[:, 1:] - vy[:, :-1]) + (vz[:, :, 1:] - vz[:, :, :-1])
    p -= cp * div


def general_step(p, vx, vy, vz, cv, cp, afl, openm, inv_b, rad_idx, rad_w, v_now,
                 patch_cells=None, patch_coef=None, v_patch=F32(0)):
    ax, ay, az = afl
    ox, oy, oz = openm
    vx[0] = 0; vx[-1] = 0; vy[:, 0] = 0; vy[:, -1] = 0; vz[:, :, 0] = 0; vz[:, :, -1] = 0
    vx[1:-1] -= cv * (p[1:] - p[:-1])
    vy[:, 1:-1] -= cv * (p[:, 1:] - p[:, :-1])
    vz[:, :, 1:-1] -= cv * (p[:, :, 1:] - p[:, :, :-1])
    vx[0] = 0; vx[-1] = 0; vy[:, 0] = 0; vy[:, -1] = 0; vz[:, :, 0] = 0; vz[:, :, -1] = 0
    vx *= ox; vy *= oy; vz *= oz
    vz.reshape(-1)[rad_idx] = rad_w * v_now
    fx, fy, fz = ax * vx, ay * vy, az * vz
    div = (fx[1:] - fx[:-1]) + (fy[:, 1:] - fy[:, :-1]) + (fz[:, :, 1:] - fz[:, :, :-1])
    if patch_cells is not None:
        d = div.reshape(-1)
        d[patch_cells] -= patch_coef * v_patch
    p -= (cp * inv_b) * div


def pulse(n, dt, sigma=F32(8.0), t0=F32(40.0)):
    t = (n + 0.5) * dt
    return F32(np.exp(-0.5 * ((t - t0) / sigma) ** 2))
