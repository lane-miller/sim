"""CFS-PML slab updates (Roden & Gedney 2000) for the acoustic Yee grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from fdtd.config import Config, Derived
from fdtd.grid import CpmlSlabs, Fields, Grid
from fdtd.kernels import UpdateCoeffs

_FLOAT32 = np.dtype(np.float32)


@dataclass(frozen=True)
class CpmlCoeffs:
    """Precomputed 1-D CFS-PML profile rows (float32), one row per slab depth index.

    *p* rows length ``cpml_cells``; normal *v* rows length ``cpml_cells + 1``.
    Neg/pos pairs index depth from the corresponding outer domain face.
    Depth ``d/L → 0`` at the CPML/interior interface and ``d/L → 1`` at the
    outer rigid face (``σ``, ``κ−1``, and ``α/α_max`` increase toward the wall).
    """

    cpml_cells: int
    sigma_max: float
    b_p: np.ndarray
    a_p: np.ndarray
    inv_kappa_m1_p: np.ndarray
    b_v: np.ndarray
    a_v: np.ndarray
    inv_kappa_m1_v: np.ndarray


def _profile_row(
    n: int,
    *,
    neg: bool,
    sigma_max: float,
    kappa_max: float,
    alpha_max: float,
    m: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(b, a, inv_kappa_m1)`` length *n* for neg (inward) or pos depth."""
    b = np.empty(n, dtype=_FLOAT32)
    a = np.empty(n, dtype=_FLOAT32)
    inv_km1 = np.empty(n, dtype=_FLOAT32)
    for q in range(n):
        if neg:
            # p[i=q]: q=0 outer wall, q=n−1 at interior interface
            d_over_L = (n - q - 0.5) / n
        else:
            # p[i=nx−n+q]: q=0 interior, q=n−1 outer wall
            d_over_L = (q + 0.5) / n
        d_over_L = min(max(d_over_L, 0.0), 1.0)
        power = d_over_L**m
        sigma = sigma_max * power
        kappa = 1.0 + (kappa_max - 1.0) * power
        alpha = alpha_max * (1.0 - d_over_L)
        decay = sigma / kappa + alpha
        b[q] = np.float32(math.exp(-decay * dt))
        if abs(sigma) < 1e-30:
            a[q] = np.float32(0.0)
        else:
            denom = sigma * kappa + kappa * kappa * alpha
            a[q] = np.float32(sigma / denom * (float(b[q]) - 1.0))
        inv_km1[q] = np.float32(1.0 / kappa - 1.0)
    return b, a, inv_km1


def _profile_row_v(
    n: int,
    *,
    neg: bool,
    sigma_max: float,
    kappa_max: float,
    alpha_max: float,
    m: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face-staggered depths: ``q / (n-1)`` spans the layer (``n = cpml+1``)."""
    b = np.empty(n, dtype=_FLOAT32)
    a = np.empty(n, dtype=_FLOAT32)
    inv_km1 = np.empty(n, dtype=_FLOAT32)
    layer = max(n - 1, 1)
    for q in range(n):
        if neg:
            # vx face i=q; q=0 rigid wall, q=layer at interior interface
            d_over_L = (layer - q) / layer
        else:
            # vx face at positive slab; q=0 interior, q=layer−1 toward wall
            d_over_L = (q + 1) / layer
        d_over_L = min(max(d_over_L, 0.0), 1.0)
        power = d_over_L**m
        sigma = sigma_max * power
        kappa = 1.0 + (kappa_max - 1.0) * power
        alpha = alpha_max * (1.0 - d_over_L)
        decay = sigma / kappa + alpha
        b[q] = np.float32(math.exp(-decay * dt))
        if abs(sigma) < 1e-30:
            a[q] = np.float32(0.0)
        else:
            dnom = sigma * kappa + kappa * kappa * alpha
            a[q] = np.float32(sigma / dnom * (float(b[q]) - 1.0))
        inv_km1[q] = np.float32(1.0 / kappa - 1.0)
    return b, a, inv_km1


def build_cpml_coeffs(
    grid: Grid, cfg: Config, derived: Derived, dt: float
) -> CpmlCoeffs:
    """CFS-PML profiles from config numerics and derived ``cpml_alpha``."""
    num = cfg.numerics
    c = derived.cpml_cells
    if c <= 0:
        z = np.zeros(0, dtype=_FLOAT32)
        return CpmlCoeffs(c, 0.0, z, z, z, z, z, z)
    L = c * derived.h
    m = num.cpml_m
    sigma_max = -(m + 1.0) * cfg.air.c * math.log(num.cpml_R) / (2.0 * L)
    alpha_max = derived.cpml_alpha
    kappa_max = num.cpml_kappa_max

    b_p, a_p, ik_p = _profile_row(
        c,
        neg=True,
        sigma_max=sigma_max,
        kappa_max=kappa_max,
        alpha_max=alpha_max,
        m=m,
        dt=dt,
    )

    nv = c + 1
    b_vn, a_vn, ik_vn = _profile_row_v(
        nv, neg=True, sigma_max=sigma_max, kappa_max=kappa_max,
        alpha_max=alpha_max, m=m, dt=dt,
    )
    return CpmlCoeffs(
        cpml_cells=c,
        sigma_max=sigma_max,
        b_p=b_p,
        a_p=a_p,
        inv_kappa_m1_p=ik_p,
        b_v=b_vn,
        a_v=a_vn,
        inv_kappa_m1_v=ik_vn,
    )


@dataclass(frozen=True)
class CpmlFaceRows:
    """Neg/pos profile rows bundled for slab kernel dispatch."""

    b_p_neg: np.ndarray
    a_p_neg: np.ndarray
    ik_p_neg: np.ndarray
    b_p_pos: np.ndarray
    a_p_pos: np.ndarray
    ik_p_pos: np.ndarray
    b_v_neg: np.ndarray
    a_v_neg: np.ndarray
    ik_v_neg: np.ndarray
    b_v_pos: np.ndarray
    a_v_pos: np.ndarray
    ik_v_pos: np.ndarray


def face_rows(cpml: CpmlCoeffs, cfg: Config, derived: Derived, dt: float) -> CpmlFaceRows:
    """Neg and pos 1-D rows (rebuild pos mirror when needed for tests)."""
    c = cpml.cpml_cells
    if c <= 0:
        z = np.zeros(0, dtype=_FLOAT32)
        return CpmlFaceRows(z, z, z, z, z, z, z, z, z, z, z, z)
    num = cfg.numerics
    L = c * derived.h
    m = num.cpml_m
    sigma_max = -(m + 1.0) * cfg.air.c * math.log(num.cpml_R) / (2.0 * L)
    alpha_max = derived.cpml_alpha
    kappa_max = num.cpml_kappa_max
    b_pn, a_pn, ik_pn = _profile_row(
        c, neg=False, sigma_max=sigma_max, kappa_max=kappa_max,
        alpha_max=alpha_max, m=m, dt=dt,
    )
    b_vp, a_vp, ik_vp = _profile_row_v(
        c + 1, neg=False, sigma_max=sigma_max, kappa_max=kappa_max,
        alpha_max=alpha_max, m=m, dt=dt,
    )
    return CpmlFaceRows(
        cpml.b_p, cpml.a_p, cpml.inv_kappa_m1_p,
        b_pn, a_pn, ik_pn,
        cpml.b_v, cpml.a_v, cpml.inv_kappa_m1_v,
        b_vp, a_vp, ik_vp,
    )


@njit(parallel=True, fastmath=False, cache=True)
def _cpml_vx_slabs(
    p: np.ndarray,
    vx: np.ndarray,
    psi_neg: np.ndarray,
    psi_pos: np.ndarray,
    cv: np.float32,
    c: int,
    nx: int,
    b_neg: np.ndarray,
    a_neg: np.ndarray,
    ik_neg: np.ndarray,
    b_pos: np.ndarray,
    a_pos: np.ndarray,
    ik_pos: np.ndarray,
) -> None:
    ny = p.shape[1]
    nz = p.shape[2]
    for j in prange(ny):
        for k in range(nz):
            for q in range(1, c + 1):
                i = q
                dp = p[i, j, k] - p[i - 1, j, k]
                ps = b_neg[q] * psi_neg[q, j, k] + a_neg[q] * dp
                psi_neg[q, j, k] = ps
                vx[i, j, k] -= cv * (ik_neg[q] * dp + ps)
            base = nx - c
            for q in range(c):
                i = base + q
                dp = p[i, j, k] - p[i - 1, j, k]
                ps = b_pos[q] * psi_pos[q, j, k] + a_pos[q] * dp
                psi_pos[q, j, k] = ps
                vx[i, j, k] -= cv * (ik_pos[q] * dp + ps)


@njit(parallel=True, fastmath=False, cache=True)
def _cpml_vy_slabs(
    p: np.ndarray,
    vy: np.ndarray,
    psi_neg: np.ndarray,
    psi_pos: np.ndarray,
    cv: np.float32,
    c: int,
    ny: int,
    b_neg: np.ndarray,
    a_neg: np.ndarray,
    ik_neg: np.ndarray,
    b_pos: np.ndarray,
    a_pos: np.ndarray,
    ik_pos: np.ndarray,
) -> None:
    nx = p.shape[0]
    nz = p.shape[2]
    for i in prange(nx):
        for k in range(nz):
            for q in range(1, c + 1):
                j = q
                dp = p[i, j, k] - p[i, j - 1, k]
                ps = b_neg[q] * psi_neg[i, q, k] + a_neg[q] * dp
                psi_neg[i, q, k] = ps
                vy[i, j, k] -= cv * (ik_neg[q] * dp + ps)
            base = ny - c
            for q in range(c):
                j = base + q
                dp = p[i, j, k] - p[i, j - 1, k]
                ps = b_pos[q] * psi_pos[i, q, k] + a_pos[q] * dp
                psi_pos[i, q, k] = ps
                vy[i, j, k] -= cv * (ik_pos[q] * dp + ps)


@njit(parallel=True, fastmath=False, cache=True)
def _cpml_vz_slabs(
    p: np.ndarray,
    vz: np.ndarray,
    psi_neg: np.ndarray,
    psi_pos: np.ndarray,
    cv: np.float32,
    c: int,
    nz: int,
    b_neg: np.ndarray,
    a_neg: np.ndarray,
    ik_neg: np.ndarray,
    b_pos: np.ndarray,
    a_pos: np.ndarray,
    ik_pos: np.ndarray,
) -> None:
    nx = p.shape[0]
    ny = p.shape[1]
    for i in prange(nx):
        for j in range(ny):
            for q in range(1, c + 1):
                kk = q
                dp = p[i, j, kk] - p[i, j, kk - 1]
                ps = b_neg[q] * psi_neg[i, j, q] + a_neg[q] * dp
                psi_neg[i, j, q] = ps
                vz[i, j, kk] -= cv * (ik_neg[q] * dp + ps)
            base = nz - c
            for q in range(c):
                kk = base + q
                dp = p[i, j, kk] - p[i, j, kk - 1]
                ps = b_pos[q] * psi_pos[i, j, q] + a_pos[q] * dp
                psi_pos[i, j, q] = ps
                vz[i, j, kk] -= cv * (ik_pos[q] * dp + ps)


@njit(parallel=True, fastmath=False, cache=True)
def _cpml_p_x_slabs(
    p: np.ndarray,
    vx: np.ndarray,
    psi_neg: np.ndarray,
    psi_pos: np.ndarray,
    cp: np.float32,
    c: int,
    nx: int,
    b_neg: np.ndarray,
    a_neg: np.ndarray,
    ik_neg: np.ndarray,
    b_pos: np.ndarray,
    a_pos: np.ndarray,
    ik_pos: np.ndarray,
) -> None:
    ny = p.shape[1]
    nz = p.shape[2]
    for j in prange(ny):
        for k in range(nz):
            for q in range(c):
                i = q
                dv = vx[i + 1, j, k] - vx[i, j, k]
                ps = b_neg[q] * psi_neg[q, j, k] + a_neg[q] * dv
                psi_neg[q, j, k] = ps
                p[i, j, k] -= cp * (ik_neg[q] * dv + ps)
            base = nx - c
            for q in range(c):
                i = base + q
                dv = vx[i + 1, j, k] - vx[i, j, k]
                ps = b_pos[q] * psi_pos[q, j, k] + a_pos[q] * dv
                psi_pos[q, j, k] = ps
                p[i, j, k] -= cp * (ik_pos[q] * dv + ps)


@njit(parallel=True, fastmath=False, cache=True)
def _cpml_p_y_slabs(
    p: np.ndarray,
    vy: np.ndarray,
    psi_neg: np.ndarray,
    psi_pos: np.ndarray,
    cp: np.float32,
    c: int,
    ny: int,
    b_neg: np.ndarray,
    a_neg: np.ndarray,
    ik_neg: np.ndarray,
    b_pos: np.ndarray,
    a_pos: np.ndarray,
    ik_pos: np.ndarray,
) -> None:
    nx = p.shape[0]
    nz = p.shape[2]
    for i in prange(nx):
        for k in range(nz):
            for q in range(c):
                j = q
                dv = vy[i, j + 1, k] - vy[i, j, k]
                ps = b_neg[q] * psi_neg[i, q, k] + a_neg[q] * dv
                psi_neg[i, q, k] = ps
                p[i, j, k] -= cp * (ik_neg[q] * dv + ps)
            base = ny - c
            for q in range(c):
                j = base + q
                dv = vy[i, j + 1, k] - vy[i, j, k]
                ps = b_pos[q] * psi_pos[i, q, k] + a_pos[q] * dv
                psi_pos[i, q, k] = ps
                p[i, j, k] -= cp * (ik_pos[q] * dv + ps)


@njit(parallel=True, fastmath=False, cache=True)
def _cpml_p_z_slabs(
    p: np.ndarray,
    vz: np.ndarray,
    psi_neg: np.ndarray,
    psi_pos: np.ndarray,
    cp: np.float32,
    c: int,
    nz: int,
    b_neg: np.ndarray,
    a_neg: np.ndarray,
    ik_neg: np.ndarray,
    b_pos: np.ndarray,
    a_pos: np.ndarray,
    ik_pos: np.ndarray,
) -> None:
    nx = p.shape[0]
    ny = p.shape[1]
    for i in prange(nx):
        for j in range(ny):
            for q in range(c):
                kk = q
                dv = vz[i, j, kk + 1] - vz[i, j, kk]
                ps = b_neg[q] * psi_neg[i, j, q] + a_neg[q] * dv
                psi_neg[i, j, q] = ps
                p[i, j, kk] -= cp * (ik_neg[q] * dv + ps)
            base = nz - c
            for q in range(c):
                kk = base + q
                dv = vz[i, j, kk + 1] - vz[i, j, kk]
                ps = b_pos[q] * psi_pos[i, j, q] + a_pos[q] * dv
                psi_pos[i, j, q] = ps
                p[i, j, kk] -= cp * (ik_pos[q] * dv + ps)


@dataclass
class CpmlState:
    """Runtime CPML: precomputed profiles plus cached neg/pos rows."""

    coeffs: CpmlCoeffs
    rows: CpmlFaceRows


def make_cpml_state(
    grid: Grid, cfg: Config, derived: Derived, dt: float
) -> CpmlState:
    coeffs = build_cpml_coeffs(grid, cfg, derived, dt)
    rows = face_rows(coeffs, cfg, derived, dt)
    return CpmlState(coeffs=coeffs, rows=rows)


def apply_v(
    fields: Fields, cpml_state: CpmlState, update: UpdateCoeffs
) -> None:
    c = cpml_state.coeffs.cpml_cells
    if c <= 0:
        return
    r = cpml_state.rows
    nx, ny, nz = fields.p.shape
    cv = update.cv
    _cpml_vx_slabs(
        fields.p, fields.vx,
        fields.psi_vx_neg, fields.psi_vx_pos,
        cv, c, nx,
        r.b_v_neg, r.a_v_neg, r.ik_v_neg,
        r.b_v_pos, r.a_v_pos, r.ik_v_pos,
    )
    _cpml_vy_slabs(
        fields.p, fields.vy,
        fields.psi_vy_neg, fields.psi_vy_pos,
        cv, c, ny,
        r.b_v_neg, r.a_v_neg, r.ik_v_neg,
        r.b_v_pos, r.a_v_pos, r.ik_v_pos,
    )
    _cpml_vz_slabs(
        fields.p, fields.vz,
        fields.psi_vz_neg, fields.psi_vz_pos,
        cv, c, nz,
        r.b_v_neg, r.a_v_neg, r.ik_v_neg,
        r.b_v_pos, r.a_v_pos, r.ik_v_pos,
    )


def apply_p(
    fields: Fields, cpml_state: CpmlState, update: UpdateCoeffs
) -> None:
    c = cpml_state.coeffs.cpml_cells
    if c <= 0:
        return
    r = cpml_state.rows
    nx, ny, nz = fields.p.shape
    cp = update.cp
    _cpml_p_x_slabs(
        fields.p, fields.vx,
        fields.psi_p_x_neg, fields.psi_p_x_pos,
        cp, c, nx,
        r.b_p_neg, r.a_p_neg, r.ik_p_neg,
        r.b_p_pos, r.a_p_pos, r.ik_p_pos,
    )
    _cpml_p_y_slabs(
        fields.p, fields.vy,
        fields.psi_p_y_neg, fields.psi_p_y_pos,
        cp, c, ny,
        r.b_p_neg, r.a_p_neg, r.ik_p_neg,
        r.b_p_pos, r.a_p_pos, r.ik_p_pos,
    )
    _cpml_p_z_slabs(
        fields.p, fields.vz,
        fields.psi_p_z_neg, fields.psi_p_z_pos,
        cp, c, nz,
        r.b_p_neg, r.a_p_neg, r.ik_p_neg,
        r.b_p_pos, r.a_p_pos, r.ik_p_pos,
    )


def apply_v_numpy(
    p: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    fields: Fields,
    update: UpdateCoeffs,
    cpml_state: CpmlState,
) -> None:
    """NumPy reference matching :func:`apply_v` loops (mutates *vx*, *vy*, *vz*, ψ)."""
    c = cpml_state.coeffs.cpml_cells
    if c <= 0:
        return
    r = cpml_state.rows
    cv = update.cv
    nx, ny, nz = p.shape

    for j in range(ny):
        for k in range(nz):
            for q in range(1, c + 1):
                i = q
                dp = p[i, j, k] - p[i - 1, j, k]
                ps = r.b_v_neg[q] * fields.psi_vx_neg[q, j, k] + r.a_v_neg[q] * dp
                fields.psi_vx_neg[q, j, k] = np.float32(ps)
                vx[i, j, k] -= cv * (r.ik_v_neg[q] * dp + ps)
            base = nx - c
            for q in range(c):
                i = base + q
                dp = p[i, j, k] - p[i - 1, j, k]
                ps = r.b_v_pos[q] * fields.psi_vx_pos[q, j, k] + r.a_v_pos[q] * dp
                fields.psi_vx_pos[q, j, k] = np.float32(ps)
                vx[i, j, k] -= cv * (r.ik_v_pos[q] * dp + ps)

    for i in range(nx):
        for k in range(nz):
            for q in range(1, c + 1):
                j = q
                dp = p[i, j, k] - p[i, j - 1, k]
                ps = r.b_v_neg[q] * fields.psi_vy_neg[i, q, k] + r.a_v_neg[q] * dp
                fields.psi_vy_neg[i, q, k] = np.float32(ps)
                vy[i, j, k] -= cv * (r.ik_v_neg[q] * dp + ps)
            base = ny - c
            for q in range(c):
                j = base + q
                dp = p[i, j, k] - p[i, j - 1, k]
                ps = r.b_v_pos[q] * fields.psi_vy_pos[i, q, k] + r.a_v_pos[q] * dp
                fields.psi_vy_pos[i, q, k] = np.float32(ps)
                vy[i, j, k] -= cv * (r.ik_v_pos[q] * dp + ps)

    for i in range(nx):
        for j in range(ny):
            for q in range(1, c + 1):
                kk = q
                dp = p[i, j, kk] - p[i, j, kk - 1]
                ps = r.b_v_neg[q] * fields.psi_vz_neg[i, j, q] + r.a_v_neg[q] * dp
                fields.psi_vz_neg[i, j, q] = np.float32(ps)
                vz[i, j, kk] -= cv * (r.ik_v_neg[q] * dp + ps)
            base = nz - c
            for q in range(c):
                kk = base + q
                dp = p[i, j, kk] - p[i, j, kk - 1]
                ps = r.b_v_pos[q] * fields.psi_vz_pos[i, j, q] + r.a_v_pos[q] * dp
                fields.psi_vz_pos[i, j, q] = np.float32(ps)
                vz[i, j, kk] -= cv * (r.ik_v_pos[q] * dp + ps)


def apply_p_numpy(
    p: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    fields: Fields,
    update: UpdateCoeffs,
    cpml_state: CpmlState,
) -> None:
    c = cpml_state.coeffs.cpml_cells
    if c <= 0:
        return
    r = cpml_state.rows
    cp = update.cp
    nx, ny, nz = p.shape

    for j in range(ny):
        for k in range(nz):
            for q in range(c):
                i = q
                dv = vx[i + 1, j, k] - vx[i, j, k]
                ps = r.b_p_neg[q] * fields.psi_p_x_neg[q, j, k] + r.a_p_neg[q] * dv
                fields.psi_p_x_neg[q, j, k] = np.float32(ps)
                p[i, j, k] -= cp * (r.ik_p_neg[q] * dv + ps)
            base = nx - c
            for q in range(c):
                i = base + q
                dv = vx[i + 1, j, k] - vx[i, j, k]
                ps = r.b_p_pos[q] * fields.psi_p_x_pos[q, j, k] + r.a_p_pos[q] * dv
                fields.psi_p_x_pos[q, j, k] = np.float32(ps)
                p[i, j, k] -= cp * (r.ik_p_pos[q] * dv + ps)

    for i in range(nx):
        for k in range(nz):
            for q in range(c):
                j = q
                dv = vy[i, j + 1, k] - vy[i, j, k]
                ps = r.b_p_neg[q] * fields.psi_p_y_neg[i, q, k] + r.a_p_neg[q] * dv
                fields.psi_p_y_neg[i, q, k] = np.float32(ps)
                p[i, j, k] -= cp * (r.ik_p_neg[q] * dv + ps)
            base = ny - c
            for q in range(c):
                j = base + q
                dv = vy[i, j + 1, k] - vy[i, j, k]
                ps = r.b_p_pos[q] * fields.psi_p_y_pos[i, q, k] + r.a_p_pos[q] * dv
                fields.psi_p_y_pos[i, q, k] = np.float32(ps)
                p[i, j, k] -= cp * (r.ik_p_pos[q] * dv + ps)

    for i in range(nx):
        for j in range(ny):
            for q in range(c):
                kk = q
                dv = vz[i, j, kk + 1] - vz[i, j, kk]
                ps = r.b_p_neg[q] * fields.psi_p_z_neg[i, j, q] + r.a_p_neg[q] * dv
                fields.psi_p_z_neg[i, j, q] = np.float32(ps)
                p[i, j, kk] -= cp * (r.ik_p_neg[q] * dv + ps)
            base = nz - c
            for q in range(c):
                kk = base + q
                dv = vz[i, j, kk + 1] - vz[i, j, kk]
                ps = r.b_p_pos[q] * fields.psi_p_z_pos[i, j, q] + r.a_p_pos[q] * dv
                fields.psi_p_z_pos[i, j, q] = np.float32(ps)
                p[i, j, kk] -= cp * (r.ik_p_pos[q] * dv + ps)
