"""Numba FDTD interior update kernels (Stage A); CPML hooks in cpml.py."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from fdtd.grid import CpmlSlabs, Fields

_FLOAT32 = np.dtype(np.float32)
_COURANT_LIMIT = np.float32(1.0 + 5e-6)


@dataclass(frozen=True)
class UpdateCoeffs:
    """Precomputed float32 leapfrog scalars (no float64 inside kernels)."""

    cv: np.float32  # dt / (rho * h)
    cp: np.float32  # rho * c**2 * dt / h


def courant(coeffs: UpdateCoeffs) -> float:
    """3-D acoustic Courant number from leapfrog scalars: ``sqrt(cv·cp)·sqrt(3)``."""
    return float(np.sqrt(float(coeffs.cv) * float(coeffs.cp)) * np.sqrt(3.0))


def make_update_coeffs(rho: float, c: float, h: float, dt: float) -> UpdateCoeffs:
    """Build update scalars in float32 (caller supplies SI *rho*, *c*, *h*, *dt*)."""
    coeffs = UpdateCoeffs(
        cv=np.float32(dt / (rho * h)),
        cp=np.float32(rho * c * c * dt / h),
    )
    nu = courant(coeffs)
    if nu > float(_COURANT_LIMIT):
        raise ValueError(
            f"3-D Courant number {nu:.8f} exceeds limit {float(_COURANT_LIMIT):.8f}; "
            "reduce dt or increase h (need dt <= h/(c*sqrt(3)))."
        )
    return coeffs


@njit(cache=True)
def zero_rigid_face_velocities(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray
) -> None:
    """Enforce ``v_n = 0`` on the six outer domain faces (rigid walls, Stage A)."""
    nx = vx.shape[0] - 1
    ny = vy.shape[1] - 1
    nz = vz.shape[2] - 1
    for j in range(vx.shape[1]):
        for k in range(vx.shape[2]):
            vx[0, j, k] = np.float32(0.0)
            vx[nx, j, k] = np.float32(0.0)
    for i in range(vy.shape[0]):
        for k in range(vy.shape[2]):
            vy[i, 0, k] = np.float32(0.0)
            vy[i, ny, k] = np.float32(0.0)
    for i in range(vz.shape[0]):
        for j in range(vz.shape[1]):
            vz[i, j, 0] = np.float32(0.0)
            vz[i, j, nz] = np.float32(0.0)


def point_source_pressure_increment(
    rho: float, c: float, dt: float, h: float, Q: float
) -> np.float32:
    """Volume-velocity point source as a pressure increment at one cell center.

    ``ΔP = ρ c² Δt Q / h³`` with *Q* in m³/s (volume velocity).
    """
    return np.float32(rho * c * c * dt * Q / (h * h * h))


@njit(parallel=True, fastmath=False, cache=True)
def update_v(
    p: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    cv: np.float32,
    i_lo: int,
    i_hi: int,
    j_lo: int,
    j_hi: int,
    k_lo: int,
    k_hi: int,
) -> None:
    """Advance staggered velocity one half step: ``v^{n+½} = v^{n-½} - cv · ∂p^n``.

    Linearised Euler on all faces except the outer rigid boundaries
    (``vx[0]``, ``vx[nx]``, etc.). CPML slab cells use the same plain
    derivatives here; :mod:`fdtd.cpml` adds convolutional corrections in slabs.

    Index ranges ``[i_lo, i_hi)`` span pressure cell indices (typically the
    full domain ``0 … nx``).
    """
    nx = p.shape[0]
    ny = p.shape[1]
    nz = p.shape[2]
    i0 = max(i_lo, 1)
    i1 = min(i_hi, nx)
    j0 = max(j_lo, 0)
    j1 = min(j_hi, ny)
    k0 = max(k_lo, 0)
    k1 = min(k_hi, nz)

    for i in prange(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                vx[i, j, k] -= cv * (p[i, j, k] - p[i - 1, j, k])

    j0y = max(j_lo, 1)
    j1y = min(j_hi, ny)
    i0y = max(i_lo, 0)
    i1y = min(i_hi, nx)
    k0y = max(k_lo, 0)
    k1y = min(k_hi, nz)

    for i in prange(i0y, i1y):
        for j in range(j0y, j1y):
            for k in range(k0y, k1y):
                vy[i, j, k] -= cv * (p[i, j, k] - p[i, j - 1, k])

    k0z = max(k_lo, 1)
    k1z = min(k_hi, nz)
    i0z = max(i_lo, 0)
    i1z = min(i_hi, nx)
    j0z = max(j_lo, 0)
    j1z = min(j_hi, ny)

    for i in prange(i0z, i1z):
        for j in range(j0z, j1z):
            for k in range(k0z, k1z):
                vz[i, j, k] -= cv * (p[i, j, k] - p[i, j, k - 1])


@njit(parallel=True, fastmath=False, cache=True)
def update_p(
    p: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    cp: np.float32,
    i_lo: int,
    i_hi: int,
    j_lo: int,
    j_hi: int,
    k_lo: int,
    k_hi: int,
) -> None:
    """Advance pressure one step: ``p^{n+1} = p^n - cp · div v^{n+½}``.

    ``div v`` is the centered difference from face velocities on all cell
    centers in the index range; CPML convolutional corrections are applied
    in ``cpml.py`` after this update.
    """
    i0 = i_lo
    i1 = i_hi
    j0 = j_lo
    j1 = j_hi
    k0 = k_lo
    k1 = k_hi

    for i in prange(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                div_v = (
                    vx[i + 1, j, k]
                    - vx[i, j, k]
                    + vy[i, j + 1, k]
                    - vy[i, j, k]
                    + vz[i, j, k + 1]
                    - vz[i, j, k]
                )
                p[i, j, k] -= cp * div_v


@njit(cache=True)
def inject_point_source(
    p: np.ndarray, i: int, j: int, k: int, value: np.float32
) -> None:
    """Add a volume-velocity point source as a pressure increment at ``(i,j,k)``.

    Pass *value* = ``ρ c² Δt Q / h³`` (see :func:`point_source_pressure_increment`).
    """
    p[i, j, k] += value


def update_v_numpy(
    p: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    cv: np.float32,
    i_lo: int,
    i_hi: int,
    j_lo: int,
    j_hi: int,
    k_lo: int,
    k_hi: int,
) -> None:
    """Reference NumPy implementation: same loops and float32 ops as :func:`update_v`."""
    nx = p.shape[0]
    i0 = max(i_lo, 1)
    i1 = min(i_hi, nx)
    j0 = max(j_lo, 0)
    j1 = min(j_hi, p.shape[1])
    k0 = max(k_lo, 0)
    k1 = min(k_hi, p.shape[2])
    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                vx[i, j, k] -= cv * (p[i, j, k] - p[i - 1, j, k])

    j0y = max(j_lo, 1)
    j1y = min(j_hi, p.shape[1])
    i0y = max(i_lo, 0)
    i1y = min(i_hi, nx)
    k0y = max(k_lo, 0)
    k1y = min(k_hi, p.shape[2])
    for i in range(i0y, i1y):
        for j in range(j0y, j1y):
            for k in range(k0y, k1y):
                vy[i, j, k] -= cv * (p[i, j, k] - p[i, j - 1, k])

    k0z = max(k_lo, 1)
    k1z = min(k_hi, p.shape[2])
    i0z = max(i_lo, 0)
    i1z = min(i_hi, nx)
    j0z = max(j_lo, 0)
    j1z = min(j_hi, p.shape[1])
    for i in range(i0z, i1z):
        for j in range(j0z, j1z):
            for k in range(k0z, k1z):
                vz[i, j, k] -= cv * (p[i, j, k] - p[i, j, k - 1])


def update_p_numpy(
    p: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    cp: np.float32,
    i_lo: int,
    i_hi: int,
    j_lo: int,
    j_hi: int,
    k_lo: int,
    k_hi: int,
) -> None:
    """Reference NumPy implementation: same loops and float32 ops as :func:`update_p`."""
    for i in range(i_lo, i_hi):
        for j in range(j_lo, j_hi):
            for k in range(k_lo, k_hi):
                div_v = (
                    vx[i + 1, j, k]
                    - vx[i, j, k]
                    + vy[i, j + 1, k]
                    - vy[i, j, k]
                    + vz[i, j, k + 1]
                    - vz[i, j, k]
                )
                p[i, j, k] -= cp * div_v


def _domain_args(
    p: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    """Full-domain cell-center index ranges for plain Yee updates."""
    nx, ny, nz = p.shape
    return 0, nx, 0, ny, 0, nz


def step(
    fields: Fields,
    coeffs: UpdateCoeffs,
    slabs: CpmlSlabs,
    cpml: object | None = None,
) -> None:
    """One FDTD leapfrog sub-step on the full grid (CPML slabs use plain derivatives).

    Time staggering (brief §2): pressure *p* is at integer time indices *n*,
    velocity *v* at half steps *n+½*. One call does

    1. ``update_v`` — advance ``v^{n-½} → v^{n+½}`` using ``p^n``
    2. optional ``cpml.apply_v`` — convolutional corrections in CPML slabs
    3. ``update_p`` — advance ``p^n → p^{n+1}`` using ``v^{n+½}``
    4. optional ``cpml.apply_p`` — CPML corrections on pressure update

    Pass ``cpml=None`` (default) to skip absorber corrections.
    """
    i_lo, i_hi, j_lo, j_hi, k_lo, k_hi = _domain_args(fields.p)
    zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
    update_v(
        fields.p,
        fields.vx,
        fields.vy,
        fields.vz,
        coeffs.cv,
        i_lo,
        i_hi,
        j_lo,
        j_hi,
        k_lo,
        k_hi,
    )
    if cpml is not None:
        from fdtd.cpml import apply_v as cpml_apply_v

        cpml_apply_v(fields, cpml, coeffs)
    zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
    update_p(
        fields.p,
        fields.vx,
        fields.vy,
        fields.vz,
        coeffs.cp,
        i_lo,
        i_hi,
        j_lo,
        j_hi,
        k_lo,
        k_hi,
    )
    if cpml is not None:
        from fdtd.cpml import apply_p as cpml_apply_p

        cpml_apply_p(fields, cpml, coeffs)


def step_numpy(
    p: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    coeffs: UpdateCoeffs,
    slabs: CpmlSlabs,
    fields: Fields | None = None,
    cpml: object | None = None,
) -> None:
    """Reference leapfrog step (same order as :func:`step`)."""
    i_lo, i_hi, j_lo, j_hi, k_lo, k_hi = _domain_args(p)
    zero_rigid_face_velocities(vx, vy, vz)
    update_v_numpy(p, vx, vy, vz, coeffs.cv, i_lo, i_hi, j_lo, j_hi, k_lo, k_hi)
    if cpml is not None:
        if fields is None:
            raise ValueError("fields required when cpml is set for step_numpy")
        from fdtd.cpml import apply_v_numpy as cpml_apply_v_numpy

        cpml_apply_v_numpy(p, vx, vy, vz, fields, coeffs, cpml)
    zero_rigid_face_velocities(vx, vy, vz)
    update_p_numpy(p, vx, vy, vz, coeffs.cp, i_lo, i_hi, j_lo, j_hi, k_lo, k_hi)
    if cpml is not None:
        from fdtd.cpml import apply_p_numpy as cpml_apply_p_numpy

        cpml_apply_p_numpy(p, vx, vy, vz, fields, coeffs, cpml)
