"""Gaussian velocity pulse and rigid-box / radiator face boundary (Stage C)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from fdtd.config import Derived
from fdtd.grid import Grid

_FLOAT32 = np.dtype(np.float32)


def velocity_at_half_step(n: int, dt: float, sigma: float, t0: float) -> float:
    """Pulse at staggered time ``(n + ½) Δt`` (float64 time; shared by apply and checks)."""
    t_half = (float(n) + 0.5) * float(dt)
    return velocity_pulse(t_half, sigma, t0)


def velocity_pulse(t: float, sigma: float, t0: float) -> float:
    """Unit-amplitude Gaussian velocity pulse (no time derivative).

    ``v(t) = exp(−(t − t0)² / (2σ²))``, peak 1 at ``t = t0``.
    ``σ`` and ``t0`` come from :func:`fdtd.config.derive`.
    """
    z = (t - t0) / sigma
    return math.exp(-0.5 * z * z)


def velocity_pulse_derivative(t: float, sigma: float, t0: float) -> float:
    """Time derivative of :func:`velocity_pulse` (analytic Gaussian)."""
    z = (t - float(t0)) / float(sigma)
    return float(velocity_pulse(t, sigma, t0) * (-z / float(sigma)))


def pulse_spectrum(
    f: np.ndarray | float,
    sigma: float,
    t0: float,
) -> np.ndarray | complex:
    """Analytic Fourier transform of :func:`velocity_pulse` (``e^{+jωt}`` convention).

    ``V(ω) = σ √(2π) exp(−ω²σ²/2) exp(+j ω t0)`` with ``ω = 2π f``.
    Amplitude at DC is ``σ √(2π)``; tests compare normalized magnitude to the
    band-limit target at ``pulse_f_factor · f_max``.

    Scalar *f* returns a Python ``complex``; array *f* returns ``ndarray``.
    """
    scalar_in = np.isscalar(f)
    f_arr = np.atleast_1d(np.asarray(f, dtype=np.float64))
    omega = 2.0 * math.pi * f_arr
    pref = sigma * math.sqrt(2.0 * math.pi)
    out = pref * np.exp(-0.5 * (omega * sigma) ** 2) * np.exp(1j * omega * t0)
    if scalar_in:
        return complex(out[0])
    if isinstance(f, np.ndarray) and f.ndim == 0:
        return complex(out[0])
    return out


def _assert_indices_1d(name: str, idx: np.ndarray, size: int) -> None:
    if idx.size == 0:
        return
    imin = int(idx.min())
    imax = int(idx.max())
    if imin < 0 or imax >= size:
        raise ValueError(
            f"{name} flat index out of range: min={imin}, max={imax}, size={size}"
        )


@njit(cache=True)
def apply_rigid(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    idx_x: np.ndarray,
    idx_y: np.ndarray,
    idx_z: np.ndarray,
) -> None:
    """Set normal velocity to zero on all indexed rigid box faces (1-D flat views)."""
    for q in range(idx_x.shape[0]):
        vx[idx_x[q]] = np.float32(0.0)
    for q in range(idx_y.shape[0]):
        vy[idx_y[q]] = np.float32(0.0)
    for q in range(idx_z.shape[0]):
        vz[idx_z[q]] = np.float32(0.0)


@njit(cache=True)
def apply_radiator(
    vz: np.ndarray,
    idx_r: np.ndarray,
    w_r: np.ndarray,
    v_now: np.float32,
) -> None:
    """Prescribe radiator face velocity: ``vz[idx] = w_i · v_now``.

    On a staggered grid each face carries flux ``h² v_z``; the area fraction
    weight belongs on the drive (``w_i``), not on the volume sum ``Q = h² Σ v_z``.
    Pressure averaging for ``Z`` uses ``⟨p⟩ = Σ w p / Σ w``.
    """
    for q in range(idx_r.shape[0]):
        vz[idx_r[q]] = w_r[q] * v_now


@dataclass(frozen=True)
class BoxBoundary:
    """Flat indices and weights for rigid box faces and the +z radiator.

    With ``drive_enabled`` (default), radiator faces get ``w · v(t)``; when off
    they are held at zero like rigid faces (obstacle / exterior-source tests).
    """

    idx_vx: np.ndarray
    idx_vy: np.ndarray
    idx_vz: np.ndarray
    idx_radiator: np.ndarray
    w_radiator: np.ndarray
    h: np.float32
    h2: np.float32
    s_grid: np.float32
    sigma: np.float32
    t0: np.float32
    dt: np.float32
    drive_enabled: bool = True

    @classmethod
    def from_grid(
        cls,
        grid: Grid,
        derived: Derived,
        *,
        drive_enabled: bool = True,
    ) -> BoxBoundary:
        if grid.cfg.geometry not in ("box", "baffle"):
            raise ValueError("BoxBoundary requires box or baffle geometry")
        if grid.cfg.geometry == "box" and grid.cfg.box is None:
            raise ValueError("BoxBoundary requires cfg.box in box geometry mode")
        if grid.rigid_vx is None or grid.radiator_face_indices is None:
            raise ValueError("grid has no box/baffle geometry")
        if grid.k_face_z is None:
            raise ValueError("grid has no k_face_z")

        vx_shape = (grid.nx + 1, grid.ny, grid.nz)
        vy_shape = (grid.nx, grid.ny + 1, grid.nz)
        vz_shape = (grid.nx, grid.ny, grid.nz + 1)

        idx_vx = np.flatnonzero(grid.rigid_vx).astype(np.int64)
        idx_vy = np.flatnonzero(grid.rigid_vy).astype(np.int64)
        idx_vz = np.flatnonzero(grid.rigid_vz).astype(np.int64)

        ir, jr, kr = grid.radiator_face_indices
        kf = int(grid.k_face_z)
        if ir.size:
            if not np.all(kr == kf):
                bad = np.nonzero(kr != kf)[0]
                raise ValueError(
                    f"radiator vz faces must lie on k_face_z={kf}; "
                    f"violations at {bad[:5]} (kr={kr[bad[:5]]})"
                )
            try:
                rad_idx = np.ravel_multi_index((ir, jr, kr), vz_shape)
            except ValueError as exc:
                raise ValueError("radiator (i,j,k) out of bounds for vz") from exc
        else:
            rad_idx = np.array([], dtype=np.int64)

        overlap = np.intersect1d(idx_vz, rad_idx)
        if overlap.size:
            raise ValueError(
                f"radiator vz indices overlap rigid_vz mask ({overlap.size} cells)"
            )

        _assert_indices_1d("idx_vx", idx_vx, int(np.prod(vx_shape)))
        _assert_indices_1d("idx_vy", idx_vy, int(np.prod(vy_shape)))
        _assert_indices_1d("idx_vz", idx_vz, int(np.prod(vz_shape)))
        _assert_indices_1d("idx_radiator", rad_idx, int(np.prod(vz_shape)))

        w = grid.radiator_weights
        if w is None:
            w = np.array([], dtype=np.float64)
        if w.shape[0] != rad_idx.shape[0]:
            raise ValueError("radiator_weights length mismatch vs radiator faces")

        s_grid = np.float32(grid.radiator_area_grid or 0.0)
        h = np.float32(grid.h)
        return cls(
            idx_vx=idx_vx,
            idx_vy=idx_vy,
            idx_vz=idx_vz,
            idx_radiator=rad_idx,
            w_radiator=w.astype(_FLOAT32),
            h=h,
            h2=np.float32(h * h),
            s_grid=s_grid,
            sigma=np.float32(derived.pulse_sigma),
            t0=np.float32(derived.pulse_t0),
            dt=np.float32(derived.dt),
            drive_enabled=bool(drive_enabled),
        )

    def v_at_step(self, n: int) -> np.float32:
        """Prescribed radiator velocity at leapfrog time ``(n + ½) Δt``."""
        v = velocity_at_half_step(
            n, float(self.dt), float(self.sigma), float(self.t0)
        )
        return np.float32(v)

    def apply(self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, n: int) -> None:
        for name, arr in (("vx", vx), ("vy", vy), ("vz", vz)):
            if not arr.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous for flat boundary writes")
        vx_f = vx.reshape(-1)
        vy_f = vy.reshape(-1)
        vz_f = vz.reshape(-1)
        if not (
            np.shares_memory(vx_f, vx)
            and np.shares_memory(vy_f, vy)
            and np.shares_memory(vz_f, vz)
        ):
            raise ValueError("boundary apply requires reshape(-1) views of vx/vy/vz")
        apply_rigid(vx_f, vy_f, vz_f, self.idx_vx, self.idx_vy, self.idx_vz)
        v_now = (
            self.v_at_step(n)
            if self.drive_enabled
            else np.float32(0.0)
        )
        apply_radiator(vz_f, self.idx_radiator, self.w_radiator, v_now)

    def volume_velocity(self, vz: np.ndarray) -> float:
        """Discrete volume velocity ``Q = h² Σ v_z`` on radiator faces (no extra *w*)."""
        if self.idx_radiator.size == 0:
            return 0.0
        vn = vz.reshape(-1)[self.idx_radiator].astype(np.float64)
        return float(self.h2 * np.sum(vn))

    def expected_volume_velocity(self, n: int) -> float:
        """``S_grid · v((n+½)Δt)`` with ``S_grid = h² Σ w`` (see :func:`velocity_at_half_step`)."""
        if not self.drive_enabled:
            return 0.0
        return float(self.s_grid * self.v_at_step(n))