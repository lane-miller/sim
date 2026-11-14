"""Analytic baffled-piston radiation impedance references (dimensionless *z*).

Time convention e^{+jωt}; *z* = ∫_S p dS / (ρc S v) with uniform normal velocity.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

_F64 = np.float64
_C128 = np.complex128
_SERIES_X_THRESH = 0.5
_SERIES_M_MAX = 20


def z_baffled_circle(ka: NDArray | float) -> NDArray:
    """Circular piston on an infinite rigid baffle.

    ``ka = k a`` with ``a = sqrt(S/π)`` and ``S`` the piston area.

    ``z = 1 − J₁(2ka)/(ka) + j H₁(2ka)/(ka)`` (Struve function ``H₁``).

    Scalar input returns a scalar; array input preserves shape.
    """
    from scipy.special import j1, struve

    scalar = np.isscalar(ka) or (isinstance(ka, np.ndarray) and ka.ndim == 0)
    x = np.atleast_1d(np.asarray(ka, dtype=_F64))
    z = np.zeros(x.shape, dtype=_C128)
    for i, val in enumerate(x.flat):
        if abs(val) < 1e-14:
            z.flat[i] = 0.0 + 0.0j
            continue
        u = 2.0 * float(val)
        z.flat[i] = 1.0 - j1(u) / float(val) + 1.0j * struve(1, u) / float(val)
    if scalar:
        return z.item()
    return z.reshape(np.asarray(ka).shape)


def _moment_series(n: int, lam: NDArray, R: float, *, m_max: int = _SERIES_M_MAX) -> NDArray:
    """``M_n = Σ_m (−λ)^m R^{n+m+1} / (m! (n+m+1))``."""
    lam = np.atleast_1d(np.asarray(lam, dtype=_C128))
    acc = np.zeros_like(lam, dtype=_C128)
    r_pow = R ** (n + 1)
    for m in range(m_max + 1):
        acc += (-lam) ** m * r_pow / (math.factorial(m) * (n + m + 1))
        r_pow *= R
    return acc


def _moment_closed(n: int, lam: NDArray, R: float) -> NDArray:
    """Closed-form ``∫₀^R r^n e^{−λr} dr`` (``λ = jk``)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=_C128))
    x = lam * R
    out = np.zeros_like(lam, dtype=_C128)
    zero = np.abs(lam) < 1e-30
    if n == 0:
        out[zero] = R
    elif n == 1:
        out[zero] = 0.5 * R * R
    else:
        out[zero] = R**3 / 3.0

    nz = ~zero
    if not np.any(nz):
        return out
    lz = lam[nz]
    xz = x[nz]
    ex = np.exp(-xz)
    if n == 0:
        out[nz] = (1.0 - ex) / lz
    elif n == 1:
        out[nz] = (1.0 - ex * (1.0 + xz)) / (lz**2)
    else:
        out[nz] = (2.0 - ex * (2.0 + 2.0 * xz + xz**2)) / (lz**3)
    return out


def _moment_r(n: int, lam: NDArray, R: float) -> NDArray:
    lam = np.atleast_1d(np.asarray(lam, dtype=_C128))
    x = lam * R
    use_series = np.abs(x) < _SERIES_X_THRESH
    out = np.zeros_like(lam, dtype=_C128)
    if np.any(use_series):
        out[use_series] = _moment_series(n, lam[use_series], R)
    if np.any(~use_series):
        out[~use_series] = _moment_closed(n, lam[~use_series], R)
    return out


def _r_segment_integral(
    k: NDArray,
    R: float,
    W: float,
    H: float,
    phi: float,
) -> NDArray:
    """``∫₀^R (W − r cosφ)(H − r sinφ) e^{−jkr} dr`` with ``λ = jk``."""
    cp = math.cos(phi)
    sp = math.sin(phi)
    lam = (1j * np.asarray(k, dtype=_F64)).astype(_C128)
    m0 = _moment_r(0, lam, R)
    m1 = _moment_r(1, lam, R)
    m2 = _moment_r(2, lam, R)
    return W * H * m0 - (W * sp + H * cp) * m1 + cp * sp * m2


def _rect_rayleigh_I(
    k: NDArray,
    W: float,
    H: float,
    *,
    n_phi: int,
) -> NDArray:
    """``∫ I_r(φ) dφ`` on ``[0,W]×[0,H]`` via Gauss–Legendre on both φ sub-intervals."""
    k = np.atleast_1d(np.asarray(k, dtype=_F64))
    phid = math.atan2(H, W)
    segments: tuple[tuple[float, float, str], ...] = (
        (0.0, phid, "w"),
        (phid, 0.5 * math.pi, "h"),
    )
    I = np.zeros(k.shape, dtype=_C128)
    for phi_lo, phi_hi, edge in segments:
        if phi_hi <= phi_lo + 1e-15:
            continue
        nodes, weights = leggauss(n_phi)
        scale = 0.5 * (phi_hi - phi_lo)
        shift = 0.5 * (phi_hi + phi_lo)
        for xi, wt in zip(nodes, weights):
            phi = scale * float(xi) + shift
            cp = math.cos(phi)
            sp = math.sin(phi)
            if edge == "w":
                rmax = W / cp if cp > 1e-14 else W
            else:
                rmax = H / sp if sp > 1e-14 else H
            I += float(wt) * scale * _r_segment_integral(k, rmax, W, H, phi)
    return I


def z_baffled_rect(
    k: NDArray | float,
    W: float,
    H: float,
    *,
    n_phi: int = 64,
) -> NDArray:
    """Rectangular baffled piston on ``[0,W]×[0,H]`` (translation invariant).

    ``S = W·H``,

        z = (2 j k / (π S)) ∫ I_r(φ) dφ,

    with ``I_r`` the closed-form ``r`` integral along each φ-ray (``λ = jk``).
    The kernel is translation invariant; no centering is required relative to FDTD.
    """
    if W <= 0.0 or H <= 0.0:
        raise ValueError("W and H must be positive")
    S = W * H
    k_arr = np.atleast_1d(np.asarray(k, dtype=_F64))
    scalar = np.isscalar(k) or (isinstance(k, np.ndarray) and k.ndim == 0)
    I = _rect_rayleigh_I(k_arr, W, H, n_phi=n_phi)
    z = (2.0j * k_arr.astype(_C128) / (math.pi * S)) * I
    if scalar:
        return z.item()
    return z