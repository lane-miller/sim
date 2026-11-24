"""Analytic acoustics references (validation-only; not public API surface)."""

from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn

_F64 = np.float64
_C128 = np.complex128


def monopole_free(
    points: np.ndarray,
    f: np.ndarray | float,
    source: np.ndarray,
    rho: float,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-field monopole ``p/Q`` with convention ``e^{+jωt}``.

    Args:
        points: ``(n_pts, 3)`` evaluation points [m].
        f: Frequency(ies) [Hz].
        source: Source position ``(3,)`` [m].
        rho: Density [kg/m³].
        c: Sound speed [m/s].

    Returns:
        ``(F, P)`` with ``F`` float64 ``(n_f,)`` and ``P`` complex128 ``(n_f, n_pts)``
        so ``P / Q`` equals ``j ω ρ exp(−jkR) / (4π R)``.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=_F64))
    src = np.asarray(source, dtype=_F64).reshape(3)
    f_arr = np.atleast_1d(np.asarray(f, dtype=_F64)).ravel()
    R = np.linalg.norm(pts - src, axis=1)
    R = np.maximum(R, 1e-30)
    k = 2.0 * np.pi * f_arr / float(c)
    omega = 2.0 * np.pi * f_arr
    phase = np.exp(-1j * k[:, np.newaxis] * R[np.newaxis, :])
    coeff = (1j * omega * float(rho) / (4.0 * np.pi))[:, np.newaxis]
    P = (coeff * phase / R[np.newaxis, :]).astype(_C128, copy=False)
    return f_arr.astype(_F64, copy=False), P


def _h2(n: int, x: np.ndarray | float, d: bool = False) -> np.ndarray:
    return spherical_jn(n, x, derivative=d) - 1j * spherical_yn(n, x, derivative=d)


def sphere_point_source(
    points: np.ndarray,
    f: float,
    center: np.ndarray,
    a: float,
    source: np.ndarray,
    rho: float,
    c: float,
    *,
    nmax: int | None = None,
    part: str = "total",
) -> np.ndarray:
    """Rigid-sphere scattering from a monopole; returns ``p/Q`` (Pa per m³/s), ``e^{+jωt}``."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pts = np.atleast_2d(np.asarray(points, float)) - np.asarray(center, float)
        s = np.asarray(source, float) - np.asarray(center, float)
        r = np.linalg.norm(pts, axis=1)
        r0 = float(np.linalg.norm(s))
        cosg = np.clip((pts @ s) / (r * r0), -1.0, 1.0)
        w = 2 * np.pi * f
        k = w / c
        ka = k * a
        if nmax is None:
            q = float(np.max(np.minimum(r, r0) / np.maximum(r, r0)))
            if q > 0.9:
                raise ValueError("probe radius within 10% of source radius: series converges too slowly")
            nmax = min(400, int(k * max(r0, a)) + 20 + int(np.log(1e-12) / np.log(q)))
        rl = np.minimum(r, r0)
        rg = np.maximum(r, r0)
        pin = np.zeros(len(r), complex)
        psc = np.zeros(len(r), complex)
        for n in range(nmax + 1):
            Pn = eval_legendre(n, cosg)
            pin += (2 * n + 1) * spherical_jn(n, k * rl) * _h2(n, k * rg) * Pn
            coef = spherical_jn(n, ka, derivative=True) / _h2(n, ka, True)
            if not np.isfinite(coef):
                coef = 0.0
            psc += (2 * n + 1) * coef * _h2(n, k * r0) * _h2(n, k * r) * Pn
        pre = 1j * w * rho
        pin *= -1j * k / (4 * np.pi) * pre
        psc *= 1j * k / (4 * np.pi) * pre
        return {"inc": pin, "scat": psc, "total": pin + psc}[part]


def sphere_surface_moments(
    f: float,
    a: float,
    r0: float,
    rho: float,
    c: float,
) -> tuple[complex, complex]:
    """Mean surface pressure ``c₀`` and axial force ``F`` for unit monopole ``Q`` (e^{+jωt}).

    ``⟨p⟩ = c₀`` on the rigid sphere of radius ``a``. Force on the sphere along the
    axis from the center toward the source at distance ``r₀`` is
    ``F = −(4π a²/3) c₁`` (series moments of :func:`sphere_point_source` at ``r = a``).
    """
    w = 2.0 * np.pi * float(f)
    k = w / float(c)
    ka = k * float(a)
    kr0 = k * float(r0)

    def cn_over_q(n: int) -> complex:
        hp = _h2(n, kr0)
        hpd = _h2(n, ka, d=True)
        num = 1j * w * float(rho) * (2 * n + 1) * (-k * hp)
        den = 4.0 * np.pi * (ka * ka) * hpd
        if abs(den) < 1e-300:
            return 0.0 + 0j
        return complex(num / den)

    c0 = cn_over_q(0)
    c1 = cn_over_q(1)
    force = -(4.0 * np.pi * float(a) ** 2 / 3.0) * c1
    return c0, force