"""Analytic acoustics references (validation-only; not public API surface)."""

from __future__ import annotations

import numpy as np

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