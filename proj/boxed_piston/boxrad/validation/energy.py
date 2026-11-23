"""Discrete energy functional for the cut-cell leapfrog scheme (step 10a)."""

from __future__ import annotations

import numpy as np

_F64 = np.float64


def discrete_energy(
    p: np.ndarray,
    v_now: tuple[np.ndarray, np.ndarray, np.ndarray],
    v_prev: tuple[np.ndarray, np.ndarray, np.ndarray],
    beta_tilde: np.ndarray,
    alpha: tuple[np.ndarray, np.ndarray, np.ndarray],
    rho: float,
    c: float,
    h: float,
) -> float:
    """Conserved cut-cell energy with physical scaling (see ``tests/ref_refstep.py``, A5).

    ``p`` is cell pressure at step *n*; ``v_prev`` and ``v_now`` are face velocities at
    *n*−½ and *n*+½; ``beta_tilde`` and ``alpha`` are cell β̃ and face fractions.
    Sums accumulate in float64; array operands keep their own dtype.
    """
    rho_f = float(rho)
    c_f = float(c)
    h_f = float(h)
    h3 = h_f**3
    e_p = np.sum(beta_tilde * p * p, dtype=_F64) / (rho_f * c_f * c_f)
    ax, ay, az = alpha
    vx0, vy0, vz0 = v_prev
    vx1, vy1, vz1 = v_now
    e_v = rho_f * (
        np.sum(ax * vx0 * vx1, dtype=_F64)
        + np.sum(ay * vy0 * vy1, dtype=_F64)
        + np.sum(az * vz0 * vz1, dtype=_F64)
    )
    return float(0.5 * h3 * (e_p + e_v))