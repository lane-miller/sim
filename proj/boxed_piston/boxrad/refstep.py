"""Energy-stable reference leapfrog on a cut-cell grid (v0.3 step 5; see tests/ref_refstep.py)."""

from __future__ import annotations

import numpy as np

_F64 = np.float64


def _div(u, alpha):
    """Divergence per cell: sum_f s_{c,f} alpha_f u_f (s=+1 on the cell's + face)."""
    ax, ay, az = alpha
    ux, uy, uz = u
    fx, fy, fz = ax * ux, ay * uy, az * uz
    return (fx[1:] - fx[:-1]) + (fy[:, 1:] - fy[:, :-1]) + (fz[:, :, 1:] - fz[:, :, :-1])


def closed_alpha(alpha):
    """Rigid domain: zero alpha on the outer boundary faces."""
    ax, ay, az = (a.copy() for a in alpha)
    ax[0], ax[-1] = 0, 0
    ay[:, 0], ay[:, -1] = 0, 0
    az[:, :, 0], az[:, :, -1] = 0, 0
    return ax, ay, az


def step(p, u, beta_t, alpha, lam, nsteps):
    """u^{n+1} = u^n - lam*grad p^{n+1/2} (alpha>0 faces);  p^{n+3/2} = p^{n+1/2} - lam*div(alpha u^{n+1})/beta_t.

    Energy E_n = 1/2 sum beta_t p^{n+1/2}^2 + 1/2 sum alpha u^n u^{n+1}  (exactly conserved).
    """
    ax, ay, az = alpha
    fluid = beta_t > 0
    inv_b = np.where(fluid, 1.0 / np.where(fluid, beta_t, 1.0), 0.0)
    ux, uy, uz = (x.copy() for x in u)
    p = p.copy()
    H = []
    for _ in range(nsteps):
        gx = np.zeros_like(ux)
        gx[1:-1] = p[1:] - p[:-1]
        gy = np.zeros_like(uy)
        gy[:, 1:-1] = p[:, 1:] - p[:, :-1]
        gz = np.zeros_like(uz)
        gz[:, :, 1:-1] = p[:, :, 1:] - p[:, :, :-1]
        nx_ = np.where(ax > 0, ux - lam * gx, 0.0)
        ny_ = np.where(ay > 0, uy - lam * gy, 0.0)
        nz_ = np.where(az > 0, uz - lam * gz, 0.0)
        E = 0.5 * np.sum(beta_t * p * p) + 0.5 * (
            np.sum(ax * ux * nx_) + np.sum(ay * uy * ny_) + np.sum(az * uz * nz_)
        )
        H.append(E)
        ux, uy, uz = nx_, ny_, nz_
        p = p - lam * inv_b * _div((ux, uy, uz), (ax, ay, az))
    return p, (ux, uy, uz), np.array(H)


def sum_alpha(alpha):
    ax, ay, az = alpha
    return ax[:-1] + ax[1:] + ay[:, :-1] + ay[:, 1:] + az[:, :, :-1] + az[:, :, 1:]


def lam_max(beta_t, alpha, iters=300, seed=1):
    """Power iteration on B^-1 L (L = weighted Laplacian) -> largest eigenvalue (rho=c=h=1)."""
    fluid = beta_t > 0
    inv_b = np.where(fluid, 1.0 / np.where(fluid, beta_t, 1.0), 0.0)
    ax, ay, az = alpha
    v = np.where(fluid, np.random.default_rng(seed).standard_normal(beta_t.shape), 0.0)
    ev = 0.0
    for _ in range(iters):
        gx = np.zeros_like(ax)
        gx[1:-1] = v[1:] - v[:-1]
        gy = np.zeros_like(ay)
        gy[:, 1:-1] = v[:, 1:] - v[:, :-1]
        gz = np.zeros_like(az)
        gz[:, :, 1:-1] = v[:, :, 1:] - v[:, :, :-1]
        Lv = -_div((gx, gy, gz), alpha)
        w = inv_b * Lv
        ev = np.sum(v * Lv) / np.sum(beta_t * v * v)
        v = w / np.sqrt(np.sum(beta_t * w * w))
    return ev


def init_velocity(alpha: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Face-centered velocity arrays matching *alpha* shapes, initially zero."""
    return tuple(np.zeros_like(a, dtype=_F64) for a in alpha)