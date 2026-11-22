"""Small-cell stability floor and thin-wall detection (v0.3 step 5)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from boxrad.config import Config, Derived
from boxrad.cutcell import Fractions, Grid
from boxrad.geom.geometry import Geometry
from boxrad.geom.mesh import inside
from boxrad.refstep import sum_alpha

_F64 = np.float64


def courant_lambda(cfg: Config, derived: Derived) -> float:
    """Courant number λ = c Δt / h from :func:`~boxrad.config.derive`."""
    return float(cfg.air.c * derived.dt / derived.h)


def apply_floor(
    fr: Fractions,
    lam: float,
    gamma: float = 1.05,
    *,
    h: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Raise β on cut cells to meet the small-cell stability floor."""
    beta = fr.beta
    shape = beta.shape
    if h is None:
        h = 1.0
    h = float(h)
    s = sum_alpha(fr.alpha)
    cut = (beta > 0.0) & (beta < 1.0)
    thresh = gamma * (lam * lam * 0.5) * s
    beta_tilde = np.where(
        cut,
        np.maximum(beta, np.minimum(1.0, thresh)),
        beta,
    )
    floored = cut & (beta_tilde > beta + 1e-15)
    n_floored = int(np.sum(floored))
    cut = (beta > 1e-12) & (beta < 1.0 - 1e-12)
    min_raw = float(np.min(beta[cut])) if np.any(cut) else 0.0
    dv = float(np.sum((beta_tilde - beta) * h**3))
    a_wall = float(np.sum(fr.patch_A)) if fr.patch_A.size else 0.0
    if a_wall > 0.0:
        delta_eq = dv / a_wall
    else:
        delta_eq = 0.0
    delta_eq_h = delta_eq / h if h > 0.0 else 0.0
    if delta_eq_h <= 0.02:
        floor_status = "ok"
    elif delta_eq_h <= 0.05:
        floor_status = "check"
    else:
        floor_status = "warn"
    worst_idx = (0, 0, 0)
    worst_beta = 0.0
    worst_tilde = 0.0
    if n_floored > 0:
        lift = (beta_tilde - beta) * floored
        flat = int(np.argmax(lift))
        worst_idx = np.unravel_index(flat, shape)
        worst_beta = float(beta[worst_idx])
        worst_tilde = float(beta_tilde[worst_idx])
    stats = {
        "n_floored": n_floored,
        "min_raw_beta_cut": min_raw,
        "delta_V": dv,
        "A_wall": a_wall,
        "delta_eq": delta_eq,
        "delta_eq_over_h": delta_eq_h,
        "floor_status": floor_status,
        "worst_cell": worst_idx,
        "worst_beta": worst_beta,
        "worst_beta_tilde": worst_tilde,
        "sum_alpha_median": float(np.median(s[cut])) if np.any(cut) else 0.0,
        "floor_level_median": float(np.median(thresh[cut])) if np.any(cut) else 0.0,
    }
    return beta_tilde, stats


def stability_ok(
    beta_tilde: np.ndarray,
    alpha: tuple[np.ndarray, ...],
    lam: float,
) -> tuple[bool, float]:
    """Check β̃_c ≥ λ²/2 Σ α on every cell with β̃ > 0."""
    shape = beta_tilde.shape
    req = (lam * lam * 0.5) * sum_alpha(alpha)
    active = beta_tilde > 0.0
    if not np.any(active):
        return True, float("inf")
    margin = beta_tilde[active] / np.maximum(req[active], 1e-30)
    worst = float(np.min(margin))
    ok = worst >= 1.0 - 1e-12
    return ok, worst


def _cell_patch_sums(fr: Fractions, nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    ncell = nx * ny * nz
    a_c = np.zeros(ncell, dtype=_F64)
    n_c = np.zeros((ncell, 3), dtype=_F64)
    for idx, nid in enumerate(fr.patch_cell):
        a = fr.patch_A[idx]
        a_c[nid] += a
        n_c[nid] += a * fr.patch_normal[idx]
    return a_c.reshape(nx, ny, nz), n_c.reshape(nx, ny, nz, 3)


@njit(cache=True)
def _subsample_points(
    x0x: float,
    x0y: float,
    x0z: float,
    h: float,
    i: int,
    j: int,
    k: int,
    pts: np.ndarray,
) -> None:
    n = 0
    for di in range(4):
        for dj in range(4):
            for dk in range(4):
                pts[n, 0] = x0x + (i + (di + 0.5) / 4.0) * h
                pts[n, 1] = x0y + (j + (dj + 0.5) / 4.0) * h
                pts[n, 2] = x0z + (k + (dk + 0.5) / 4.0) * h
                n += 1


def _face_components(fluid: np.ndarray) -> list[tuple[int, frozenset[int]]]:
    """6-connected components of fluid voxels; faces: 0=x-,1=x+,2=y-,3=y+,4=z-,5=z+."""
    n = 4
    visited = np.zeros((n, n, n), dtype=np.bool_)
    components: list[tuple[int, frozenset[int]]] = []

    for di in range(n):
        for dj in range(n):
            for dk in range(n):
                if not fluid[di, dj, dk] or visited[di, dj, dk]:
                    continue
                stack = [(di, dj, dk)]
                visited[di, dj, dk] = True
                faces: set[int] = set()
                while stack:
                    ci, cj, ck = stack.pop()
                    if ci == 0:
                        faces.add(0)
                    if ci == n - 1:
                        faces.add(1)
                    if cj == 0:
                        faces.add(2)
                    if cj == n - 1:
                        faces.add(3)
                    if ck == 0:
                        faces.add(4)
                    if ck == n - 1:
                        faces.add(5)
                    for ni, nj, nk in (
                        (ci - 1, cj, ck),
                        (ci + 1, cj, ck),
                        (ci, cj - 1, ck),
                        (ci, cj + 1, ck),
                        (ci, cj, ck - 1),
                        (ci, cj, ck + 1),
                    ):
                        if 0 <= ni < n and 0 <= nj < n and 0 <= nk < n:
                            if fluid[ni, nj, nk] and not visited[ni, nj, nk]:
                                visited[ni, nj, nk] = True
                                stack.append((ni, nj, nk))
                components.append((len(components), frozenset(faces)))
    return components


def _is_thin_cell(mesh, grid: Grid, i: int, j: int, k: int) -> bool:
    pts = np.zeros((64, 3), dtype=_F64)
    _subsample_points(
        float(grid.x0[0]),
        float(grid.x0[1]),
        float(grid.x0[2]),
        float(grid.h),
        i,
        j,
        k,
        pts,
    )
    fluid = ~inside(mesh, pts)
    fluid_grid = fluid.reshape(4, 4, 4)
    if not np.any(fluid_grid):
        return False
    comps = _face_components(fluid_grid)
    if len(comps) < 2:
        return False
    face_sets = [c[1] for c in comps if c[1]]
    if len(face_sets) < 2:
        return False
    for a in range(len(face_sets)):
        for b in range(a + 1, len(face_sets)):
            fi = face_sets[a] | face_sets[b]
            if (0 in fi and 1 in fi) or (2 in fi and 3 in fi) or (4 in fi and 5 in fi):
                return True
    return False


def thin_wall_cells(
    fr: Fractions,
    geometry: Geometry,
    grid: Grid,
    max_thin: int = 0,
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """Detect opposing-wall (thin) cut cells; raise if count exceeds *max_thin*."""
    from boxrad.cutcell import _STATE_CUT

    cut = _STATE_CUT
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    h = float(grid.h)
    h2 = h * h
    a_c, n_c = _cell_patch_sums(fr, nx, ny, nz)
    state = fr.state
    candidates: list[tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if state[i, j, k] != cut:
                    continue
                ac = a_c[i, j, k]
                if ac <= 0.5 * h2:
                    continue
                nvec = n_c[i, j, k]
                if float(np.linalg.norm(nvec)) >= 0.2 * ac:
                    continue
                candidates.append((i, j, k))
    confirmed: list[tuple[int, int, int]] = []
    mesh = geometry.solid
    for cell in candidates:
        if _is_thin_cell(mesh, grid, cell[0], cell[1], cell[2]):
            confirmed.append(cell)
    stats = {"n_candidates": len(candidates), "n_confirmed": len(confirmed)}
    if len(confirmed) > max_thin:
        positions = []
        for i, j, k in confirmed:
            c = grid.x0 + (np.array([i, j, k], dtype=_F64) + 0.5) * h
            positions.append(f"({c[0]:.6g}, {c[1]:.6g}, {c[2]:.6g})")
        raise ValueError(
            f"thin-wall cells ({len(confirmed)} > max_thin={max_thin}): " + ", ".join(positions)
        )
    return confirmed, stats