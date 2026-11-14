"""Exterior Kirchhoff–Helmholtz integration over a closed box surface.

Time convention e^{+jωt}; Green's function G = e^{−jkR}/(4πR) with k = ω/c.

Normals ``n`` point outward from the box volume into the air (same as the FDTD
radiation-side face normal). All arrays use ``numpy.complex128`` for spectral data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

_C128 = np.complex128
_F64 = np.float64


@dataclass(frozen=True)
class SurfacePatches:
    """Axis-aligned box surface quadrature patches (face-center rule)."""

    centers: np.ndarray  # (N, 3) float64
    normals: np.ndarray  # (N, 3) float64 unit outward
    areas: np.ndarray  # (N,) float64
    h: float  # characteristic patch edge length (m)


def _axis_aligned_box_patches(
    ox: float,
    oy: float,
    oz: float,
    W: float,
    H: float,
    D: float,
    h: float,
) -> SurfacePatches:
    """h×h patches on six faces of [ox, ox+W]×[oy, oy+H]×[oz, oz+D] at Yee v-face centers."""
    h = float(h)

    def _axis_centers(x0: float, length: float) -> np.ndarray:
        n = max(1, int(round(length / h)))
        return x0 + (np.arange(n, dtype=_F64) + 0.5) * h

    xs_in = _axis_centers(ox, W)
    ys_in = _axis_centers(oy, H)
    zs_in = _axis_centers(oz, D)

    centers_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    areas_list: list[float] = []

    def _add_face(
        fixed_coord: float,
        axis_fixed: int,
        u_centers: np.ndarray,
        v_centers: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        nu, nv = len(u_centers), len(v_centers)
        pts = np.zeros((nu * nv, 3), dtype=_F64)
        idx = {0: fixed_coord, 1: None, 2: None}
        tang = [i for i in range(3) if i != axis_fixed]
        uu, vv = np.meshgrid(u_centers, v_centers, indexing="ij")
        pts[:, axis_fixed] = fixed_coord
        pts[:, tang[0]] = uu.ravel()
        pts[:, tang[1]] = vv.ravel()
        centers_list.append(pts)
        normals_list.append(np.tile(normal, (nu * nv, 1)))
        areas_list.extend([h * h] * (nu * nv))

    ex = np.array([1.0, 0.0, 0.0], dtype=_F64)
    ey = np.array([0.0, 1.0, 0.0], dtype=_F64)
    ez = np.array([0.0, 0.0, 1.0], dtype=_F64)

    _add_face(ox, 0, ys_in, zs_in, -ex)
    _add_face(ox + W, 0, ys_in, zs_in, ex)
    _add_face(oy, 1, xs_in, zs_in, -ey)
    _add_face(oy + H, 1, xs_in, zs_in, ey)
    _add_face(oz, 2, xs_in, ys_in, -ez)
    _add_face(oz + D, 2, xs_in, ys_in, ez)

    centers = np.vstack(centers_list)
    normals = np.vstack(normals_list)
    areas = np.asarray(areas_list, dtype=_F64)
    return SurfacePatches(centers=centers, normals=normals, areas=areas, h=h)


def box_surface_patches(
    W: float,
    H: float,
    D: float,
    h: float,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SurfacePatches:
    """Build h×h patches on all six faces of the physical (rigid) box.

    ``origin`` is the minimum corner (x₀, y₀, z₀); the box occupies
    [x₀, x₀+W] × [y₀, y₀+H] × [z₀, z₀+D]. Patch centers are Yee velocity-face
    positions: (i+½)h along each in-face edge, with the face normal along a grid axis.
    """
    ox, oy, oz = origin
    return _axis_aligned_box_patches(ox, oy, oz, W, H, D, h)


def face_p_from_cells(
    p_m3: np.ndarray,
    p_m1: np.ndarray,
    p_p1: np.ndarray,
    p_p3: np.ndarray,
    *,
    order: int = 4,
) -> np.ndarray:
    """Pressure on a velocity face from staggered cell-center samples along the normal.

    Samples lie at −3h/2, −h/2, +h/2, +3h/2 from the face (outward normal ``n``).
    ``order=4`` uses the cubic face rule (−1, 9, 9, −1)/16 (O(h⁴)); ``order=2`` is
    the two-point mean of the ±h/2 centers (O(h²)).

    Preserves input kind: real samples yield a real array or scalar; complex samples
    yield complex. Scalar inputs return a NumPy scalar (``.item()``-equivalent).
    """
    raw = (p_m3, p_m1, p_p1, p_p3)
    scalar_in = all(np.isscalar(x) for x in raw)
    arrays = [np.asarray(x) for x in raw]
    is_complex = any(a.dtype.kind == "c" for a in arrays)
    if is_complex:
        dtype = np.result_type(*arrays)
        if dtype.kind != "c":
            dtype = _C128
    else:
        dtype = np.result_type(*arrays)
        if dtype.kind == "c":
            dtype = _F64
    p_m3, p_m1, p_p1, p_p3 = [np.asarray(x, dtype=dtype) for x in raw]
    if order == 2:
        out = 0.5 * (p_m1 + p_p1)
    elif order == 4:
        out = (-p_m3 + 9.0 * p_m1 + 9.0 * p_p1 - p_p3) / 16.0
    else:
        raise ValueError(f"order must be 2 or 4, got {order!r}")
    if scalar_in:
        return np.asarray(out).item()
    return out


def huygens_box_patches(
    W: float,
    H: float,
    D: float,
    h: float,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    offset_cells: int = 3,
) -> SurfacePatches:
    """Closed Huygens surface for KH, offset outside the rigid box by ``offset_cells·h``.

    The physical box is [x₀, x₀+W]×[y₀, y₀+H]×[z₀, z₀+D]. The Huygens
    parallelepiped expands by ``offset_cells·h`` on every side. Patch centers
    coincide with staggered-grid *velocity* face points (h×h faces, normal ∥ axis).

    **FDTD pairing (e^{+jωt}):** ``v_n`` is recorded at these face centers. Face
    pressure should use ``face_p_from_cells`` with cell-center ``p`` at −3h/2, −h/2,
    +h/2, +3h/2 along the outward normal (weights (−1, 9, 9, −1)/16). That stencil
    needs two pressure cells on each side of the face, so place the Huygens surface
    **≥ 2 cells** from the rigid box and from the CPML inner edge (default offset 3).

    Far-field KH uses this Huygens data, not the rigid box faces.
    """
    ox, oy, oz = origin
    pad = float(offset_cells) * float(h)
    return _axis_aligned_box_patches(
        ox - pad,
        oy - pad,
        oz - pad,
        W + 2.0 * pad,
        H + 2.0 * pad,
        D + 2.0 * pad,
        h,
    )


def dpdn_from_vn(
    vn: np.ndarray,
    omega: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Normal pressure gradient from outward normal velocity: ∂p/∂n = −jωρ v_n."""
    vn = np.asarray(vn, dtype=_C128)
    omega = np.asarray(omega, dtype=_F64)
    return -1j * omega[:, np.newaxis] * rho * vn


# ---------------------------------------------------------------------------
# Sign convention (exterior KH with box-outward normal n)
# ---------------------------------------------------------------------------
# Green's identity on a source-free exterior volume V⁺ (outside closed surface S)
# uses the outward normal n⁺ pointing *into* S. With G = e^{−jkR}/(4πR),
#
#   p(x) = ∫_S [ G ∂p/∂n⁺ − p ∂G/∂n⁺ ] dS .
#
# FDTD stores the radiation-side normal n = −n⁺ (outward from the box into air).
# Since ∂f/∂n⁺ = −∂f/∂n,
#
#   p(x) = ∫_S [ p ∂G/∂n − G ∂p/∂n ] dS .                    (1)
#
# Source-point derivative (y ∈ S, observation x ∉ S), R = |x−y|, r̂ = (x−y)/R:
#
#   ∂G/∂n_y = ∇_y G·n = (n·(x−y)/R) (1+jkR) e^{−jkR}/(4πR²) .  (2)
#
# The plus sign in (2) is fixed by ∇_y R = (y−x)/R and ∂G/∂R = −(1+jkR) e^{−jkR}/(4πR²).
# ---------------------------------------------------------------------------


def _green(k: float | np.ndarray, R: np.ndarray) -> np.ndarray:
    """G = e^{−jkR}/(4πR); scalar ``k`` or ``k`` (F,) with ``R`` (F, …) or (…)."""
    R = np.maximum(np.asarray(R, dtype=_F64), 1e-30)
    k_arr = np.asarray(k, dtype=_F64)
    if k_arr.ndim == 0 or k_arr.size == 1:
        kf = float(k_arr.ravel()[0])
        return np.exp(-1j * kf * R) / (4.0 * np.pi * R)
    return np.exp(-1j * k_arr[:, np.newaxis] * R) / (4.0 * np.pi * R)


def _green_dn(
    k: float | np.ndarray,
    R: np.ndarray,
    rdx: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """∂G/∂n_y for (2); ``rdx`` = x−y, ``normals`` (N, 3)."""
    R = np.maximum(np.asarray(R, dtype=_F64), 1e-30)
    if rdx.ndim == 3:
        cos_nr = np.sum(rdx * normals[np.newaxis, :, :], axis=-1) / R
    else:
        cos_nr = np.sum(normals * rdx, axis=-1) / R
    k_arr = np.asarray(k, dtype=_F64)
    if k_arr.ndim == 0 or k_arr.size == 1:
        kf = float(k_arr.ravel()[0])
        return (
            cos_nr
            * (1.0 + 1j * kf * R)
            * np.exp(-1j * kf * R)
            / (4.0 * np.pi * R * R)
        )
    return (
        cos_nr
        * (1.0 + 1j * k_arr[:, np.newaxis] * R)
        * np.exp(-1j * k_arr[:, np.newaxis] * R)
        / (4.0 * np.pi * R * R)
    )


def kh_pressure(
    points: np.ndarray,
    patches: SurfacePatches,
    p: np.ndarray,
    dpdn: np.ndarray,
    k: np.ndarray,
    *,
    m_chunk: int = 64,
    n_chunk: int = 512,
    near_subdivide: bool = True,
    near_threshold: float | None = None,
    subdivide_grid: int = 4,
) -> np.ndarray:
    """Exterior KH pressure at ``points`` (M, 3); returns (F, M) complex128.

    ``p`` and ``dpdn`` are (F, N) over patch centers; ``k`` is (F,).

    Points within ``near_threshold`` of the box (default 2h) optionally use
    ``subdivide_grid``×``subdivide_grid`` sub-quadrature on patches whose
    center lies within that distance of the evaluation point.
    """
    points = np.asarray(points, dtype=_F64)
    p = np.asarray(p, dtype=_C128)
    dpdn = np.asarray(dpdn, dtype=_C128)
    k = np.asarray(k, dtype=_F64)
    y0 = patches.centers
    n_hat = patches.normals
    areas = patches.areas
    h = patches.h

    F, N = p.shape
    M = points.shape[0]
    out = np.zeros((F, M), dtype=_C128)

    thr = (2.0 * h if near_threshold is None else float(near_threshold)) if near_subdivide else -1.0

    cmin = y0.min(axis=0)
    cmax = y0.max(axis=0)

    for m0 in range(0, M, m_chunk):
        m1 = min(m0 + m_chunk, M)
        pts = points[m0:m1]

        for f_i, kf in enumerate(k):
            chunk_acc = np.zeros(m1 - m0, dtype=_C128)

            for n0 in range(0, N, n_chunk):
                n1 = min(n0 + n_chunk, N)
                yc = y0[n0:n1]
                nc = n_hat[n0:n1]
                ac = areas[n0:n1]
                pc = p[f_i, n0:n1]
                dpc = dpdn[f_i, n0:n1]

                rdx = pts[:, np.newaxis, :] - yc[np.newaxis, :, :]
                R = np.linalg.norm(rdx, axis=-1)

                G = _green(kf, R)
                dGdn = _green_dn(kf, R, rdx, nc)

                contrib = (pc[np.newaxis, :] * dGdn - G * dpc[np.newaxis, :]) * ac[np.newaxis, :]
                chunk_acc += np.sum(contrib, axis=-1)

                if near_subdivide and thr > 0:
                    for mi in range(m1 - m0):
                        dist_box = _dist_to_box(pts[mi], cmin, cmax)
                        if dist_box >= thr:
                            continue
                        for ni in range(n1 - n0):
                            if np.linalg.norm(pts[mi] - yc[ni]) >= thr:
                                continue
                            sub_c, sub_a = _subdivide_patch(yc[ni], nc[ni], h, subdivide_grid)
                            rdx_s = pts[mi] - sub_c
                            R_s = np.linalg.norm(rdx_s, axis=-1)
                            G_s = _green(kf, R_s)
                            dGdn_s = _green_dn(kf, R_s, rdx_s, nc[ni])
                            fine = np.sum(
                                (pc[ni] * dGdn_s - G_s * dpc[ni]) * sub_a
                            )
                            chunk_acc[mi] += fine - contrib[mi, ni]

            out[f_i, m0:m1] = chunk_acc

    return out


def kh_pressure_loop(
    points: np.ndarray,
    patches: SurfacePatches,
    p: np.ndarray,
    dpdn: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """Reference KH quadrature: explicit loops, full Green's kernel, no subdivision."""
    points = np.asarray(points, dtype=_F64)
    p = np.asarray(p, dtype=_C128)
    dpdn = np.asarray(dpdn, dtype=_C128)
    k = np.asarray(k, dtype=_F64)
    y0 = patches.centers
    n_hat = patches.normals
    areas = patches.areas

    F, N = p.shape
    M = points.shape[0]
    out = np.zeros((F, M), dtype=_C128)

    for f_i, kf in enumerate(k):
        for mi in range(M):
            acc = 0.0 + 0.0j
            x = points[mi]
            for ni in range(N):
                rdx = x - y0[ni]
                R = float(np.linalg.norm(rdx))
                R = max(R, 1e-30)
                G = np.exp(-1j * kf * R) / (4.0 * np.pi * R)
                cos_nr = float(np.dot(n_hat[ni], rdx) / R)
                dGdn = (
                    cos_nr
                    * (1.0 + 1j * kf * R)
                    * np.exp(-1j * kf * R)
                    / (4.0 * np.pi * R * R)
                )
                acc += (p[f_i, ni] * dGdn - G * dpdn[f_i, ni]) * areas[ni]
            out[f_i, mi] = acc
    return out


def _dist_to_box(point: np.ndarray, cmin: np.ndarray, cmax: np.ndarray) -> float:
    """Unsigned distance from ``point`` to the axis-aligned box [cmin, cmax]."""
    d = np.maximum(cmin - point, np.maximum(point - cmax, 0.0))
    return float(np.linalg.norm(d))


def _subdivide_patch(
    center: np.ndarray,
    normal: np.ndarray,
    h: float,
    grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``grid``×``grid`` face-centered sub-points on the patch tangent plane."""
    n = normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)

    offsets = (np.arange(grid, dtype=_F64) + 0.5) * (h / grid) - h / 2.0
    uu, vv = np.meshgrid(offsets, offsets, indexing="ij")
    sub = center + uu[..., np.newaxis] * t1 + vv[..., np.newaxis] * t2
    sub_a = np.full(grid * grid, (h / grid) ** 2, dtype=_F64)
    return sub.reshape(-1, 3), sub_a


def kh_far_field(
    directions: np.ndarray,
    patches: SurfacePatches,
    p: np.ndarray,
    dpdn: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """Far-field amplitude A(d) = lim_{r→∞} p(r d)·r·e^{jkr}, shape (F, D).

    Uses the Fraunhofer form of (1):
    A(d) = (1/4π) ∫_S [ jk p (n·d) − ∂p/∂n ] e^{+j k d·y} dS,
    which matches a radiating monopole (jωρQ)/(4π) for a compact source.
    Normalization: convert to pressure at radius ``R_ref`` via ``far_field_at_radius``.
    """
    directions = np.asarray(directions, dtype=_F64)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    p = np.asarray(p, dtype=_C128)
    dpdn = np.asarray(dpdn, dtype=_C128)
    k = np.asarray(k, dtype=_F64)

    y = patches.centers
    n_hat = patches.normals
    areas = patches.areas

    D = directions.shape[0]
    F = k.shape[0]
    nd = directions @ n_hat.T  # (D, N)
    phase = np.exp(1j * k[:, np.newaxis, np.newaxis] * (directions @ y.T))

    integrand = (
        1j * k[:, np.newaxis, np.newaxis] * p[:, np.newaxis, :] * nd[np.newaxis, :, :]
        - dpdn[:, np.newaxis, :]
    )
    return np.sum(integrand * phase * areas[np.newaxis, np.newaxis, :], axis=-1) / (4.0 * np.pi)


def far_field_at_radius(
    amplitude: np.ndarray,
    k: np.ndarray,
    R_ref: float = 1.0,
) -> np.ndarray:
    """Pressure at distance ``R_ref`` along each far-field direction: p = A e^{−jkR}/R."""
    amplitude = np.asarray(amplitude, dtype=_C128)
    k = np.asarray(k, dtype=_F64)
    return amplitude * np.exp(-1j * k[:, np.newaxis] * R_ref) / R_ref