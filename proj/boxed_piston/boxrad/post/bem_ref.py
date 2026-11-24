"""Exterior Neumann Burton–Miller BEM reference (bempp-cl).

FDTD coordinates: radiator center at the origin, front face at *z* = 0, box interior
in −*z*. Time-harmonic data from bempp-cl use e^{−iωt}; map to the project
convention e^{+jωt} via :func:`to_plus_jwt` (complex conjugation) in one place
before any comparison with FDTD or published references.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import bempp_cl.api as bempp
import numpy as np
from bempp_cl.api.linalg import gmres as bempp_gmres
from scipy.sparse.linalg import gmres as scipy_gmres

RADIATOR_TAG = 1
BOX_TAG = 2
_DENSE_MAX_GB = 3.0
_GMRES_RTOL = 1e-5
_DEFAULT_EPW = 12
_DEFAULT_H_MAX_M = 0.02
_F64 = np.float64
_C128 = np.complex128


def h_target_from_frequency(
    f_hz: float,
    c: float,
    *,
    epw: float = _DEFAULT_EPW,
    h_max: float = _DEFAULT_H_MAX_M,
) -> float:
    """Mesh target length min(λ/epw, h_max) for time-harmonic frequency *f_hz*."""
    f_hz = float(f_hz)
    if f_hz <= 0.0:
        raise ValueError("f_hz must be positive")
    wavelength = float(c) / f_hz
    return min(wavelength / float(epw), float(h_max))


def h_target_sphere(ka: float, a: float, *, epw: float = _DEFAULT_EPW) -> float:
    """Sphere mesh target: min(λ/epw, a/4) with λ = 2πa/ka.

    At low *ka*, icosphere faceting—not wavelength—limits accuracy; the *a*/4
    cap keeps enough elements per radius for the pulsating-sphere checks.
    """
    ka = float(ka)
    a = float(a)
    if ka <= 0.0 or a <= 0.0:
        raise ValueError("ka and a must be positive")
    wavelength = 2.0 * math.pi * a / ka
    return min(wavelength / float(epw), 0.25 * a)


_ICO_FACES: tuple[tuple[int, int, int], ...] = (
    (0, 11, 5),
    (0, 5, 1),
    (0, 1, 7),
    (0, 7, 10),
    (0, 10, 11),
    (1, 5, 9),
    (5, 11, 4),
    (11, 10, 2),
    (10, 7, 6),
    (7, 1, 8),
    (3, 9, 4),
    (3, 4, 2),
    (3, 2, 6),
    (3, 6, 8),
    (3, 8, 9),
    (4, 9, 5),
    (2, 4, 11),
    (6, 2, 10),
    (8, 6, 7),
    (9, 8, 1),
)


def _icosahedron_unit_vertices() -> list[np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ]
    return [
        (np.asarray(row, dtype=_F64) / np.linalg.norm(np.asarray(row, dtype=_F64)))
        for row in raw
    ]


def _mesh_max_edge(verts: Sequence[np.ndarray], faces: Sequence[tuple[int, int, int]]) -> float:
    mx = 0.0
    for i0, i1, i2 in faces:
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            mx = max(mx, float(np.linalg.norm(verts[b] - verts[a])))
    return mx


def _subdivide_triangles(
    verts: list[np.ndarray],
    faces: Sequence[tuple[int, int, int]],
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    edge_mid: dict[tuple[int, int], int] = {}
    out_verts = list(verts)
    out_faces: list[tuple[int, int, int]] = []

    def _midpoint(i: int, j: int) -> int:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in edge_mid:
            edge_mid[(a, b)] = len(out_verts)
            out_verts.append(0.5 * (out_verts[a] + out_verts[b]))
        return edge_mid[(a, b)]

    for i0, i1, i2 in faces:
        m01 = _midpoint(i0, i1)
        m12 = _midpoint(i1, i2)
        m20 = _midpoint(i2, i0)
        out_faces.extend(
            (
                (i0, m01, m20),
                (i1, m12, m01),
                (i2, m20, m12),
                (m01, m12, m20),
            )
        )
    return out_verts, out_faces


_ICOSPHERE_MAX_LEVEL = 6
_ICOSPHERE_MAX_ELEMENTS = 20 * (4**_ICOSPHERE_MAX_LEVEL)


def _vertices_on_sphere(verts: Sequence[np.ndarray], r: float) -> list[np.ndarray]:
    return [v / (np.linalg.norm(v) + 1e-300) * float(r) for v in verts]


def icosphere(r: float, h: float) -> bempp.Grid:
    """Closed sphere surface: regular icosahedron with midpoint subdivision to edge length ≤ *h*."""
    r = float(r)
    h = float(h)
    if r <= 0.0 or h <= 0.0:
        raise ValueError("r and h must be positive")
    verts = _icosahedron_unit_vertices()
    faces = list(_ICO_FACES)
    level = 0
    while _mesh_max_edge(_vertices_on_sphere(verts, r), faces) > h:
        if level >= _ICOSPHERE_MAX_LEVEL:
            raise ValueError(
                f"icosphere(r={r}, h={h}) requires more than {_ICOSPHERE_MAX_LEVEL} "
                f"subdivision levels (> {_ICOSPHERE_MAX_ELEMENTS} elements); "
                f"increase h or decrease r"
            )
        verts, faces = _subdivide_triangles(verts, faces)
        level += 1
    verts = _vertices_on_sphere(verts, r)
    elements: list[list[int]] = []
    for i0, i1, i2 in faces:
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        cross = np.cross(v1 - v0, v2 - v0)
        n_hat = cross / (np.linalg.norm(cross) + 1e-300)
        centroid = (v0 + v1 + v2) / 3.0
        if float(np.dot(centroid, n_hat)) <= 0.0:
            i1, i2 = i2, i1
            v0, v1, v2 = verts[i0], verts[i1], verts[i2]
            cross = np.cross(v1 - v0, v2 - v0)
            n_hat = cross / (np.linalg.norm(cross) + 1e-300)
            centroid = (v0 + v1 + v2) / 3.0
        assert float(np.dot(centroid, n_hat)) > 0.0
        elements.append([i0, i1, i2])
    v_arr = np.array(verts, dtype=_F64).T
    elems = np.array(elements, dtype=np.int32).T
    tags = np.full(elems.shape[1], RADIATOR_TAG, dtype=np.int32)
    return bempp.Grid(v_arr, elems, domain_indices=tags)


def dedupe_frequencies_hz(freqs: Sequence[float], *, sep_hz: float = 1.0) -> np.ndarray:
    """Sort ascending; drop entries within *sep_hz* of the previous kept frequency."""
    if len(freqs) == 0:
        return np.asarray([], dtype=_F64)
    arr = np.sort(np.asarray(freqs, dtype=_F64))
    out: list[float] = [float(arr[0])]
    for f in arr[1:]:
        if f - out[-1] > float(sep_hz):
            out.append(float(f))
    return np.asarray(out, dtype=_F64)


class BoxMeshCache:
    """Reuse :func:`box_mesh` grids keyed by rounded (*h_target*, *h_rad*) (m)."""

    def __init__(self) -> None:
        self._grids: dict[tuple[float, float], bempp.Grid] = {}

    def get(
        self,
        W: float,
        H: float,
        D: float,
        rad_rect: tuple[float, float],
        origin: tuple[float, float, float],
        h_target: float,
        *,
        h_rad: float | None = None,
    ) -> bempp.Grid:
        rw, rh = float(rad_rect[0]), float(rad_rect[1])
        h_rad_eff = min(rw, rh) / 8.0 if h_rad is None else float(h_rad)
        key = (round(float(h_target), 12), round(h_rad_eff, 12))
        if key not in self._grids:
            self._grids[key] = box_mesh(
                W, H, D, rad_rect, origin, h_target, h_rad=h_rad_eff
            )
        return self._grids[key]


@dataclass(frozen=True)
class BmSolveStats:
    n_dofs: int
    assemble_s: float
    solve_s: float
    gmres_iters: int
    dense_gb: float


def format_bem_progress_line(
    *,
    index: int,
    total: int,
    f_hz: float,
    h_m: float,
    stats: BmSolveStats,
    elapsed_s: float,
    eta_s: float | None,
    prefix: str = "bem",
) -> str:
    eta = "—" if eta_s is None else f"{eta_s:.1f}"
    return (
        f"[{prefix} {index}/{total}] f={f_hz:.2f} Hz h={h_m * 1e3:.2f} mm "
        f"DOF={stats.n_dofs} assemble={stats.assemble_s:.2f}s "
        f"solve={stats.solve_s:.2f}s iters={stats.gmres_iters} "
        f"elapsed={elapsed_s:.1f}s ETA={eta}s"
    )


def to_plus_jwt(z: np.ndarray | complex) -> np.ndarray | complex:
    """Map bempp e^{−iωt} phasors to this project's e^{+jωt} convention."""
    if np.isscalar(z):
        return np.conj(z)
    return np.conj(np.asarray(z, dtype=_C128))


def _grid_vertices_3d(grid: object) -> np.ndarray:
    """Vertex coordinates as (n_vertices, 3)."""
    return np.asarray(grid.vertices, dtype=_F64).T


def _grid_elements(grid: object) -> np.ndarray:
    """Triangle vertex indices (n_elements, 3); row e matches BEM local DOF columns."""
    return np.asarray(grid.elements, dtype=np.int32).T


def _triangle_barycentric_weights(
    points: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Barycentric weights (N, E, 3); λ[..., i] is the weight of vertex v_i (Σλ = 1)."""
    pts = points[:, np.newaxis, :]
    e10 = v1[np.newaxis, :, :] - v0[np.newaxis, :, :]
    e20 = v2[np.newaxis, :, :] - v0[np.newaxis, :, :]
    w = pts - v0[np.newaxis, :, :]
    dot00 = np.sum(e10 * e10, axis=-1)
    dot01 = np.sum(e10 * e20, axis=-1)
    dot02 = np.sum(e10 * w, axis=-1)
    dot11 = np.sum(e20 * e20, axis=-1)
    dot12 = np.sum(e20 * w, axis=-1)
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-300)
    lam1 = (dot11 * dot02 - dot01 * dot12) * inv_denom
    lam2 = (dot00 * dot12 - dot01 * dot02) * inv_denom
    lam0 = 1.0 - lam1 - lam2
    return np.stack([lam0, lam1, lam2], axis=-1)


def _triangle_max_edge(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    e01 = np.linalg.norm(v1 - v0, axis=-1)
    e12 = np.linalg.norm(v2 - v1, axis=-1)
    e20 = np.linalg.norm(v0 - v2, axis=-1)
    return np.maximum(np.maximum(e01, e12), e20)


def _locate_points_on_surface_mesh(
    points: np.ndarray,
    grid: object,
    *,
    tol_scale: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (element_index, barycentric_weights) per point; raise if off-surface."""
    verts = _grid_vertices_3d(grid)
    elems = _grid_elements(grid)
    v0 = verts[elems[:, 0]]
    v1 = verts[elems[:, 1]]
    v2 = verts[elems[:, 2]]
    lams = _triangle_barycentric_weights(points, v0, v1, v2)
    h_el = _triangle_max_edge(v0, v1, v2)
    cross = np.cross(v1 - v0, v2 - v0, axis=-1)
    n_hat = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-300)
    delta = points[:, np.newaxis, :] - v0[np.newaxis, :, :]
    d = np.abs(np.sum(delta * n_hat[np.newaxis, :, :], axis=-1))
    plane_tol = float(tol_scale) * h_el
    lam_tol = 1e-9
    inside = (d <= plane_tol[np.newaxis, :]) & np.all(lams >= -lam_tol, axis=-1)
    elem_idx = np.full(points.shape[0], -1, dtype=np.int32)
    lam_out = np.zeros((points.shape[0], 3), dtype=_F64)
    offending: list[tuple[float, ...]] = []
    for i in range(points.shape[0]):
        matches = np.flatnonzero(inside[i])
        if matches.size == 0:
            offending.append(tuple(points[i].tolist()))
            continue
        e = int(matches[np.argmin(d[i, matches])])
        elem_idx[i] = e
        lam_out[i] = lams[i, e]
    if offending:
        raise ValueError(
            "point(s) not on the surface mesh (plane distance / barycentric tolerance): "
            f"{offending}"
        )
    return elem_idx, lam_out


def p1_interpolate_at(
    space: object,
    grid: object,
    coefficients: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate a P1 GridFunction at surface points via barycentric interpolation."""
    pts = np.asarray(points, dtype=_F64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    coeffs = np.asarray(coefficients, dtype=_C128).ravel()
    elem_idx, lams = _locate_points_on_surface_mesh(pts, grid)
    l2g = np.asarray(space.local2global, dtype=np.int32)[elem_idx]
    mult = np.asarray(space.local_multipliers, dtype=_F64)[elem_idx]
    dof_vals = coeffs[l2g]
    weights = lams * mult
    return np.sum(weights * dof_vals, axis=-1)


def _estimate_dense_gb(n_dofs: int) -> float:
    return (n_dofs**2 * 16) / 1e9


def _subdivide_axis(
    edges: Sequence[float],
    h_target: float,
    *,
    h_cap: float | None = None,
    cap_lo: float | None = None,
    cap_hi: float | None = None,
) -> np.ndarray:
    """Sorted 1-D nodes; interval length ≤ *h_target* (or ≤ min(*h_target*, *h_cap*) in [cap_lo, cap_hi])."""
    h_target = float(h_target)
    uniq = sorted(set(float(x) for x in edges))
    if len(uniq) < 2:
        return np.asarray(uniq, dtype=_F64)
    out: list[float] = [uniq[0]]
    for a, b in zip(uniq[:-1], uniq[1:]):
        length = b - a
        if length <= 0.0:
            continue
        h_eff = h_target
        if (
            h_cap is not None
            and cap_lo is not None
            and cap_hi is not None
            and a >= float(cap_lo) - 1e-15
            and b <= float(cap_hi) + 1e-15
        ):
            h_eff = min(h_target, float(h_cap))
        n_seg = max(1, int(math.ceil(length / h_eff)))
        for s in range(1, n_seg):
            out.append(a + length * s / n_seg)
        out.append(b)
    return np.asarray(out, dtype=_F64)


def box_mesh(
    W: float,
    H: float,
    D: float,
    rad_rect: tuple[float, float],
    origin: tuple[float, float, float],
    h_target: float,
    *,
    h_rad: float | None = None,
) -> bempp.Grid:
    """Structured triangle mesh of a closed box surface (outward normals).

    *origin* is the minimum corner (x₀, y₀, z₀); the box is
    [x₀, x₀+W]×[y₀, y₀+H]×[z₀, z₀+D] with the front face at z = z₀+D (typically 0).
    *rad_rect* is (width, height) of the radiator rectangle on the front face,
    centered at the origin in the horizontal plane. Triangles on the radiator
    carry :data:`RADIATOR_TAG`; all others use :data:`BOX_TAG`.
    """
    W, H, D = float(W), float(H), float(D)
    rw, rh = float(rad_rect[0]), float(rad_rect[1])
    ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))
    x0, x1 = ox, ox + W
    y0, y1 = oy, oy + H
    z0, z1 = oz, oz + D
    rx0, rx1 = -0.5 * rw, 0.5 * rw
    ry0, ry1 = -0.5 * rh, 0.5 * rh
    if h_rad is None:
        h_rad = min(rw, rh) / 8.0
    h_rad = float(h_rad)

    xs = _subdivide_axis(
        [x0, x1, rx0, rx1], h_target, h_cap=h_rad, cap_lo=rx0, cap_hi=rx1
    )
    ys = _subdivide_axis(
        [y0, y1, ry0, ry1], h_target, h_cap=h_rad, cap_lo=ry0, cap_hi=ry1
    )
    zs = _subdivide_axis([z0, z1], h_target)

    vertices: list[np.ndarray] = []
    vmap: dict[tuple[float, float, float], int] = {}
    elements: list[list[int]] = []
    domain: list[int] = []

    def _radiator_element(i: int, j: int, u: np.ndarray, v: np.ndarray) -> int:
        uc = 0.5 * (u[i] + u[i + 1])
        vc = 0.5 * (v[j] + v[j + 1])
        if abs(uc) <= 0.5 * rw + 1e-15 and abs(vc) <= 0.5 * rh + 1e-15:
            return RADIATOR_TAG
        return BOX_TAG

    def _add_face_tagged(u, v, fixed, flip: bool) -> None:
        axis, coord = fixed
        nu, nv = len(u), len(v)
        tang = [t for t in range(3) if t != axis]
        u_arr, v_arr = u, v

        def _vid(x: float, y: float, z: float) -> int:
            key = (x, y, z)
            if key not in vmap:
                vmap[key] = len(vertices)
                vertices.append(np.array([x, y, z], dtype=_F64))
            return vmap[key]

        for i in range(nu - 1):
            for j in range(nv - 1):
                corners = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    pt = [0.0, 0.0, 0.0]
                    pt[axis] = coord
                    pt[tang[0]] = float(u_arr[i + du])
                    pt[tang[1]] = float(v_arr[j + dv])
                    corners.append(_vid(*pt))
                a, b, c, d = corners
                tag = _radiator_element(i, j, u_arr, v_arr) if axis == 2 and coord == z1 else BOX_TAG
                if flip:
                    elements.append([a, c, b])
                    elements.append([a, d, c])
                else:
                    elements.append([a, b, c])
                    elements.append([a, c, d])
                domain.extend([tag, tag])

    # x = x0 (−x normal), x = x1 (+x)
    _add_face_tagged(ys, zs, (0, x0), flip=True)
    _add_face_tagged(ys, zs, (0, x1), flip=False)
    # y = y0 (−y), y = y1 (+y); tangents (x, z) ⇒ flip=False is −y, flip=True is +y
    _add_face_tagged(xs, zs, (1, y0), flip=False)
    _add_face_tagged(xs, zs, (1, y1), flip=True)
    # z = z0 (rear, −z), z = z1 (front, +z)
    _add_face_tagged(xs, ys, (2, z0), flip=True)
    _add_face_tagged(xs, ys, (2, z1), flip=False)

    verts = np.array(vertices, dtype=_F64).T
    elems = np.array(elements, dtype=np.int32).T
    tags = np.array(domain, dtype=np.int32)
    return bempp.Grid(verts, elems, domain_indices=tags)


@dataclass
class NeumannBmResult:
    """Exterior Neumann Burton–Miller solution (P1 pressure, DP0 Neumann data)."""

    grid: object
    space_p: object
    space_q: object
    p_gf: object
    neumann_fun: object
    k: float
    rho: float
    c: float
    vn_P0: float
    stats: BmSolveStats | None = None

    def zrad(self) -> complex:
        """Dimensionless z = ⟨p⟩_rad / (ρ c v_n) in e^{+jωt}."""
        domain = self.grid.domain_indices.astype(int)
        areas = self.grid.volumes.astype(float)
        rad = domain == RADIATOR_TAG
        if not np.any(rad):
            rad = np.ones(domain.shape, dtype=bool)
        p_el = self.p_gf.evaluate_on_element_centers().ravel()
        num = np.sum(p_el[rad] * areas[rad])
        den = np.sum(areas[rad])
        z_bem = num / (den * self.rho * self.c * self.vn_P0)
        return complex(to_plus_jwt(z_bem).item())

    def radiator_mean_pressure(self) -> complex:
        """Area-weighted mean radiator pressure (Pa), e^{+jωt} (element quadrature)."""
        domain = self.grid.domain_indices.astype(int)
        areas = self.grid.volumes.astype(float)
        rad = domain == RADIATOR_TAG
        if not np.any(rad):
            raise RuntimeError("radiator_mean_pressure: no radiator elements")
        p_el = self.p_gf.evaluate_on_element_centers().ravel()
        num = np.sum(p_el[rad] * areas[rad])
        den = np.sum(areas[rad])
        return complex(to_plus_jwt(num / den).item())

    def surface_pressure_at(self, points: np.ndarray) -> np.ndarray:
        """Interpolate solved P1 boundary pressure at *points* (N,3) in e^{+jωt}."""
        raw = p1_interpolate_at(
            self.space_p, self.grid, self.p_gf.coefficients, points
        )
        return np.asarray(to_plus_jwt(raw), dtype=_C128)

    def field_at(self, points: np.ndarray) -> np.ndarray:
        """Exterior pressure at *points* (N,3) via D[p] − S[q] (outward normals).

        *q* is ∂p/∂n in bempp's e^{−iωt} form (set in :func:`solve_neumann_bm`).
        Do not use on the boundary; use :meth:`surface_pressure_at` there instead.
        """
        pts = np.asarray(points, dtype=_F64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        eval_pts = pts.T
        k = self.k
        slp_pot = bempp.operators.potential.helmholtz.single_layer(
            self.space_q, eval_pts, k
        )
        dlp_pot = bempp.operators.potential.helmholtz.double_layer(
            self.space_p, eval_pts, k
        )
        p_ext = (dlp_pot @ self.p_gf - slp_pot @ self.neumann_fun).ravel()
        return np.asarray(to_plus_jwt(p_ext), dtype=_C128)

    def transfer_at(self, points: np.ndarray, *, on_surface: bool) -> np.ndarray:
        """Dimensionless transfer p/(ρ c v_n) (``vn_P0`` as in the Burton–Miller solve)."""
        scale = self.rho * self.c * self.vn_P0
        if on_surface:
            p = self.surface_pressure_at(points)
        else:
            p = self.field_at(points)
        return np.asarray(p / scale, dtype=_C128)


def solve_neumann_bm(
    grid: bempp.Grid,
    k: float,
    vn_P0: float,
    rho: float,
    c: float,
    *,
    rtol: float = _GMRES_RTOL,
    use_scipy_gmres: bool = True,
    pulsating_sphere: bool = False,
    log_dof: bool = False,
) -> NeumannBmResult:
    """Solve exterior Neumann (Burton–Miller): prescribed normal velocity on the radiator."""
    k = float(k)
    omega = k * c
    space_p = bempp.function_space(grid, "P", 1)
    space_q = bempp.function_space(grid, "DP", 0)
    id_pp = bempp.operators.boundary.sparse.identity(space_p, space_p, space_p)
    id_qp = bempp.operators.boundary.sparse.identity(space_q, space_p, space_p)

    n_dofs = space_p.global_dof_count
    dense_gb = _estimate_dense_gb(n_dofs)
    if log_dof:
        print(
            f"  BEM DOFs={n_dofs}  est. dense={dense_gb:.2f} GB",
            flush=True,
        )
    if dense_gb > _DENSE_MAX_GB:
        raise RuntimeError(
            f"BEM dense matrix estimate {dense_gb:.2f} GB exceeds {_DENSE_MAX_GB} GB limit"
        )

    alpha = 1j / k

    @bempp.complex_callable
    def neumann_callable(point, normal, domain_index, result):
        if pulsating_sphere or int(domain_index) == RADIATOR_TAG:
            # Project (e^{+jωt}): ∂p/∂n = −jωρ v_n. bempp-cl (e^{−iωt}): ∂p/∂n = +iωρ v_n.
            # Prescribe the latter; map solved fields to +jωt once via to_plus_jwt.
            result[0] = 1j * omega * rho * vn_P0
        else:
            result[0] = 0.0

    neumann_fun = bempp.GridFunction(space_q, fun=neumann_callable)

    slp = bempp.operators.boundary.helmholtz.single_layer(space_q, space_p, space_p, k)
    dlp = bempp.operators.boundary.helmholtz.double_layer(space_p, space_p, space_p, k)
    adlp = bempp.operators.boundary.helmholtz.adjoint_double_layer(space_q, space_p, space_p, k)
    hyp = bempp.operators.boundary.helmholtz.hypersingular(space_p, space_p, space_p, k)

    # bempp-cl hypersingular sign: +α W (not −α W).
    lhs = 0.5 * id_pp - dlp + alpha * hyp
    rhs_op = -slp - alpha * (adlp + 0.5 * id_qp)

    gmres_iters = 0
    if use_scipy_gmres:
        t_asm = time.perf_counter()
        lhs_w = lhs.weak_form()
        rhs_proj = (rhs_op * neumann_fun).projections(lhs.dual_to_range)
        assemble_s = time.perf_counter() - t_asm

        def _iter_cb(_: object) -> None:
            nonlocal gmres_iters
            gmres_iters += 1

        t_slv = time.perf_counter()
        x, info = scipy_gmres(
            lhs_w,
            rhs_proj,
            rtol=rtol,
            callback=_iter_cb,
            callback_type="legacy",
        )
        solve_s = time.perf_counter() - t_slv
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")
        p_gf = bempp.GridFunction(space_p, coefficients=x)
    else:
        assemble_s = 0.0
        t_slv = time.perf_counter()
        p_gf, info = bempp_gmres(
            lhs,
            rhs_op * neumann_fun,
            rtol=rtol,
            use_strong_form=True,
            callback_type="legacy",
        )
        solve_s = time.perf_counter() - t_slv
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")

    stats = BmSolveStats(
        n_dofs=n_dofs,
        assemble_s=assemble_s,
        solve_s=solve_s,
        gmres_iters=gmres_iters,
        dense_gb=dense_gb,
    )

    return NeumannBmResult(
        grid=grid,
        space_p=space_p,
        space_q=space_q,
        p_gf=p_gf,
        neumann_fun=neumann_fun,
        k=k,
        rho=rho,
        c=c,
        vn_P0=vn_P0,
        stats=stats,
    )


def zrad_bem(sol: NeumannBmResult) -> complex:
    """Dimensionless radiation impedance from a :class:`NeumannBmResult` (e^{+jωt})."""
    return sol.zrad()


def zrad_sphere_analytic(ka: float) -> complex:
    """Pulsating sphere z = j ka / (1 + j ka), e^{+jωt}, unit normal velocity."""
    jka = 1j * float(ka)
    return jka / (1.0 + jka)


def pulsating_sphere_H_transfer(r_m: float, a: float, ka: float) -> complex:
    """Transfer H = p/(ρ c v_n) on sphere axis, e^{+jωt}.

    H(r) = z_rad(ka) (a/r) e^{−jk(r−a)} with k = ka/a.
    """
    r_m = float(r_m)
    a = float(a)
    ka = float(ka)
    if r_m <= a:
        raise ValueError("r must lie outside the sphere")
    k = ka / a
    z = zrad_sphere_analytic(ka)
    return z * (a / r_m) * np.exp(-1j * k * (r_m - a))


_DEFAULT_RAD_DIV = 8


def c09_geometry_signature(
    W: float,
    H: float,
    D: float,
    rad_rect: tuple[float, float],
    origin: tuple[float, float, float],
    *,
    h_max: float,
    rho: float,
    c: float,
    rad_div: int = _DEFAULT_RAD_DIV,
) -> dict[str, Any]:
    """Stable dict for cache keys (geometry + fluid, not frequency)."""
    sig: dict[str, Any] = {
        "W": round(float(W), 12),
        "H": round(float(H), 12),
        "D": round(float(D), 12),
        "rw": round(float(rad_rect[0]), 12),
        "rh": round(float(rad_rect[1]), 12),
        "ox": round(float(origin[0]), 12),
        "oy": round(float(origin[1]), 12),
        "oz": round(float(origin[2]), 12),
        "h_max": round(float(h_max), 12),
        "rho": round(float(rho), 12),
        "c": round(float(c), 12),
    }
    if int(rad_div) != _DEFAULT_RAD_DIV:
        sig["rad_div"] = int(rad_div)
    return sig


def c09_bem_cache_path(
    cache_dir: Path,
    geo: dict[str, Any],
    *,
    epw: float,
    f_hz: float,
) -> Path:
    payload = {
        **geo,
        "cache_version": _BEM_CACHE_VERSION,
        "epw": float(epw),
        "f_hz": round(float(f_hz), 8),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return cache_dir / f"c09_{digest}.npz"


_BEM_CACHE_VERSION = 5


def save_c09_bem_cache(
    path: Path,
    *,
    f_hz: float,
    epw: float,
    h_m: float,
    n_dof: int,
    p1_coefficients: np.ndarray,
    z_rad: complex | None = None,
    surface_names: Sequence[str] | None = None,
    surface_p: Sequence[complex] | None = None,
    g2_planes: Sequence[str] | None = None,
    g2_theta: np.ndarray | None = None,
    g2_p_by_plane: dict[str, np.ndarray] | None = None,
    post_complete: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p1 = np.asarray(p1_coefficients, dtype=_C128).ravel()
    arrays: dict[str, Any] = {
        "version": _BEM_CACHE_VERSION,
        "f_hz": float(f_hz),
        "epw": float(epw),
        "h_m": float(h_m),
        "n_dof": int(n_dof),
        "post_complete": bool(post_complete),
        "p1_coeff_real": np.real(p1).astype(_F64),
        "p1_coeff_imag": np.imag(p1).astype(_F64),
    }
    if z_rad is not None:
        arrays["z_real"] = float(z_rad.real)
        arrays["z_imag"] = float(z_rad.imag)
    if surface_names is not None and surface_p is not None:
        arrays["surface_names"] = np.asarray(list(surface_names), dtype=object)
        arrays["surface_p_real"] = np.asarray(
            [float(np.real(p)) for p in surface_p], dtype=_F64
        )
        arrays["surface_p_imag"] = np.asarray(
            [float(np.imag(p)) for p in surface_p], dtype=_F64
        )
    if g2_planes is not None and g2_theta is not None and g2_p_by_plane is not None:
        arrays["g2_planes"] = np.asarray(list(g2_planes), dtype=object)
        arrays["g2_theta"] = np.asarray(g2_theta, dtype=_F64)
        for plane in g2_planes:
            p_row = np.asarray(g2_p_by_plane[plane], dtype=_C128).ravel()
            arrays[f"g2_p_{plane}_real"] = np.real(p_row).astype(_F64)
            arrays[f"g2_p_{plane}_imag"] = np.imag(p_row).astype(_F64)
    if path.is_file():
        with np.load(path, allow_pickle=True) as prior:
            for key in prior.files:
                if key not in arrays:
                    arrays[key] = prior[key]
    np.savez_compressed(path, **arrays)


def load_c09_bem_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as zf:
        version = int(zf["version"])
        if version not in (1, 2, 3, 4, _BEM_CACHE_VERSION):
            raise ValueError(f"BEM cache version mismatch in {path}")
        if version < _BEM_CACHE_VERSION:
            raise ValueError(
                f"BEM cache v{version} stale (need v{_BEM_CACHE_VERSION}); delete {path}"
            )
        out: dict[str, Any] = {
            "f_hz": float(zf["f_hz"]),
            "epw": float(zf["epw"]),
            "h_m": float(zf["h_m"]),
            "n_dof": int(zf["n_dof"]),
            "post_complete": bool(zf["post_complete"]) if "post_complete" in zf else True,
        }
        if "p1_coeff_real" in zf:
            out["p1_coefficients"] = (
                np.asarray(zf["p1_coeff_real"], dtype=_F64)
                + 1j * np.asarray(zf["p1_coeff_imag"], dtype=_F64)
            )
        else:
            out["p1_coefficients"] = None
        if "z_real" in zf:
            out["z_rad"] = complex(float(zf["z_real"]), float(zf["z_imag"]))
        elif out["post_complete"]:
            raise ValueError(f"BEM cache missing z_rad in {path}")
        else:
            out["z_rad"] = None
        if "surface_names" in zf:
            out["surface_names"] = [str(x) for x in zf["surface_names"].tolist()]
        else:
            out["surface_names"] = []
        out["surface_p"] = (
            [
                complex(r, i)
                for r, i in zip(
                    zf["surface_p_real"], zf["surface_p_imag"], strict=True
                )
            ]
            if "surface_p_real" in zf
            else []
        )
        if "g2_planes" in zf:
            planes = [str(p) for p in zf["g2_planes"].tolist()]
            theta = np.asarray(zf["g2_theta"], dtype=_F64)
            p_by: dict[str, np.ndarray] = {}
            for plane in planes:
                re = np.asarray(zf[f"g2_p_{plane}_real"], dtype=_F64)
                im = np.asarray(zf[f"g2_p_{plane}_imag"], dtype=_F64)
                p_by[plane] = re + 1j * im
            out["g2_planes"] = planes
            out["g2_theta"] = theta
            out["g2_p_by_plane"] = p_by
        return out