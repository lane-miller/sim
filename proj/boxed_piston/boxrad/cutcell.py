"""Cut-cell volume and face fractions (v0.3 step 4).

Kernel index conventions (match ``boxrad.grid`` Yee layout):

* ``beta[i, j, k]`` — fluid volume fraction at cell center of cell ``(i, j, k)``.
* ``alpha_x[i, j, k]`` — x-normal face area fraction at plane ``x = x0 + i*h``,
  shape ``(nx + 1, ny, nz)`` (same staggering as ``vx``).
* ``alpha_y[i, j, k]`` — y face at ``y = y0 + j*h``, shape ``(nx, ny + 1, nz)``.
* ``alpha_z[i, j, k]`` — z face at ``z = z0 + k*h``, shape ``(nx, ny, nz + 1)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit

from boxrad.geom.geometry import Geometry
from boxrad.geom.mesh import TriMesh, inside
from boxrad.geomconfig import PlaneSpec

_F64 = np.float64
_I64 = np.int64
_STATE_SOLID = np.uint8(0)
_STATE_FLUID = np.uint8(1)
_STATE_CUT = np.uint8(2)
_TOL = 1e-9
_MAX_POLY = 16
_AXIS = ("x", "y", "z")


def _side_key(axis: int, side: str) -> str:
    return f"{_AXIS[axis]}{side}"


@dataclass(frozen=True)
class Grid:
    """Structured background grid for fraction computation (not the FDTD ``grid.Grid``)."""

    x0: np.ndarray
    h: float
    shape: tuple[int, int, int]
    rigid_sides: dict[str, bool]
    h_nominal: float = field(default=0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x0", np.asarray(self.x0, dtype=_F64).reshape(3))
        if self.h_nominal <= 0.0:
            object.__setattr__(self, "h_nominal", float(self.h))

    @property
    def nx(self) -> int:
        return int(self.shape[0])

    @property
    def ny(self) -> int:
        return int(self.shape[1])

    @property
    def nz(self) -> int:
        return int(self.shape[2])

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    def cell_center(self, i: int, j: int, k: int) -> np.ndarray:
        return self.x0 + (np.array([i, j, k], dtype=_F64) + 0.5) * self.h

    def face_plane(self, axis: int, idx: int) -> float:
        return float(self.x0[axis] + idx * self.h)

    def world_to_index(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=_F64).reshape(3)
        return np.floor((x - self.x0) / self.h).astype(_I64)

    def flat_index(self, i: int, j: int, k: int) -> int:
        return (i * self.ny + j) * self.nz + k

    def absorbing_side(self, axis: int) -> str | None:
        neg = _side_key(axis, "-")
        pos = _side_key(axis, "+")
        neg_rigid = self.rigid_sides.get(neg, False)
        pos_rigid = self.rigid_sides.get(pos, False)
        if neg_rigid and pos_rigid:
            return None
        if neg_rigid:
            return "+"
        if pos_rigid:
            return "-"
        return "-"


@dataclass
class Fractions:
    alpha: tuple[np.ndarray, np.ndarray, np.ndarray]
    beta: np.ndarray
    state: np.ndarray
    patch_cell: np.ndarray
    patch_A: np.ndarray
    patch_centroid: np.ndarray
    patch_normal: np.ndarray
    radiator_cell: list[np.ndarray]
    radiator_a: list[np.ndarray]
    radiator_S: list[float]
    stats: dict[str, Any]
    radiator_centroid: list[np.ndarray] = field(default_factory=list)


def _pose_axis_normal(radiator) -> np.ndarray | None:
    n = radiator.pose.R[:, 2].astype(_F64)
    ln = float(np.linalg.norm(n))
    if ln < 1e-30:
        return None
    n = n / ln
    for ax in range(3):
        if abs(n[ax]) > 1.0 - 1e-12 and abs(n[1 - (ax != 0)] - 0) < 1e-12:
            return n
    return None


def _anchors_for_axis(geometry: Geometry, axis: int) -> list[float]:
    anchors: list[float] = []
    for pl in geometry.planes:
        if pl.axis == _AXIS[axis]:
            anchors.append(float(pl.offset))
    if geometry.radiators:
        rd = geometry.radiators[0]
        n = _pose_axis_normal(rd)
        if n is not None and abs(n[axis]) > 0.5:
            anchors.append(float(rd.pose.position[axis]))
    if not anchors:
        anchors.append(0.0)
    return anchors


def grid_for_geometry(
    geometry: Geometry,
    h: float,
    *,
    margin_cells: int = 3,
) -> Grid:
    """Build a uniform grid enclosing *geometry* (showcase / tests; step 6 uses derive)."""
    h0 = float(h)
    lo = np.min(geometry.solid.vertices, axis=0)
    hi = np.max(geometry.solid.vertices, axis=0)
    h_use = h0
    for axis in range(3):
        anc = sorted(set(_anchors_for_axis(geometry, axis)))
        if len(anc) == 2:
            d = abs(anc[1] - anc[0])
            ncell = int(math.ceil(d / h0))
            h_use = min(h_use, d / max(ncell, 1))
    rigid: dict[str, bool] = {}
    for pl in geometry.planes:
        rigid[f"{pl.axis}{pl.side}"] = True
    x0 = np.zeros(3, dtype=_F64)
    shape = [0, 0, 0]
    for axis in range(3):
        anc = sorted(set(_anchors_for_axis(geometry, axis)))
        ref = anc[0]
        lo_a = lo[axis]
        hi_a = hi[axis]
        neg_key = _side_key(axis, "-")
        pos_key = _side_key(axis, "+")
        if rigid.get(neg_key, False):
            plane_off = next(
                float(p.offset) for p in geometry.planes if p.axis == _AXIS[axis] and p.side == "-"
            )
            if lo_a < plane_off - 1e-9 * h_use - 1e-12:
                raise ValueError(
                    f"solid crosses rigid plane {_AXIS[axis]}- by more than tolerance"
                )
            lo_a = plane_off
        else:
            lo_a -= margin_cells * h_use
        if rigid.get(pos_key, False):
            plane_off = next(
                float(p.offset) for p in geometry.planes if p.axis == _AXIS[axis] and p.side == "+"
            )
            if hi_a > plane_off + 1e-9 * h_use + 1e-12:
                raise ValueError(
                    f"solid crosses rigid plane {_AXIS[axis]}+ by more than tolerance"
                )
            hi_a = plane_off
        else:
            hi_a += margin_cells * h_use
        k0 = int(math.floor((lo_a - ref) / h_use))
        k1 = int(math.ceil((hi_a - ref) / h_use))
        x0[axis] = ref + k0 * h_use
        shape[axis] = max(1, k1 - k0)
    return Grid(
        x0=x0,
        h=h_use,
        shape=(shape[0], shape[1], shape[2]),
        rigid_sides=rigid,
        h_nominal=h0,
    )


def _snap_mesh(mesh: TriMesh, grid: Grid) -> TriMesh:
    snap = 1e-9 * grid.h
    v = mesh.vertices.copy()
    for ax in range(3):
        t = np.round((v[:, ax] - grid.x0[ax]) / grid.h) * grid.h + grid.x0[ax]
        mask = np.abs(v[:, ax] - t) <= snap
        v[mask, ax] = t[mask]
    return TriMesh(v, mesh.faces.copy())


@njit(cache=False)
def _clip_halfspace(
    poly: np.ndarray,
    n_in: int,
    ax: int,
    val: float,
    keep_ge: bool,
    out: np.ndarray,
) -> int:
    if n_in == 0:
        return 0
    m = 0
    for i in range(n_in):
        P = poly[i]
        Q = poly[(i + 1) % n_in]
        if keep_ge:
            dp = P[ax] - val
            dq = Q[ax] - val
        else:
            dp = val - P[ax]
            dq = val - Q[ax]
        if dp >= 0.0:
            out[m] = P
            m += 1
        if (dp >= 0.0) != (dq >= 0.0):
            t = dp / (dp - dq + 1e-30)
            out[m, 0] = P[0] + t * (Q[0] - P[0])
            out[m, 1] = P[1] + t * (Q[1] - P[1])
            out[m, 2] = P[2] + t * (Q[2] - P[2])
            m += 1
    return m


@njit(cache=False)
def _poly_area_centroid(poly, n_in):
    if n_in < 3:
        return 0.0, 0.0, 0.0, 0.0
    p0 = poly[0]
    A = 0.0
    cx = 0.0
    cy = 0.0
    cz = 0.0
    for i in range(1, n_in - 1):
        a1 = poly[i] - p0
        a2 = poly[i + 1] - p0
        c0 = a1[1] * a2[2] - a1[2] * a2[1]
        c1 = a1[2] * a2[0] - a1[0] * a2[2]
        c2 = a1[0] * a2[1] - a1[1] * a2[0]
        a = 0.5 * math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
        A += a
        cx += a * (p0[0] + poly[i, 0] + poly[i + 1, 0]) / 3.0
        cy += a * (p0[1] + poly[i, 1] + poly[i + 1, 1]) / 3.0
        cz += a * (p0[2] + poly[i, 2] + poly[i + 1, 2]) / 3.0
    if A > 0.0:
        return A, cx / A, cy / A, cz / A
    return 0.0, cx, cy, cz


@njit(cache=False)
def _clip_polygon_cell(
    v0,
    v1,
    v2,
    xmin,
    ymin,
    zmin,
    xmax,
    ymax,
    zmax,
    skip_axis,
    poly_a,
    poly_b,
):
    poly_a[0] = v0
    poly_a[1] = v1
    poly_a[2] = v2
    n = 3
    for ax, lo, hi in (
        (0, xmin, xmax),
        (1, ymin, ymax),
        (2, zmin, zmax),
    ):
        if skip_axis >= 0 and ax == skip_axis:
            continue
        n = _clip_halfspace(poly_a, n, ax, lo, True, poly_b)
        if n == 0:
            return 0
        n = _clip_halfspace(poly_b, n, ax, hi, False, poly_a)
        if n == 0:
            return 0
    return n


@njit(cache=False)
def _tri_area_normal(v0, v1, v2):
    a = v1 - v0
    b = v2 - v0
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    area = 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
    if area < 1e-30:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    nx = cx / (2.0 * area)
    ny = cy / (2.0 * area)
    nz = cz / (2.0 * area)
    cx2 = (v0[0] + v1[0] + v2[0]) / 3.0
    cy2 = (v0[1] + v1[1] + v2[1]) / 3.0
    cz2 = (v0[2] + v1[2] + v2[2]) / 3.0
    return area, nx, ny, nz, cx2, cy2, cz2


@njit(cache=False)
def _clip_triangle_aabb(
    v0,
    v1,
    v2,
    xmin,
    ymin,
    zmin,
    xmax,
    ymax,
    zmax,
    out_tris,
    out_n,
):
    poly_a = np.zeros((_MAX_POLY, 3), dtype=np.float64)
    poly_b = np.zeros((_MAX_POLY, 3), dtype=np.float64)
    n = _clip_polygon_cell(v0, v1, v2, xmin, ymin, zmin, xmax, ymax, zmax, -1, poly_a, poly_b)
    if n < 3:
        return 0
    out_n_local = 0
    for t in range(1, n - 1):
        o = out_n + out_n_local
        out_tris[o, 0] = poly_a[0]
        out_tris[o, 1] = poly_a[t]
        out_tris[o, 2] = poly_a[t + 1]
        out_n_local += 1
    return out_n_local


@njit(cache=False)
def _detect_on_plane(v0, v1, v2, x0x, x0y, x0z, h, plane_tol):
    """All three vertices on the same grid plane of some axis (not normal-based)."""
    for axis in range(3):
        x0a = x0x if axis == 0 else (x0y if axis == 1 else x0z)
        s0 = (v0[axis] - x0a) / h
        s1 = (v1[axis] - x0a) / h
        s2 = (v2[axis] - x0a) / h
        r0 = round(s0)
        r1 = round(s1)
        r2 = round(s2)
        if (
            abs(s0 - r0) < plane_tol
            and abs(s1 - r1) < plane_tol
            and abs(s2 - r2) < plane_tol
            and r0 == r1
            and r1 == r2
        ):
            return axis, int(r0)
    return -1, -1


@njit(cache=False)
def _patch_accumulate(
    i,
    j,
    k,
    area,
    cx,
    cy,
    cz,
    fn0,
    fn1,
    fn2,
    x0x,
    x0y,
    x0z,
    h,
    ny,
    nz,
    A_c,
    S_c,
    N_c,
    W_c,
):
    if area < 1e-30:
        return
    nid = (i * ny + j) * nz + k
    A_c[nid] += area
    S_c[nid, 0] += area * cx
    S_c[nid, 1] += area * cy
    S_c[nid, 2] += area * cz
    N_c[nid, 0] += area * fn0
    N_c[nid, 1] += area * fn1
    N_c[nid, 2] += area * fn2
    ccx = x0x + (i + 0.5) * h
    ccy = x0y + (j + 0.5) * h
    ccz = x0z + (k + 0.5) * h
    dx, dy, dz = cx - ccx, cy - ccy, cz - ccz
    W_c[nid] += area * (dx * fn0 + dy * fn1 + dz * fn2)


@njit(cache=False)
def _accumulate_wall(
    vertices,
    faces,
    fnorm,
    x0x,
    x0y,
    x0z,
    h,
    nx,
    ny,
    nz,
    N_c,
    A_c,
    S_c,
    W_c,
):
    poly_a = np.zeros((_MAX_POLY, 3), dtype=np.float64)
    poly_b = np.zeros((_MAX_POLY, 3), dtype=np.float64)
    inv_h = 1.0 / h
    for fi in range(faces.shape[0]):
        i0, i1, i2 = faces[fi, 0], faces[fi, 1], faces[fi, 2]
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        _, fn0, fn1, fn2, _, _, _ = _tri_area_normal(v0, v1, v2)
        if fn0 == 0.0 and fn1 == 0.0 and fn2 == 0.0:
            continue
        lo_x = min(v0[0], v1[0], v2[0])
        hi_x = max(v0[0], v1[0], v2[0])
        lo_y = min(v0[1], v1[1], v2[1])
        hi_y = max(v0[1], v1[1], v2[1])
        lo_z = min(v0[2], v1[2], v2[2])
        hi_z = max(v0[2], v1[2], v2[2])
        i0x = max(0, int(math.floor((lo_x - x0x) * inv_h)))
        i1x = min(nx - 1, int(math.floor((hi_x - x0x) * inv_h)))
        i0y = max(0, int(math.floor((lo_y - x0y) * inv_h)))
        i1y = min(ny - 1, int(math.floor((hi_y - x0y) * inv_h)))
        i0z = max(0, int(math.floor((lo_z - x0z) * inv_h)))
        i1z = min(nz - 1, int(math.floor((hi_z - x0z) * inv_h)))
        on_axis, pi = _detect_on_plane(v0, v1, v2, x0x, x0y, x0z, h, 1e-9)
        if on_axis >= 0:
            na = fn0 if on_axis == 0 else (fn1 if on_axis == 1 else fn2)
            on_layer = pi if na > 0.0 else pi - 1
            if on_axis == 0:
                i = max(0, min(nx - 1, on_layer))
                lo_y = min(min(v0[1], v1[1]), v2[1])
                hi_y = max(max(v0[1], v1[1]), v2[1])
                lo_z = min(min(v0[2], v1[2]), v2[2])
                hi_z = max(max(v0[2], v1[2]), v2[2])
                j0 = max(0, int(math.floor((lo_y - x0y) * inv_h)))
                j1 = min(ny - 1, int(math.floor((hi_y - x0y) * inv_h)))
                k0 = max(0, int(math.floor((lo_z - x0z) * inv_h)))
                k1 = min(nz - 1, int(math.floor((hi_z - x0z) * inv_h)))
                for j in range(j0, j1 + 1):
                    for k in range(k0, k1 + 1):
                        ymin = x0y + j * h
                        ymax = ymin + h
                        zmin = x0z + k * h
                        zmax = zmin + h
                        nv = _clip_polygon_tangential(
                            v0, v1, v2, 1, 2, ymin, ymax, zmin, zmax, poly_a, poly_b
                        )
                        area, cx, cy, cz = _poly_area_centroid(poly_a, nv)
                        if area > 0.0:
                            _patch_accumulate(
                                i, j, k, area, cx, cy, cz, fn0, fn1, fn2,
                                x0x, x0y, x0z, h, ny, nz, A_c, S_c, N_c, W_c,
                            )
            elif on_axis == 1:
                j = max(0, min(ny - 1, on_layer))
                lo_x = min(min(v0[0], v1[0]), v2[0])
                hi_x = max(max(v0[0], v1[0]), v2[0])
                lo_z = min(min(v0[2], v1[2]), v2[2])
                hi_z = max(max(v0[2], v1[2]), v2[2])
                i0 = max(0, int(math.floor((lo_x - x0x) * inv_h)))
                i1 = min(nx - 1, int(math.floor((hi_x - x0x) * inv_h)))
                k0 = max(0, int(math.floor((lo_z - x0z) * inv_h)))
                k1 = min(nz - 1, int(math.floor((hi_z - x0z) * inv_h)))
                for i in range(i0, i1 + 1):
                    for k in range(k0, k1 + 1):
                        xmin = x0x + i * h
                        xmax = xmin + h
                        zmin = x0z + k * h
                        zmax = zmin + h
                        nv = _clip_polygon_tangential(
                            v0, v1, v2, 0, 2, xmin, xmax, zmin, zmax, poly_a, poly_b
                        )
                        area, cx, cy, cz = _poly_area_centroid(poly_a, nv)
                        if area > 0.0:
                            _patch_accumulate(
                                i, j, k, area, cx, cy, cz, fn0, fn1, fn2,
                                x0x, x0y, x0z, h, ny, nz, A_c, S_c, N_c, W_c,
                            )
            else:
                k = max(0, min(nz - 1, on_layer))
                lo_x = min(min(v0[0], v1[0]), v2[0])
                hi_x = max(max(v0[0], v1[0]), v2[0])
                lo_y = min(min(v0[1], v1[1]), v2[1])
                hi_y = max(max(v0[1], v1[1]), v2[1])
                i0 = max(0, int(math.floor((lo_x - x0x) * inv_h)))
                i1 = min(nx - 1, int(math.floor((hi_x - x0x) * inv_h)))
                j0 = max(0, int(math.floor((lo_y - x0y) * inv_h)))
                j1 = min(ny - 1, int(math.floor((hi_y - x0y) * inv_h)))
                for i in range(i0, i1 + 1):
                    for j in range(j0, j1 + 1):
                        xmin = x0x + i * h
                        xmax = xmin + h
                        ymin = x0y + j * h
                        ymax = ymin + h
                        nv = _clip_polygon_tangential(
                            v0, v1, v2, 0, 1, xmin, xmax, ymin, ymax, poly_a, poly_b
                        )
                        area, cx, cy, cz = _poly_area_centroid(poly_a, nv)
                        if area > 0.0:
                            _patch_accumulate(
                                i, j, k, area, cx, cy, cz, fn0, fn1, fn2,
                                x0x, x0y, x0z, h, ny, nz, A_c, S_c, N_c, W_c,
                            )
            continue

        for i in range(i0x, i1x + 1):
            for j in range(i0y, i1y + 1):
                for k in range(i0z, i1z + 1):
                    xmin = x0x + i * h
                    xmax = xmin + h
                    ymin = x0y + j * h
                    ymax = ymin + h
                    zmin = x0z + k * h
                    zmax = zmin + h
                    nv = _clip_polygon_cell(
                        v0, v1, v2, xmin, ymin, zmin, xmax, ymax, zmax, -1, poly_a, poly_b
                    )
                    area, cx, cy, cz = _poly_area_centroid(poly_a, nv)
                    if area > 0.0:
                        _patch_accumulate(
                            i, j, k, area, cx, cy, cz, fn0, fn1, fn2,
                            x0x, x0y, x0z, h, ny, nz, A_c, S_c, N_c, W_c,
                        )


@njit(cache=False)
def _sweep_axis(Nc, alpha_faces, start_neg, axis, nx, ny, nz, h):
    hsq = h * h
    max_res = 0.0
    if axis == 0:
        for j in range(ny):
            for k in range(nz):
                if start_neg:
                    cur = 1.0
                    alpha_faces[0, j, k] = cur
                    for i in range(nx):
                        cur = cur + Nc[(i * ny + j) * nz + k, 0] / hsq
                        alpha_faces[i + 1, j, k] = cur
                    max_res = max(max_res, abs(cur - 1.0))
                else:
                    cur = 1.0
                    alpha_faces[nx, j, k] = cur
                    for ii in range(nx):
                        i = nx - 1 - ii
                        cur = cur - Nc[(i * ny + j) * nz + k, 0] / hsq
                        alpha_faces[i, j, k] = cur
                    max_res = max(max_res, abs(cur - 1.0))
    elif axis == 1:
        for i in range(nx):
            for k in range(nz):
                if start_neg:
                    cur = 1.0
                    alpha_faces[i, 0, k] = cur
                    for j in range(ny):
                        cur = cur + Nc[(i * ny + j) * nz + k, 1] / hsq
                        alpha_faces[i, j + 1, k] = cur
                    max_res = max(max_res, abs(cur - 1.0))
                else:
                    cur = 1.0
                    alpha_faces[i, ny, k] = cur
                    for jj in range(ny):
                        j = ny - 1 - jj
                        cur = cur - Nc[(i * ny + j) * nz + k, 1] / hsq
                        alpha_faces[i, j, k] = cur
                    max_res = max(max_res, abs(cur - 1.0))
    else:
        for i in range(nx):
            for j in range(ny):
                if start_neg:
                    cur = 1.0
                    alpha_faces[i, j, 0] = cur
                    for k in range(nz):
                        cur = cur + Nc[(i * ny + j) * nz + k, 2] / hsq
                        alpha_faces[i, j, k + 1] = cur
                    max_res = max(max_res, abs(cur - 1.0))
                else:
                    cur = 1.0
                    alpha_faces[i, j, nz] = cur
                    for kk in range(nz):
                        k = nz - 1 - kk
                        cur = cur - Nc[(i * ny + j) * nz + k, 2] / hsq
                        alpha_faces[i, j, k] = cur
                    max_res = max(max_res, abs(cur - 1.0))
    return max_res


@njit(cache=False)
def _clip_polygon_tangential(
    v0,
    v1,
    v2,
    ax_a,
    ax_b,
    lo_a,
    hi_a,
    lo_b,
    hi_b,
    poly_a,
    poly_b,
):
    poly_a[0] = v0
    poly_a[1] = v1
    poly_a[2] = v2
    n = 3
    for ax, lo, hi in ((ax_a, lo_a, hi_a), (ax_b, lo_b, hi_b)):
        n = _clip_halfspace(poly_a, n, ax, lo, True, poly_b)
        if n == 0:
            return 0
        n = _clip_halfspace(poly_b, n, ax, hi, False, poly_a)
        if n == 0:
            return 0
    return n


@njit(cache=False)
def _clip_triangle_tangential(
    v0,
    v1,
    v2,
    ax_a,
    ax_b,
    lo_a,
    hi_a,
    lo_b,
    hi_b,
    out_tris,
    out_n,
):
    poly_a = np.zeros((_MAX_POLY, 3), dtype=np.float64)
    poly_b = np.zeros((_MAX_POLY, 3), dtype=np.float64)
    n = _clip_polygon_tangential(v0, v1, v2, ax_a, ax_b, lo_a, hi_a, lo_b, hi_b, poly_a, poly_b)
    if n < 3:
        return 0
    out_n_local = 0
    for t in range(1, n - 1):
        o = out_n + out_n_local
        out_tris[o, 0] = poly_a[0]
        out_tris[o, 1] = poly_a[t]
        out_tris[o, 2] = poly_a[t + 1]
        out_n_local += 1
    return out_n_local


@njit(cache=False)
def _compute_beta(alpha_x, alpha_y, alpha_z, W_c, h, nx, ny, nz, beta):
    h3 = h * h * h
    face_flux = (h * 0.5) * (h * h)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                nid = (i * ny + j) * nz + k
                af = (
                    alpha_x[i, j, k]
                    + alpha_x[i + 1, j, k]
                    + alpha_y[i, j, k]
                    + alpha_y[i, j + 1, k]
                    + alpha_z[i, j, k]
                    + alpha_z[i, j, k + 1]
                )
                beta[i, j, k] = (face_flux * af - W_c[nid]) / (3.0 * h3)


@njit(cache=False)
def _accumulate_radiator(
    vertices,
    faces,
    ere,
    x0x,
    x0y,
    x0z,
    h,
    nx,
    ny,
    nz,
    a_cell,
    m_cell,
):
    poly_a = np.zeros((_MAX_POLY, 3), dtype=np.float64)
    poly_b = np.zeros((_MAX_POLY, 3), dtype=np.float64)
    tmp_tris = np.zeros((8, 3, 3), dtype=np.float64)
    inv_h = 1.0 / h
    for fi in range(faces.shape[0]):
        i0, i1, i2 = faces[fi, 0], faces[fi, 1], faces[fi, 2]
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        _, fn0, fn1, fn2, _, _, _ = _tri_area_normal(v0, v1, v2)
        edot = ere[0] * fn0 + ere[1] * fn1 + ere[2] * fn2
        lo_x = min(v0[0], v1[0], v2[0])
        hi_x = max(v0[0], v1[0], v2[0])
        lo_y = min(v0[1], v1[1], v2[1])
        hi_y = max(v0[1], v1[1], v2[1])
        lo_z = min(v0[2], v1[2], v2[2])
        hi_z = max(v0[2], v1[2], v2[2])
        i0x = max(0, int(math.floor((lo_x - x0x) * inv_h)))
        i1x = min(nx - 1, int(math.floor((hi_x - x0x) * inv_h)))
        i0y = max(0, int(math.floor((lo_y - x0y) * inv_h)))
        i1y = min(ny - 1, int(math.floor((hi_y - x0y) * inv_h)))
        i0z = max(0, int(math.floor((lo_z - x0z) * inv_h)))
        i1z = min(nz - 1, int(math.floor((hi_z - x0z) * inv_h)))
        on_axis, pi = _detect_on_plane(v0, v1, v2, x0x, x0y, x0z, h, 1e-9)
        if on_axis >= 0:
            _, fn0, fn1, fn2, _, _, _ = _tri_area_normal(v0, v1, v2)
            na = fn0 if on_axis == 0 else (fn1 if on_axis == 1 else fn2)
            on_layer = pi if na > 0.0 else pi - 1
            if on_axis == 0:
                ii = max(0, min(nx - 1, on_layer))
                j0 = max(0, int(math.floor((lo_y - x0y) * inv_h)))
                j1 = min(ny - 1, int(math.floor((hi_y - x0y) * inv_h)))
                k0 = max(0, int(math.floor((lo_z - x0z) * inv_h)))
                k1 = min(nz - 1, int(math.floor((hi_z - x0z) * inv_h)))
                for jj in range(j0, j1 + 1):
                    for kk in range(k0, k1 + 1):
                        ymin = x0y + jj * h
                        ymax = ymin + h
                        zmin = x0z + kk * h
                        zmax = zmin + h
                        nv = _clip_polygon_tangential(
                            v0, v1, v2, 1, 2, ymin, ymax, zmin, zmax, poly_a, poly_b
                        )
                        area, cx, cy, cz = _poly_area_centroid(poly_a, nv)
                        if area >= 1e-12 * h * h:
                            idx = (ii * ny + jj) * nz + kk
                            w = area * edot
                            a_cell[idx] += w
                            m_cell[idx, 0] += w * cx
                            m_cell[idx, 1] += w * cy
                            m_cell[idx, 2] += w * cz
            elif on_axis == 1:
                jj = max(0, min(ny - 1, on_layer))
                i0 = max(0, int(math.floor((lo_x - x0x) * inv_h)))
                i1 = min(nx - 1, int(math.floor((hi_x - x0x) * inv_h)))
                k0 = max(0, int(math.floor((lo_z - x0z) * inv_h)))
                k1 = min(nz - 1, int(math.floor((hi_z - x0z) * inv_h)))
                for ii in range(i0, i1 + 1):
                    for kk in range(k0, k1 + 1):
                        xmin = x0x + ii * h
                        xmax = xmin + h
                        zmin = x0z + kk * h
                        zmax = zmin + h
                        nv = _clip_polygon_tangential(
                            v0, v1, v2, 0, 2, xmin, xmax, zmin, zmax, poly_a, poly_b
                        )
                        area, cx, cy, cz = _poly_area_centroid(poly_a, nv)
                        if area >= 1e-12 * h * h:
                            idx = (ii * ny + jj) * nz + kk
                            w = area * edot
                            a_cell[idx] += w
                            m_cell[idx, 0] += w * cx
                            m_cell[idx, 1] += w * cy
                            m_cell[idx, 2] += w * cz
            else:
                kk = max(0, min(nz - 1, on_layer))
                i0 = max(0, int(math.floor((lo_x - x0x) * inv_h)))
                i1 = min(nx - 1, int(math.floor((hi_x - x0x) * inv_h)))
                j0 = max(0, int(math.floor((lo_y - x0y) * inv_h)))
                j1 = min(ny - 1, int(math.floor((hi_y - x0y) * inv_h)))
                for ii in range(i0, i1 + 1):
                    for jj in range(j0, j1 + 1):
                        xmin = x0x + ii * h
                        xmax = xmin + h
                        ymin = x0y + jj * h
                        ymax = ymin + h
                        nv = _clip_polygon_tangential(
                            v0, v1, v2, 0, 1, xmin, xmax, ymin, ymax, poly_a, poly_b
                        )
                        area, cx, cy, cz = _poly_area_centroid(poly_a, nv)
                        if area >= 1e-12 * h * h:
                            idx = (ii * ny + jj) * nz + kk
                            w = area * edot
                            a_cell[idx] += w
                            m_cell[idx, 0] += w * cx
                            m_cell[idx, 1] += w * cy
                            m_cell[idx, 2] += w * cz
            continue
        for i in range(i0x, i1x + 1):
            for j in range(i0y, i1y + 1):
                for k in range(i0z, i1z + 1):
                    xmin = x0x + i * h
                    xmax = xmin + h
                    ymin = x0y + j * h
                    ymax = ymin + h
                    zmin = x0z + k * h
                    zmax = zmin + h
                    nsub = _clip_triangle_aabb(
                        v0, v1, v2, xmin, ymin, zmin, xmax, ymax, zmax, tmp_tris, 0
                    )
                    for t in range(nsub):
                        area, _, _, _, _, _, _ = _tri_area_normal(
                            tmp_tris[t, 0], tmp_tris[t, 1], tmp_tris[t, 2]
                        )
                        if area < 1e-12 * h * h:
                            continue
                        nid = (i * ny + j) * nz + k
                        t0v = tmp_tris[t, 0]
                        t1v = tmp_tris[t, 1]
                        t2v = tmp_tris[t, 2]
                        cx = (t0v[0] + t1v[0] + t2v[0]) / 3.0
                        cy = (t0v[1] + t1v[1] + t2v[1]) / 3.0
                        cz = (t0v[2] + t1v[2] + t2v[2]) / 3.0
                        w = area * edot
                        a_cell[nid] += w
                        m_cell[nid, 0] += w * cx
                        m_cell[nid, 1] += w * cy
                        m_cell[nid, 2] += w * cz


@njit(cache=False)
def _clamp_alphas(alpha_x, alpha_y, alpha_z):
    for i in range(alpha_x.shape[0]):
        for j in range(alpha_x.shape[1]):
            for k in range(alpha_x.shape[2]):
                v = alpha_x[i, j, k]
                if v < -1e-9 or v > 1.0 + 1e-9:
                    raise ValueError(1)
                if v < 1e-10:
                    alpha_x[i, j, k] = 0.0
                elif v > 1.0 - 1e-10:
                    alpha_x[i, j, k] = 1.0
    for i in range(alpha_y.shape[0]):
        for j in range(alpha_y.shape[1]):
            for k in range(alpha_y.shape[2]):
                v = alpha_y[i, j, k]
                if v < -1e-9 or v > 1.0 + 1e-9:
                    raise ValueError(1)
                if v < 1e-10:
                    alpha_y[i, j, k] = 0.0
                elif v > 1.0 - 1e-10:
                    alpha_y[i, j, k] = 1.0
    for i in range(alpha_z.shape[0]):
        for j in range(alpha_z.shape[1]):
            for k in range(alpha_z.shape[2]):
                v = alpha_z[i, j, k]
                if v < -1e-9 or v > 1.0 + 1e-9:
                    raise ValueError(1)
                if v < 1e-10:
                    alpha_z[i, j, k] = 0.0
                elif v > 1.0 - 1e-10:
                    alpha_z[i, j, k] = 1.0


@njit(cache=False)
def _flood_fill(
    alpha_x,
    alpha_y,
    alpha_z,
    beta,
    start_neg_x,
    start_neg_y,
    start_neg_z,
    nx,
    ny,
    nz,
):
    visited = np.zeros(nx * ny * nz, dtype=np.uint8)
    stack_i = np.zeros(nx * ny * nz, dtype=np.int64)
    stack_j = np.zeros(nx * ny * nz, dtype=np.int64)
    stack_k = np.zeros(nx * ny * nz, dtype=np.int64)
    sp = 0
    removed_count = 0
    removed_vol = 0.0

    for j in range(ny):
        for k in range(nz):
            ii = 0 if start_neg_x else nx - 1
            nid = (ii * ny + j) * nz + k
            if not visited[nid]:
                visited[nid] = 1
                stack_i[sp] = ii
                stack_j[sp] = j
                stack_k[sp] = k
                sp += 1
    for i in range(nx):
        for k in range(nz):
            jj = 0 if start_neg_y else ny - 1
            nid = (i * ny + jj) * nz + k
            if not visited[nid]:
                visited[nid] = 1
                stack_i[sp] = i
                stack_j[sp] = jj
                stack_k[sp] = k
                sp += 1
    for i in range(nx):
        for j in range(ny):
            kk = 0 if start_neg_z else nz - 1
            nid = (i * ny + j) * nz + kk
            if not visited[nid]:
                visited[nid] = 1
                stack_i[sp] = i
                stack_j[sp] = j
                stack_k[sp] = kk
                sp += 1

    while sp > 0:
        sp -= 1
        i = int(stack_i[sp])
        j = int(stack_j[sp])
        k = int(stack_k[sp])
        if i + 1 < nx and alpha_x[i + 1, j, k] > 0.0:
            nid = ((i + 1) * ny + j) * nz + k
            if not visited[nid]:
                visited[nid] = 1
                stack_i[sp] = i + 1
                stack_j[sp] = j
                stack_k[sp] = k
                sp += 1
        if i > 0 and alpha_x[i, j, k] > 0.0:
            nid = ((i - 1) * ny + j) * nz + k
            if not visited[nid]:
                visited[nid] = 1
                stack_i[sp] = i - 1
                stack_j[sp] = j
                stack_k[sp] = k
                sp += 1
        if j + 1 < ny and alpha_y[i, j + 1, k] > 0.0:
            nid = (i * ny + j + 1) * nz + k
            if not visited[nid]:
                visited[nid] = 1
                stack_i[sp] = i
                stack_j[sp] = j + 1
                stack_k[sp] = k
                sp += 1
        if j > 0 and alpha_y[i, j, k] > 0.0:
            nid = (i * ny + j - 1) * nz + k
            if not visited[nid]:
                visited[nid] = 1
                stack_i[sp] = i
                stack_j[sp] = j - 1
                stack_k[sp] = k
                sp += 1
        if k + 1 < nz and alpha_z[i, j, k + 1] > 0.0:
            nid = (i * ny + j) * nz + k + 1
            if not visited[nid]:
                visited[nid] = 1
                stack_i[sp] = i
                stack_j[sp] = j
                stack_k[sp] = k + 1
                sp += 1
        if k > 0 and alpha_z[i, j, k] > 0.0:
            nid = (i * ny + j) * nz + k - 1
            if not visited[nid]:
                visited[nid] = 1
                stack_i[sp] = i
                stack_j[sp] = j
                stack_k[sp] = k - 1
                sp += 1

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                nid = (i * ny + j) * nz + k
                if not visited[nid] and beta[i, j, k] > 0.5:
                    removed_vol += beta[i, j, k]
                    beta[i, j, k] = 0.0
                    removed_count += 1
                    alpha_x[i, j, k] = 0.0
                    if i + 1 < nx + 1:
                        alpha_x[i + 1, j, k] = 0.0
                    alpha_y[i, j, k] = 0.0
                    if j + 1 < ny + 1:
                        alpha_y[i, j + 1, k] = 0.0
                    alpha_z[i, j, k] = 0.0
                    if k + 1 < nz + 1:
                        alpha_z[i, j, k + 1] = 0.0
    return removed_count, removed_vol


def _natural_grid_for_box(geometry: Geometry, cfg, h: float) -> Grid:
    """Align grid planes to the box mesh for legacy ``from_box`` tests."""
    from boxrad.config import derive

    d = derive(cfg)
    xc, yc = cfg.radiator.x_c, cfg.radiator.y_c
    W, H, D = d.realized_W, d.realized_H, d.realized_D
    x0 = np.array([-xc - 0.5 * W, -yc - 0.5 * H, -D], dtype=_F64)
    nx = int(round(W / h))
    ny = int(round(H / h))
    nz = int(round(D / h))
    rigid: dict[str, bool] = {}
    for pl in geometry.planes:
        rigid[f"{pl.axis}{pl.side}"] = True
    margin = 3
    lo = np.min(geometry.solid.vertices, axis=0)
    hi = np.max(geometry.solid.vertices, axis=0)
    g = grid_for_geometry(geometry, h, margin_cells=margin)
    # snap x0 to box corner for aligned tests
    if abs(lo[0] - x0[0]) < h and abs(lo[1] - x0[1]) < h and abs(lo[2] - x0[2]) < h:
        return Grid(x0=x0, h=h, shape=(nx, ny, nz), rigid_sides=rigid or g.rigid_sides, h_nominal=h)
    return g


def compute_fractions(geometry: Geometry | None, grid: Grid) -> Fractions:
    if geometry is None:
        raise ValueError("baffle geometry is not supported for compute_fractions")
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if geometry.solid.faces.shape[0] == 0:
        alpha_x = np.ones((nx + 1, ny, nz), dtype=_F64)
        alpha_y = np.ones((nx, ny + 1, nz), dtype=_F64)
        alpha_z = np.ones((nx, ny, nz + 1), dtype=_F64)
        beta = np.ones((nx, ny, nz), dtype=_F64)
        state = np.full((nx, ny, nz), _STATE_FLUID, dtype=np.uint8)
        return Fractions(
            alpha=(alpha_x, alpha_y, alpha_z),
            beta=beta,
            state=state,
            patch_cell=np.zeros(0, dtype=_I64),
            patch_A=np.zeros(0, dtype=_F64),
            patch_centroid=np.zeros((0, 3), dtype=_F64),
            patch_normal=np.zeros((0, 3), dtype=_F64),
            radiator_cell=[np.zeros(0, dtype=_I64) for _ in geometry.radiators],
            radiator_a=[np.zeros(0, dtype=_F64) for _ in geometry.radiators],
            radiator_S=[0.0 for _ in geometry.radiators],
            radiator_centroid=[np.zeros((0, 3), dtype=_F64) for _ in geometry.radiators],
            stats={
                "n_cut_cells": 0,
                "n_cut_faces": 0,
                "min_beta_positive": 1.0,
                "fluid_volume": float(nx * ny * nz * grid.h**3),
                "mesh_volume": 0.0,
                "volume_rel_error": 0.0,
                "volume_rel_error_raw": 0.0,
                "max_line_closure_residual": 0.0,
                "voids_removed_count": 0,
                "voids_removed_volume": 0.0,
                "radiator_S": [0.0 for _ in geometry.radiators],
                "wall_time_s": 0.0,
                "h_shrunk": grid.h if grid.h != grid.h_nominal else None,
            },
        )
    t0 = time.perf_counter()
    mesh = _snap_mesh(geometry.solid, grid)
    fn_dummies = mesh.face_normals  # unused; normals from triangle cross product in kernel
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    ncell = nx * ny * nz
    N_c = np.zeros((ncell, 3), dtype=_F64)
    A_c = np.zeros(ncell, dtype=_F64)
    S_c = np.zeros((ncell, 3), dtype=_F64)
    W_c = np.zeros(ncell, dtype=_F64)
    _accumulate_wall(
        mesh.vertices,
        mesh.faces,
        fn_dummies,
        grid.x0[0],
        grid.x0[1],
        grid.x0[2],
        grid.h,
        nx,
        ny,
        nz,
        N_c,
        A_c,
        S_c,
        W_c,
    )
    alpha_x = np.ones((nx + 1, ny, nz), dtype=_F64)
    alpha_y = np.ones((nx, ny + 1, nz), dtype=_F64)
    alpha_z = np.ones((nx, ny, nz + 1), dtype=_F64)
    max_closure = 0.0
    for axis in range(3):
        side = grid.absorbing_side(axis)
        if side is None:
            continue
        start_neg = side == "-"
        if axis == 0:
            res = _sweep_axis(N_c, alpha_x, start_neg, axis, nx, ny, nz, grid.h)
        elif axis == 1:
            res = _sweep_axis(N_c, alpha_y, start_neg, axis, nx, ny, nz, grid.h)
        else:
            res = _sweep_axis(N_c, alpha_z, start_neg, axis, nx, ny, nz, grid.h)
        if res > 1e-9:
            raise ValueError(
                f"non-watertight mesh: line closure residual {res:.3e} on axis {_AXIS[axis]}"
            )
        max_closure = max(max_closure, res)
    try:
        _clamp_alphas(alpha_x, alpha_y, alpha_z)
    except ValueError as exc:
        raise ValueError("alpha out of [0,1] bounds") from exc
    beta = np.zeros((nx, ny, nz), dtype=_F64)
    _compute_beta(alpha_x, alpha_y, alpha_z, W_c, grid.h, nx, ny, nz, beta)
    state = np.full((nx, ny, nz), _STATE_FLUID, dtype=np.uint8)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                nid = (i * ny + j) * nz + k
                has_patch = A_c[nid] > 0.0
                b = beta[i, j, k]
                if b < -1e-9 or b > 1.0 + 1e-9:
                    raise ValueError(f"beta out of bounds in cell ({i},{j},{k}): {b}")
                if b < _TOL:
                    beta[i, j, k] = 0.0
                elif b > 1.0 - _TOL:
                    beta[i, j, k] = 1.0
                b = beta[i, j, k]
                if b <= _TOL and not has_patch:
                    state[i, j, k] = _STATE_SOLID
                elif b >= 1.0 - _TOL and not has_patch:
                    state[i, j, k] = _STATE_FLUID
                else:
                    state[i, j, k] = _STATE_CUT
    rigid_starts = []
    for ax in range(3):
        side = grid.absorbing_side(ax)
        rigid_starts.append(side == "-" if side is not None else True)
    void_count, void_vol = _flood_fill(
        alpha_x,
        alpha_y,
        alpha_z,
        beta,
        rigid_starts[0],
        rigid_starts[1],
        rigid_starts[2],
        nx,
        ny,
        nz,
    )
    h3 = grid.h ** 3
    fluid_vol = float(np.sum((1.0 - beta) * h3))
    mesh_vol = geometry.solid.volume
    voids_removed_volume = float(void_vol * h3)
    target_vol = mesh_vol + voids_removed_volume
    rel_vol_err = abs(fluid_vol - target_vol) / max(abs(target_vol), 1e-30)
    rel_vol_err_raw = abs(fluid_vol - mesh_vol) / max(abs(mesh_vol), 1e-30)
    patch_mask = A_c > 0.0
    patch_cell = np.where(patch_mask)[0].astype(_I64)
    patch_A = A_c[patch_mask].astype(_F64)
    patch_centroid = np.zeros((patch_cell.shape[0], 3), dtype=_F64)
    patch_normal = np.zeros((patch_cell.shape[0], 3), dtype=_F64)
    for idx, nid in enumerate(patch_cell):
        a = A_c[nid]
        patch_centroid[idx] = S_c[nid] / a
        patch_normal[idx] = N_c[nid] / a
    n_cut = int(np.sum((beta > _TOL) & (beta < 1.0 - _TOL)))
    n_cut_faces = 0
    for arr in (alpha_x, alpha_y, alpha_z):
        n_cut_faces += int(np.sum((arr > _TOL) & (arr < 1.0 - _TOL)))
    min_beta = float(np.min(beta[beta > 0.0])) if np.any(beta > 0.0) else 0.0
    rad_cell: list[np.ndarray] = []
    rad_a: list[np.ndarray] = []
    rad_S: list[float] = []
    rad_centroid: list[np.ndarray] = []
    for rd in geometry.radiators:
        ere = rd.pose.R[:, 2]
        ere = ere / np.linalg.norm(ere)
        a_c = np.zeros(ncell, dtype=_F64)
        m_c = np.zeros((ncell, 3), dtype=_F64)
        v = _snap_mesh(TriMesh(rd.vertices_world, rd.triangles), grid).vertices
        _accumulate_radiator(
            v,
            rd.triangles,
            ere.astype(_F64),
            grid.x0[0],
            grid.x0[1],
            grid.x0[2],
            grid.h,
            nx,
            ny,
            nz,
            a_c,
            m_c,
        )
        mask = a_c != 0.0
        rad_cell.append(np.where(mask)[0].astype(_I64))
        rad_a.append(a_c[mask].astype(_F64))
        rad_centroid.append((m_c[mask] / a_c[mask, np.newaxis]).astype(_F64))
        rad_S.append(float(np.sum(a_c)))
    wall_s = time.perf_counter() - t0
    stats = {
        "n_cut_cells": n_cut,
        "n_cut_faces": n_cut_faces,
        "min_beta_positive": min_beta,
        "fluid_volume": fluid_vol,
        "mesh_volume": mesh_vol,
        "volume_rel_error": rel_vol_err,
        "volume_rel_error_raw": rel_vol_err_raw,
        "max_line_closure_residual": max_closure,
        "voids_removed_count": void_count,
        "voids_removed_volume": voids_removed_volume,
        "radiator_S": rad_S,
        "wall_time_s": wall_s,
        "h_shrunk": grid.h if grid.h != grid.h_nominal else None,
    }
    return Fractions(
        alpha=(alpha_x, alpha_y, alpha_z),
        beta=beta,
        state=state,
        patch_cell=patch_cell,
        patch_A=patch_A,
        patch_centroid=patch_centroid,
        patch_normal=patch_normal,
        radiator_cell=rad_cell,
        radiator_a=rad_a,
        radiator_S=rad_S,
        radiator_centroid=rad_centroid,
        stats=stats,
    )