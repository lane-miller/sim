"""Closed triangle meshes (numpy + numba)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

_F64 = np.float64
_I64 = np.int64

_GEOM_IMPORT_ERR = (
    "parametric geometry needs: python -m pip install -e proj/boxed_piston[geom]"
)


def _arc_segments(radius: float, tol: float) -> int:
    if radius <= 0.0:
        return 16
    c = 1.0 - tol / radius
    c = min(1.0, max(-1.0, c))
    ang = math.acos(c)
    n = int(math.ceil(math.pi / ang))
    return max(16, n)


def polygonal_cylinder_volume(n: int, r: float, length: float) -> float:
    return 0.5 * n * r * r * math.sin(2.0 * math.pi / n) * length


def box_volume_exact(sx: float, sy: float, sz: float) -> float:
    return sx * sy * sz


@dataclass(frozen=True)
class TriMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=_F64)
        f = np.asarray(self.faces, dtype=_I64)
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "faces", f)

    @property
    def bbox(self) -> tuple[float, float, float, float, float, float]:
        v = self.vertices
        return (
            float(np.min(v[:, 0])),
            float(np.min(v[:, 1])),
            float(np.min(v[:, 2])),
            float(np.max(v[:, 0])),
            float(np.max(v[:, 1])),
            float(np.max(v[:, 2])),
        )

    def _bbox_diag(self) -> float:
        lo = np.min(self.vertices, axis=0)
        hi = np.max(self.vertices, axis=0)
        return float(np.linalg.norm(hi - lo))

    @property
    def face_areas(self) -> np.ndarray:
        v = self.vertices
        f = self.faces
        a = v[f[:, 1]] - v[f[:, 0]]
        b = v[f[:, 2]] - v[f[:, 0]]
        return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)

    @property
    def face_normals(self) -> np.ndarray:
        v = self.vertices
        f = self.faces
        a = v[f[:, 1]] - v[f[:, 0]]
        b = v[f[:, 2]] - v[f[:, 0]]
        n = np.cross(a, b)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln = np.maximum(ln, 1e-30)
        return n / ln

    @property
    def area(self) -> float:
        return float(np.sum(self.face_areas))

    @property
    def volume(self) -> float:
        v = self.vertices
        f = self.faces
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        return float(np.sum(np.sum(v0 * np.cross(v1, v2), axis=1)) / 6.0)

    def transformed(self, R: np.ndarray, t: np.ndarray) -> TriMesh:
        R = np.asarray(R, dtype=_F64)
        t = np.asarray(t, dtype=_F64).reshape(3)
        return TriMesh(self.vertices @ R.T + t, self.faces.copy())

    def is_closed(self) -> bool:
        pending: dict[tuple[int, int], bool] = {}
        for row in self.faces:
            for a, b in (
                (int(row[0]), int(row[1])),
                (int(row[1]), int(row[2])),
                (int(row[2]), int(row[0])),
            ):
                rev = (b, a)
                if rev in pending:
                    del pending[rev]
                else:
                    pending[(a, b)] = True
        return len(pending) == 0


def _assert_outward(mesh: TriMesh) -> None:
    if mesh.volume <= 0.0:
        raise ValueError("mesh faces must be outward-oriented (positive volume)")
    center = np.mean(mesh.vertices, axis=0)
    fn = mesh.face_normals
    fc = np.mean(mesh.vertices[mesh.faces], axis=1)
    dots = np.sum((fc - center) * fn, axis=1)
    if float(np.min(dots)) <= 0.0:
        raise ValueError("mesh face normals must point away from the centroid")


def weld(vertices: np.ndarray, faces: np.ndarray, tol: float | None = None) -> TriMesh:
    v = np.asarray(vertices, dtype=_F64)
    f = np.asarray(faces, dtype=_I64)
    lo = np.min(v, axis=0)
    hi = np.max(v, axis=0)
    L = float(np.linalg.norm(hi - lo))
    if tol is None:
        tol = 1e-9 * max(L, 1.0)
    cell = max(tol, 1e-30)
    keys = np.floor(v / cell).astype(np.int64)
    unique: dict[tuple[int, int, int], int] = {}
    new_v: list[np.ndarray] = []
    remap = np.zeros(v.shape[0], dtype=_I64)
    for i in range(v.shape[0]):
        k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        if k not in unique:
            unique[k] = len(new_v)
            new_v.append(v[i])
        remap[i] = unique[k]
    nf = remap[f]
    keep = np.ones(nf.shape[0], dtype=bool)
    for i, row in enumerate(nf):
        if row[0] == row[1] or row[1] == row[2] or row[2] == row[0]:
            keep[i] = False
    return TriMesh(np.asarray(new_v, dtype=_F64), nf[keep])


@njit(cache=False)
def _solid_angle_tri(a, b, c, p):
    ax = a[0] - p[0]
    ay = a[1] - p[1]
    az = a[2] - p[2]
    bx = b[0] - p[0]
    by = b[1] - p[1]
    bz = b[2] - p[2]
    cx = c[0] - p[0]
    cy = c[1] - p[1]
    cz = c[2] - p[2]
    la = math.sqrt(ax * ax + ay * ay + az * az)
    lb = math.sqrt(bx * bx + by * by + bz * bz)
    lc = math.sqrt(cx * cx + cy * cy + cz * cz)
    if la < 1e-30 or lb < 1e-30 or lc < 1e-30:
        return 0.0
    bxc = by * cz - bz * cy
    byc = bz * cx - bx * cz
    bzc = bx * cy - by * cx
    num = ax * bxc + ay * byc + az * bzc
    den = (
        la * lb * lc
        + (ax * bx + ay * by + az * bz) * lc
        + (bx * cx + by * cy + bz * cz) * la
        + (cx * ax + cy * ay + cz * az) * lb
    )
    return 2.0 * math.atan2(num, den)


@njit(parallel=True, cache=False)
def winding_number(vertices, faces, pts):
    n_pts = pts.shape[0]
    out = np.zeros(n_pts, dtype=_F64)
    for i in prange(n_pts):
        p = pts[i]
        total = 0.0
        for j in range(faces.shape[0]):
            i0, i1, i2 = faces[j, 0], faces[j, 1], faces[j, 2]
            total += _solid_angle_tri(vertices[i0], vertices[i1], vertices[i2], p)
        out[i] = total / (4.0 * math.pi)
    return out


def inside(mesh: TriMesh, pts: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(np.asarray(pts, dtype=_F64))
    w = winding_number(mesh.vertices, mesh.faces, p)
    return w > 0.5


def box_mesh(size: np.ndarray | list[float]) -> TriMesh:
    s = np.asarray(size, dtype=_F64).reshape(3)
    sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    verts = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=_F64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=_I64,
    )
    mesh = TriMesh(verts, faces)
    _assert_outward(mesh)
    return mesh


def cylinder_mesh(d: float, length: float, tol: float = 1e-6) -> TriMesh:
    r = 0.5 * d
    n = _arc_segments(r, tol)
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=_F64)
    z0, z1 = -0.5 * length, 0.5 * length
    ring0 = np.column_stack((r * np.cos(t), r * np.sin(t), np.full(n, z0)))
    ring1 = np.column_stack((r * np.cos(t), r * np.sin(t), np.full(n, z1)))
    center0 = np.array([[0.0, 0.0, z0]], dtype=_F64)
    center1 = np.array([[0.0, 0.0, z1]], dtype=_F64)
    verts = np.vstack((center0, ring0, center1, ring1))
    faces: list[list[int]] = []
    o0, o1 = 0, n + 1
    for i in range(n):
        j = (i + 1) % n
        i0, j0 = 1 + i, 1 + j
        i1, j1 = n + 2 + i, n + 2 + j
        faces.append([i0, j0, j1])
        faces.append([i0, j1, i1])
        faces.append([o0, j0, i0])
        faces.append([o1, i1, j1])
    faces_arr = np.asarray(faces, dtype=_I64)
    mesh = TriMesh(verts, faces_arr)
    _assert_outward(mesh)
    return mesh


def _icosphere_base() -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=_F64,
    )
    verts /= np.linalg.norm(verts[0])
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=_I64,
    )
    return verts, faces


def _subdivide(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    midpoint_cache: dict[tuple[int, int], int] = {}
    new_verts = list(verts)

    def mid(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in midpoint_cache:
            return midpoint_cache[key]
        m = 0.5 * (verts[a] + verts[b])
        m /= np.linalg.norm(m)
        idx = len(new_verts)
        midpoint_cache[key] = idx
        new_verts.append(m)
        return idx

    new_faces: list[list[int]] = []
    for i0, i1, i2 in faces:
        m01 = mid(int(i0), int(i1))
        m12 = mid(int(i1), int(i2))
        m20 = mid(int(i2), int(i0))
        new_faces.extend(
            [
                [i0, m01, m20],
                [i1, m12, m01],
                [i2, m20, m12],
                [m01, m12, m20],
            ]
        )
    return np.asarray(new_verts, dtype=_F64), np.asarray(new_faces, dtype=_I64)


def _max_sagitta(verts: np.ndarray, r: float) -> float:
    radii = np.linalg.norm(verts, axis=1)
    return float(np.max(np.abs(radii - r)))


def _max_face_sagitta(verts: np.ndarray, faces: np.ndarray, r: float) -> float:
    """Max radial gap between the sphere and triangle centroids (flat-face inset)."""
    fc = np.mean(verts[faces], axis=1)
    radial = np.linalg.norm(fc, axis=1)
    return float(np.max(r - radial))


def sphere_mesh(d: float, tol: float = 1e-6) -> TriMesh:
    r = 0.5 * d
    verts, faces = _icosphere_base()
    for _ in range(7):
        verts, faces = _subdivide(verts, faces)
        verts = r * verts / np.linalg.norm(verts, axis=1, keepdims=True)
        if _max_sagitta(verts, r) <= tol and _max_face_sagitta(verts, faces, r) <= tol:
            break
    mesh = TriMesh(verts, faces)
    _assert_outward(mesh)
    return mesh


def trimesh_to_manifold(mesh: TriMesh):
    import manifold3d

    return manifold3d.Manifold(
        manifold3d.Mesh64(
            np.asarray(mesh.vertices, dtype=_F64),
            np.asarray(mesh.faces, dtype=np.uint64),
        )
    )


def concat_meshes(meshes: list[TriMesh]) -> TriMesh:
    if len(meshes) == 1:
        return meshes[0]
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    off = 0
    for m in meshes:
        verts.append(m.vertices)
        faces.append(m.faces + off)
        off += m.vertices.shape[0]
    return TriMesh(np.vstack(verts), np.vstack(faces))


def _bbox_separation(lo_a, hi_a, lo_b, hi_b) -> float:
    gaps = []
    for k in range(3):
        g = max(float(lo_b[k] - hi_a[k]), float(lo_a[k] - hi_b[k]), 0.0)
        gaps.append(g)
    return float(math.sqrt(sum(g * g for g in gaps)))


def parts_disjoint(meshes: list[TriMesh], *, gap_tol: float = 1e-9) -> bool:
    if len(meshes) < 2:
        return True
    for i in range(len(meshes)):
        lo_i = np.min(meshes[i].vertices, axis=0)
        hi_i = np.max(meshes[i].vertices, axis=0)
        for j in range(i + 1, len(meshes)):
            lo_j = np.min(meshes[j].vertices, axis=0)
            hi_j = np.max(meshes[j].vertices, axis=0)
            if _bbox_separation(lo_i, hi_i, lo_j, hi_j) <= gap_tol:
                return False
    return True


@njit(cache=False)
def _ray_tri_moller(
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    v0x: float,
    v0y: float,
    v0z: float,
    v1x: float,
    v1y: float,
    v1z: float,
    v2x: float,
    v2y: float,
    v2z: float,
) -> float:
    eps = 1e-30
    e1x, e1y, e1z = v1x - v0x, v1y - v0y, v1z - v0z
    e2x, e2y, e2z = v2x - v0x, v2y - v0y, v2z - v0z
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if det > -eps and det < eps:
        return math.inf
    inv = 1.0 / det
    tx, ty, tz = ox - v0x, oy - v0y, oz - v0z
    u = (tx * px + ty * py + tz * pz) * inv
    if u < 0.0 or u > 1.0:
        return math.inf
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v < 0.0 or u + v > 1.0:
        return math.inf
    t = (e2x * qx + e2y * qy + e2z * qz) * inv
    if t <= 1e-9:
        return math.inf
    return t


@njit(cache=False)
def _ray_mesh_nearest_dist(
    ox: float,
    oy: float,
    oz: float,
    dx: float,
    dy: float,
    dz: float,
    vertices,
    faces,
) -> float:
    best = math.inf
    nv = vertices.shape[0]
    for fi in range(faces.shape[0]):
        i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        t = _ray_tri_moller(
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            vertices[i0, 0],
            vertices[i0, 1],
            vertices[i0, 2],
            vertices[i1, 0],
            vertices[i1, 1],
            vertices[i1, 2],
            vertices[i2, 0],
            vertices[i2, 1],
            vertices[i2, 2],
        )
        if t < best:
            best = t
    return best


def ray_mesh_nearest(
    origin: np.ndarray,
    direction: np.ndarray,
    mesh: TriMesh,
) -> float:
    o = np.asarray(origin, dtype=_F64).reshape(3)
    d = np.asarray(direction, dtype=_F64).reshape(3)
    ln = float(np.linalg.norm(d))
    if ln < 1e-30:
        return math.inf
    d = d / ln
    return float(
        _ray_mesh_nearest_dist(
            float(o[0]),
            float(o[1]),
            float(o[2]),
            float(d[0]),
            float(d[1]),
            float(d[2]),
            mesh.vertices,
            mesh.faces,
        )
    )


def union_meshes(meshes: list[TriMesh]) -> TriMesh:
    if len(meshes) == 1:
        return meshes[0]
    try:
        import manifold3d
    except ImportError as exc:
        raise ImportError(_GEOM_IMPORT_ERR) from exc
    acc = trimesh_to_manifold(meshes[0])
    for m in meshes[1:]:
        acc = acc + trimesh_to_manifold(m)
    out = acc.to_mesh64()
    verts = np.asarray(out.vert_properties[:, :3], dtype=_F64)
    faces = np.asarray(out.tri_verts, dtype=_I64)
    return TriMesh(verts, faces)