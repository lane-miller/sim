"""2-D polygon utilities (numpy only)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from boxrad.geomconfig import ShapeSpec

_F64 = np.float64


def _arc_segments(radius: float, tol: float) -> int:
    if radius <= 0.0:
        return 16
    c = 1.0 - tol / radius
    if c <= -1.0:
        return 16
    if c >= 1.0:
        return 16
    ang = math.acos(c)
    n = int(math.ceil(math.pi / ang))
    return max(16, n)


def signed_area(ring: np.ndarray) -> float:
    """Signed area of an open ring (shoelace)."""
    r = np.asarray(ring, dtype=_F64)
    if r.shape[0] < 3:
        return 0.0
    x = r[:, 0]
    y = r[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def centroid(ring: np.ndarray) -> np.ndarray:
    """Area centroid of a simple ring."""
    r = np.asarray(ring, dtype=_F64)
    a = signed_area(r)
    if abs(a) < 1e-30:
        return np.mean(r, axis=0)
    x = r[:, 0]
    y = r[:, 1]
    cross = x * np.roll(y, -1) - y * np.roll(x, -1)
    cx = float(np.sum((x + np.roll(x, -1)) * cross))
    cy = float(np.sum((y + np.roll(y, -1)) * cross))
    factor = 1.0 / (6.0 * a)
    return np.array([cx * factor, cy * factor], dtype=_F64)


def orient(ring: np.ndarray, *, ccw: bool = True) -> np.ndarray:
    r = np.asarray(ring, dtype=_F64)
    if r.shape[0] < 3:
        return r.copy()
    if ccw and signed_area(r) < 0.0:
        return r[::-1].copy()
    if not ccw and signed_area(r) > 0.0:
        return r[::-1].copy()
    return r.copy()


def _winding_point(ring: np.ndarray, x: float, y: float) -> int:
    r = np.asarray(ring, dtype=_F64)
    w = 0
    n = r.shape[0]
    for i in range(n):
        x1, y1 = r[i]
        x2, y2 = r[(i + 1) % n]
        if y1 <= y:
            if y2 > y:
                cross = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
                if cross > 0.0:
                    w += 1
        else:
            if y2 <= y:
                cross = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
                if cross < 0.0:
                    w -= 1
    return w


@dataclass(frozen=True)
class Outline:
    outer: np.ndarray
    holes: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        outer = orient(self.outer, ccw=True)
        holes = tuple(orient(h, ccw=False) for h in self.holes)
        object.__setattr__(self, "outer", np.asarray(outer, dtype=_F64))
        object.__setattr__(self, "holes", holes)

    @property
    def area(self) -> float:
        a = abs(signed_area(self.outer))
        for h in self.holes:
            a -= abs(signed_area(h))
        return float(a)

    @property
    def centroid(self) -> np.ndarray:
        outer_a = signed_area(self.outer)
        cx, cy = 0.0, 0.0
        total = 0.0
        if abs(outer_a) > 1e-30:
            c = centroid(self.outer)
            cx += c[0] * outer_a
            cy += c[1] * outer_a
            total += outer_a
        for h in self.holes:
            ha = signed_area(h)
            if abs(ha) > 1e-30:
                c = centroid(h)
                cx -= c[0] * abs(ha)
                cy -= c[1] * abs(ha)
                total -= abs(ha)
        if abs(total) < 1e-30:
            return centroid(self.outer)
        return np.array([cx / total, cy / total], dtype=_F64)

    def contains(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=_F64))
        out = np.zeros(p.shape[0], dtype=bool)
        for k in range(p.shape[0]):
            x, y = float(p[k, 0]), float(p[k, 1])
            inside = _winding_point(self.outer, x, y) != 0
            for h in self.holes:
                if _winding_point(h, x, y) != 0:
                    inside = False
                    break
            out[k] = inside
        return out if pts.ndim == 2 else out[0]


def _sample_arc(cx: float, cy: float, r: float, t0: float, t1: float, n: int) -> np.ndarray:
    t = np.linspace(t0, t1, n, endpoint=False, dtype=_F64)
    return np.column_stack((cx + r * np.cos(t), cy + r * np.sin(t)))


def rect(w: float, h: float) -> np.ndarray:
    hw, hh = 0.5 * w, 0.5 * h
    return np.array(
        [
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh],
        ],
        dtype=_F64,
    )


def circle(d: float, tol: float = 1e-6) -> np.ndarray:
    r = 0.5 * d
    n = _arc_segments(r, tol)
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=_F64)
    return np.column_stack((r * np.cos(t), r * np.sin(t)))


def racetrack(w: float, h: float, tol: float = 1e-6) -> np.ndarray:
    r = 0.5 * h
    n = _arc_segments(r, tol)
    straight = w - h
    if straight < 0.0:
        raise ValueError("racetrack requires w >= h")
    cx_l = -0.5 * straight
    cx_r = 0.5 * straight
    left = _sample_arc(cx_l, 0.0, r, 0.5 * math.pi, 1.5 * math.pi, n)
    right = _sample_arc(cx_r, 0.0, r, -0.5 * math.pi, 0.5 * math.pi, n)
    return np.vstack((left, right))


def ellipse(a: float, b: float, tol: float = 1e-6) -> np.ndarray:
    r = max(a, b)
    n = _arc_segments(r, tol)
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=_F64)
    return np.column_stack((a * np.cos(t), b * np.sin(t)))


def _ring_from_shape(spec: ShapeSpec, tol: float) -> np.ndarray:
    k = spec.kind
    p = spec.params
    if k == "rect":
        return rect(p["w"], p["h"])
    if k == "circle":
        return circle(p["d"], tol)
    if k == "racetrack":
        return racetrack(p["w"], p["h"], tol)
    if k == "ellipse":
        return ellipse(p["a"], p["b"], tol)
    raise ValueError(f"unknown shape {k!r}")


def from_shape_spec(spec: ShapeSpec, tol: float = 1e-6) -> Outline:
    outer = _ring_from_shape(spec, tol)
    holes: tuple[np.ndarray, ...] = ()
    if spec.hole is not None:
        hole_ring = _ring_from_shape(spec.hole, tol)
        holes = (hole_ring,)
        ol = Outline(outer, holes)
        if not np.all(ol.contains(hole_ring)):
            raise ValueError("hole is not strictly inside the outer")
        if abs(signed_area(hole_ring)) >= abs(signed_area(outer)):
            raise ValueError("hole is not strictly inside the outer")
    return Outline(outer, holes)


def _cross_z(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


def _segment_intersect_line(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    ab = b - a
    cd = d - c
    denom = ab[0] * cd[1] - ab[1] * cd[0]
    if abs(denom) < 1e-30:
        return a.copy()
    t = ((c[0] - a[0]) * cd[1] - (c[1] - a[1]) * cd[0]) / denom
    return a + t * ab


def _clip_to_edge(ring: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray) -> np.ndarray:
    r = np.asarray(ring, dtype=_F64)
    n = r.shape[0]
    if n == 0:
        return r
    out: list[np.ndarray] = []
    for i in range(n):
        cur = r[i]
        nxt = r[(i + 1) % n]
        cur_in = _cross_z(edge_a, edge_b, cur) >= -1e-15
        nxt_in = _cross_z(edge_a, edge_b, nxt) >= -1e-15
        if cur_in and nxt_in:
            out.append(nxt)
        elif cur_in and not nxt_in:
            out.append(_segment_intersect_line(cur, nxt, edge_a, edge_b))
        elif not cur_in and nxt_in:
            out.append(_segment_intersect_line(cur, nxt, edge_a, edge_b))
            out.append(nxt)
    if not out:
        return np.zeros((0, 2), dtype=_F64)
    return np.asarray(out, dtype=_F64)


def clip_rect(
    ring: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> float:
    r = orient(np.asarray(ring, dtype=_F64), ccw=True)
    window = np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=_F64,
    )
    for i in range(4):
        a, b = window[i], window[(i + 1) % 4]
        r = _clip_to_edge(r, a, b)
        if r.shape[0] < 3:
            return 0.0
    return abs(signed_area(r))


def clip_outline_rect(
    outline: Outline,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> float:
    a = clip_rect(outline.outer, xmin, xmax, ymin, ymax)
    for h in outline.holes:
        a -= clip_rect(h, xmin, xmax, ymin, ymax)
    return float(a)


def _is_convex(ring: np.ndarray) -> bool:
    r = np.asarray(ring, dtype=_F64)
    n = r.shape[0]
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        p0 = r[(i - 1) % n]
        p1 = r[i]
        p2 = r[(i + 1) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        s = 1 if cross > 1e-15 else (-1 if cross < -1e-15 else 0)
        if s == 0:
            continue
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def triangulate_convex_fan(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(ring, dtype=_F64)
    verts = r.copy()
    if r.shape[0] < 3:
        return verts, np.zeros((0, 3), dtype=np.int64)
    tris = np.column_stack(
        (
            np.zeros(r.shape[0] - 2, dtype=np.int64),
            np.arange(1, r.shape[0] - 1, dtype=np.int64),
            np.arange(2, r.shape[0], dtype=np.int64),
        )
    )
    return verts, tris


def triangulate(outline: Outline) -> tuple[np.ndarray, np.ndarray]:
    if not outline.holes and _is_convex(outline.outer):
        return triangulate_convex_fan(outline.outer)
    try:
        import manifold3d
    except ImportError as exc:
        raise ImportError(
            "parametric geometry needs: python -m pip install -e proj/boxed_piston[geom]"
        ) from exc
    polys = [outline.outer]
    for h in outline.holes:
        polys.append(h)
    tri_idx = manifold3d.triangulate(polys)
    verts = np.vstack(polys)
    tris = tri_idx.astype(np.int64)
    a_sum = 0.0
    for row in tris:
        v0, v1, v2 = verts[row]
        a_sum += abs(signed_area(np.array([v0, v1, v2, v0], dtype=_F64)))
    if outline.area > 0.0:
        rel = abs(a_sum - outline.area) / outline.area
        if rel > 1e-12:
            raise AssertionError(f"triangulation area mismatch rel={rel}")
    return verts.astype(_F64), tris