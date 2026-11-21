"""Tests for boxrad.geom.poly2d."""

from __future__ import annotations

import math

import numpy as np
import pytest
from matplotlib.path import Path

from boxrad.geom import poly2d
from boxrad.geomconfig import ShapeSpec

RTOL = 1e-12
TOL = 1e-6
BRUTE_N = 2000
BRUTE_RTOL = 1e-3


def _ring_area_exact_circle(d: float, tol: float) -> float:
    r = 0.5 * d
    n = poly2d._arc_segments(r, tol)
    return 0.5 * n * r * r * math.sin(2.0 * math.pi / n)


def test_areas_and_centroids():
    w, h = 0.12, 0.08
    r = poly2d.rect(w, h)
    assert poly2d.signed_area(r) == pytest.approx(w * h, rel=RTOL)
    c = poly2d.centroid(r)
    assert np.allclose(c, [0.0, 0.0], atol=1e-14)

    d = 0.1
    circ = poly2d.circle(d, TOL)
    a_poly = abs(poly2d.signed_area(circ))
    a_exact = _ring_area_exact_circle(d, TOL)
    assert a_poly == pytest.approx(a_exact, rel=RTOL)
    sagitta = TOL
    a_analytic = math.pi * (0.5 * d) ** 2
    assert abs(a_poly - a_analytic) <= sagitta * (2.0 * math.pi * 0.5 * d)

    rt = poly2d.racetrack(0.15, 0.05, TOL)
    a_rt_poly = abs(poly2d.signed_area(rt))
    w0, h0 = 0.15, 0.05
    d0 = min(w0, h0)
    r0 = 0.5 * d0
    a_rt = h0 * (w0 - d0) + math.pi * r0 * r0
    assert abs(a_rt_poly - a_rt) <= TOL * (2.0 * math.pi * r0 + w0)

    ell = poly2d.ellipse(0.08, 0.05, TOL)
    assert abs(poly2d.signed_area(ell)) == pytest.approx(math.pi * 0.08 * 0.05, rel=5e-3)

    hole = poly2d.circle(0.04, TOL)
    outer = poly2d.circle(0.12, TOL)
    ann = poly2d.Outline(outer, (hole,))
    assert ann.area == pytest.approx(abs(poly2d.signed_area(outer)) - abs(poly2d.signed_area(hole)), rel=RTOL)


def _brute_clip_area(ring, xmin, xmax, ymin, ymax, n: int = BRUTE_N) -> float:
    dx = (xmax - xmin) / n
    dy = (ymax - ymin) / n
    xs = xmin + (np.arange(n, dtype=np.float64) + 0.5) * dx
    ys = ymin + (np.arange(n, dtype=np.float64) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    rect_in = (
        (pts[:, 0] >= xmin)
        & (pts[:, 0] <= xmax)
        & (pts[:, 1] >= ymin)
        & (pts[:, 1] <= ymax)
    )
    inside = Path(np.asarray(ring, dtype=np.float64)).contains_points(pts)
    mask = inside & rect_in
    return float(np.count_nonzero(mask)) * dx * dy


def _rel_err(got: float, ref: float) -> float:
    return abs(got - ref) / max(abs(ref), 1e-30)


def test_clip_rect_cases():
    sq = poly2d.rect(0.2, 0.2)
    assert poly2d.clip_rect(sq, -0.5, 0.5, -0.5, 0.5) == pytest.approx(0.04, rel=RTOL)
    assert poly2d.clip_rect(sq, 0.5, 1.0, 0.5, 1.0) == pytest.approx(0.0, abs=1e-15)

    worst = 0.0
    part = poly2d.clip_rect(sq, -0.05, 0.15, -0.05, 0.15)
    brute = _brute_clip_area(sq, -0.05, 0.15, -0.05, 0.15)
    worst = max(worst, _rel_err(part, brute))

    l_shape = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.0], [0.0, 0.0], [0.0, 0.1], [-0.1, 0.1]], dtype=float)
    a = poly2d.clip_rect(l_shape, -0.05, 0.05, -0.05, 0.05)
    b = _brute_clip_area(l_shape, -0.05, 0.05, -0.05, 0.05)
    worst = max(worst, _rel_err(a, b))

    outer = poly2d.circle(0.2, 1e-3)
    hole = poly2d.circle(0.08, 1e-3)
    ann = poly2d.Outline(outer, (hole,))
    clipped = poly2d.clip_outline_rect(ann, -0.05, 0.05, -0.05, 0.05)
    b_ann = _brute_clip_area(outer, -0.05, 0.05, -0.05, 0.05)
    b_hole = _brute_clip_area(hole, -0.05, 0.05, -0.05, 0.05)
    worst = max(worst, _rel_err(clipped, b_ann - b_hole))

    assert worst <= BRUTE_RTOL, f"worst_rel_clip={worst}"


def test_clip_partition():
    ring = poly2d.rect(0.3, 0.2)
    outline = poly2d.Outline(ring, ())
    total = 0.0
    for i in range(6):
        for j in range(6):
            xmin, xmax = -0.15 + i * 0.05, -0.15 + (i + 1) * 0.05
            ymin, ymax = -0.10 + j * 0.05, -0.10 + (j + 1) * 0.05
            total += poly2d.clip_outline_rect(outline, xmin, xmax, ymin, ymax)
    assert total == pytest.approx(outline.area, rel=RTOL)


def test_triangulate_convex():
    ring = poly2d.rect(0.1, 0.05)
    v, t = poly2d.triangulate(poly2d.Outline(ring, ()))
    assert t.shape[0] == 2
    assert abs(poly2d.signed_area(ring)) == pytest.approx(
        sum(abs(poly2d.signed_area(v[row])) for row in t),
        rel=RTOL,
    )


@pytest.mark.skipif(
    not pytest.importorskip("manifold3d", reason="geom extra"),
    reason="manifold3d",
)
def test_triangulate_hole():
    outline = poly2d.Outline(
        poly2d.rect(0.2, 0.2),
        (poly2d.rect(0.1, 0.1),),
    )
    v, t = poly2d.triangulate(outline)
    assert t.shape[0] > 0


def test_bad_hole_raises():
    spec = ShapeSpec(
        "circle",
        {"d": 0.1},
        ShapeSpec("circle", {"d": 0.2}, None),
    )
    with pytest.raises(ValueError, match="hole"):
        poly2d.from_shape_spec(spec, TOL)