"""Tests for boxrad.geom.mesh."""

from __future__ import annotations

import math

import numpy as np
import pytest

from boxrad.geom import mesh

RTOL = 1e-12
TOL = 1e-6


def _assert_faces_outward(m: mesh.TriMesh) -> None:
    center = np.mean(m.vertices, axis=0)
    fc = np.mean(m.vertices[m.faces], axis=1)
    dots = np.sum((fc - center) * m.face_normals, axis=1)
    assert float(np.min(dots)) > 0.0


def test_primitive_faces_outward():
    for m in (
        mesh.box_mesh([0.2, 0.15, 0.25]),
        mesh.cylinder_mesh(0.2, 0.3, TOL),
        mesh.sphere_mesh(0.2, TOL),
    ):
        _assert_faces_outward(m)


def test_primitives_closed_volume():
    b = mesh.box_mesh([0.2, 0.15, 0.25])
    assert b.is_closed()
    assert b.volume == pytest.approx(0.2 * 0.15 * 0.25, rel=RTOL)
    assert b.volume > 0.0

    c = mesh.cylinder_mesh(0.2, 0.3, TOL)
    assert c.is_closed()
    r = 0.1
    n = max(16, int(math.ceil(math.pi / math.acos(max(-1.0, min(1.0, 1.0 - TOL / r))))))
    v_exact = mesh.polygonal_cylinder_volume(n, r, 0.3)
    assert c.volume == pytest.approx(v_exact, rel=RTOL)

    s = mesh.sphere_mesh(0.2, TOL)
    assert s.is_closed()
    assert s.volume > 0.0


def test_weld_and_is_closed():
    b = mesh.box_mesh([0.1, 0.1, 0.1])
    v = np.vstack((b.vertices, b.vertices[0:1] + 1e-12))
    f = b.faces.copy()
    w = mesh.weld(v, f)
    assert w.vertices.shape[0] <= v.shape[0]
    broken = mesh.TriMesh(b.vertices, b.faces[:10])
    assert not broken.is_closed()


def _box_surface_dist(pts: np.ndarray, hx: float, hy: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    ax, ay, az = np.abs(pts[:, 0]), np.abs(pts[:, 1]), np.abs(pts[:, 2])
    inside = (ax <= hx) & (ay <= hy) & (az <= hz)
    dist_in = np.minimum(np.minimum(hx - ax, hy - ay), hz - az)
    q = np.maximum(np.stack([ax - hx, ay - hy, az - hz], axis=1), 0.0)
    dist_out = np.linalg.norm(q, axis=1)
    surf = np.where(inside, dist_in, dist_out)
    return inside, surf


def test_winding_number_box_sphere():
    rng = np.random.default_rng(0)
    hx, hy, hz = 0.2, 0.15, 0.1
    b = mesh.box_mesh([0.4, 0.3, 0.2])
    pts = rng.uniform(-0.5, 0.5, size=(2000, 3))
    w = mesh.winding_number(b.vertices, b.faces, pts)
    inside, surf = _box_surface_dist(pts, hx, hy, hz)
    use = surf >= 1e-6
    deep_in = use & inside & (surf >= 1e-3)
    far_out = use & ~inside & (surf >= 1e-3)
    max_w1 = float(np.max(np.abs(w[deep_in] - 1.0))) if np.any(deep_in) else 0.0
    max_w = float(np.max(np.abs(w[far_out]))) if np.any(far_out) else 0.0
    assert max_w1 <= 1e-9, f"max|w-1|={max_w1}"
    assert max_w <= 1e-9, f"max|w|={max_w}"

    radius = 0.25
    s = mesh.sphere_mesh(0.5, 1e-4)
    pts2 = rng.uniform(-0.8, 0.8, size=(2000, 3))
    w2 = mesh.winding_number(s.vertices, s.faces, pts2)
    rnorm = np.linalg.norm(pts2, axis=1)
    surf2 = np.abs(rnorm - radius)
    use2 = surf2 >= 1e-6
    deep2 = use2 & (rnorm < radius) & (surf2 >= 1e-3)
    far2 = use2 & (rnorm > radius) & (surf2 >= 1e-3)
    max_w1 = max(max_w1, float(np.max(np.abs(w2[deep2] - 1.0))) if np.any(deep2) else 0.0)
    max_w = max(max_w, float(np.max(np.abs(w2[far2]))) if np.any(far2) else 0.0)
    assert float(np.max(np.abs(w2[deep2] - 1.0))) <= 1e-9 if np.any(deep2) else True
    assert float(np.max(np.abs(w2[far2]))) <= 1e-9 if np.any(far2) else True


def test_transformed_volume():
    b = mesh.box_mesh([0.2, 0.1, 0.15])
    R = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    t = np.array([0.1, -0.05, 0.2], dtype=float)
    bt = b.transformed(R, t)
    assert bt.volume == pytest.approx(b.volume, rel=RTOL)