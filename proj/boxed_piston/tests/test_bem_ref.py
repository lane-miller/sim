"""Unit tests for fdtd.post.bem_ref (BEM box mesh and pulsating-sphere check)."""

from __future__ import annotations

import math

import bempp_cl.api as bempp
import numpy as np
import pytest

from fdtd.post.bem_ref import (
    BOX_TAG,
    RADIATOR_TAG,
    BoxMeshCache,
    box_mesh,
    dedupe_frequencies_hz,
    h_target_from_frequency,
    h_target_sphere,
    solve_neumann_bm,
    zrad_sphere_analytic,
)


def _box_origin(W: float, H: float, D: float, x_c: float, y_c: float) -> tuple[float, float, float]:
    return (-x_c - 0.5 * W, -y_c - 0.5 * H, -D)


def _edge_key(v0: np.ndarray, v1: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    a = tuple(np.round(v0, 12))
    b = tuple(np.round(v1, 12))
    return (a, b) if a < b else (b, a)


def _dominant_face_label(normal: np.ndarray) -> str:
    n = np.asarray(normal, dtype=float).ravel()
    axis = int(np.argmax(np.abs(n)))
    sign = "+" if n[axis] >= 0.0 else "-"
    return f"{sign}{'xyz'[axis]}"


def test_box_mesh_topology_and_normals():
    W, H, D = 0.20, 0.15, 0.25
    rw, rh = 0.08, 0.06
    origin = _box_origin(W, H, D, 0.03, 0.02)
    h_tgt = 0.02
    grid = box_mesh(W, H, D, (rw, rh), origin, h_tgt)

    verts = np.asarray(grid.vertices).T
    elems = np.asarray(grid.elements).T
    n_tri = elems.shape[0]
    assert n_tri > 0

    edge_count: dict[tuple, int] = {}
    for tri in elems:
        for e in range(3):
            v0, v1 = verts[tri[e]], verts[tri[(e + 1) % 3]]
            key = _edge_key(v0, v1)
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary = [k for k, c in edge_count.items() if c == 1]
    assert boundary == [], "surface mesh should be closed (no boundary edges)"
    interior = [c for c in edge_count.values() if c != 2]
    assert not interior, "non-manifold edges present"

    centers = np.asarray(grid.centroids)
    normals = np.asarray(grid.normals)
    ox, oy, oz = origin
    box_center = np.array([ox + 0.5 * W, oy + 0.5 * H, oz + 0.5 * D])
    outward = np.sum((centers - box_center) * normals, axis=1)
    inward_by_face: dict[str, int] = {}
    for i, dot in enumerate(outward):
        if dot > -1e-10:
            continue
        face = _dominant_face_label(normals[i])
        inward_by_face[face] = inward_by_face.get(face, 0) + 1
    if inward_by_face:
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(inward_by_face.items()))
        pytest.fail(f"inward triangle count by face ({detail})")


def test_h_target_and_dedupe_frequencies():
    h_cap = h_target_from_frequency(1000.0, 343.0, epw=10, h_max=0.02)
    assert abs(h_cap - 0.02) < 1e-12
    h_epw = h_target_from_frequency(3000.0, 343.0, epw=10, h_max=0.02)
    assert abs(h_epw - 343.0 / 3000.0 / 10.0) < 1e-12
    f = dedupe_frequencies_hz([100.0, 100.4, 101.5, 105.0], sep_hz=1.0)
    assert f.tolist() == [100.0, 101.5, 105.0]


def test_box_mesh_cache_reuses_h():
    W, H, D = 0.20, 0.15, 0.25
    origin = _box_origin(W, H, D, 0.0, 0.0)
    cache = BoxMeshCache()
    g1 = cache.get(W, H, D, (0.08, 0.06), origin, 0.02)
    g2 = cache.get(W, H, D, (0.08, 0.06), origin, 0.02)
    assert g1 is g2
    g3 = cache.get(W, H, D, (0.08, 0.06), origin, 0.015)
    assert g3 is not g1


def test_box_mesh_areas():
    W, H, D = 0.20, 0.15, 0.25
    rw, rh = 0.08, 0.06
    origin = _box_origin(W, H, D, 0.0, 0.0)
    grid = box_mesh(W, H, D, (rw, rh), origin, 0.01)
    areas = grid.volumes.astype(float)
    total = float(np.sum(areas))
    a_box = 2.0 * (W * H + W * D + H * D)
    assert abs(total - a_box) <= 1e-10

    tags = grid.domain_indices.astype(int)
    a_rad_mesh = float(np.sum(areas[tags == RADIATOR_TAG]))
    assert abs(a_rad_mesh - rw * rh) <= 1e-10
    assert np.any(tags == RADIATOR_TAG)
    assert np.any(tags == BOX_TAG)


@pytest.mark.slow
@pytest.mark.parametrize("ka", [0.2, 1.0, math.pi, 4.0])
def test_pulsating_sphere_zrad(ka: float):
    """Closed-form z = jka/(1+jka); π case exercises irregular-frequency neighborhood."""
    a = 0.05
    rho, c = 1.204, 343.0
    k = ka / a
    h_mesh = h_target_sphere(ka, a, epw=10)
    grid = bempp.shapes.sphere(r=a, h=h_mesh)
    sol = solve_neumann_bm(grid, k, vn_P0=1.0, rho=rho, c=c, pulsating_sphere=True)
    z_bem = sol.zrad()
    z_ref = zrad_sphere_analytic(ka)
    assert abs(z_bem - z_ref) <= 0.01