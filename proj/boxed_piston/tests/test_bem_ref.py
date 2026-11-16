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
    p1_interpolate_at,
    pulsating_sphere_H_transfer,
    solve_neumann_bm,
    zrad_sphere_analytic,
    _grid_elements,
    _locate_points_on_surface_mesh,
    _triangle_barycentric_weights,
)


def _triangle_plane_distance(
    point: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> float:
    cross = np.cross(v1 - v0, v2 - v0)
    n_hat = cross / (np.linalg.norm(cross) + 1e-300)
    return float(abs(np.dot(n_hat, point - v0)))


def _assert_located_plane_distances(
    points: np.ndarray, grid: object, verts: np.ndarray, elems: np.ndarray
) -> None:
    elem_idx, _ = _locate_points_on_surface_mesh(points, grid)
    for i, e in enumerate(elem_idx):
        tri = verts[elems[e]]
        d = _triangle_plane_distance(points[i], tri[0], tri[1], tri[2])
        assert d <= 1e-12


def _random_points_one_per_face(
    rng: np.random.Generator,
    ox: float,
    oy: float,
    oz: float,
    W: float,
    H: float,
    D: float,
    verts: np.ndarray,
    elems: np.ndarray,
) -> np.ndarray:
    faces = (
        (0, ox),
        (0, ox + W),
        (1, oy),
        (1, oy + H),
        (2, oz),
        (2, oz + D),
    )
    tri_verts = verts[elems]
    pts: list[np.ndarray] = []
    for ax, coord in faces:
        on_face = np.all(np.abs(tri_verts[:, :, ax] - coord) <= 1e-12, axis=1)
        candidates = np.flatnonzero(on_face)
        assert candidates.size > 0
        e = int(rng.choice(candidates))
        lam = rng.random(3)
        lam /= lam.sum()
        pts.append(lam @ tri_verts[e])
    return np.asarray(pts, dtype=np.float64)


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
    h_cap = h_target_from_frequency(1000.0, 343.0, epw=12, h_max=0.02)
    assert abs(h_cap - 0.02) < 1e-12
    h_epw = h_target_from_frequency(3000.0, 343.0, epw=12, h_max=0.02)
    assert abs(h_epw - 343.0 / 3000.0 / 12.0) < 1e-12
    h_def = h_target_from_frequency(3000.0, 343.0, h_max=0.02)
    assert abs(h_def - h_epw) < 1e-12
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


def test_box_mesh_radiator_axis_spacing():
    W, H, D = 0.20, 0.15, 0.25
    rw, rh = 0.08, 0.06
    origin = _box_origin(W, H, D, 0.0, 0.0)
    h_tgt = 0.02
    h_rad = min(rw, rh) / 8.0
    grid = box_mesh(W, H, D, (rw, rh), origin, h_tgt)
    verts = np.asarray(grid.vertices).T
    z_front = origin[2] + D
    on_front = verts[np.abs(verts[:, 2] - z_front) <= 1e-12]
    in_rad = on_front[
        (np.abs(on_front[:, 0]) <= 0.5 * rw + 1e-12)
        & (np.abs(on_front[:, 1]) <= 0.5 * rh + 1e-12)
    ]
    xs = np.unique(np.round(np.sort(in_rad[:, 0]), 12))
    ys = np.unique(np.round(np.sort(in_rad[:, 1]), 12))
    for coords in (xs, ys):
        for a, b in zip(coords[:-1], coords[1:]):
            assert b - a <= h_rad * (1.0 + 1e-9)


def test_grid_elements_vertex_order():
    W, H, D = 0.08, 0.06, 0.04
    rw, rh = 0.04, 0.03
    origin = _box_origin(W, H, D, 0.0, 0.0)
    grid = box_mesh(W, H, D, (rw, rh), origin, 0.02)
    elems = _grid_elements(grid)
    assert np.array_equal(elems, np.asarray(grid.elements, dtype=np.int32).T)
    space_p = bempp.function_space(grid, "P", 1)
    l2g = np.asarray(space_p.local2global, dtype=np.int32)
    mult = np.asarray(space_p.local_multipliers, dtype=np.float64)
    assert l2g.shape == elems.shape
    assert mult.shape == elems.shape


def test_triangle_barycentric_weights():
    v0 = np.array([[0.0, 0.0, 0.0]])
    v1 = np.array([[1.0, 0.0, 0.0]])
    v2 = np.array([[0.0, 1.0, 0.0]])
    centroid = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]])
    mid01 = np.array([[0.5, 0.0, 0.0]])

    for k, vk in enumerate((v0, v1, v2)):
        pts = vk.copy()
        lam = _triangle_barycentric_weights(pts, v0, v1, v2)[0, 0]
        expected = np.zeros(3)
        expected[k] = 1.0
        assert np.max(np.abs(lam - expected)) <= 1e-12
        assert abs(lam.sum() - 1.0) <= 1e-12

    lam_c = _triangle_barycentric_weights(centroid, v0, v1, v2)[0, 0]
    assert np.max(np.abs(lam_c - np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))) <= 1e-12

    lam_m = _triangle_barycentric_weights(mid01, v0, v1, v2)[0, 0]
    assert np.max(np.abs(lam_m - np.array([0.5, 0.5, 0.0]))) <= 1e-12


def test_p1_surface_pressure_interpolation():
    """P1 barycentric interpolation on box surface matches linear f = x + 2y + 3z."""
    W, H, D = 0.08, 0.06, 0.04
    rw, rh = 0.04, 0.03
    origin = _box_origin(W, H, D, 0.0, 0.0)
    grid = box_mesh(W, H, D, (rw, rh), origin, 0.02)
    space_p = bempp.function_space(grid, "P", 1)
    verts = np.asarray(grid.vertices).T
    elems = np.asarray(grid.elements).T
    l2g = np.asarray(space_p.local2global, dtype=np.int32)
    mult = np.asarray(space_p.local_multipliers, dtype=np.float64)

    def linear_f(xyz: np.ndarray) -> float:
        return float(xyz[0] + 2.0 * xyz[1] + 3.0 * xyz[2])

    coeffs = np.zeros(int(space_p.global_dof_count), dtype=np.float64)
    for e in range(elems.shape[0]):
        for i in range(3):
            m = float(mult[e, i])
            if m == 0.0:
                continue
            g = int(l2g[e, i])
            coeffs[g] = linear_f(verts[elems[e, i]]) / m

    rng = np.random.default_rng(0)
    pts = verts[rng.choice(verts.shape[0], size=5, replace=False)].copy()
    ox, oy, oz = origin
    z_front = oz + D
    pts = np.vstack(
        [
            pts,
            np.array(
                [
                    [ox, oy, oz],
                    [ox + W, oy + H, z_front],
                    [0.5 * (ox + ox + W), oy, z_front],
                ],
                dtype=np.float64,
            ),
        ]
    )
    while pts.shape[0] < 20:
        tri = verts[elems[int(rng.integers(0, elems.shape[0]))]]
        lam = rng.random(3)
        lam /= lam.sum()
        pts = np.vstack([pts, (lam @ tri)[np.newaxis, :]])

    expected = pts[:, 0] + 2.0 * pts[:, 1] + 3.0 * pts[:, 2]
    got = np.real(p1_interpolate_at(space_p, grid, coeffs, pts))
    assert np.max(np.abs(got - expected)) <= 1e-10
    _assert_located_plane_distances(pts, grid, verts, elems)

    face_pts = _random_points_one_per_face(rng, ox, oy, oz, W, H, D, verts, elems)
    face_expected = (
        face_pts[:, 0] + 2.0 * face_pts[:, 1] + 3.0 * face_pts[:, 2]
    )
    face_got = np.real(p1_interpolate_at(space_p, grid, coeffs, face_pts))
    assert np.max(np.abs(face_got - face_expected)) <= 1e-10
    _assert_located_plane_distances(face_pts, grid, verts, elems)


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


def _assert_H_close(H_bem: complex, H_ref: complex, *, mag_pct: float, phase_deg: float) -> None:
    mag_err = abs(abs(H_bem) - abs(H_ref)) / max(abs(H_ref), 1e-30)
    assert mag_err <= mag_pct, f"|H| err {100 * mag_err:.2f}% > {100 * mag_pct:.1f}%"
    ph = float(np.rad2deg(np.angle(H_bem) - np.angle(H_ref)))
    while ph > 180.0:
        ph -= 360.0
    while ph < -180.0:
        ph += 360.0
    assert abs(ph) <= phase_deg, f"phase err {ph:.2f}° > {phase_deg:.1f}°"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("ka", "r_mult"),
    [
        (0.2, 2.0),
        (0.2, 10.0),
        (1.0, 2.0),
        (1.0, 10.0),
        (math.pi, 2.0),
        (math.pi, 10.0),
        (4.0, 2.0),
        (4.0, 10.0),
    ],
)
def test_pulsating_sphere_field_at(ka: float, r_mult: float) -> None:
    """Exterior D[p]−S[q] agrees with analytic transfer on +z axis."""
    a = 0.05
    rho, c = 1.204, 343.0
    k = ka / a
    r_m = r_mult * a
    h_mesh = h_target_sphere(ka, a, epw=12)
    grid = bempp.shapes.sphere(r=a, h=h_mesh)
    sol = solve_neumann_bm(
        grid, k, vn_P0=1.0, rho=rho, c=c, pulsating_sphere=True, log_dof=False
    )
    pt = np.array([[0.0, 0.0, r_m]], dtype=np.float64)
    H_bem = complex(sol.transfer_at(pt, on_surface=False)[0])
    H_ref = pulsating_sphere_H_transfer(r_m, a, ka)
    _assert_H_close(H_bem, H_ref, mag_pct=0.01, phase_deg=1.0)


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