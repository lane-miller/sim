"""Cut-cell fraction engine (v0.3 step 4)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config, derive
from boxrad.cutcell import (
    Grid,
    _STATE_CUT,
    _STATE_FLUID,
    _STATE_SOLID,
    compute_fractions,
    grid_for_geometry,
)
from boxrad.geom import geometry, mesh, poly2d
from boxrad.grid import _radiator_face_weights, radiator_area_exact

PKG = Path(__file__).resolve().parents[1]
CONFIGS = PKG / "configs"
SHOWCASE = CONFIGS / "showcase"
H = 0.005
RTOL = 1e-12


def test_unit_cube_hand():
    """Unit cube [0,1]^3 on h=1 grid; analytic patches, sweep, and beta (step-04 debug)."""
    solid = mesh.box_mesh([1.0, 1.0, 1.0]).transformed(np.eye(3), np.array([0.5, 0.5, 0.5]))
    geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
    grid = Grid(x0=np.array([-2.0, -2.0, -2.0]), h=1.0, shape=(5, 5, 5), rigid_sides={})
    fr = compute_fractions(geom, grid)
    ax, ay, az = fr.alpha
    # x-planes at indices 2 (x=0) and 3 (x=1) for y,z in cells 2,3 covering [0,1]
    assert ax[2, 2, 2] == pytest.approx(0.0, abs=RTOL)
    assert ax[3, 2, 2] == pytest.approx(0.0, abs=RTOL)
    assert ax[4, 2, 2] == pytest.approx(1.0, abs=RTOL)
    assert fr.beta[2, 2, 2] == pytest.approx(0.0, abs=RTOL)
    assert fr.beta[3, 2, 2] == pytest.approx(1.0, abs=RTOL)
    assert fr.stats["fluid_volume"] == pytest.approx(1.0, rel=RTOL)
    assert fr.stats["n_cut_cells"] == 0
    for arr in (ax, ay, az):
        u = np.unique(np.round(arr.ravel(), 12))
        assert set(u.tolist()).issubset({0.0, 1.0})


def _c09_geom():
    cfg = Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8"))
    return cfg, geometry.from_box(cfg)


def _natural_c09_grid(cfg: Config, geom) -> Grid:
    d = derive(cfg)
    h = d.h
    margin = 2
    lo = np.min(geom.solid.vertices, axis=0)
    hi = np.max(geom.solid.vertices, axis=0)
    x0 = lo - margin * h
    shape = tuple(
        int(math.ceil((hi[ax] - lo[ax]) / h)) + 2 * margin for ax in range(3)
    )
    return Grid(x0=x0, h=h, shape=shape, rigid_sides={})


def test_aligned_box_c09_natural_grid():
    cfg, geom = _c09_geom()
    grid = _natural_c09_grid(cfg, geom)
    fr = compute_fractions(geom, grid)
    d = derive(cfg)
    vol = d.realized_W * d.realized_H * d.realized_D
    assert fr.stats["n_cut_cells"] == 0
    assert fr.stats["volume_rel_error"] <= RTOL
    assert fr.stats["fluid_volume"] == pytest.approx(vol, rel=RTOL)
    for ax in fr.alpha:
        u = np.unique(np.round(ax.ravel(), 12))
        assert set(u.tolist()).issubset({0.0, 1.0})
    assert set(np.unique(fr.beta).tolist()).issubset({0.0, 1.0})


def test_aligned_box_on_grid_planes():
    cfg, _ = _c09_geom()
    geom = geometry.from_box(cfg)
    grid = _natural_c09_grid(cfg, geom)
    fr = compute_fractions(geom, grid)
    assert fr.stats["n_cut_cells"] == 0


def test_shifted_box():
    """Shifted 4³ box on h=1 grid (x0=−2, 9³); exact face and β checks."""
    from ref_cutcell import box_mesh as ref_box_mesh  # test-only reference

    lo, hi = 0.37, 4.37
    v, f = ref_box_mesh([lo, lo, lo], [hi, hi, hi])
    solid = mesh.TriMesh(v, f)
    geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
    grid = Grid(x0=np.array([-2.0, -2.0, -2.0]), h=1.0, shape=(9, 9, 9), rigid_sides={})
    fr = compute_fractions(geom, grid)
    vol = (hi - lo) ** 3
    assert fr.stats["fluid_volume"] == pytest.approx(vol, rel=RTOL)
    ax = fr.alpha[0]
    assert ax[4, 2, 4] == pytest.approx(0.37, abs=RTOL)
    assert ax[3, 2, 2] == pytest.approx(1.0 - 0.63**2, abs=RTOL)
    assert fr.beta[2, 2, 2] == pytest.approx(1.0 - 0.63**3, abs=RTOL)


def test_matches_reference():
    """Small grids: numba fractions match ref_cutcell (test-only oracle)."""
    from ref_cutcell import (  # test-only reference
        box_mesh as ref_box_mesh,
        cylinder_mesh,
        fractions as ref_fractions,
        mesh_volume,
    )

    def _check(V, F, x0, h, shape):
        ref = ref_fractions(V, F, np.asarray(x0, float), h, shape)
        solid = mesh.TriMesh(np.asarray(V, float), np.asarray(F, int))
        geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
        grid = Grid(x0=np.asarray(x0, float), h=h, shape=tuple(shape), rigid_sides={})
        fr = compute_fractions(geom, grid)
        for a in range(3):
            d = np.max(np.abs(fr.alpha[a] - ref["alpha"][a]))
            assert d <= RTOL, f"alpha axis {a} max err {d}"
        db = np.max(np.abs(fr.beta - ref["beta"]))
        assert db <= RTOL, f"beta max err {db}"
        mesh_area = 0.0
        P = V[F]
        for tri in P:
            mesh_area += 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        patch_area = float(np.sum(fr.patch_A))
        assert patch_area == pytest.approx(mesh_area, rel=RTOL)

    V, F = ref_box_mesh([0.37, 0.37, 0.37], [4.37, 4.37, 4.37])
    _check(V, F, [-2.0, -2.0, -2.0], 1.0, (9, 9, 9))

    V, F = cylinder_mesh(1.0, 0.0, 1.0, 32)
    _check(V, F, [-2.0, -2.0, -2.0], 0.5, (12, 12, 8))

    cfg = Config.from_json((CONFIGS / "c09_box.json").read_text())
    geom0 = geometry.from_box(cfg)
    cz, sz = math.cos(math.radians(37)), math.sin(math.radians(37))
    cx, sx = math.cos(math.radians(20)), math.sin(math.radians(20))
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    solid = geom0.solid.transformed(Rz @ Rx, np.zeros(3))
    grid = grid_for_geometry(
        geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={}),
        0.05,
        margin_cells=2,
    )
    assert grid.n_cells <= 12**3
    _check(solid.vertices, solid.faces, grid.x0, grid.h, grid.shape)

    sph = mesh.sphere_mesh(0.25, 2.5 * H)
    _check(sph.vertices, sph.faces, [-0.6, -0.6, -0.6], 0.1, (12, 12, 12))
    assert abs(mesh_volume(sph.vertices, sph.faces) - sph.volume) / sph.volume < RTOL


@pytest.mark.parametrize("kind", ["rotated_cube", "sphere"])
def test_rotated_or_sphere(kind: str):
    if kind == "rotated_cube":
        cfg = Config.from_json((CONFIGS / "c09_box.json").read_text())
        geom = geometry.from_box(cfg)
        cz, sz = math.cos(math.radians(37)), math.sin(math.radians(37))
        cx, sx = math.cos(math.radians(20)), math.sin(math.radians(20))
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        R = Rz @ Rx
        solid = geom.solid.transformed(R, np.zeros(3))
        geom = geometry.Geometry(
            solid=solid, radiators=geom.radiators, planes=geom.planes, provenance={}
        )
        mesh_vol = abs(solid.volume)
    else:
        sph = mesh.sphere_mesh(0.1, 1e-4)
        geom = geometry.Geometry(solid=sph, radiators=[], planes=(), provenance={})
        mesh_vol = sph.volume
    grid = grid_for_geometry(geom, H, margin_cells=4)
    fr = compute_fractions(geom, grid)
    assert fr.stats["volume_rel_error"] <= RTOL
    assert fr.stats["max_line_closure_residual"] <= 1e-9
    cut_idx = np.argwhere(fr.state == _STATE_CUT)
    if cut_idx.shape[0] > 0:
        rng = np.random.default_rng(0)
        take = min(60, cut_idx.shape[0])
        pick = rng.choice(cut_idx.shape[0], size=take, replace=False)
        sub = 8
        offs = (np.arange(sub) + 0.5) / sub
        errors = []
        h = grid.h
        for r in pick:
            i, j, k = int(cut_idx[r, 0]), int(cut_idx[r, 1]), int(cut_idx[r, 2])
            c = grid.cell_center(i, j, k)
            pts = []
            for ox in offs:
                for oy in offs:
                    for oz in offs:
                        pts.append(c + (np.array([ox, oy, oz]) - 0.5) * h)
            pts = np.asarray(pts)
            inside = ~geom.inside(pts)
            beta_oracle = float(np.mean(inside.astype(np.float64)))
            errors.append(abs(fr.beta[i, j, k] - beta_oracle))
        # 8³ sampling error ~1/8 per edge cell.
        assert max(errors) <= 0.08
        assert float(np.mean(errors)) <= 0.02


def test_radiator_c9_weights():
    cfg, geom = _c09_geom()
    grid = _natural_c09_grid(cfg, geom)
    fr = compute_fractions(geom, grid)
    d = derive(cfg)
    g_yee = __import__("boxrad.grid", fromlist=["Grid"]).Grid(cfg, d)
    ib, jb, kf = g_yee.i_box, g_yee.j_box, g_yee.k_face_z
    x = g_yee.x_p[ib : ib + d.box_nx]
    y = g_yee.y_p[jb : jb + d.box_ny]
    w_ref = _radiator_face_weights(cfg.radiator, x, y, grid.h, 32)
    margin = 2
    by_ij: dict[tuple[int, int], float] = {}
    for nid, a in zip(fr.radiator_cell[0], fr.radiator_a[0]):
        i = nid // (grid.ny * grid.nz)
        r = nid % (grid.ny * grid.nz)
        j = r // grid.nz
        by_ij[(i - margin, j - margin)] = a
    # 1e-5: v0.2 Yee grid stores cell centers as float32; radiator weights use those coords.
    for wi in range(w_ref.shape[0]):
        for wj in range(w_ref.shape[1]):
            w = w_ref[wi, wj]
            if w <= 1e-5:
                continue
            assert (wi, wj) in by_ij, f"missing radiator cell for w_ref[{wi},{wj}]={w}"
            assert by_ij[(wi, wj)] / grid.h ** 2 == pytest.approx(w, abs=1e-5)
    assert sum(fr.radiator_a[0]) == pytest.approx(grid.h ** 2 * w_ref.sum(), rel=1e-5)


def test_showcase_radiator_areas():
    cases = {
        "S1": SHOWCASE / "s1_cylinder.json",
        "S4": SHOWCASE / "s4_tilted.json",
    }
    for name, path in cases.items():
        cfg = Config.from_json(path.read_text())
        geom = geometry.build_geometry(cfg)
        assert geom is not None
        grid = grid_for_geometry(geom, H)
        fr = compute_fractions(geom, grid)
        exact = sum(rd.area for rd in geom.radiators)
        assert sum(fr.radiator_S) == pytest.approx(exact, rel=RTOL)


def test_s2_grid_extents():
    cfg = Config.from_json((SHOWCASE / "s2_bookshelf.json").read_text())
    geom = geometry.build_geometry(cfg)
    grid = grid_for_geometry(geom, H)
    assert grid.x0[2] == pytest.approx(0.0, abs=1e-9)
    assert grid.x0[1] == pytest.approx(-0.425, abs=1e-9)
    fr = compute_fractions(geom, grid)
    assert fr.stats["volume_rel_error"] <= 1e-10


def test_s6_layers():
    cfg = Config.from_json((SHOWCASE / "s6_downfire.json").read_text())
    geom = geometry.build_geometry(cfg)
    grid = grid_for_geometry(geom, H)
    fr = compute_fractions(geom, grid)
    assert fr.radiator_S[0] == pytest.approx(geom.radiators[0].area, rel=RTOL)
    gap_cells = round(0.03 / grid.h)
    fluid_along_z = np.sum(fr.beta > 0.5, axis=(0, 1))
    assert int(np.max(fluid_along_z)) >= gap_cells - 1


def test_non_watertight_raises():
    cfg, geom = _c09_geom()
    faces = geom.solid.faces[1:].copy()
    solid = mesh.TriMesh(geom.solid.vertices, faces)
    bad = geometry.Geometry(solid=solid, radiators=geom.radiators, planes=geom.planes, provenance={})
    grid = grid_for_geometry(geom, H, margin_cells=2)
    with pytest.raises(ValueError, match="non-watertight"):
        compute_fractions(bad, grid)


def test_flood_fill_pocket():
    outer = mesh.box_mesh([0.2, 0.2, 0.2]).transformed(np.eye(3), np.zeros(3))
    inner = mesh.box_mesh([0.08, 0.08, 0.08]).transformed(np.eye(3), np.zeros(3))
    inner_r = mesh.TriMesh(inner.vertices, inner.faces[:, ::-1].copy())
    both = mesh.concat_meshes([outer, inner_r])
    solid = mesh.weld(both.vertices, both.faces)
    geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
    grid = grid_for_geometry(geom, H, margin_cells=3)
    fr = compute_fractions(geom, grid)
    assert fr.stats["voids_removed_count"] > 0
    assert fr.stats["voids_removed_volume"] == pytest.approx(0.08**3, rel=1e-12)


def test_box_path_no_manifold_in_compute_fractions():
    code = (
        "import sys; from pathlib import Path; "
        "from boxrad.config import Config; "
        "from boxrad.geom.geometry import from_box; "
        "from boxrad.cutcell import compute_fractions, grid_for_geometry; "
        f"p=Path({repr(str(CONFIGS / 'c09_box.json'))}); "
        "cfg=Config.from_json(p.read_text()); "
        "g=from_box(cfg); "
        "gr=grid_for_geometry(g, 0.005); "
        "compute_fractions(g, gr); "
        "assert 'manifold3d' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(PKG))


@pytest.mark.slow
def test_s5_performance():
    cfg = Config.from_json((SHOWCASE / "s5_platform.json").read_text())
    geom = geometry.build_geometry(cfg)
    grid = grid_for_geometry(geom, H)
    fr = compute_fractions(geom, grid)
    assert fr.stats["wall_time_s"] < 20.0