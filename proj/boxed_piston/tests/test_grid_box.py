"""Unit tests for boxed-piston grid geometry, rigid masks, and radiator weights."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import (  # noqa: E402
    Band,
    Box,
    Config,
    Numerics,
    Output,
    Radiator,
    derive,
    validate,
)
from boxrad.grid import (  # noqa: E402
    Grid,
    _radiator_face_weights,
    validate_box_radiator_on_grid,
)


def _box_cfg(
    *,
    box: Box = Box(W=0.2, H=0.2, D=0.15),
    radiator: Radiator | None = None,
    f_max: float = 8_000.0,
    cpml_cells: int = 6,
    gap_cells: int = 5,
    ppw: float = 12.0,
    cells_across: int = 10,
    min_edge_cells: int = 3,
) -> tuple[Config, object]:
    if radiator is None:
        radiator = Radiator("circle", dims=[0.05], x_c=0.0, y_c=0.0)
    cfg = Config(
        box=box,
        radiator=radiator,
        band=Band(f_min=20.0, f_max=f_max),
        numerics=Numerics(
            cpml_cells=cpml_cells,
            gap_cells=gap_cells,
            ppw=ppw,
            cells_across_radiator=cells_across,
            min_edge_cells=min_edge_cells,
        ),
        output=Output(path="/tmp/grid_box_test.h5"),
    )
    return cfg, derive(cfg)


def _grid(cfg: Config, d: object) -> Grid:
    return Grid(cfg, d, radiator_supersample=32)


def test_realized_box_dimensions_and_cell_count():
    box = Box(W=0.203, H=0.197, D=0.151)
    cfg, d = _box_cfg(box=box)
    g = _grid(cfg, d)
    h = d.h
    assert d.realized_W == pytest.approx(round(box.W / h) * h)
    assert d.realized_H == pytest.approx(round(box.H / h) * h)
    assert d.realized_D == pytest.approx(round(box.D / h) * h)
    assert g.realized_W == d.realized_W
    n_inside = int(np.count_nonzero(g.box_cells))
    assert n_inside == d.box_nx * d.box_ny * d.box_nz


def _iter_box_surface_faces(g: Grid) -> list[tuple[str, int, int, int]]:
    ib, jb, kb = g.i_box, g.j_box, g.k_box
    nbx, nby, nbz = g.nbx, g.nby, g.nbz
    out: list[tuple[str, int, int, int]] = []
    for j in range(jb, jb + nby):
        for k in range(kb, kb + nbz):
            out.append(("vx", ib, j, k))
            out.append(("vx", ib + nbx, j, k))
    for i in range(ib, ib + nbx):
        for k in range(kb, kb + nbz):
            out.append(("vy", i, jb, k))
            out.append(("vy", i, jb + nby, k))
    for i in range(ib, ib + nbx):
        for j in range(jb, jb + nby):
            out.append(("vz", i, j, kb))
            out.append(("vz", i, j, kb + nbz))
    return out


def _radiator_vz_face_set(g: Grid) -> set[tuple[int, int, int]]:
    ir, jr, kr = g.radiator_face_indices
    return {tuple(t) for t in zip(ir.tolist(), jr.tolist(), kr.tolist())}


def test_rigid_masks_surface_only_and_radiator_excluded():
    cfg, d = _box_cfg()
    g = _grid(cfg, d)
    radiator_vz = _radiator_vz_face_set(g)
    for comp, i, j, k in _iter_box_surface_faces(g):
        if comp == "vz" and (i, j, k) in radiator_vz:
            continue  # driven radiator patches, not rigid
        mask = getattr(g, f"rigid_{comp}")
        assert mask[i, j, k], f"{comp}[{i},{j},{k}] should be rigid"

    kf = g.k_face_z
    ir, jr, kr = g.radiator_face_indices
    for i, j, k in zip(ir, jr, kr):
        assert k == kf
        assert not g.rigid_vz[i, j, k]

    # Interior air: no rigid faces inside the box volume.
    ib, jb, kb = g.i_box, g.j_box, g.k_box
    if g.nbx > 1:
        assert not g.rigid_vx[ib + 1, jb, kb]
    if g.nbz > 1:
        assert not g.rigid_vz[ib, jb, kb + 1]

    # Nothing rigid outside the box surface shell.
    for name, arr in (("vx", g.rigid_vx), ("vy", g.rigid_vy), ("vz", g.rigid_vz)):
        surf = set()
        for comp, i, j, k in _iter_box_surface_faces(g):
            if comp == name:
                surf.add((i, j, k))
        ii, jj, kk = np.where(arr)
        for i, j, k in zip(ii, jj, kk):
            assert (i, j, k) in surf, f"rigid {name} outside box surface at ({i},{j},{k})"


@pytest.mark.parametrize("shape,dims", [
    ("circle", [0.05]),
    ("rect", [0.06, 0.04]),
    ("racetrack", [0.07, 0.04]),
])
def test_radiator_area_shapes(shape: str, dims: list[float]):
    cfg, d = _box_cfg(radiator=Radiator(shape, dims=dims, x_c=0.0, y_c=0.0))
    g = _grid(cfg, d)
    rel = abs(g.radiator_area_grid - g.radiator_area_exact) / g.radiator_area_exact
    assert rel <= 1e-3


@pytest.mark.parametrize("shape,dims", [
    ("circle", [0.05]),
    ("rect", [0.06, 0.04]),
    ("racetrack", [0.07, 0.04]),
])
def test_radiator_area_off_grid_center(shape: str, dims: list[float]):
    _, d0 = _box_cfg()
    h = d0.h
    cfg, d = _box_cfg(
        radiator=Radiator(shape, dims=dims, x_c=0.17 * h, y_c=-0.23 * h),
    )
    g = _grid(cfg, d)
    rel = abs(g.radiator_area_grid - g.radiator_area_exact) / g.radiator_area_exact
    assert rel <= 1e-3


def test_radiator_area_small_circle_three_cells():
    box = Box(W=0.2, H=0.2, D=0.15)
    diam = 0.03
    cfg, d = _box_cfg(
        box=box,
        radiator=Radiator("circle", dims=[diam]),
        cells_across=3,
        ppw=4.0,
        f_max=8_000.0,
    )
    assert d.h == pytest.approx(diam / 3.0, rel=1e-9)
    g = _grid(cfg, d)
    rel = abs(g.radiator_area_grid - g.radiator_area_exact) / g.radiator_area_exact
    assert rel <= 1e-3


def test_radiator_weight_symmetry_centered_circle():
    cfg, d = _box_cfg(radiator=Radiator("circle", dims=[0.05]))
    g = _grid(cfg, d)
    ib, jb = g.i_box, g.j_box
    x = g.x_p[ib : ib + g.nbx]
    y = g.y_p[jb : jb + g.nby]
    w = _radiator_face_weights(cfg.radiator, x, y, g.h, 32)
    for i in range(g.nbx):
        for j in range(g.nby):
            i2 = g.nbx - 1 - i
            j2 = g.nby - 1 - j
            assert w[i, j] == pytest.approx(w[i2, j], abs=1e-12)
            assert w[i, j] == pytest.approx(w[i, j2], abs=1e-12)


def test_origin_at_radiator_center_and_face_plane():
    cfg, d = _box_cfg(radiator=Radiator("circle", dims=[0.05], x_c=0.011, y_c=-0.009))
    g = _grid(cfg, d)
    assert float(g.z_vz[g.k_face_z]) == pytest.approx(0.0, abs=1e-9)

    ib, jb = g.i_box, g.j_box
    x = g.x_p[ib : ib + g.nbx]
    y = g.y_p[jb : jb + g.nby]
    w = _radiator_face_weights(cfg.radiator, x, y, g.h, 32)
    wt = float(np.sum(w))
    cx = float(np.sum(w * x[:, np.newaxis])) / wt
    cy = float(np.sum(w * y[np.newaxis, :])) / wt
    # Supersampling at 32×32 leaves centroid noise ~1e-3·h; use 5e-3·h margin.
    tol = 5e-3 * g.h
    assert cx == pytest.approx(0.0, abs=tol)
    assert cy == pytest.approx(0.0, abs=tol)

    x_fc = -cfg.radiator.x_c
    y_fc = -cfg.radiator.y_c
    assert cx - x_fc == pytest.approx(cfg.radiator.x_c, abs=tol)
    assert cy - y_fc == pytest.approx(cfg.radiator.y_c, abs=tol)


def test_near_edge_radiator_validation_error():
    box = Box(W=0.2, H=0.2, D=0.15)
    cfg, d = _box_cfg(
        box=box,
        radiator=Radiator("rect", dims=[0.18, 0.04], x_c=0.0, y_c=0.0),
        min_edge_cells=3,
    )
    errs, _ = validate(cfg, d)
    assert any("min_edge_cells" in e or "does not fit" in e for e in errs)
    gerrs = validate_box_radiator_on_grid(cfg, d)
    assert gerrs


def test_free_field_mode_unchanged():
    cfg = Config(
        box=None,
        radiator=Radiator("circle", dims=[0.05]),
        band=Band(f_min=20.0, f_max=5_000.0),
        numerics=Numerics(cpml_cells=4, gap_cells=2, ppw=10.0),
        output=Output(path="/tmp/grid_ff.h5"),
    )
    d = derive(cfg)
    g = Grid(cfg, d)
    assert g.box_cells is None
    assert g.rigid_vx is None
    h = g.h
    assert float(g.x_p[d.nx // 2]) == pytest.approx(0.0, abs=h)
    assert float(g.x_vx[0]) == pytest.approx(-0.5 * d.nx * h, abs=1e-6)