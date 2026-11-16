"""Unit tests for boxrad.grid (Stage A free-field)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import (  # noqa: E402
    Band,
    Config,
    Numerics,
    Output,
    Radiator,
    derive,
    estimate_memory_bytes,
)
from boxrad.grid import (  # noqa: E402
    Grid,
    allocate_fields,
    memory_bytes,
)

_FLOAT32 = np.dtype(np.float32)


def _small_free_field(cpml_cells: int = 4, gap_cells: int = 2) -> tuple[Config, object]:
    cfg = Config(
        box=None,
        radiator=Radiator("circle", dims=[0.05]),
        band=Band(f_min=20.0, f_max=5_000.0),
        numerics=Numerics(cpml_cells=cpml_cells, gap_cells=gap_cells, ppw=10.0),
        output=Output(path="/tmp/grid_test.h5"),
    )
    return cfg, derive(cfg)


def test_staggered_field_shapes():
    _, d = _small_free_field()
    g = Grid(Config(box=None, radiator=Radiator("circle", dims=[0.05]), output=Output()), d)
    f = allocate_fields(g)
    assert f.p.shape == (d.nx, d.ny, d.nz)
    assert f.vx.shape == (d.nx + 1, d.ny, d.nz)
    assert f.vy.shape == (d.nx, d.ny + 1, d.nz)
    assert f.vz.shape == (d.nx, d.ny, d.nz + 1)


def test_cpml_slab_thickness_and_interior():
    cpml = 4
    _, d = _small_free_field(cpml_cells=cpml)
    g = Grid(
        Config(box=None, radiator=Radiator("circle", dims=[0.05]), output=Output()),
        d,
    )
    s = g.cpml_slabs
    assert s.p_x_neg.thickness() == cpml
    assert s.p_x_pos.thickness() == cpml
    assert s.i_inner.lo == s.p_x_neg.hi == cpml
    assert s.i_inner.hi == d.nx - cpml
    assert s.j_inner.lo == cpml and s.j_inner.hi == d.ny - cpml
    assert s.k_inner.lo == cpml and s.k_inner.hi == d.nz - cpml
    assert s.i_inner.lo >= s.p_x_neg.hi
    assert s.i_inner.hi <= s.p_x_pos.lo
    assert s.vx_x_neg.thickness() == cpml + 1
    assert s.vx_x_pos.hi == d.nx + 1


def test_index_of_coords_of_roundtrip():
    cfg, d = _small_free_field()
    g = Grid(cfg, d)
    for i, j, k in ((0, 0, 0), (d.nx // 2, d.ny // 2, d.nz // 2), (d.nx - 1, d.ny - 1, d.nz - 1)):
        x, y, z = g.coords_of(i, j, k)
        ii, jj, kk = g.index_of(x, y, z)
        assert (ii, jj, kk) == (i, j, k)


def test_memory_bytes_vs_derived_estimate():
    cfg, d = _small_free_field()
    g = Grid(cfg, d)
    f = allocate_fields(g)
    actual = memory_bytes(f)
    est = int(d.memory_bytes)
    assert est == estimate_memory_bytes(d.nx, d.ny, d.nz, d.cpml_cells)
    assert actual == est
    assert actual == pytest.approx(est, rel=0.1)


def test_float32_c_contiguous():
    cfg, d = _small_free_field()
    g = Grid(cfg, d)
    f = allocate_fields(g)
    for name in f.__dataclass_fields__:
        arr = getattr(f, name)
        assert arr.dtype == _FLOAT32, name
        assert arr.flags["C_CONTIGUOUS"], name


def test_coordinate_vectors_length_and_origin():
    _, d = _small_free_field()
    g = Grid(
        Config(box=None, radiator=Radiator("circle", dims=[0.05]), output=Output()),
        d,
    )
    assert len(g.x_p) == d.nx
    assert len(g.x_vx) == d.nx + 1
    assert float(g.x_p[d.nx // 2]) == pytest.approx(0.0, abs=g.h)
    h = g.h
    nx = d.nx
    assert float(g.x_vx[0]) == pytest.approx(-0.5 * nx * h, abs=1e-6)
    assert float(g.x_vx[-1]) == pytest.approx(0.5 * nx * h, abs=1e-6)
    np.testing.assert_allclose(g.x_p, g.x_vx[:-1] + 0.5 * h, atol=1e-6)
