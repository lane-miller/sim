"""Tests for boxrad.viz.plot_geometry and plot_face_mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D

from boxrad.config import Config, derive
from boxrad.grid import Grid
from boxrad.viz import (
    _face_bounds,
    _face_weight_plane,
    _outline_centroid_xy,
    _radiator_outline_xy,
    _weight_centroid_xy,
    plot_face_mesh,
    plot_geometry,
)

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
_DEMO_CONFIGS = _CONFIGS / "demo"

_WEIGHT_EPS = 1e-6


def _plot_axes(fig):
    return [a for a in fig.axes if a.get_label() != "<colorbar>"]


def _assert_weight_centroid_matches_outline(cfg: Config) -> None:
    derived = derive(cfg)
    grid = Grid(cfg, derived)
    h = float(derived.h)
    i0, i1 = grid.i_box, grid.i_box + grid.nbx
    j0, j1 = grid.j_box, grid.j_box + grid.nby
    w = _face_weight_plane(cfg, grid, h)
    cx_w, cy_w = _weight_centroid_xy(w, grid.x_p[i0:i1], grid.y_p[j0:j1])
    cx_o, cy_o = _outline_centroid_xy(_radiator_outline_xy(cfg))
    assert abs(cx_w - cx_o) <= h
    assert abs(cy_w - cy_o) <= h
    x0, x1, y0, y1, _ = _face_bounds(grid)
    for loop in _radiator_outline_xy(cfg):
        assert np.all(loop[:, 0] >= x0 - 1e-12)
        assert np.all(loop[:, 0] <= x1 + 1e-12)
        assert np.all(loop[:, 1] >= y0 - 1e-12)
        assert np.all(loop[:, 1] <= y1 + 1e-12)


@pytest.mark.parametrize(
    "cfg_name",
    ("c09_box.json", "c06_baffle.json"),
)
def test_plot_geometry_two_panels(cfg_name: str) -> None:
    cfg = _CONFIGS / cfg_name
    fig = plot_geometry(cfg, show=False)
    axes = _plot_axes(fig)
    assert len(axes) == 2
    assert isinstance(axes[0], Axes3D)
    assert axes[0].get_xlabel() == "x [m]"
    assert axes[1].get_xlabel() == "x [m]"
    assert axes[1].get_ylabel() == "y [m]"
    assert "origin = radiator center" in (fig.get_suptitle() or "")


def test_plot_geometry_c09_box_writes_png(tmp_path: Path) -> None:
    cfg = _CONFIGS / "c09_box.json"
    out = tmp_path / "c09.png"
    plot_geometry(cfg, out=out, show=False)
    assert out.is_file()
    assert out.stat().st_size > 0


@pytest.mark.parametrize("cfg_name", ("c09_box.json", "c06_baffle.json"))
def test_weight_centroid_matches_outline(cfg_name: str) -> None:
    cfg = Config.from_json((_CONFIGS / cfg_name).read_text(encoding="utf-8"))
    _assert_weight_centroid_matches_outline(cfg)


def test_plot_face_mesh_c09_nonzero_cells(tmp_path: Path) -> None:
    cfg_path = _CONFIGS / "c09_box.json"
    cfg = Config.from_json(cfg_path.read_text(encoding="utf-8"))
    derived = derive(cfg)
    grid = Grid(cfg, derived)
    h = float(derived.h)
    w = _face_weight_plane(cfg, grid, h)
    n_w = int(np.count_nonzero(w > _WEIGHT_EPS))
    s_grid = float(np.sum(w) * h * h)
    fig = plot_face_mesh(cfg_path, show=False)
    assert fig is not None
    assert n_w > 0
    assert abs(float(np.sum(w) * h * h) - s_grid) < 1e-9
    axes = _plot_axes(fig)
    assert len(axes) == 1
    title = axes[0].get_title()
    assert f"{n_w} cells" in title
    assert "Σw" in title


@pytest.mark.parametrize(
    "cfg_name",
    (
        "demo_circle_box.json",
        "demo_racetrack_box.json",
        "demo_coarse_circle.json",
    ),
)
def test_demo_configs_geometry_plot(cfg_name: str) -> None:
    path = _DEMO_CONFIGS / cfg_name
    text = path.read_text(encoding="utf-8")
    cfg = Config.from_json(text)
    derive(cfg)
    fig = plot_geometry(path, show=False)
    assert len(_plot_axes(fig)) == 2
    _assert_weight_centroid_matches_outline(cfg)