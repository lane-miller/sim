"""Step 12: geom_report and geometry plot (fast)."""

from __future__ import annotations

import copy
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from boxrad.config import Config, derive
from boxrad.geom.geometry import build_geometry, radiator_clearance
from boxrad.geom.mesh import box_mesh
from boxrad.geomrun import prepare_geom_run
from boxrad.plot.geometry import plot_geometry, prepare_geometry_plot, slice_image, triangle_plane_segments
from boxrad.report import format_geom_report, geom_report, geom_report_exit_code
from boxrad.runner import _dry_run_summary, _print_dry_run_summary, run

_PKG = Path(__file__).resolve().parents[1]
_SHOWCASE = _PKG / "configs" / "showcase"


def _load(path: Path) -> Config:
    return Config.from_json(path.read_text(encoding="utf-8"))


def test_geom_report_c13_matches_fractions_and_cfl():
    raw = copy.deepcopy(
        json.loads((_PKG / "configs" / "c13_sphere_ps.json").read_text(encoding="utf-8"))
    )
    raw["numerics"].pop("cfl_safety", None)
    cfg = Config.from_dict(raw)
    run = prepare_geom_run(cfg)
    fr = run.info["fractions_stats"]
    floor = run.info["floor_stats"]
    rep = geom_report(cfg)
    assert rep["fractions"]["n_cut_cells"] == fr["n_cut_cells"]
    assert rep["fractions"]["n_floored"] == floor["n_floored"]
    assert rep["time"]["cfl_safety_source"] == "auto_cutcell"
    assert rep["fractions"]["delta_eq_over_h_status"] == "OK"


def test_geom_report_s6_clearance_warning():
    cfg = _load(_SHOWCASE / "s6_downfire.json")
    geometry = build_geometry(cfg)
    assert geometry is not None
    rep = geom_report(cfg, cfg_path=_SHOWCASE / "s6_downfire.json")
    h = rep["grid"]["h_m"]
    rd = geometry.radiators[0]
    cmin, _ = radiator_clearance(geometry, rd, n=64)
    warnings = rep["warnings"]
    if cmin / h < 5.0:
        h_need = cmin / 5.0
        assert any(f"h needed {h_need:g} m" in w for w in warnings)
    else:
        assert not any("clearance" in w and "h needed" in w for w in warnings)


def test_probe_in_cpml_listed_not_raised():
    cfg = _load(_PKG / "configs" / "c13_sphere_ps.json")
    rep0 = geom_report(cfg)
    ok_probe = next(
        (p for p in rep0["recording"]["probes"] if p["status"] == "OK"),
        None,
    )
    assert ok_probe is not None
    cfg_ok = copy.deepcopy(cfg)
    cfg_ok.output.probes = [list(ok_probe["xyz_m"])]
    rep_ok = geom_report(cfg_ok)
    assert rep_ok["recording"]["probes"][0]["status"] == "OK"
    assert geom_report_exit_code(rep_ok) == 0

    cfg_bad = copy.deepcopy(cfg)
    in_cpml = next(
        (p for p in rep0["recording"]["probes"] if p["status"] == "in CPML"),
        None,
    )
    if in_cpml is not None:
        cfg_bad.output.probes = [list(in_cpml["xyz_m"])]
    else:
        from boxrad.geom.geometry import build_geometry
        from boxrad.geomrun import _derive_geom_with_cfl_resolution

        geometry = build_geometry(cfg_bad)
        _, gg, _, _ = _derive_geom_with_cfl_resolution(cfg_bad, geometry)
        h = gg.h
        ctr = gg.x0 + np.array([0.5, 0.5, 0.5], dtype=np.float64) * h
        cfg_bad.output.probes = [[float(ctr[0]), float(ctr[1]), float(ctr[2])]]
    rep_bad = geom_report(cfg_bad)
    assert rep_bad["recording"]["probes"][0]["status"] == "in CPML"
    assert geom_report_exit_code(rep_bad) == 2


def test_box_dry_run_text_unchanged():
    cfg = _load(_PKG / "configs" / "c09_box.json")
    derived = derive(cfg)
    summary = _dry_run_summary(cfg, derived)
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_dry_run_summary(summary)
    text = buf.getvalue()
    assert text.startswith("dry run summary:\n")
    assert "  grid_shape:" in text
    assert "  h:" in text
    assert "  zrad_source:" in text
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        result = run(cfg, dry_run=True, progress=False)
    assert isinstance(result, dict)
    assert "box_dims" in result
    assert buf2.getvalue() == text


def test_plot_geometry_sphere_writes_png():
    cfg_path = _PKG / "configs" / "c13_sphere_ps.json"
    cfg = _load(cfg_path)
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "geo.png"
        fig = plot_geometry(cfg_path, out=png, h_override=0.01)
        assert (fig.get_suptitle() or "").startswith(cfg_path.stem)
        assert png.stat().st_size > 10_000
        from boxrad.plot.geometry import prepare_geometry_plot

        ctx = prepare_geometry_plot(cfg, h_override=0.01)
        cells, _, centroids, _ = __import__(
            "boxrad.cutcell", fromlist=["wetted_wall_arrays"]
        ).wetted_wall_arrays(
            ctx["fractions"], ctx["beta_tilde"], ctx["derived"].h
        )
        n_pts = centroids.shape[0]
        stride = max(1, int(np.ceil(n_pts / 20_000)))
        assert n_pts // stride <= 20_000 or stride == 1


def test_triangle_plane_unit_cube():
    mesh = box_mesh([1.0, 1.0, 1.0])
    verts, tris = mesh.vertices, mesh.faces

    segs = triangle_plane_segments(verts, tris, axis=2, offset=0.0)
    total = sum(float(np.linalg.norm(b - a)) for a, b in segs)
    assert len(segs) == 8
    assert total == pytest.approx(4.0, abs=1e-12)
    for a, b in segs:
        assert float(np.linalg.norm(b - a)) > 0

    segs_top = triangle_plane_segments(verts, tris, axis=2, offset=0.5)
    for a, b in segs_top:
        assert float(np.linalg.norm(b - a)) > 0


def test_slice_image_x_panel_box_solid_orientation():
    sx, sy, sz = 0.2, 0.1, 0.05
    h = 0.01
    half_z = 0.5 * sz
    cfg = Config.from_dict(
        {
            "geometry": "parametric",
            "air": {"c": 343.0, "rho": 1.204},
            "box": None,
            "band": {"f_min": 100.0, "f_max": 2000.0},
            "numerics": {
                "h_override": h,
                "cells_across_radiator": 8,
                "cfl_safety": 0.99,
                "gap_cells": 4,
                "cpml_cells": 8,
                "cpml_alpha_factor": 1.0,
                "cpml_m": 3.0,
                "cpml_R": 1e-6,
                "cpml_kappa_max": 1.0,
                "cell_budget": 60_000_000,
                "pulse_db": -40.0,
                "pulse_f_factor": 1.2,
                "n_transits": 2.0,
                "min_edge_cells": 3,
            },
            "output": {"path": "box_slice_test.json"},
            "parts": [
                {
                    "box": {"size": [sx, sy, sz]},
                    "position": [0.0, 0.0, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                }
            ],
            "radiators": [
                {
                    "name": "d",
                    "circle": {"d": 0.04},
                    "position": [0.0, 0.0, half_z],
                    "normal": [0.0, 0.0, 1.0],
                }
            ],
            "radiator": {"shape": "circle", "dims": [0.04]},
        }
    )
    ctx = prepare_geometry_plot(cfg, h_override=h)
    gg = ctx["gg"]
    state = ctx["fractions"].state
    rad_mask = np.zeros_like(state, dtype=bool)
    x0 = gg.x0
    ii = int(np.clip(int(np.floor((0.0 - float(x0[0])) / h)), 0, gg.nx - 1))
    panel = slice_image(
        state,
        ctx["beta_tilde"],
        ctx["floored"],
        rad_mask,
        "x",
        ii,
        gg,
    )
    solid = panel["state"] == 0
    rows, cols = np.where(solid)
    assert rows.size > 0
    y_lo = float(x0[1]) + cols.min() * h
    y_hi = float(x0[1]) + (cols.max() + 1) * h
    z_lo = float(x0[2]) + rows.min() * h
    z_hi = float(x0[2]) + (rows.max() + 1) * h
    assert y_lo == pytest.approx(-0.5 * sy, abs=h)
    assert y_hi == pytest.approx(0.5 * sy, abs=h)
    assert z_lo == pytest.approx(-0.5 * sz, abs=h)
    assert z_hi == pytest.approx(0.5 * sz, abs=h)