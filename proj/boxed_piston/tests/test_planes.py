"""Environment planes (step 7a)."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config
from boxrad.geom.geometry import build_geometry
from boxrad.geomderive import derive_geom
from boxrad.geomrec import huygens_patches_planes
from boxrad.geomrun import dry_run_geom, prepare_geom_run, simulate_geom, step_geom
from boxrad.post.kirchhoff import huygens_box_patches
from boxrad.recorder import resolve_huygens_offset
from boxrad.result import Result

SHOWCASE = Path(__file__).resolve().parents[1] / "configs" / "showcase"
_F64 = np.float64

_C19_CX = 0.0
_C19_CY = 0.0
_C19_FULL = "c19_sym_full.json"


def _load(name: str) -> Config:
    return Config.from_json((SHOWCASE / name).read_text(encoding="utf-8"))


def _derive(cfg: Config):
    geom = build_geometry(cfg)
    assert geom is not None
    return derive_geom(cfg, geom)


def test_s2_truncation_derive() -> None:
    cfg = _load("s2_bookshelf.json")
    derived, gg = _derive(cfg)
    sides = derived.cpml_sides
    assert sides[4] is False and sides[2] is False
    assert float(gg.x0[2]) == pytest.approx(0.0, abs=1e-12)
    assert float(gg.x0[1]) == pytest.approx(-0.425, abs=1e-12)


def test_grid_coincidence_c19_halves() -> None:
    cfgs = {
        "full": _load(_C19_FULL),
        "half": _load("c19_half_x.json"),
        "quarter": _load("c19_quarter_xy.json"),
    }
    derived = {k: _derive(v) for k, v in cfgs.items()}
    d0, g0 = derived["full"]
    d_h, g_h = derived["half"]
    d_q, g_q = derived["quarter"]
    rtol = 1e-12
    assert d_h.h == pytest.approx(d0.h, rel=rtol)
    assert d_q.h == pytest.approx(d0.h, rel=rtol)
    assert d_h.dt == pytest.approx(d0.dt, rel=rtol)
    assert d_h.n_steps == d0.n_steps
    assert d_h.nz == d0.nz
    h = float(d0.h)
    x_faces_full = g0.x0[0] + np.arange(g0.nx + 1) * h
    x_faces_half = g_h.x0[0] + np.arange(g_h.nx + 1) * h
    pos_full = x_faces_full[x_faces_full >= _C19_CX - 1e-12]
    shift = pos_full[0] - x_faces_half[0]
    assert abs(shift / h - round(shift / h)) < 1e-12
    assert np.allclose(
        x_faces_half,
        pos_full[: x_faces_half.size],
        atol=1e-12,
    )
    n_pos = int(np.sum((x_faces_full[:-1] + x_faces_full[1:]) * 0.5 > _C19_CX))
    assert g_h.nx == n_pos
    y_faces_full = g0.x0[1] + np.arange(g0.ny + 1) * h
    y_faces_q = g_q.x0[1] + np.arange(g_q.ny + 1) * h
    pos_y = y_faces_full[y_faces_full >= _C19_CY - 1e-12]
    shift_y = pos_y[0] - y_faces_q[0]
    assert abs(shift_y / h - round(shift_y / h)) < 1e-12
    n_y_pos = int(np.sum((y_faces_full[:-1] + y_faces_full[1:]) * 0.5 > _C19_CY))
    assert g_q.ny == n_y_pos
    assert np.allclose(
        y_faces_q,
        pos_y[: y_faces_q.size],
        atol=1e-12,
    )


def test_s_grid_halves() -> None:
    full = _load(_C19_FULL)
    half = _load("c19_half_x.json")
    quarter = _load("c19_quarter_xy.json")
    s_full = dry_run_geom(full)["S_grid"]
    s_half = dry_run_geom(half)["S_grid"]
    s_quarter = dry_run_geom(quarter)["S_grid"]
    assert s_half == pytest.approx(0.5 * s_full, rel=1e-9)
    assert s_quarter == pytest.approx(0.25 * s_full, rel=1e-9)


def test_plane_crossing_raises() -> None:
    cfg = _load("s3p_c09_parametric.json")
    bad = copy.deepcopy(cfg)
    pl = copy.deepcopy(bad.geometry_spec.environment.planes)
    _ = pl
    raw = json.loads(cfg.to_json())
    raw["environment"] = {"planes": [{"axis": "x", "side": "-", "offset": 0.08}]}
    with pytest.raises(ValueError, match="crosses rigid plane"):
        _derive(Config.from_dict(raw))


def test_h_shrink_commensurate() -> None:
    raw = {
        "geometry": "parametric",
        "air": {"c": 343.0, "rho": 1.204},
        "band": {"f_min": 50.0, "f_max": 500.0},
        "numerics": {
            "h_override": 0.01,
            "cells_across_radiator": 6,
            "cfl_safety": 0.99,
            "cpml_cells": 8,
            "cell_budget": 5000000,
            "pulse_db": -40.0,
            "n_transits": 2.0,
        },
        "environment": {"planes": [{"axis": "z", "side": "-", "offset": 0.0}]},
        "parts": [
            {
                "box": {"size": [0.08, 0.08, 0.1237]},
                "position": [0.0, 0.0, 0.06185],
                "normal": [0, 0, 1],
            }
        ],
        "radiators": [
            {
                "name": "rd",
                "rect": {"w": 0.04, "h": 0.04},
                "position": [0.0, 0.0, 0.1237],
                "normal": [0.0, 0.0, 1.0],
            }
        ],
    }
    derived, gg = _derive(Config.from_dict(raw))
    assert derived.h <= 0.01 + 1e-15
    assert derived.h == pytest.approx(0.1237 / 13, rel=1e-12)
    assert gg.h_nominal == pytest.approx(0.01, rel=1e-12)
    d = abs(0.1237 - 0.0)
    assert abs(d / derived.h - round(d / derived.h)) < 1e-9


def _open_surface_lengths(
    lo_s: np.ndarray,
    hi_s: np.ndarray,
    pad: float,
    rigid_sides: dict[str, bool],
) -> tuple[float, float, float]:
    """L_a = open-surface max − min (matches :func:`huygens_patches_planes`)."""
    lo_s = np.asarray(lo_s, dtype=_F64).reshape(3)
    hi_s = np.asarray(hi_s, dtype=_F64).reshape(3)
    mins = np.empty(3, dtype=_F64)
    maxs = np.empty(3, dtype=_F64)
    for axis, key in enumerate(("x", "y", "z")):
        if rigid_sides.get(f"{key}-", False):
            mins[axis] = float(lo_s[axis])
        else:
            mins[axis] = float(lo_s[axis]) - pad
        if rigid_sides.get(f"{key}+", False):
            maxs[axis] = float(hi_s[axis])
        else:
            maxs[axis] = float(hi_s[axis]) + pad
    span = maxs - mins
    return float(span[0]), float(span[1]), float(span[2])


def _open_huygens_area(
    lx: float,
    ly: float,
    lz: float,
    h: float,
    rigid_sides: dict[str, bool],
) -> float:
    n_x = max(1, int(round(lx / h)))
    n_y = max(1, int(round(ly / h)))
    n_z = max(1, int(round(lz / h)))
    h2 = h * h
    area = 0.0
    if not rigid_sides.get("x-", False):
        area += n_y * n_z * h2
    if not rigid_sides.get("x+", False):
        area += n_y * n_z * h2
    if not rigid_sides.get("y-", False):
        area += n_x * n_z * h2
    if not rigid_sides.get("y+", False):
        area += n_x * n_z * h2
    if not rigid_sides.get("z-", False):
        area += n_x * n_y * h2
    if not rigid_sides.get("z+", False):
        area += n_x * n_y * h2
    return area


def test_huygens_open_surface() -> None:
    full = _load(_C19_FULL)
    half = _load("c19_half_x.json")
    derived, gg = _derive(full)
    offset = resolve_huygens_offset(full)
    h = derived.h
    gap_cpml = (derived.gap_cells + derived.cpml_cells) * h
    lo_s = gg.x0.astype(_F64) + gap_cpml
    closed = huygens_box_patches(
        derived.realized_W,
        derived.realized_H,
        derived.realized_D,
        h,
        lo_s,
        offset_cells=offset,
    )
    open_same = huygens_patches_planes(
        lo_s,
        lo_s + np.array([derived.realized_W, derived.realized_H, derived.realized_D]),
        h,
        offset,
        {},
        gg.x0,
        gg.x0 + np.array([gg.nx, gg.ny, gg.nz]) * h,
    )
    assert np.allclose(closed.centers, open_same.centers, atol=0.0, rtol=0.0)
    assert np.allclose(closed.normals, open_same.normals)
    assert np.allclose(closed.areas, open_same.areas)

    d_h, g_h = _derive(half)
    rigid = dict(g_h.rigid_sides or {})
    gap_h = (d_h.gap_cells + d_h.cpml_cells) * h
    loff = g_h.x0.astype(_F64) + gap_h * np.array(
        [0.0 if rigid.get("x-", False) else 1.0, 1.0, 1.0]
    )
    hi_b = loff + np.array([d_h.realized_W, d_h.realized_H, d_h.realized_D])
    hy = huygens_patches_planes(
        loff,
        hi_b,
        h,
        offset,
        rigid,
        g_h.x0,
        g_h.x0 + np.array([g_h.nx, g_h.ny, g_h.nz]) * h,
    )
    assert not np.any(hy.normals[:, 0] < -0.5)
    assert np.all(hy.centers[:, 0] >= _C19_CX - 1e-12)
    pad = offset * h
    lx, ly, lz = _open_surface_lengths(loff, hi_b, pad, rigid)
    area_expected = _open_huygens_area(lx, ly, lz, h, rigid)
    assert float(np.sum(hy.areas)) == pytest.approx(area_expected, rel=1e-9)


def test_symmetry_pavg_short_run() -> None:
    from boxrad.geomrec import GeomRecorder

    full = _load(_C19_FULL)
    half = _load("c19_half_x.json")
    n_max = 150
    with tempfile.TemporaryDirectory() as td:
        p_full = Path(td) / "full.h5"
        p_half = Path(td) / "half.h5"
        for cfg, path in ((full, p_full), (half, p_half)):
            run = prepare_geom_run(copy.deepcopy(cfg))
            rec = GeomRecorder(
                cfg,
                run.derived,
                run.gg,
                run.geometry,
                run.info,
                path,
                buffer_samples=32,
            )
            for n in range(n_max):
                step_geom(run, n)
                rec.sample(run.fields, n)
            rec.close()
        with Result(p_full) as rf, Result(p_half) as rh:
            pavg_f = np.asarray(rf._file()["radiator/p_avg"][()], dtype=_F64)
            pavg_h = np.asarray(rh._file()["radiator/p_avg"][()], dtype=_F64)
    denom = float(np.max(np.abs(pavg_f)))
    num = float(np.max(np.abs(pavg_f - pavg_h)))
    assert num <= 1e-5 * max(denom, 1e-30)


def test_rigid_edge_vx_zero() -> None:
    half = _load("c19_half_x.json")
    run = simulate_geom(copy.deepcopy(half), n_steps=150)
    vx0 = run.fields.vx[0, :, :]
    assert float(np.max(np.abs(vx0))) == 0.0