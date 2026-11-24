"""Step 10b1: numerics.test_source validation monopole."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from boxrad.config import Config
from boxrad.cutkernel import step_cut
from boxrad.geomrun import prepare_geom_run, step_geom
from boxrad.grid import allocate_fields
from boxrad.kernels import make_update_coeffs
from boxrad.source import velocity_at_half_step
from boxrad.validation.analytic import monopole_free

_PKG = Path(__file__).resolve().parents[1]
_C19 = _PKG / "configs" / "showcase" / "c19_sym_full.json"


def _cfg(d: dict) -> Config:
    return Config.from_dict(d)


def test_reject_box_baffle_test_source():
    base = {"numerics": {"test_source": {"position": [0.0, 0.0, 0.0]}}}
    with pytest.raises(ValueError, match="test_source"):
        _cfg({"geometry": "box", "radiator": {"shape": "circle", "dims": [0.05]}, **base})
    with pytest.raises(ValueError, match="test_source"):
        _cfg(
            {
                "geometry": "baffle",
                "radiator": {"shape": "circle", "dims": [0.05]},
                **base,
            }
        )


def test_reject_empty_parametric_without_test_source():
    with pytest.raises(ValueError, match="part"):
        _cfg({"geometry": "parametric", "parts": [], "radiators": []})


def test_null_equals_absent_c19():
    raw = json.loads(_C19.read_text(encoding="utf-8"))
    a = _cfg(raw)
    b = _cfg(copy.deepcopy(raw))
    b.numerics.test_source = None
    b_dict = copy.deepcopy(raw)
    b_dict.setdefault("numerics", {})["test_source"] = None
    c = Config.from_dict(b_dict)
    assert c.numerics.test_source is None
    ra = prepare_geom_run(a)
    rb = prepare_geom_run(b)
    rc = prepare_geom_run(c)
    for run in (rb, rc):
        assert np.array_equal(run.op.inv_beta, ra.op.inv_beta)
        assert run.derived.nx == ra.derived.nx
        assert np.allclose(run.gg.x0, ra.gg.x0)
        for ax in range(3):
            np.testing.assert_array_equal(
                getattr(run.op, f"afl_{'xyz'[ax]}"),
                getattr(ra.op, f"afl_{'xyz'[ax]}"),
            )


def test_empty_geometry_all_fluid_and_anchor():
    cfg = _cfg(
        {
            "geometry": "parametric",
            "parts": [],
            "radiators": [],
            "band": {"f_min": 50.0, "f_max": 1000.0},
            "numerics": {
                "h_override": 0.01,
                "n_transits": 2.0,
                "test_source": {"position": [0.02, -0.03, 0.01]},
            },
            "output": {"path": ""},
        }
    )
    run = prepare_geom_run(cfg)
    fr = run.info["fractions"]
    assert np.allclose(fr.beta, 1.0)
    assert run.derived.test_source_cell_center is not None
    cc = np.array(run.derived.test_source_cell_center, dtype=np.float64)
    req = np.array(cfg.numerics.test_source.position, dtype=np.float64)
    np.testing.assert_allclose(cc, req, atol=1e-12)


def test_injection_unit_scalar():
    h = 0.02
    cfg = _cfg(
        {
            "geometry": "parametric",
            "parts": [],
            "radiators": [],
            "band": {"f_min": 50.0, "f_max": 1000.0},
            "air": {"c": 343.0, "rho": 1.204},
            "numerics": {
                "h_override": h,
                "n_transits": 1.0,
                "gap_cells": 3,
                "cpml_cells": 4,
                "test_source": {"position": [0.0, 0.0, 0.0]},
            },
            "output": {"path": ""},
        }
    )
    run = prepare_geom_run(cfg)
    fields = allocate_fields(run.gg)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, run.derived.h, run.derived.dt)
    flat, coeff, _ = run.test_source
    assert flat is not None
    v_q = np.float32(1e-6)
    step_cut(fields, coeffs, run.op, run.cpml, [], test_source=(flat, coeff, v_q))
    rho = cfg.air.rho
    c = cfg.air.c
    dt = run.derived.dt
    expected = rho * c * c * dt * 1e-6 / h**3
    got = float(fields.p.reshape(-1)[flat])
    rtol = expected * 1e-6 + 1e-12
    assert abs(got - expected) <= max(rtol, 1e-12)
    flat_p = fields.p.reshape(-1)
    mask = np.ones(flat_p.shape[0], dtype=bool)
    mask[flat] = False
    assert float(np.max(np.abs(flat_p[mask]))) == 0.0


def test_monopole_free_hand_value():
    pt = np.array([[0.1, 0.0, 0.0]], dtype=np.float64)
    src = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    f = 500.0
    rho = 1.204
    c = 343.0
    _, p = monopole_free(pt, f, src, rho, c)
    k = 2.0 * np.pi * f / c
    r = 0.1
    ref = 1j * 2.0 * np.pi * f * rho * np.exp(-1j * k * r) / (4.0 * np.pi * r)
    assert abs(p[0, 0] - ref) <= 1e-12 * abs(ref)


def test_source_inside_solid_rejected():
    cfg = _cfg(
        {
            "geometry": "parametric",
            "parts": [
                {
                    "box": {"size": [0.2, 0.2, 0.2]},
                    "position": [0.0, 0.0, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                    "x_axis": [1.0, 0.0, 0.0],
                }
            ],
            "radiators": [],
            "band": {"f_min": 50.0, "f_max": 1000.0},
            "numerics": {
                "h_override": 0.02,
                "n_transits": 2.0,
                "test_source": {"position": [0.0, 0.0, 0.0]},
            },
            "output": {"path": ""},
        }
    )
    with pytest.raises(ValueError, match="test_source cell"):
        prepare_geom_run(cfg)


@pytest.mark.slow
def test_coarse_free_field_vs_analytic(tmp_path):
    cfg = _cfg(
        {
            "geometry": "parametric",
            "parts": [],
            "radiators": [],
            "band": {"f_min": 50.0, "f_max": 1000.0},
            "air": {"c": 343.0, "rho": 1.204},
            "numerics": {
                "h_override": 0.02,
                "n_transits": 6.0,
                "gap_cells": 8,
                "cpml_cells": 12,
                "test_source": {"position": [0.0, 0.0, 0.0]},
            },
            "output": {
                "path": "",
                "probes": [[0.1, 0.0, 0.0]],
            },
        }
    )
    from boxrad.geomrec import GeomRecorder

    run = prepare_geom_run(cfg)
    out = tmp_path / "free.h5"
    rec = GeomRecorder(cfg, run.derived, run.gg, run.geometry, run.info, out)
    n_cap = min(800, run.derived.n_steps)
    for n in range(n_cap):
        step_geom(run, n)
        rec.sample(run.fields, n)
    rec.close()

    import h5py

    from boxrad import Result
    from boxrad.post.transfer import h_transfer_err, probe_fdtd_pressure_spectrum

    freqs = [300.0, 500.0, 800.0]
    src = np.array([0.0, 0.0, 0.0])
    pt = np.array([0.1, 0.0, 0.0])
    with h5py.File(out, "r") as f:
        t_pr = np.asarray(f["probes/times_p"][()], dtype=np.float64)
        p_ts = np.asarray(f["probes/p"][()], dtype=np.float64)
    with Result(out) as res:
        for f_hz in freqs:
            p_fd = probe_fdtd_pressure_spectrum(p_ts, t_pr, f_hz, 0)
            _, p_ref = monopole_free(pt[np.newaxis, :], f_hz, src, cfg.air.rho, cfg.air.c)
            h_fd = p_fd / res._radiator_v_at_f(f_hz)
            db, deg = h_transfer_err(p_ref[0, 0], h_fd)
            assert db <= 0.3 and abs(deg) <= 2.0