"""Alias-aware Huygens decimation from :func:`boxrad.config.derive`."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from boxrad.config import (
    Band,
    Box,
    Config,
    Numerics,
    Output,
    Radiator,
    decimation_m_cap,
    derive,
    pulse_params,
    validate,
    worst_huygens_alias_level_db,
)
from boxrad.grid import Grid
from boxrad.post.transfer import load_huygens_spectra
from boxrad.post.zrad import zrad_at_frequencies
from boxrad.runner import run
from boxrad.source import BoxBoundary


def _bandedge_cfg() -> Config:
    text = (
        Path(__file__).resolve().parents[1] / "configs" / "demo" / "bandedge_box.json"
    ).read_text(encoding="utf-8")
    return Config.from_json(text)


def test_choose_decimation_meets_alias_rejection():
    cfg = _bandedge_cfg()
    d = derive(cfg)
    a_db = float(cfg.numerics.alias_rejection_db)
    assert d.alias_level_db <= -a_db + 1e-6
    assert d.decimation_m >= 1
    m_cap = decimation_m_cap(d.f_max, d.dt)
    m = d.decimation_m
    sigma, t0 = pulse_params(cfg, d)
    if m < m_cap:
        level_m1 = worst_huygens_alias_level_db(
            d.f_max, sigma, t0, d.dt, m + 1
        )
        assert level_m1 > -a_db + 1e-6
    else:
        assert m == m_cap


def test_decimation_override_respected_and_warn_level():
    cfg = _bandedge_cfg()
    d_def = derive(cfg)
    m0 = d_def.decimation_m

    cfg_hi = Config.from_json(cfg.to_json())
    cfg_hi.numerics.decimation_override = max(1, 2 * m0)
    d_hi = derive(cfg_hi)
    assert d_hi.decimation_m == cfg_hi.numerics.decimation_override
    assert d_hi.alias_level_db > d_def.alias_level_db
    _, warnings_hi = validate(cfg_hi, d_hi)
    assert any("Huygens alias level" in w for w in warnings_hi)

    cfg_lo = Config.from_json(cfg.to_json())
    cfg_lo.numerics.decimation_override = max(1, m0 // 2)
    d_lo = derive(cfg_lo)
    assert d_lo.decimation_m == cfg_lo.numerics.decimation_override
    assert d_lo.alias_level_db < d_def.alias_level_db
    _, warnings_lo = validate(cfg_lo, d_lo)
    assert not any("Huygens alias level" in w for w in warnings_lo)


def test_smoke_decimation_finite():
    path = Path(__file__).resolve().parents[1] / "configs" / "smoke.json"
    cfg = Config.from_json(path.read_text(encoding="utf-8"))
    d = derive(cfg)
    assert d.decimation_m >= 1
    assert math.isfinite(d.alias_level_db)
    assert d.alias_level_db <= -cfg.numerics.alias_rejection_db + 1e-6


def _tiny_bandedge_cfg() -> Config:
    return Config(
        box=Box(W=0.08, H=0.08, D=0.06),
        radiator=Radiator("circle", dims=[0.03], x_c=0.0, y_c=0.0),
        band=Band(f_min=200.0, f_max=800.0),
        numerics=Numerics(
            gap_cells=6,
            cpml_cells=8,
            ppw=12.0,
            cells_across_radiator=8,
            cell_budget=10**9,
            cfl_safety=0.99,
            n_transits=1.0,
        ),
        output=Output(path=""),
    )


def test_huygens_flux_matches_circuit_at_fmax():
    cfg = _tiny_bandedge_cfg()
    d = derive(cfg)
    f_hz = float(d.f_max)
    with tempfile.TemporaryDirectory() as tmp:
        h5 = Path(tmp) / "flux.h5"
        run(cfg, out_path=h5, progress=False)
        grid = Grid(cfg, d)
        s_grid = float(BoxBoundary.from_grid(grid, d).s_grid)
        _, z = zrad_at_frequencies(h5, np.asarray([f_hz]))
        z_ac = float(cfg.air.rho) * float(cfg.air.c) * complex(z[0]) / s_grid
        p_face = 0.5 * float(np.real(z_ac)) * s_grid**2
        patches, v_ref, p_h, vn = load_huygens_spectra(h5, d, f_hz)
        areas = np.asarray(patches.areas, dtype=np.float64)
        p_huyg = float(
            0.5
            * np.real(np.sum((p_h / v_ref) * np.conj(vn / v_ref) * areas))
        )
        rel = abs(p_huyg / max(p_face, 1e-30) - 1.0)
        assert rel <= 0.005, f"P_huyg/P_face−1={100*rel:.3f}% at f_max"