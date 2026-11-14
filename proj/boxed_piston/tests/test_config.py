"""Unit tests for fdtd.config."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fdtd.config import (  # noqa: E402
    Air,
    Band,
    Box,
    Config,
    Numerics,
    Output,
    Radiator,
    derive,
    validate,
)
from fdtd.run import run  # noqa: E402

CONFIGS = ROOT / "configs"

SQRT3 = math.sqrt(3.0)


def _base_cfg(**kwargs) -> Config:
    defaults = dict(
        air=Air(),
        box=Box(W=0.3, H=0.3, D=0.2),
        radiator=Radiator("circle", dims=[0.05]),
        band=Band(f_min=20.0),
        numerics=Numerics(),
        output=Output(path="/tmp/out.h5"),
    )
    defaults.update(kwargs)
    return Config(**defaults)


def test_grid_spacing_and_dt_at_20khz():
    cfg = _base_cfg(
        band=Band(f_min=20.0, f_max=20_000.0),
        numerics=Numerics(ppw=20.0, cfl_safety=1.0),
    )
    d = derive(cfg)
    assert d.h == pytest.approx(343.0 / 20_000.0 / 20.0, rel=1e-5)
    assert d.h == pytest.approx(0.8575e-3, rel=1e-3)
    expected_dt = d.h / (343.0 * SQRT3)
    assert d.dt == pytest.approx(expected_dt, rel=1e-5)
    assert d.dt == pytest.approx(1.443e-6, rel=1e-3)


def test_default_f_max_from_ka10():
    cfg = _base_cfg(radiator=Radiator("circle", dims=[0.2]))
    d = derive(cfg)
    expected = 10.0 * 343.0 / (math.pi * 0.2)
    assert expected == pytest.approx(5459.0, rel=1e-3)
    assert d.f_max == pytest.approx(expected, rel=1e-5)


def test_zrad_fallback_tiny_radiator_large_box():
    cfg = _base_cfg(
        box=Box(W=0.5, H=0.5, D=0.5),
        radiator=Radiator("circle", dims=[0.002]),
        numerics=Numerics(cell_budget=500_000, ppw=20.0),
    )
    d = derive(cfg)
    assert d.zrad_fallback is True
    _, warnings = validate(cfg, d)
    assert any("fallback" in w.lower() for w in warnings)


def test_radiator_too_close_to_edge_is_error():
    box = Box(W=0.2, H=0.2, D=0.1)
    R = 0.025
    cfg_probe = Config(
        box=box,
        radiator=Radiator("circle", dims=[2 * R], x_c=0.0),
        band=Band(f_min=20.0, f_max=5_000.0),
        output=Output(path="/tmp/x.h5"),
    )
    h = derive(cfg_probe).h
    x_c = box.W / 2.0 - R - 2.0 * h
    cfg = Config(
        box=box,
        radiator=Radiator("circle", dims=[2 * R], x_c=x_c),
        band=Band(f_min=20.0, f_max=5_000.0),
        numerics=Numerics(min_edge_cells=3),
        output=Output(path="/tmp/x.h5"),
    )
    d = derive(cfg)
    errors, _ = validate(cfg, d)
    assert errors
    assert any("clearance" in e for e in errors)


def test_json_roundtrip():
    cfg = _base_cfg()
    d1 = derive(cfg)
    cfg2 = Config.from_json(cfg.to_json())
    d2 = derive(cfg2)
    assert cfg2.to_dict() == cfg.to_dict()
    assert d2.h == d1.h
    assert d2.n_steps == d1.n_steps


def test_baffle_derives_without_box():
    cfg = Config(
        geometry="baffle",
        box=None,
        radiator=Radiator("circle", dims=[0.1]),
        band=Band(f_min=100.0, f_max=6000.0),
        output=Output(path="/tmp/baffle.h5"),
    )
    d = derive(cfg)
    assert d.geometry == "baffle"
    from fdtd.config import CPML_SIDES_BAFFLE

    assert d.cpml_sides == CPML_SIDES_BAFFLE
    assert d.box_nz == 0
    assert d.total_cells > 0
    errors, _ = validate(cfg, d)
    assert not errors


def test_box_mode_derive_unchanged_smoke():
    """Regression guard: box geometry derive must stay bit-identical to pre-C6."""
    cfg = Config.from_json((CONFIGS / "smoke.json").read_text(encoding="utf-8"))
    d = derive(cfg)
    assert (d.nx, d.ny, d.nz) == (44, 44, 42)
    assert d.n_steps == 149
    assert d.h == pytest.approx(0.005, rel=1e-6)
    assert d.geometry == "box"
    from fdtd.config import CPML_SIDES_ALL

    assert d.cpml_sides == CPML_SIDES_ALL


def test_default_json_matches_config_defaults():
    loaded = Config.from_json((CONFIGS / "default.json").read_text(encoding="utf-8"))
    assert loaded.to_dict() == Config().to_dict()


def test_smoke_json_derive_validate_and_dry_run():
    cfg = Config.from_json((CONFIGS / "smoke.json").read_text(encoding="utf-8"))
    derived = derive(cfg)
    errors, _warnings = validate(cfg, derived)
    assert not errors
    summary = run(cfg, dry_run=True, progress=False)
    assert isinstance(summary, dict)
    assert summary["n_steps"] == derived.n_steps
