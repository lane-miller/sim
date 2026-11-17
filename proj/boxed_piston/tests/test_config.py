"""Unit tests for boxrad.config."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from boxrad.config import (
    Air,
    Band,
    Box,
    Config,
    Derived,
    Numerics,
    Output,
    Radiator,
    derive,
    validate,
)
from boxrad.runner import run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

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


def test_cell_budget_exceeded_is_error_tiny_radiator_large_box():
    cfg = _base_cfg(
        box=Box(W=0.5, H=0.5, D=0.5),
        radiator=Radiator("circle", dims=[0.002]),
        numerics=Numerics(cell_budget=500_000, ppw=20.0),
    )
    d = derive(cfg)
    assert d.cell_budget_exceeded is True
    assert d.h == pytest.approx(0.002 / 10, rel=1e-12)
    errors, warnings = validate(cfg, d)
    assert any("cell_budget" in e for e in errors)
    assert not any("fallback" in w.lower() for w in warnings)
    with pytest.raises(ValueError, match="cell_budget"):
        run(cfg, dry_run=True, progress=False)


def test_derived_from_dict_ignores_unknown_keys():
    d = derive(_base_cfg())
    blob = json.loads(json.dumps(asdict(d)))
    blob.pop("cell_budget_exceeded")
    blob["zrad_fallback"] = False
    d2 = Derived.from_dict(blob)
    assert d2.h == d.h
    assert d2.cpml_sides == d.cpml_sides
    assert d2.cell_budget_exceeded is False


def test_within_cell_budget_no_error():
    cfg = _base_cfg()
    d = derive(cfg)
    assert d.cell_budget_exceeded is False
    errors, _ = validate(cfg, d)
    assert not any("cell_budget" in e for e in errors)


def _baffle_cfg(x_c: float = 0.0, y_c: float = 0.0) -> Config:
    return Config(
        geometry="baffle",
        box=None,
        radiator=Radiator("circle", dims=[0.1], x_c=x_c, y_c=y_c),
        band=Band(f_min=100.0, f_max=6000.0),
        output=Output(path="/tmp/baffle.h5"),
    )


def test_baffle_offset_subcell_warns():
    h = derive(_baffle_cfg()).h
    cfg = _baffle_cfg(x_c=0.25 * h)
    d = derive(cfg)
    errors, warnings = validate(cfg, d)
    assert not errors
    assert any("baffle mode" in w and "clips" in w for w in warnings)


def test_baffle_offset_over_one_cell_is_error():
    h = derive(_baffle_cfg()).h
    cfg = _baffle_cfg(x_c=2.0 * h)
    d = derive(cfg)
    errors, _ = validate(cfg, d)
    assert any("baffle mode" in e and "clips" in e for e in errors)
    with pytest.raises(ValueError, match="clips"):
        run(cfg, dry_run=True, progress=False)


def test_baffle_centered_no_clip_message():
    cfg = _baffle_cfg()
    d = derive(cfg)
    errors, warnings = validate(cfg, d)
    assert not errors
    assert not any("clips" in w for w in warnings)


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
    from boxrad.config import CPML_SIDES_BAFFLE

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
    from boxrad.config import CPML_SIDES_ALL

    assert d.cpml_sides == CPML_SIDES_ALL


def test_h_override_sets_spacing_and_ppw_realized():
    cfg = _base_cfg(
        box=Box(W=0.2, H=0.2, D=0.2),
        band=Band(f_min=50.0, f_max=3000.0),
        numerics=Numerics(h_override=0.2 / 36.0, ppw=999.0),
    )
    d = derive(cfg)
    assert d.h == pytest.approx(0.2 / 36.0, rel=1e-12)
    assert d.realized_W == pytest.approx(0.2, abs=1e-9)
    assert d.realized_H == pytest.approx(0.2, abs=1e-9)
    assert d.realized_D == pytest.approx(0.2, abs=1e-9)
    assert d.ppw_realized == pytest.approx(d.lambda_min / d.h, rel=1e-9)
    _, warnings = validate(cfg, d)
    assert any("h_override" in w for w in warnings)


def test_default_json_loads_validates_derives():
    cfg = Config.from_json((CONFIGS / "default.json").read_text(encoding="utf-8"))
    assert cfg.geometry == "box"
    assert cfg.box is not None
    assert cfg.numerics.h_override is None
    d = derive(cfg)
    errors, warnings = validate(cfg, d)
    assert not errors
    assert not warnings
    assert d.h == pytest.approx(343.0 / 3000.0 / 20.0, rel=1e-12)
    assert (d.box_nx, d.box_ny, d.box_nz) == (35, 26, 44)
    assert d.cell_budget_exceeded is False
    summary = run(cfg, dry_run=True, progress=False)
    assert summary["n_steps"] == d.n_steps


def test_smoke_json_derive_validate_and_dry_run():
    cfg = Config.from_json((CONFIGS / "smoke.json").read_text(encoding="utf-8"))
    derived = derive(cfg)
    errors, _warnings = validate(cfg, derived)
    assert not errors
    summary = run(cfg, dry_run=True, progress=False)
    assert isinstance(summary, dict)
    assert summary["n_steps"] == derived.n_steps
