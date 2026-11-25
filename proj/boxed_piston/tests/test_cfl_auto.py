"""Auto CFL for cut-cell parametric grids (step 10b3)."""

from __future__ import annotations

import pytest
import copy
import json
from pathlib import Path

from boxrad.config import (
    CFL_CUTCELL,
    CFL_DEFAULT,
    Config,
    SQRT3,
    derive,
)
from boxrad.geomrun import dry_run_geom, prepare_geom_run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHOWCASE = CONFIGS / "showcase"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sphere_cfg(*, include_cfl_key: bool = True, cfl_safety: float = CFL_DEFAULT) -> Config:
    raw = copy.deepcopy(_load_json(CONFIGS / "c13_sphere_ps.json"))
    if not include_cfl_key:
        raw["numerics"].pop("cfl_safety", None)
    else:
        raw["numerics"]["cfl_safety"] = cfl_safety
    return Config.from_dict(raw)


def test_legacy_box_auto_cfl_matches_explicit_default():
    raw = copy.deepcopy(_load_json(CONFIGS / "default.json"))
    raw["numerics"].pop("cfl_safety", None)
    cfg_auto = Config.from_dict(raw)
    cfg_explicit = Config.from_dict(
        {**raw, "numerics": {**raw["numerics"], "cfl_safety": CFL_DEFAULT}}
    )
    d_auto = derive(cfg_auto)
    d_explicit = derive(cfg_explicit)
    assert d_auto.dt == d_explicit.dt
    assert d_auto.cfl_safety_resolved == CFL_DEFAULT
    assert d_auto.cfl_safety_source == "default"
    assert d_explicit.cfl_safety_source == "user"


def test_parametric_sphere_auto_cutcell():
    cfg = _sphere_cfg(include_cfl_key=False)
    assert cfg.numerics.cfl_safety is None
    assert cfg.numerics.h_override == pytest.approx(0.05 / 9, rel=1e-12)
    run = prepare_geom_run(cfg)
    d = run.derived
    assert d.cfl_safety_resolved == CFL_CUTCELL
    assert d.cfl_safety_source == "auto_cutcell"
    expected_dt = CFL_CUTCELL * d.h / (cfg.air.c * SQRT3)
    assert d.dt == expected_dt
    assert run.info["floor_stats"]["delta_eq_over_h"] < 0.01


def test_parametric_sphere_user_cfl_099():
    cfg = _sphere_cfg(include_cfl_key=True, cfl_safety=CFL_DEFAULT)
    run = prepare_geom_run(cfg)
    d = run.derived
    assert d.cfl_safety_resolved == CFL_DEFAULT
    assert d.cfl_safety_source == "user"
    assert run.info["floor_stats"]["delta_eq_over_h"] > 0.03


def test_parametric_aligned_box_no_cutcell():
    raw = copy.deepcopy(_load_json(SHOWCASE / "c19_sym_full.json"))
    raw["numerics"].pop("cfl_safety", None)
    cfg = Config.from_dict(raw)
    run = prepare_geom_run(cfg)
    fr = run.info["fractions_stats"]
    assert fr.get("n_cut_cells", 0) == 0
    assert run.derived.cfl_safety_resolved == CFL_DEFAULT
    assert run.derived.cfl_safety_source == "default"


def test_to_json_cfl_round_trip():
    cfg = Config.from_dict(
        {
            "geometry": "box",
            "box": {"W": 0.2, "H": 0.15, "D": 0.25},
            "radiator": {"shape": "circle", "dims": [0.05]},
            "band": {"f_min": 50.0, "f_max": 3000.0},
            "numerics": {"ppw": 20.0},
            "output": {"path": "out.h5"},
        }
    )
    assert cfg.numerics.cfl_safety is None
    text = cfg.to_json()
    blob = json.loads(text)
    assert blob["numerics"].get("cfl_safety") is None
    cfg2 = Config.from_json(text)
    assert cfg2.numerics.cfl_safety is None

    cfg3 = Config.from_dict(
        {
            **json.loads(text),
            "numerics": {**json.loads(text)["numerics"], "cfl_safety": 0.7},
        }
    )
    text7 = cfg3.to_json()
    assert json.loads(text7)["numerics"]["cfl_safety"] == 0.7
    assert Config.from_json(text7).numerics.cfl_safety == 0.7


def test_dry_run_geom_sphere_auto_reports_07():
    cfg = _sphere_cfg(include_cfl_key=False)
    est = dry_run_geom(cfg)
    assert est["cfl_safety_resolved"] == CFL_CUTCELL
    assert est["cfl_safety_source"] == "auto_cutcell"


_SHOWCASE_FIXTURES = (
    ("s1_cylinder.json", CFL_CUTCELL, "auto_cutcell", False),
    ("s2_bookshelf.json", CFL_DEFAULT, "default", True),
    ("s3p_c09_parametric.json", CFL_DEFAULT, "default", True),
    ("s4_tilted.json", CFL_CUTCELL, "auto_cutcell", False),
    ("s5_platform.json", CFL_DEFAULT, "default", True),
    ("s6_downfire.json", CFL_CUTCELL, "auto_cutcell", False),
)


@pytest.mark.parametrize(
    "fixture,resolved,source,dt_parity",
    _SHOWCASE_FIXTURES,
    ids=[t[0].replace(".json", "") for t in _SHOWCASE_FIXTURES],
)
def test_showcase_s1_s6_auto_cfl(
    fixture: str,
    resolved: float,
    source: str,
    dt_parity: bool,
) -> None:
    path = SHOWCASE / fixture
    raw = _load_json(path)
    assert "cfl_safety" not in raw.get("numerics", {})
    cfg = Config.from_json(path.read_text())
    assert cfg.numerics.cfl_safety is None
    est = dry_run_geom(cfg)
    assert est["cfl_safety_resolved"] == resolved
    assert est["cfl_safety_source"] == source
    if dt_parity:
        raw99 = copy.deepcopy(raw)
        raw99["numerics"]["cfl_safety"] = CFL_DEFAULT
        est99 = dry_run_geom(Config.from_dict(raw99))
        assert est["dt"] == est99["dt"]