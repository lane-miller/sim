"""Multi-radiator :class:`boxrad.Result` API (step 8b)."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from boxrad import Result
from boxrad.config import Config
from boxrad.geom.geometry import build_geometry
from boxrad.geomrun import multirad_output_path, prepare_geom_run, resolve_drive_indices, step_geom
from boxrad.geomrec import GeomRecorder

SHOWCASE = Path(__file__).resolve().parents[1] / "configs" / "showcase"
_F64 = np.float64
_C128 = np.complex128
_RHO = 1.204
_C = 343.0


def _load(name: str) -> Config:
    return Config.from_json((SHOWCASE / name).read_text(encoding="utf-8"))


def _s2_free_coarse() -> Config:
    raw = json.loads(_load("s2_bookshelf.json").to_json())
    raw.pop("environment", None)
    raw["numerics"]["h_override"] = 0.01
    raw["numerics"]["n_transits"] = 0.35
    return Config.from_dict(raw)


def _write_truncated_multirad(cfg: Config, out: Path, *, n_max: int = 150) -> None:
    import h5py

    geom = build_geometry(cfg)
    assert geom is not None
    names = [r.name for r in geom.radiators]
    drive = resolve_drive_indices(cfg)
    for j in drive:
        run = prepare_geom_run(copy.deepcopy(cfg), driven=j)
        dest = multirad_output_path(out, j, names[j])
        rec = GeomRecorder(
            cfg,
            run.derived,
            run.gg,
            run.geometry,
            run.info,
            dest,
            buffer_samples=32,
            driven=j,
        )
        for n in range(n_max):
            step_geom(run, n)
            rec.sample(run.fields, n)
        rec.close()
    run_files: list[str | None] = [None] * len(names)
    for j in drive:
        run_files[j] = multirad_output_path(out, j, names[j]).name
    with h5py.File(out, "a") as f_out:
        f_out["meta"].attrs["run_files_json"] = json.dumps(run_files)


def test_names_and_radiator_siblings(tmp_path: Path) -> None:
    cfg = _s2_free_coarse()
    out = tmp_path / "out.h5"
    _write_truncated_multirad(cfg, out)
    with Result(out) as r0:
        assert r0.names == ["woofer", "tweeter"]
        assert r0.radiator(0) is r0
        r1 = r0.radiator(1)
        assert r1.radiator("tweeter") is r1
        assert r0.radiator("tweeter").path == r1.path
        assert r1.radiator(0).path.resolve() == out.resolve()


def test_missing_sibling_and_undriven(tmp_path: Path) -> None:
    cfg = _s2_free_coarse()
    out = tmp_path / "out.h5"
    _write_truncated_multirad(cfg, out)
    sib = tmp_path / "out.r1_tweeter.h5"
    assert sib.is_file()
    sib.unlink()
    with Result(out) as r0:
        with pytest.raises(FileNotFoundError, match="out.r1_tweeter.h5"):
            r0.radiator(1)

    raw = json.loads(cfg.to_json())
    raw["numerics"]["drive_radiators"] = ["woofer"]
    cfg_one = Config.from_dict(raw)
    out2 = tmp_path / "woofer_only.h5"
    _write_truncated_multirad(cfg_one, out2)
    with Result(out2) as r_w:
        with pytest.raises(ValueError, match="tweeter was not driven"):
            r_w.radiator(1)


def test_Z_matrix_from_zrad_row(tmp_path: Path) -> None:
    cfg = _s2_free_coarse()
    out = tmp_path / "out.h5"
    _write_truncated_multirad(cfg, out)
    import h5py

    with h5py.File(out, "r") as f0:
        f_log = np.asarray(f0["results/zrad_row"].attrs["f"], dtype=_F64)
    pick = [int(len(f_log) // 4), int(len(f_log) // 2), int(3 * len(f_log) // 4)]
    freqs = f_log[pick]
    with Result(out) as r0:
        z_mat = r0.Z_matrix(freqs)
        s_list = r0.S_list
        for j in (0, 1):
            path_j = multirad_output_path(out, j, r0.names[j])
            with h5py.File(path_j, "r") as fj:
                z_row = np.asarray(fj["results/zrad_row"][()], dtype=_C128)[pick]
            ref = (_RHO * _C * z_row / s_list[j]).astype(_C128)
            num = np.max(np.abs(z_mat[:, :, j] - ref))
            den = max(float(np.max(np.abs(ref))), 1e-30)
            assert num / den <= 1e-12


def test_field_sum_superposition(tmp_path: Path) -> None:
    cfg = _s2_free_coarse()
    out = tmp_path / "out.h5"
    _write_truncated_multirad(cfg, out)
    pts = np.array([[0.35, 0.30, 0.20], [0.50, 0.10, 0.35]], dtype=_F64)
    freqs = np.array([200.0, 500.0], dtype=_F64)
    rtol = 1e-12
    with Result(out, frame="world") as r0:
        s0, s1 = float(r0.S_list[0]), float(r0.S_list[1])
        h0 = r0.field(pts, freqs, frame="world") / s0
        h1 = r0.radiator(1).field(pts, freqs, frame="world") / s1
        f_a = r0.field_sum(pts, freqs, U=(1.0, 0.0), frame="world")
        f_b = r0.field_sum(pts, freqs, U=(0.0, 1.0), frame="world")
        f_ab = r0.field_sum(pts, freqs, U=(1.0, 1.0), frame="world")
        den = max(float(np.max(np.abs(h0))), 1e-30)
        assert float(np.max(np.abs(f_a - h0))) / den <= rtol
        den = max(float(np.max(np.abs(h1))), 1e-30)
        assert float(np.max(np.abs(f_b - h1))) / den <= rtol
        den = max(float(np.max(np.abs(f_ab))), 1e-30)
        assert float(np.max(np.abs(f_ab - (h0 + h1)))) / den <= rtol


def test_inside_box_and_floor(tmp_path: Path) -> None:
    cfg = _s2_free_coarse()
    out = tmp_path / "out.h5"
    _write_truncated_multirad(cfg, out)
    with Result(out, frame="world") as r0:
        center = np.array([[0.0, 0.0, 0.15]], dtype=_F64)
        outside = np.array([[0.0, 0.136, 0.15]], dtype=_F64)
        assert r0.inside(center, frame="world").all()
        assert not r0.inside(outside, frame="world").any()

    raw_floor = json.loads(_load("c19b_floor.json").to_json())
    raw_floor["numerics"]["n_transits"] = 0.35
    floor_cfg = Config.from_dict(raw_floor)
    out_floor = tmp_path / "floor.h5"
    _write_truncated_multirad(floor_cfg, out_floor, n_max=80)
    with Result(out_floor, frame="world") as r_f:
        behind = np.array([[0.0, 0.0, -0.01]], dtype=_F64)
        assert r_f.inside(behind, frame="world").all()