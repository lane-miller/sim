"""Result local/world frames (step 6b2b)."""

from __future__ import annotations

import copy
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from boxrad.config import Config, derive
from boxrad.cpml import make_cpml_state
from boxrad.geomrec import GeomRecorder
from boxrad.geomrun import prepare_geom_run, step_geom
from boxrad.grid import Grid, allocate_fields
from boxrad.kernels import make_update_coeffs, step
from boxrad.recorder import Recorder
from boxrad.result import Result
from boxrad.source import BoxBoundary

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHOWCASE = CONFIGS / "showcase"
_F64 = np.float64
_N_STEPS = 220
_R_FAR = 2000.0


def _load_showcase(name: str) -> Config:
    return Config.from_json((SHOWCASE / name).read_text(encoding="utf-8"))


def _coarse(cfg: Config, *, h: float = 0.012, n_transits: float = 0.35) -> Config:
    c = copy.deepcopy(cfg)
    c.numerics.h_override = h
    c.numerics.n_transits = n_transits
    return c


def _fast_transits(cfg: Config, *, n_transits: float = 0.35) -> Config:
    c = copy.deepcopy(cfg)
    c.numerics.n_transits = n_transits
    return c


def _dir_samples() -> tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0], dtype=_F64))
    ph = np.deg2rad(np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0], dtype=_F64))
    return th, ph


def _unit_dirs(th: np.ndarray, ph: np.ndarray) -> np.ndarray:
    st = np.sin(th)
    return np.column_stack([st * np.cos(ph), st * np.sin(ph), np.cos(th)])


@pytest.fixture(scope="module")
def s4_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = _coarse(_load_showcase("s4_tilted.json"))
    run = prepare_geom_run(cfg)
    path = tmp_path_factory.mktemp("s4_frames") / "s4.h5"
    rec = GeomRecorder(cfg, run.derived, run.gg, run.geometry, run.info, path)
    for n in range(_N_STEPS):
        step_geom(run, n)
        rec.sample(run.fields, n)
    rec.close()
    return path


def test_far_field_phase_local(s4_h5: Path) -> None:
    freqs = np.array([250.0, 600.0], dtype=_F64)
    th, ph = _dir_samples()
    r_hat = _unit_dirs(th, ph)
    pts = _R_FAR * r_hat
    with Result(s4_h5) as res:
        c = float(res.config.air.c)
        pv = res.field(pts, freqs, frame="local")
        d_loc = res.directivity(th, ph, freqs, frame="local")
        for fi, f_hz in enumerate(freqs):
            k = 2.0 * math.pi * f_hz / c
            ref = d_loc[fi] * np.exp(-1j * k * _R_FAR) / _R_FAR
            err = np.abs(pv[fi] - ref) / np.maximum(np.abs(ref), 1e-30)
            assert float(np.max(err)) <= 2.0e-3


def test_far_field_phase_world(s4_h5: Path) -> None:
    freqs = np.array([250.0, 600.0], dtype=_F64)
    th, ph = _dir_samples()
    r_hat = _unit_dirs(th, ph)
    pts = _R_FAR * r_hat
    with Result(s4_h5) as res:
        c = float(res.config.air.c)
        pv = res.field(pts, freqs, frame="world")
        d_w = res.directivity(th, ph, freqs, frame="world")
        for fi, f_hz in enumerate(freqs):
            k = 2.0 * math.pi * f_hz / c
            ref = d_w[fi] * np.exp(-1j * k * _R_FAR) / _R_FAR
            err = np.abs(pv[fi] - ref) / np.maximum(np.abs(ref), 1e-30)
            assert float(np.max(err)) <= 2.0e-3


def test_frame_consistency_field(s4_h5: Path) -> None:
    pts_l = np.array(
        [
            [0.3, 0.0, 0.4],
            [-0.2, 0.25, 0.35],
            [0.15, -0.3, 0.5],
            [0.0, 0.2, 0.6],
        ],
        dtype=_F64,
    )
    f_hz = 400.0
    with Result(s4_h5) as res:
        r_mat, t = res.pose
        pts_w = pts_l @ r_mat.T + t
        pl = res.field(pts_l, f_hz, frame="local")
        pw = res.field(pts_w, f_hz, frame="world")
        denom = np.maximum(np.abs(pl), 1e-30)
        rel = np.abs(pl - pw) / denom
        assert float(np.max(rel)) <= 1e-12


def test_box_file_frames_identical() -> None:
    c09 = _fast_transits(Config.from_json((CONFIGS / "c09_box.json").read_text(encoding="utf-8")))
    leg = derive(c09)
    grid = Grid(c09, leg)
    fields = allocate_fields(grid)
    coeffs = make_update_coeffs(c09.air.rho, c09.air.c, leg.h, leg.dt)
    cpml = make_cpml_state(grid, c09, leg, leg.dt)
    boundary = BoxBoundary.from_grid(grid, leg)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c09.h5"
        rec = Recorder(c09, leg, grid, boundary, path)
        for n in range(_N_STEPS):
            step(fields, coeffs, grid.cpml_slabs, cpml, boundary, n)
            rec.sample(fields, n)
        rec.close()
        theta = np.deg2rad(np.linspace(0.0, 180.0, 13))
        freqs = np.array([300.0, 700.0], dtype=_F64)
        with Result(path) as res:
            r_mat, t = res.pose
            np.testing.assert_allclose(r_mat, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(t, 0.0, atol=1e-12)
            d0 = res.directivity(theta, 0.0, freqs)
            dl = res.directivity(theta, 0.0, freqs, frame="local")
            dw = res.directivity(theta, 0.0, freqs, frame="world")
            np.testing.assert_allclose(dl, d0, rtol=0, atol=1e-12)
            np.testing.assert_allclose(dw, d0, rtol=0, atol=1e-12)


def test_bad_pose_raises(s4_h5: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.h5"
        shutil.copy(s4_h5, bad)
        with h5py.File(bad, "a") as f:
            x_bad = np.asarray(f["meta"].attrs["radiator0_x_axis"], dtype=_F64)
            x_bad[0] *= 1.1
            f["meta"].attrs["radiator0_x_axis"] = x_bad
        with Result(bad) as res:
            with pytest.raises(ValueError, match="orthonormal|determinant"):
                _ = res.pose


def test_invalid_frame_raises(s4_h5: Path) -> None:
    with Result(s4_h5) as res:
        with pytest.raises(ValueError, match="frame must"):
            res.field([0.5, 0.0, 0.5], 400.0, frame="sideways")  # type: ignore[arg-type]