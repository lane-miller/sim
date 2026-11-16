"""Unit tests for boxrad.runner (driver and dry run)."""

from __future__ import annotations

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
    derive,
)
from boxrad.kernels import step as kernel_step  # noqa: E402
from boxrad.recorder import decimation_factor  # noqa: E402
from boxrad.runner import run  # noqa: E402
from boxrad.source import velocity_pulse  # noqa: E402


def _tiny_config() -> Config:
    return Config(
        box=Box(W=0.06, H=0.06, D=0.05),
        radiator=Radiator("circle", dims=[0.03], x_c=0.0, y_c=0.0),
        band=Band(f_min=20.0, f_max=4_000.0),
        numerics=Numerics(
            gap_cells=8,
            cpml_cells=8,
            ppw=8.0,
            cells_across_radiator=6,
            cell_budget=10**9,
            cfl_safety=0.99,
            n_transits=0.08,
        ),
        output=Output(path=""),
    )


@pytest.fixture
def patch_few_steps(monkeypatch):
    n_steps = 12
    real_derive = derive

    def wrapped(cfg: Config):
        d = real_derive(cfg)
        d.n_steps = n_steps
        d.run_time_s = n_steps * d.dt
        return d

    monkeypatch.setattr("boxrad.runner.derive", wrapped)
    return n_steps


def test_run_hdf5_structure(patch_few_steps):
    cfg = _tiny_config()
    n_steps = patch_few_steps
    d = derive(cfg)
    m = decimation_factor(d.f_max, d.dt)
    n_dec = (n_steps + m - 1) // m

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.h5"
        out = run(cfg, out_path=path, progress=False)
        assert out == path
        assert path.is_file()

        with h5py.File(path, "r") as f:
            assert "meta" in f and "radiator" in f and "huygens" in f and "box_surface" in f
            assert f["radiator/v_pulse_integer"].shape == (n_steps,)
            assert f["radiator/v_pulse_half"].shape == (n_steps,)
            assert f["radiator/p_avg"].shape == (n_steps,)
            assert f["huygens/p_face"].shape[0] == n_dec
            assert f["box_surface/p"].shape[0] == n_dec
            assert f["meta"].attrs["n_steps"] == n_steps
            assert f["meta"].attrs["decimation_factor"] == m
            assert "wall_time_s" in f["meta"].attrs
            assert "git_commit" in f["meta"].attrs
            assert "tail_db" in f["meta"].attrs
            assert "boxrad_version" in f["meta"].attrs
            assert "results/zrad" in f
            assert "meta/accuracy" in f


def test_dry_run_no_file(monkeypatch):
    cfg = _tiny_config()

    def _no_alloc(*args, **kwargs):
        raise AssertionError("allocate_fields called during dry_run")

    monkeypatch.setattr("boxrad.runner.allocate_fields", _no_alloc)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "would.h5"
        summary = run(cfg, out_path=path, dry_run=True, progress=False)
        assert isinstance(summary, dict)
        assert summary["n_steps"] > 0
        assert not path.exists()


def test_validation_before_output():
    cfg = _tiny_config()
    cfg.radiator = Radiator("circle", dims=[0.03], x_c=0.029, y_c=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.h5"
        with pytest.raises(ValueError, match="clearance|radiator"):
            run(cfg, out_path=path, progress=False)
        assert not path.exists()


def test_stability_guard_aborts(monkeypatch):
    cfg = _tiny_config()
    real_derive = derive

    def wrapped(c: Config):
        d = real_derive(c)
        d.n_steps = 8
        d.run_time_s = 8 * d.dt
        return d

    monkeypatch.setattr("boxrad.runner.derive", wrapped)
    monkeypatch.setattr("boxrad.runner.STABILITY_CHECK_INTERVAL", 1)

    orig = kernel_step

    def nan_step(fields, coeffs, slabs, cpml, boundary, n):
        orig(fields, coeffs, slabs, cpml, boundary, n)
        fields.p[0, 0, 0] = np.float32(np.nan)

    monkeypatch.setattr("boxrad.runner.step", nan_step)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "abort.h5"
        with pytest.raises(RuntimeError, match="stability guard"):
            run(cfg, out_path=path, progress=False)
        assert path.is_file()
        with h5py.File(path, "r") as f:
            assert bool(f["meta"].attrs.get("aborted"))
            assert "aborted_at_step" in f["meta"].attrs


def test_deterministic_p_avg(patch_few_steps):
    cfg = _tiny_config()
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.h5"
        p2 = Path(tmp) / "b.h5"
        run(cfg, out_path=p1, progress=False)
        run(cfg, out_path=p2, progress=False)
        with h5py.File(p1, "r") as f1, h5py.File(p2, "r") as f2:
            a = f1["radiator/p_avg"][()]
            b = f2["radiator/p_avg"][()]
            np.testing.assert_array_equal(a, b)


def test_v_pulse_half_matches_source(patch_few_steps):
    cfg = _tiny_config()
    d = derive(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v.h5"
        run(cfg, out_path=path, progress=False)
        with h5py.File(path, "r") as f:
            v_half = f["radiator/v_pulse_half"][()]
            for n, val in enumerate(v_half):
                t_half = (float(n) + 0.5) * d.dt
                ref = velocity_pulse(t_half, d.pulse_sigma, d.pulse_t0)
                assert val == pytest.approx(ref, rel=0, abs=1e-12)