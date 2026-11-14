"""Synthetic HDF5 tests for fdtd.post.zrad (no FDTD run)."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

h5py = pytest.importorskip("h5py")

from fdtd.config import Air, Band, Box, Config, Numerics, Output, Radiator, derive  # noqa: E402
from fdtd.post.zrad import zrad_from_h5  # noqa: E402
from fdtd.recorder import velocity_pulse_derivative  # noqa: E402
from fdtd.source import pulse_spectrum, velocity_pulse  # noqa: E402


def _write_synthetic_h5(path: Path, p_avg: np.ndarray, cfg: Config, derived) -> None:
    n = int(p_avg.shape[0])
    dt = float(derived.dt)
    times = (np.arange(n, dtype=np.float64) + 1.0) * dt
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["dt"] = dt
        meta.attrs["h"] = derived.h
        meta.create_dataset(
            "config_json",
            data=np.asarray(cfg.to_json(), dtype="S"),
        )
        meta.create_dataset(
            "derived_json",
            data=np.asarray(json.dumps(asdict(derived)), dtype="S"),
        )
        rad = f.create_group("radiator")
        rad.create_dataset("times_p", data=times)
        rad.create_dataset("p_avg", data=p_avg.astype(np.float64))
        rad.create_dataset("v_pulse_integer", data=np.zeros(n))
        rad.create_dataset("v_pulse_half", data=np.zeros(n))


def _base_cfg() -> Config:
    return Config(
        air=Air(),
        box=Box(W=0.2, H=0.2, D=0.2),
        radiator=Radiator("circle", dims=[0.1]),
        band=Band(f_min=20.0, f_max=1500.0),
        numerics=Numerics(cell_budget=10**9),
        output=Output(path=""),
    )


def _in_band_mask(freqs: np.ndarray, *, dt: float, sigma: float, t0: float) -> np.ndarray:
    v_spec = pulse_spectrum(freqs, sigma, t0)
    mag = np.abs(v_spec)
    pk = float(np.max(mag))
    floor = pk * 10.0 ** (-40.0 / 20.0)
    return (freqs > 0.0) & (mag >= floor)


def test_zrad_constant_impedance():
    cfg = _base_cfg()
    d = derive(cfg)
    rho, c = cfg.air.rho, cfg.air.c
    z0 = 0.37
    n = 4096
    times = (np.arange(n, dtype=np.float64) + 1.0) * d.dt
    v = np.array([velocity_pulse(float(t), d.pulse_sigma, d.pulse_t0) for t in times])
    p_avg = rho * c * z0 * v

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "syn.h5"
        _write_synthetic_h5(path, p_avg, cfg, d)
        freqs, z = zrad_from_h5(path, df_hz=0.5)
        mask = _in_band_mask(freqs, dt=d.dt, sigma=d.pulse_sigma, t0=d.pulse_t0)
        rel = np.abs(z[mask] - z0) / z0
        assert np.all(rel < 1e-6)


def test_zrad_derivative_impedance():
    cfg = _base_cfg()
    d = derive(cfg)
    rho, c = cfg.air.rho, cfg.air.c
    L = 0.08
    n = 4096
    times = (np.arange(n, dtype=np.float64) + 1.0) * d.dt
    dvdt = np.array(
        [
            velocity_pulse_derivative(float(t), d.pulse_sigma, d.pulse_t0)
            for t in times
        ]
    )
    p_avg = rho * L * dvdt

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "syn.h5"
        _write_synthetic_h5(path, p_avg, cfg, d)
        freqs, z = zrad_from_h5(path, df_hz=0.5)
        mask = _in_band_mask(freqs, dt=d.dt, sigma=d.pulse_sigma, t0=d.pulse_t0)
        k = 2.0 * np.pi * freqs[mask] / c
        z_ref = 1j * k * L
        rel = np.abs(z[mask] - z_ref) / np.max(np.abs(z_ref))
        assert np.all(rel < 1e-6)


def test_zrad_time_shift_phase():
    cfg = _base_cfg()
    d = derive(cfg)
    rho, c = cfg.air.rho, cfg.air.c
    z0 = 0.42
    n = 2048
    times = (np.arange(n, dtype=np.float64) + 1.0) * d.dt
    v = np.array([velocity_pulse(float(t), d.pulse_sigma, d.pulse_t0) for t in times])
    p_avg = rho * c * z0 * v
    shift = 0.5 * d.dt

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "syn.h5"
        _write_synthetic_h5(path, p_avg, cfg, d)
        f0, z0_spec = zrad_from_h5(path, df_hz=1.0, time_shift_s=0.0)
        _, z1_spec = zrad_from_h5(path, df_hz=1.0, time_shift_s=shift)
        mask = _in_band_mask(f0, dt=d.dt, sigma=d.pulse_sigma, t0=d.pulse_t0)
        omega = 2.0 * np.pi * f0[mask]
        expected = z0_spec[mask] * np.exp(1j * omega * shift)
        rel = np.abs(z1_spec[mask] - expected) / np.maximum(np.abs(expected), 1e-30)
        assert np.all(rel < 1e-6)