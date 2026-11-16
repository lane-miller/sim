"""Tests for boxrad.post.spectra."""

from __future__ import annotations

import numpy as np
import pytest

from boxrad.post.spectra import (
    _DEFAULT_MEMORY_CAP,
    sample_dt_weights,
    spectrum,
    spectrum_rows,
)
from boxrad.source import velocity_pulse


def test_spectrum_fft_matches_direct_uniform():
    n = 128
    dt = 2e-5
    times = (np.arange(n, dtype=np.float64) + 1.0) * dt
    sigma, t0 = 0.0008, 6.0 * 0.0008
    x = np.array([velocity_pulse(float(t), sigma, t0) for t in times], dtype=np.float64)
    freqs = np.array([300.0, 1200.0, 3400.0], dtype=np.float64)
    x_fft = spectrum(x, times, freqs)
    dt_w = sample_dt_weights(times)
    direct = np.empty(freqs.size, dtype=np.complex128)
    for i, f in enumerate(freqs):
        phase = np.exp(-1j * 2.0 * np.pi * f * times)
        direct[i] = np.sum(x * dt_w * phase)
    rel = np.abs(x_fft - direct) / np.maximum(np.abs(direct), 1e-30)
    assert np.all(rel < 1e-10)


def test_spectrum_uniform_shifted_t0_phase():
    n = 64
    dt = 1e-4
    t0_shift = 7.5 * dt
    times = t0_shift + np.arange(n, dtype=np.float64) * dt
    f_hz = 250.0
    x = np.cos(2.0 * np.pi * f_hz * times)
    X = complex(spectrum(x, times, f_hz))
    dt_w = sample_dt_weights(times)
    direct = np.sum(x * dt_w * np.exp(-1j * 2.0 * np.pi * f_hz * times))
    rel = abs(X - direct) / max(abs(direct), 1e-30)
    assert rel < 1e-10


def test_spectrum_full_grid_large_n_within_cap():
    n_t = 2**16
    dt = 1e-5
    times = (np.arange(n_t, dtype=np.float64) + 1.0) * dt
    x = np.sin(2.0 * np.pi * 180.0 * times) * np.exp(-times / 0.02)
    cap = _DEFAULT_MEMORY_CAP
    X = np.asarray(spectrum(x, times, f=None, memory_cap_bytes=cap), dtype=np.complex128)
    assert X.size == n_t // 2 + 1
    k = int(round(180.0 * n_t * dt))
    assert abs(X[k]) > 0.0


def test_spectrum_memory_error_reports_size():
    n = 4096
    dt = 1e-5
    times = np.arange(n, dtype=np.float64) * dt
    x = np.ones(n, dtype=np.float64)
    cap = 1024
    with pytest.raises(MemoryError, match="bytes"):
        spectrum(x, times, f=None, memory_cap_bytes=cap)


def test_spectrum_rows_scalar_frequency():
    n, m = 32, 5
    dt = 1e-4
    times = (np.arange(n, dtype=np.float64) + 1.0) * dt
    x = np.random.default_rng(0).standard_normal((n, m))
    f = 500.0
    row = np.asarray(spectrum_rows(x, times, f), dtype=np.complex128)
    assert row.shape == (m,)
    cols = np.array([complex(spectrum(x[:, j], times, f)) for j in range(m)])
    rel = np.abs(row - cols) / np.maximum(np.abs(cols), 1e-30)
    assert np.all(rel < 1e-10)


def test_spectrum_matches_fft_dt_for_uniform_shifted_times():
    n = 256
    dt = 2e-5
    times = (np.arange(n, dtype=np.float64) + 1.0) * dt
    sigma, t0 = 0.0008, 6.0 * 0.0008
    x = np.array([velocity_pulse(float(t), sigma, t0) for t in times], dtype=np.float64)
    f = 500.0
    x_d = spectrum(x, times, f)
    dt_w = sample_dt_weights(times)
    ref = np.sum(x * dt_w * np.exp(-1j * 2.0 * np.pi * f * times))
    assert abs(x_d - ref) / max(abs(ref), 1e-30) < 1e-10


def test_spectrum_exponential_mode():
    tau = 0.001
    dt = 1e-4
    times = (np.arange(400, dtype=np.float64) + 1.0) * dt
    x = np.exp(-times / tau)
    f = 80.0
    X = spectrum(x, times, f)
    Xa = tau / (1.0 + 1j * 2.0 * np.pi * f * tau)
    dt_w = sample_dt_weights(times)
    direct = np.sum(x * dt_w * np.exp(-1j * 2.0 * np.pi * f * times))
    rel = abs(X - direct) / max(abs(direct), 1e-30)
    assert rel < 1e-10