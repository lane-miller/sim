"""Radiation impedance z(f) from recorded radiator-averaged pressure."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from fdtd.config import Config
from fdtd.post.spectra import spectrum
from fdtd.source import velocity_pulse

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

_F64 = np.float64
_C128 = np.complex128
_SOURCE_BAND_DB = 40.0
_DEFAULT_DF_HZ = 1.0
_END_TAPER_FRAC = 0.25


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError("h5py is required for fdtd.post.zrad")
    return h5py


def causal_end_taper(
    n: int,
    *,
    taper_frac: float = _END_TAPER_FRAC,
    n_taper: int | None = None,
) -> np.ndarray:
    """Unit gain at the start; raised-cosine taper on the last samples only."""
    if n <= 1:
        return np.ones(max(n, 1), dtype=_F64)
    w = np.ones(n, dtype=_F64)
    if taper_frac <= 0.0:
        return w
    nt = n_taper if n_taper is not None else int(max(1, math.floor(taper_frac * n)))
    nt = min(nt, n - 1)
    if nt <= 0:
        return w
    x = np.linspace(0.0, math.pi, nt)
    w[-nt:] = 0.5 * (1.0 + np.cos(x))
    return w


def _load_meta(path: Path) -> dict[str, Any]:
    h5 = _require_h5py()
    with h5.File(path, "r") as f:
        meta = f["meta"]
        dt = float(meta.attrs["dt"])
        cfg = Config.from_json(bytes(f["meta/config_json"][()]).decode("utf-8"))
        derived = json.loads(bytes(f["meta/derived_json"][()]).decode("utf-8"))
        p_avg = np.asarray(f["radiator/p_avg"][()], dtype=_F64)
        rad = f["radiator"]
        if "times_p" in rad:
            times = np.asarray(rad["times_p"][()], dtype=_F64)
        else:
            raise ValueError("radiator/times_p missing; cannot infer pressure sample times")
    if times.shape[0] != p_avg.shape[0]:
        raise ValueError(
            f"p_avg length {p_avg.shape[0]} != radiator/times_p length {times.shape[0]}"
        )
    return {
        "dt": dt,
        "cfg": cfg,
        "derived": derived,
        "p_avg": p_avg,
        "times": times,
    }


def _n_fft_for_df(n_samples: int, dt: float, df_hz: float) -> int:
    span = max(n_samples * dt, dt)
    n_min = int(math.ceil(1.0 / (df_hz * dt)))
    return int(max(n_samples, n_min, 64))


def _pad_for_fft(times: np.ndarray, x: np.ndarray, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad ``x`` and extend ``times`` with the final sample spacing."""
    n = int(times.size)
    t_pad = np.empty(n_fft, dtype=_F64)
    x_pad = np.zeros(n_fft, dtype=_F64)
    t_pad[:n] = times
    x_pad[:n] = x
    if n_fft > n:
        step = float(times[-1] - times[-2]) if n > 1 else float(times[0])
        t_pad[n:] = times[-1] + step * np.arange(1, n_fft - n + 1, dtype=_F64)
    return x_pad, t_pad


def zrad_from_h5(
    path: str | Path,
    f: np.ndarray | None = None,
    *,
    df_hz: float = _DEFAULT_DF_HZ,
    use_taper: bool = True,
    time_shift_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(f, z)`` with ``z = ⟨p⟩ / (ρ c V)`` (brief §1).

    Spectra use :func:`~fdtd.post.spectra.spectrum` (Δt-weighted sum at stored
    ``radiator/times_p``). ``V(f)`` is the spectrum of ``velocity_pulse`` at those
    same times (not the analytic continuous ``pulse_spectrum``).
    """
    path = Path(path)
    blob = _load_meta(path)
    rho = float(blob["cfg"].air.rho)
    c = float(blob["cfg"].air.c)
    sigma = float(blob["derived"]["pulse_sigma"])
    t0 = float(blob["derived"]["pulse_t0"])
    p_avg = blob["p_avg"]
    times = blob["times"]
    n = int(p_avg.shape[0])
    if n == 0:
        return np.empty(0, _F64), np.empty(0, _C128)

    win = causal_end_taper(n) if use_taper else np.ones(n, dtype=_F64)
    dt_med = float(np.median(np.diff(times))) if n > 1 else float(blob["dt"])
    n_fft = _n_fft_for_df(n, dt_med, df_hz)
    v_time = np.array([velocity_pulse(float(t), sigma, t0) for t in times], dtype=_F64)
    buf_p, t_pad = _pad_for_fft(times, p_avg * win, n_fft)
    buf_v, _ = _pad_for_fft(times, v_time * win, n_fft)

    f_rfft = np.fft.rfftfreq(n_fft, dt_med)
    pos = f_rfft > 0.0
    f_pos = f_rfft[pos]
    p_spec = np.asarray(spectrum(buf_p, t_pad, f=None, taper=None), dtype=_C128)[pos]
    v_spec = np.asarray(spectrum(buf_v, t_pad, f=None, taper=None), dtype=_C128)[pos]

    if time_shift_s != 0.0:
        omega = 2.0 * math.pi * f_pos
        p_spec = p_spec * np.exp(1j * omega * float(time_shift_s))

    denom = rho * c * v_spec
    z_spec = np.full(f_pos.shape[0], np.nan + 0j, dtype=_C128)
    mag_v = np.abs(v_spec)
    pk = float(np.max(mag_v)) if mag_v.size else 0.0
    floor = pk * 10.0 ** (-_SOURCE_BAND_DB / 20.0)
    band = mag_v >= floor
    z_spec[band] = p_spec[band] / denom[band]

    if f is None:
        return f_pos, z_spec
    f_arr = np.asarray(f, dtype=_F64)
    z_out = np.full(f_arr.shape, np.nan + 0j, dtype=np.complex128)
    for i, fz in enumerate(f_arr.flat):
        j = int(np.argmin(np.abs(f_pos - float(fz))))
        if f_pos[j] > 0.0 and mag_v[j] >= floor:
            z_out.flat[i] = z_spec[j]
    return f_arr, z_out.reshape(np.asarray(f, dtype=_F64).shape)