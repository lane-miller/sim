"""Discrete-time spectra with explicit sample times (e^{+jωt} convention)."""

from __future__ import annotations

import math

import numpy as np

_F64 = np.float64
_C128 = np.complex128

_DIRECT_SUM_MAX_F = 64
_CHUNK_TEMP_BYTES = 64 * 1024 * 1024
_DEFAULT_MEMORY_CAP = 512 * 1024 * 1024
_COMPLEX128_BYTES = 16
_UNIFORM_DT_RTOL = 1e-9


def _check_temp_bytes(nbytes: int, cap: int, *, what: str) -> None:
    if nbytes > cap:
        raise MemoryError(f"{what} would allocate {nbytes} bytes (cap {cap})")


def uniform_dt(times: np.ndarray) -> tuple[float, float] | None:
    """Return ``(t_0, Δt)`` when samples are uniform; else ``None``."""
    t = np.asarray(times, dtype=_F64).ravel()
    if t.size < 2:
        return float(t[0]) if t.size == 1 else (0.0, 0.0)
    diffs = np.diff(t)
    dt = float(diffs[0])
    if abs(dt) < np.finfo(_F64).tiny:
        if np.max(np.abs(diffs)) <= _UNIFORM_DT_RTOL * np.finfo(_F64).eps:
            return float(t[0]), dt
        return None
    if float(np.max(np.abs(diffs - dt))) > _UNIFORM_DT_RTOL * abs(dt):
        return None
    return float(t[0]), dt


def sample_dt_weights(times: np.ndarray) -> np.ndarray:
    """Trapezoidal Δt_n weights for samples at ``times`` (length N)."""
    t = np.asarray(times, dtype=_F64).ravel()
    n = t.size
    if n == 0:
        return np.empty(0, dtype=_F64)
    if n == 1:
        return np.ones(1, dtype=_F64)
    dt = np.empty(n, dtype=_F64)
    diffs = np.diff(t)
    dt[0] = diffs[0]
    dt[-1] = diffs[-1]
    if n > 2:
        dt[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return dt


def _n_pad_uniform(n: int, dt: float, df: float | None) -> int:
    if df is None or df <= 0.0:
        return n
    n_min = int(math.ceil(1.0 / (df * dt)))
    return int(max(n, n_min, 1))


def _n_pad_for_frequencies(
    n: int, dt: float, f_arr: np.ndarray, df: float | None
) -> int:
    """Choose zero-pad length so ``f_arr`` lies on the rfft grid when possible."""
    n_pad = _n_pad_uniform(n, dt, df)
    if f_arr.size == 0:
        return n_pad
    if _freq_bins_on_rfft_grid(f_arr, n_pad, dt) is not None:
        return n_pad
    f_sorted = np.sort(f_arr)
    diffs = np.diff(f_sorted)
    pos = diffs[diffs > 0.0]
    if pos.size:
        for df_est in (float(np.min(pos)), float(np.median(pos))):
            n_try = _n_pad_uniform(n, dt, df_est)
            for n_cand in (n_try, n_try - 1, n_try + 1):
                if n_cand >= n and _freq_bins_on_rfft_grid(f_arr, n_cand, dt) is not None:
                    return n_cand
    return n_pad


def _spectrum_fft_uniform(
    x: np.ndarray,
    t0: float,
    dt: float,
    n_pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``X_k = Δt · e^{−jω_k t_0} · rfft(x, n_pad)`` on the rfft frequency grid."""
    x_pad = np.zeros(n_pad, dtype=_F64)
    n = min(int(x.size), n_pad)
    x_pad[:n] = x[:n]
    spec = np.fft.rfft(x_pad, n=n_pad)
    f = np.fft.rfftfreq(n_pad, dt)
    phase = np.exp(-1j * 2.0 * np.pi * f * t0)
    return (dt * phase * spec).astype(_C128), f


def _freq_bins_on_rfft_grid(
    f_arr: np.ndarray, n_pad: int, dt: float
) -> np.ndarray | None:
    """Integer rfft bin indices if every ``f`` lies on ``k/(n_pad·Δt)`` (interpolate-free)."""
    if f_arr.size == 0:
        return np.empty(0, dtype=np.intp)
    inv = n_pad * dt
    if inv <= 0.0:
        return None
    k = np.rint(f_arr * inv)
    f_grid = k / inv
    rtol = _UNIFORM_DT_RTOL
    ok = np.abs(f_arr - f_grid) <= rtol * np.maximum(np.abs(f_arr), 1.0)
    if not np.all(ok):
        return None
    k_i = k.astype(np.intp)
    n_bins = n_pad // 2 + 1
    if np.any(k_i < 0) or np.any(k_i >= n_bins):
        return None
    return k_i


def _direct_sum(
    x: np.ndarray,
    t: np.ndarray,
    dt_w: np.ndarray,
    f_arr: np.ndarray,
    *,
    memory_cap_bytes: int,
) -> np.ndarray:
    denom = max(t.size * _COMPLEX128_BYTES, 1)
    chunk_f = max(1, min(f_arr.size, _CHUNK_TEMP_BYTES // denom, memory_cap_bytes // denom))
    if chunk_f < 1:
        nbytes = t.size * _COMPLEX128_BYTES
        raise MemoryError(
            f"spectrum phase row would allocate {nbytes} bytes (cap {memory_cap_bytes})"
        )
    weights = x * dt_w
    out = np.empty(f_arr.size, dtype=_C128)
    for i0 in range(0, f_arr.size, chunk_f):
        i1 = min(i0 + chunk_f, f_arr.size)
        f_chunk = f_arr[i0:i1]
        nbytes = f_chunk.size * t.size * _COMPLEX128_BYTES
        _check_temp_bytes(nbytes, memory_cap_bytes, what="spectrum phase chunk")
        phase = np.exp(-1j * 2.0 * np.pi * f_chunk[:, np.newaxis] * t[np.newaxis, :])
        out[i0:i1] = phase @ weights
    return out


def spectrum(
    x: np.ndarray,
    times: np.ndarray,
    f: np.ndarray | float | None = None,
    *,
    df: float | None = None,
    taper: np.ndarray | None = None,
    memory_cap_bytes: int = _DEFAULT_MEMORY_CAP,
) -> np.ndarray | complex:
    """Discrete spectrum ``X(f) ≈ Σ_n x(t_n) e^{-jω t_n} Δt_n``.

    For **uniform** sample times (``max |Δt_n − Δt| ≤ 10^{-9} Δt``), the exact
    relation ``X(f_k) = Δt · e^{-j2π f_k t_0} · \\mathrm{rfft}(x, n_{pad})_k`` holds
    with ``t_0 = times[0]`` and ``f_k = k/(n_{pad} Δt)``. Pass ``f is None`` to
    return the full positive rfft grid (no ``n_f × n_t`` temporary). Requested
    frequencies that lie on that grid are taken by bin index without interpolation.

    Off-grid requests on a uniform grid allow at most 64 frequencies (chunked
    direct sum, ~64 MiB temporaries). Non-uniform times use a chunked direct sum
    for any length. Exceeding ``memory_cap_bytes`` raises :class:`MemoryError``
    instead of allocating.
    """
    x_arr = np.asarray(x, dtype=_F64).ravel()
    t_arr = np.asarray(times, dtype=_F64).ravel()
    if x_arr.shape[0] != t_arr.shape[0]:
        raise ValueError(f"x length {x_arr.shape[0]} != times length {t_arr.shape[0]}")
    if taper is not None:
        w = np.asarray(taper, dtype=_F64).ravel()
        if w.shape[0] != x_arr.shape[0]:
            raise ValueError("taper length mismatch")
        x_arr = x_arr * w

    uniform = uniform_dt(t_arr)
    scalar_in = f is not None and np.isscalar(f)
    f_none = f is None

    if uniform is not None:
        t0, dt = uniform
        f_for_pad = np.empty(0, dtype=_F64) if f_none else np.atleast_1d(np.asarray(f, dtype=_F64))
        n_pad = _n_pad_for_frequencies(x_arr.size, dt, f_for_pad, df)
        full, f_grid = _spectrum_fft_uniform(x_arr, t0, dt, n_pad)
        if f_none:
            _check_temp_bytes(
                full.size * _COMPLEX128_BYTES, memory_cap_bytes, what="spectrum output"
            )
            return full
        f_arr = np.atleast_1d(np.asarray(f, dtype=_F64))
        bins = _freq_bins_on_rfft_grid(f_arr, n_pad, dt)
        if bins is not None:
            out = full[bins]
            if scalar_in:
                return complex(out[0])
            return out
        if f_arr.size > _DIRECT_SUM_MAX_F:
            raise MemoryError(
                f"{f_arr.size} off-grid frequencies on a uniform grid; use f=None "
                f"for the FFT grid, on-grid f, or at most {_DIRECT_SUM_MAX_F} explicit bins"
            )
        dt_w = sample_dt_weights(t_arr)
        out = _direct_sum(
            x_arr, t_arr, dt_w, f_arr, memory_cap_bytes=memory_cap_bytes
        )
        if scalar_in:
            return complex(out[0])
        return out

    if f_none:
        raise ValueError("f=None requires uniformly spaced times")

    f_arr = np.atleast_1d(np.asarray(f, dtype=_F64))
    dt_w = sample_dt_weights(t_arr)
    out = _direct_sum(x_arr, t_arr, dt_w, f_arr, memory_cap_bytes=memory_cap_bytes)
    if scalar_in:
        return complex(out[0])
    return out


def _spectrum_rows_fft(
    x_in: np.ndarray,
    t0: float,
    dt: float,
    n_pad: int,
    f_arr: np.ndarray,
    *,
    memory_cap_bytes: int,
) -> np.ndarray:
    """rfft along time axis (rows), shape ``(F, M)`` or ``(M,)`` for one bin."""
    n_t, n_m = x_in.shape
    x_pad = np.zeros((n_pad, n_m), dtype=_F64)
    n = min(n_t, n_pad)
    x_pad[:n, :] = x_in[:n, :]
    spec = np.fft.rfft(x_pad, n=n_pad, axis=0)
    f_grid = np.fft.rfftfreq(n_pad, dt)
    phase = np.exp(-1j * 2.0 * np.pi * f_grid * t0)[:, np.newaxis]
    full = (dt * phase * spec).astype(_C128)
    bins = _freq_bins_on_rfft_grid(f_arr, n_pad, dt)
    if bins is None:
        raise ValueError("spectrum_rows requires frequencies on the uniform FFT grid")
    if f_arr.size == 1:
        return full[bins[0], :]
    return full[bins, :]


def _spectrum_rows_direct(
    x_in: np.ndarray,
    t_arr: np.ndarray,
    dt_w: np.ndarray,
    f_arr: np.ndarray,
    *,
    memory_cap_bytes: int,
) -> np.ndarray:
    n_t, n_m = x_in.shape
    n_f = f_arr.size
    row_chunk = max(
        1,
        min(
            n_m,
            _CHUNK_TEMP_BYTES // max(n_t * _COMPLEX128_BYTES, 1),
        ),
    )
    f_chunk = max(
        1,
        min(
            n_f,
            _CHUNK_TEMP_BYTES // max(n_t * _COMPLEX128_BYTES, 1),
        ),
    )
    out = np.empty((n_f, n_m), dtype=_C128)
    xw = x_in * dt_w[:, np.newaxis]
    for j0 in range(0, n_m, row_chunk):
        j1 = min(j0 + row_chunk, n_m)
        block = xw[:, j0:j1]
        for i0 in range(0, n_f, f_chunk):
            i1 = min(i0 + f_chunk, n_f)
            f_c = f_arr[i0:i1]
            nbytes = f_c.size * n_t * _COMPLEX128_BYTES
            _check_temp_bytes(nbytes, memory_cap_bytes, what="spectrum_rows phase chunk")
            phase = np.exp(-1j * 2.0 * np.pi * f_c[:, np.newaxis] * t_arr[np.newaxis, :])
            out[i0:i1, j0:j1] = phase @ block
    if n_f == 1:
        return out[0]
    return out


def spectrum_rows(
    x: np.ndarray,
    times: np.ndarray,
    f: np.ndarray | float,
    *,
    taper: np.ndarray | None = None,
    memory_cap_bytes: int = _DEFAULT_MEMORY_CAP,
) -> np.ndarray:
    """Like :func:`spectrum` for ``x`` shaped ``(N,)`` or ``(N, M)`` → ``(F,)`` or ``(F, M)``.

    Requires an explicit frequency list (no full-grid evaluation). For uniform
    ``times``, uses an FFT along rows when every ``f`` lies on the rfft grid;
    otherwise a chunked direct sum. Scalar ``f`` returns shape ``(M,)``.
    """
    if f is None:
        raise ValueError("spectrum_rows does not support f=None; pass explicit frequencies")
    x_in = np.asarray(x, dtype=_F64)
    if x_in.ndim == 1:
        return np.asarray(
            spectrum(x_in, times, f, taper=taper, memory_cap_bytes=memory_cap_bytes),
            dtype=_C128,
        )
    if x_in.ndim != 2:
        raise ValueError("x must be 1-D or 2-D")
    t_arr = np.asarray(times, dtype=_F64).ravel()
    if x_in.shape[0] != t_arr.size:
        raise ValueError("x rows must match times length")
    if taper is not None:
        w = np.asarray(taper, dtype=_F64).ravel()
        x_in = x_in * w[:, np.newaxis]

    f_arr = np.atleast_1d(np.asarray(f, dtype=_F64))
    uniform = uniform_dt(t_arr)
    if uniform is not None:
        t0, dt = uniform
        n_pad = _n_pad_for_frequencies(x_in.shape[0], dt, f_arr, None)
        bins = _freq_bins_on_rfft_grid(f_arr, n_pad, dt)
        if bins is not None:
            return _spectrum_rows_fft(
                x_in, t0, dt, n_pad, f_arr, memory_cap_bytes=memory_cap_bytes
            )

    dt_w = sample_dt_weights(t_arr)
    return _spectrum_rows_direct(
        x_in, t_arr, dt_w, f_arr, memory_cap_bytes=memory_cap_bytes
    )