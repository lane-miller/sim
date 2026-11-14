"""Shared validation helpers (timing, windowing, spectra) for Stage A checks."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fdtd.config import DEFAULT_RAM_LIMIT_BYTES, Config, SQRT3, estimate_memory_bytes
from fdtd.grid import Grid

TUKEY_ALPHA = 0.25
END_TAPER_FRAC = TUKEY_ALPHA
# ``p_trace[i]`` is ``p^{i+1}`` (sampled after step *i*); ``q_trace[i]`` drives ``p^i → p^{i+1}``.
PRESSURE_TRACE_INDEX_SHIFT = -1
FFT_PAD_FACTOR = 8
WIN_SIGMA_SPAN = 5.0
WIN_TAIL_CELLS = 10.0
MIN_WINDOW_SAMPLES = 32
PULSE_DELAY_SIGMA_FACTOR = 4.0
SOURCE_BAND_DB = 30.0
DEFAULT_TAIL_CELLS = 20.0
ECHO_CLEARANCE_MARGIN = 1.15


def memory_estimate_cube(n: int, cpml_cells: int) -> int:
    return estimate_memory_bytes(n, n, n, cpml_cells)


class RamLimitExceeded(RuntimeError):
    """Planned domain exceeds the configured RAM budget."""

    def __init__(self, label: str, n: int, est_bytes: float, limit_bytes: float) -> None:
        self.label = label
        self.n = n
        self.est_bytes = est_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"{label}: grid {n}³ estimated {fmt_bytes(est_bytes)} "
            f"exceeds RAM limit {fmt_bytes(limit_bytes)}"
        )


def require_ram(
    n: int,
    cpml_cells: int,
    limit_bytes: float,
    *,
    label: str,
) -> int:
    """Abort before allocation if the cubic domain estimate exceeds *limit_bytes*."""
    est = memory_estimate_cube(n, cpml_cells)
    if est > limit_bytes:
        raise RamLimitExceeded(label, n, float(est), float(limit_bytes))
    return est


class RunTimer:
    """Wall-clock phases from script start."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.phases: dict[str, float] = {}
        self._phase_t0 = self.t0

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def banner(self, tag: str, detail: str = "") -> None:
        msg = f"[{tag}] +{self.elapsed():.1f}s"
        if detail:
            msg += f"  {detail}"
        print(msg, flush=True)

    def begin(self, phase: str) -> None:
        self._phase_t0 = time.perf_counter()

    def end(self, phase: str) -> float:
        dt = time.perf_counter() - self._phase_t0
        self.phases[phase] = dt
        return dt


def fmt_bytes(b: float | int) -> str:
    b = float(b)
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b / 1e3:.1f} kB"


def emit_sim_progress(line: str, *, tty: bool, final: bool) -> None:
    if tty and not final:
        print("\r" + line.ljust(96), end="", flush=True)
    else:
        print(line, flush=True)


def sync_dt(cfg: Config, d: object, cfl: float | None = None) -> float:
    cfl_safety = cfg.numerics.cfl_safety if cfl is None else cfl
    dt = cfl_safety * d.h / (cfg.air.c * SQRT3)
    d.dt = dt
    return dt


def pulse_q(t: float, sigma: float, t0: float) -> float:
    return float(np.exp(-0.5 * ((t - t0) / sigma) ** 2))


def apply_h_from_ppw(d: object, cfg: Config, *, f_ref: float, ppw: float) -> None:
    """Set ``h = c/(f_ref·ppw)`` (Stage A override; ignore radiator ``D/10`` cap)."""
    c = cfg.air.c
    h = c / (f_ref * ppw)
    d.h = h
    d.ppw_realised = ppw
    d.lambda_min = c / f_ref
    att = math.pow(10.0, -cfg.numerics.pulse_db / 20.0)
    d.pulse_sigma = math.sqrt(2.0 * math.log(att)) / (
        2.0 * math.pi * cfg.numerics.pulse_f_factor * d.f_max
    )
    d.pulse_t0 = PULSE_DELAY_SIGMA_FACTOR * d.pulse_sigma
    sync_dt(cfg, d)


def centre_index(n: int) -> int:
    return n // 2


def coords_m(g: Grid, i: int, j: int, k: int) -> tuple[float, float, float]:
    return g.coords_of(i, j, k)


def domain_half_extent_m(n: int, h: float) -> float:
    return 0.5 * n * h


def arrival_time_s(pulse_t0: float, r_m: float, c: float) -> float:
    return pulse_t0 + r_m / c


def first_reflection_time_s(
    g: Grid,
    cfg: Config,
    *,
    i_src: int,
    j_src: int,
    k_src: int,
    i_rec: int,
    j_rec: int,
    k_rec: int,
    margin: float = 0.9,
) -> float:
    """Earliest specular return from any domain face (before CPML contamination)."""
    c = cfg.air.c
    x_s, y_s, z_s = coords_m(g, i_src, j_src, k_src)
    x_r, y_r, z_r = coords_m(g, i_rec, j_rec, k_rec)
    hx = domain_half_extent_m(g.nx, g.h)
    hy = domain_half_extent_m(g.ny, g.h)
    hz = domain_half_extent_m(g.nz, g.h)
    times = [
        (x_s + hx + x_r + hx) / c,
        (hx - x_s + hx - x_r) / c,
        (y_s + hy + y_r + hy) / c,
        (hy - y_s + hy - y_r) / c,
        (z_s + hz + z_r + hz) / c,
        (hz - z_s + hz - z_r) / c,
    ]
    return margin * min(times)


def tukey(n: int, alpha: float = TUKEY_ALPHA) -> np.ndarray:
    """Symmetric Tukey (legacy); prefer :func:`causal_end_taper` for causal records."""
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    if alpha <= 0.0:
        return np.ones(n, dtype=np.float64)
    if alpha >= 1.0:
        return np.hanning(n).astype(np.float64)
    w = np.ones(n, dtype=np.float64)
    n_taper = int(max(1, math.floor(alpha * (n - 1) / 2.0)))
    x = np.linspace(0.0, math.pi / 2.0, n_taper, endpoint=False)
    taper = 0.5 * (1.0 - np.cos(x))
    w[:n_taper] = taper
    w[-n_taper:] = taper[::-1]
    return w


def causal_end_taper(
    n: int,
    *,
    taper_frac: float = END_TAPER_FRAC,
    n_taper: int | None = None,
) -> np.ndarray:
    """Unit gain at the start; raised-cosine taper on the last ``n_taper`` samples only."""
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    if taper_frac <= 0.0:
        return w
    nt = n_taper if n_taper is not None else int(max(1, math.floor(taper_frac * n)))
    nt = min(nt, n - 1)
    if nt <= 0:
        return w
    x = np.linspace(0.0, math.pi, nt)
    w[-nt:] = 0.5 * (1.0 + np.cos(x))
    return w


def trace_segment(
    trace: np.ndarray,
    i_start: int,
    i_end: int,
    *,
    index_shift: int = 0,
) -> np.ndarray:
    """Length ``i_end - i_start`` segment with optional index shift (out-of-range → 0)."""
    n_win = i_end - i_start
    seg = np.zeros(n_win, dtype=np.float64)
    for k in range(n_win):
        j = i_start + k + index_shift
        if 0 <= j < len(trace):
            seg[k] = float(trace[j])
    return seg


def print_leapfrog_timing_convention(*, dt: float) -> None:
    """Document indexing used for Q vs p spectra (no separate phase correction)."""
    print(
        "\n=== FDTD timing convention (C3-cpml) ===",
        flush=True,
    )
    print(
        "  At step i: t=i·Δt; Q(t) injected into p^i before step(); "
        "step() advances p^i→p^{i+1}; samples are p^{i+1}.",
        flush=True,
    )
    print(
        f"  q_trace[i] ↔ t=i·Δt; ⟨p⟩ trace is (⟨p⟩_pre+⟨p⟩_post)/2 at step i (paired with Q^i, "
        f"no index shift in Z FFTs).",
        flush=True,
    )
    print(
        f"  Receiver p traces use index_shift={PRESSURE_TRACE_INDEX_SHIFT} "
        f"so slot i aligns p^i (Δt={dt:.6e} s).",
        flush=True,
    )


def cube_geometric_centre_m(g: Grid, geom: object) -> tuple[float, float, float]:
    """Mean cell-centre position of the source cube (true centroid; even s may sit between cells)."""
    i0 = int(geom.i0)  # type: ignore[attr-defined]
    j0 = int(geom.j0)
    k0 = int(geom.k0)
    s = int(geom.s)
    sx = sy = sz = 0.0
    n = float(s**3)
    for i in range(i0, i0 + s):
        for j in range(j0, j0 + s):
            for k in range(k0, k0 + s):
                x, y, z = coords_m(g, i, j, k)
                sx += x
                sy += y
                sz += z
    return sx / n, sy / n, sz / n


def path_length_from_cube_centre_m(
    g: Grid, geom: object, i1: int, j1: int, k1: int
) -> float:
    xs, ys, zs = cube_geometric_centre_m(g, geom)
    xr, yr, zr = coords_m(g, i1, j1, k1)
    return math.hypot(xr - xs, yr - ys, zr - zs)


def fft_len(trace_len: int, i_end_max: int, pad_factor: int = FFT_PAD_FACTOR) -> int:
    return int(
        2 ** math.ceil(math.log2(max(trace_len, i_end_max * pad_factor, 64)))
    )


def comparison_n_fft(
    i_start: int,
    i_end: int,
    pad_factor: int = FFT_PAD_FACTOR,
) -> int:
    """FFT size for ref vs small comparison: pad to ≥ *pad_factor* × window length."""
    n_win = max(i_end - i_start, 1)
    return int(2 ** math.ceil(math.log2(max(64, n_win * pad_factor))))


def assert_same_length(
    *pairs: tuple[np.ndarray, str],
    context: str,
) -> None:
    """Require every array in *pairs* to share the same shape[0]."""
    if not pairs:
        return
    n0 = pairs[0][0].shape[0]
    for arr, name in pairs:
        if arr.shape[0] != n0:
            raise AssertionError(
                f"{context}: length mismatch — {pairs[0][1]} has {n0} bins, "
                f"{name} has {arr.shape[0]} bins"
            )


def spectrum_absolute(
    trace: np.ndarray,
    dt: float,
    *,
    i_start: int,
    i_end: int,
    n_fft: int,
    win: np.ndarray | None = None,
    index_shift: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Causal window on ``[i_start, i_end)``; flat start, cosine taper at end only."""
    n_win = i_end - i_start
    if win is None:
        win = causal_end_taper(n_win)
    elif win.shape[0] != n_win:
        raise ValueError(f"win length {win.shape[0]} != window {n_win}")
    buf = np.zeros(n_fft, dtype=np.float64)
    seg = trace_segment(trace, i_start, i_end, index_shift=index_shift) * win
    buf[i_start:i_end] = seg
    spec = np.fft.fft(buf, n=n_fft)
    freqs = np.fft.fftfreq(n_fft, dt)
    return freqs, spec, n_win


def paired_spectra_absolute(
    q_trace: np.ndarray,
    p_trace: np.ndarray,
    dt: float,
    *,
    i_start: int,
    i_end: int,
    n_fft: int,
    win: np.ndarray | None = None,
    q_index_shift: int = 0,
    p_index_shift: int = PRESSURE_TRACE_INDEX_SHIFT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identical causal window on Q and p; default shifts align leapfrog sample times."""
    n_win = i_end - i_start
    if win is None:
        win = causal_end_taper(n_win)
    freqs, q_spec, _ = spectrum_absolute(
        q_trace,
        dt,
        i_start=i_start,
        i_end=i_end,
        n_fft=n_fft,
        win=win,
        index_shift=q_index_shift,
    )
    _, p_spec, _ = spectrum_absolute(
        p_trace,
        dt,
        i_start=i_start,
        i_end=i_end,
        n_fft=n_fft,
        win=win,
        index_shift=p_index_shift,
    )
    assert_same_length(
        (freqs, "freqs"),
        (q_spec, "q_spec"),
        (p_spec, "p_spec"),
        context="paired_spectra_absolute",
    )
    return freqs, q_spec, p_spec, win


def source_spectrum_Q(
    dt: float,
    *,
    sigma: float,
    t0: float,
    i_start: int,
    i_end: int,
    n_fft: int,
    win: np.ndarray | None = None,
) -> np.ndarray:
    n = n_fft
    t = np.arange(n, dtype=np.float64) * dt
    q = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    if i_end > i_start:
        n_win = i_end - i_start
        w = win if win is not None else causal_end_taper(n_win)
        buf = np.zeros(n, dtype=np.float64)
        buf[i_start:i_end] = q[i_start:i_end] * w[:n_win]
        q = buf
    return np.fft.fft(q, n=n_fft)


def analysis_mask(
    freqs: np.ndarray,
    q_spec: np.ndarray,
    *,
    f_min: float,
    f_max: float,
    floor_db: float = SOURCE_BAND_DB,
) -> np.ndarray:
    mag = np.abs(q_spec)
    pos = freqs > 0.0
    pk = float(np.max(mag[pos])) if np.any(pos) else float(np.max(mag))
    floor = pk * 10.0 ** (-floor_db / 20.0)
    return pos & (freqs >= f_min) & (freqs <= f_max) & (mag >= floor)


def json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def to_json_safe(obj: Any, *, _seen: set[int] | None = None) -> Any:
    """Deep-convert to JSON-serializable plain scalars, lists, and dicts (no shared refs)."""
    if _seen is None:
        _seen = set()
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return float(obj) if isinstance(obj, float) else int(obj)
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        if isinstance(obj, np.bool_):
            return bool(obj)
        v = obj.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist(), _seen=_seen)
    if is_dataclass(obj):
        return to_json_safe(asdict(obj), _seen=_seen)
    oid = id(obj)
    if isinstance(obj, dict):
        if oid in _seen:
            return None
        _seen.add(oid)
        return {str(k): to_json_safe(v, _seen=_seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if oid in _seen:
            return None
        _seen.add(oid)
        return [to_json_safe(v, _seen=_seen) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def path_length_m(g: Grid, i0: int, j0: int, k0: int, i1: int, j1: int, k1: int) -> float:
    xs, ys, zs = coords_m(g, i0, j0, k0)
    xr, yr, zr = coords_m(g, i1, j1, k1)
    return math.hypot(xr - xs, yr - ys, zr - zs)


def radii_match_m(a: float, b: float, *, h: float) -> bool:
    """True if two path lengths (m) match within float noise at grid scale."""
    return bool(np.isclose(a, b, rtol=1e-6, atol=1e-6 * h)) # float32 coordinate rounding (~1e-7 rel); integer offsets are the exact check


def comparison_window_seconds(
    cfg: Config,
    d: object,
    g: Grid,
    *,
    i_src: int,
    j_src: int,
    k_src: int,
    receivers: tuple[tuple[int, int, int], ...],
    tail_cells: float = DEFAULT_TAIL_CELLS,
) -> tuple[float, float, float]:
    """Return ``(T_window, t_pulse_end, t_prop_max)`` in seconds.

    ``T_window = (t₀ + 6σ) + max receiver propagation + tail_cells·h/c`` (no ``1/f_min`` scaling).
    """
    c = cfg.air.c
    h = d.h
    sigma, t0 = d.pulse_sigma, d.pulse_t0
    t_pulse = t0 + 6.0 * sigma
    t_prop = 0.0
    for pt in receivers:
        r_m = path_length_m(g, i_src, j_src, k_src, pt[0], pt[1], pt[2])
        t_prop = max(t_prop, r_m / c)
    t_tail = tail_cells * h / c
    t_window = t_pulse + t_prop + t_tail
    return t_window, t_pulse, t_prop


def window_sample_range(
    d: object,
    *,
    t_window_s: float,
    trace_len: int | None = None,
) -> tuple[int, int]:
    dt = d.dt
    i0 = 0
    i1 = int(math.ceil(t_window_s / dt))
    if trace_len is not None:
        i1 = min(i1, trace_len)
    i1 = max(i1, i0 + MIN_WINDOW_SAMPLES)
    return i0, i1


def f_resolution_hz(t_window_s: float) -> float:
    if t_window_s <= 0.0:
        return float("inf")
    return 2.0 / t_window_s


def min_echo_time_s(
    g: Grid,
    cfg: Config,
    *,
    i_src: int,
    j_src: int,
    k_src: int,
    points: tuple[tuple[int, int, int], ...],
    margin: float = 0.9,
) -> float:
    """Earliest wall echo among round trips from source to each point."""
    times = [
        first_reflection_time_s(
            g,
            cfg,
            i_src=i_src,
            j_src=j_src,
            k_src=k_src,
            i_rec=pt[0],
            j_rec=pt[1],
            k_rec=pt[2],
            margin=margin,
        )
        for pt in points
    ]
    return min(times)


def print_runtime_summary(run_timer: RunTimer, phases: tuple[str, ...]) -> None:
    total = run_timer.elapsed()
    print("\n=== Runtime (wall) ===", flush=True)
    for name in phases:
        t = run_timer.phases.get(name)
        if t is not None:
            print(f"  {name:<10} {t:7.2f} s", flush=True)
    print(f"  {'total':<10} {total:7.2f} s", flush=True)
