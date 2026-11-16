"""Shared validation helpers (timing, windowing, spectra) for Stage A checks."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable, TypeVar

import numpy as np

_RunKey = TypeVar("_RunKey")

from boxrad.config import DEFAULT_RAM_LIMIT_BYTES, Config, SQRT3, estimate_memory_bytes
from boxrad.grid import Grid

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
_H5_DISK_MAX_FRAC = 0.5
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class H5DiskLimitExceeded(RuntimeError):
    """Planned HDF5 output would exceed the disk budget."""

    def __init__(
        self,
        label: str,
        planned_bytes: float,
        free_bytes: float,
        parent: Path,
    ) -> None:
        self.label = label
        self.planned_bytes = planned_bytes
        self.free_bytes = free_bytes
        self.parent = parent
        super().__init__(
            f"{label}: planned HDF5 {fmt_bytes(planned_bytes)} exceeds "
            f"{_H5_DISK_MAX_FRAC:.0%} of free space on {parent} "
            f"({fmt_bytes(free_bytes)} free)"
        )


def _h5_bytes_under(root: Path) -> int:
    total = 0
    for p in root.rglob("*.h5"):
        if p.is_file():
            total += p.stat().st_size
    return total


def workdir_disk_parent(*, keep: bool) -> Path:
    """Directory whose free space is checked before a validation workdir is created."""
    if keep:
        return _PROJECT_ROOT / "outputs"
    return Path(tempfile.gettempdir())


def planned_h5_bytes(cfgs: Iterable[Config]) -> int:
    from boxrad.config import derive
    from boxrad.run import estimate_output_bytes

    total = 0
    for cfg in cfgs:
        d = derive(cfg)
        total += estimate_output_bytes(cfg, d)
    return int(total)


def require_h5_disk(planned_bytes: int, parent: Path, *, label: str) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    free = float(shutil.disk_usage(parent).free)
    print(
        f"[disk] {label}: planned HDF5 ≈{fmt_bytes(planned_bytes)}  "
        f"free on {parent} ≈{fmt_bytes(free)}",
        flush=True,
    )
    if float(planned_bytes) > _H5_DISK_MAX_FRAC * free:
        raise H5DiskLimitExceeded(label, float(planned_bytes), free, parent)


@contextmanager
def run_workdir(label: str, *, keep: bool = False) -> Generator[Path, None, None]:
    """Scratch or kept directory for validation FDTD HDF5 (removed when *keep* is false)."""
    tmp_mgr: tempfile.TemporaryDirectory[str] | None = None
    if keep:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = _PROJECT_ROOT / "outputs" / f"{label}_{stamp}"
        path.mkdir(parents=True, exist_ok=True)
        kind = "kept"
    else:
        tmp_mgr = tempfile.TemporaryDirectory(prefix=f"bp_{label}_")
        path = Path(tmp_mgr.name)
        kind = "temporary"
    print(f"[workdir] {path} ({kind})", flush=True)
    try:
        yield path
    finally:
        if tmp_mgr is not None:
            h5_total = _h5_bytes_under(path)
            tmp_mgr.cleanup()
            print(f"[workdir] removed  HDF5 total {fmt_bytes(h5_total)}", flush=True)


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


@contextmanager
def keep_awake(label: str) -> Generator[None, None, None]:
    """On macOS, run ``caffeinate -i -w <pid>`` for the wrapped block; no-op elsewhere."""
    _ = label
    proc: subprocess.Popen[bytes] | None = None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        proc = subprocess.Popen(
            ["caffeinate", "-i", "-w", str(os.getpid())],
        )
        print(f"[keep_awake] caffeinate started (pid {proc.pid})", flush=True)
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            print(f"[keep_awake] caffeinate stopped (pid {proc.pid})", flush=True)


@contextmanager
def timed_section(label: str, *, min_report_s: float = 2.0) -> Generator[None, None, None]:
    """Print start/done lines when the block takes longer than *min_report_s*."""
    t0 = time.perf_counter()
    print(f"[{label}] start", flush=True)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if dt >= min_report_s:
            print(f"[{label}] done  {dt:.1f} s", flush=True)


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
    d.ppw_realized = ppw
    d.lambda_min = c / f_ref
    att = math.pow(10.0, -cfg.numerics.pulse_db / 20.0)
    d.pulse_sigma = math.sqrt(2.0 * math.log(att)) / (
        2.0 * math.pi * cfg.numerics.pulse_f_factor * d.f_max
    )
    d.pulse_t0 = PULSE_DELAY_SIGMA_FACTOR * d.pulse_sigma
    sync_dt(cfg, d)


def center_index(n: int) -> int:
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


def cube_geometric_center_m(g: Grid, geom: object) -> tuple[float, float, float]:
    """Mean cell-center position of the source cube (true centroid; even s may sit between cells)."""
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


def path_length_from_cube_center_m(
    g: Grid, geom: object, i1: int, j1: int, k1: int
) -> float:
    xs, ys, zs = cube_geometric_center_m(g, geom)
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


def design_band_mask(
    freqs: np.ndarray,
    *,
    f_min: float,
    f_max: float,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Design band ``[f_min, f_max]``; optional ``np.isfinite(z)`` for Z gates."""
    m = (freqs >= f_min) & (freqs <= f_max)
    if z is not None:
        m &= np.isfinite(z)
    return m


def interp_vs_ka(
    ka_tgt: np.ndarray,
    ka_src: np.ndarray,
    z_src: np.ndarray,
) -> np.ndarray:
    """Interpolate complex ``z_src(ka_src)`` onto ``ka_tgt`` (sorted by ``ka``)."""
    if ka_tgt.size == 0:
        return np.asarray([], dtype=np.complex128)
    order = np.argsort(ka_src)
    ks = ka_src[order]
    zs = z_src[order]
    re = np.interp(ka_tgt, ks, np.real(zs))
    im = np.interp(ka_tgt, ks, np.imag(zs))
    return re + 1j * im


def interp_vs_ka_real(
    ka_tgt: float,
    ka_src: np.ndarray,
    y_src: np.ndarray,
) -> float:
    """Scalar linear interp of real ``y_src`` at ``ka_tgt``."""
    if ka_src.size == 0:
        return float("nan")
    order = np.argsort(ka_src)
    return float(np.interp(ka_tgt, ka_src[order], y_src[order]))


def runs_on_common_ka(
    runs: dict[_RunKey, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    ref_key: _RunKey,
    label: str,
    a_by_key: dict[_RunKey, float] | None = None,
    restrict_to_band: bool = False,
    key_label: str = "key",
) -> tuple[np.ndarray, dict[_RunKey, np.ndarray], dict[str, Any]]:
    """Interpolate runs onto reference ``ka`` where all runs overlap.

    Each run value is ``(ka, z, band)`` with ``band`` the design band mask on that
    run's frequency grid. If ``restrict_to_band`` is True (C8 gates), overlap uses
    only ``band & isfinite(z)``; otherwise only finite ``z`` (C7 legacy G3 alignment).
    """
    print(f"  [{label}] common-ka alignment:", flush=True)
    valid: dict[_RunKey, tuple[np.ndarray, np.ndarray]] = {}
    spans: list[tuple[float, float]] = []
    meta_runs: dict[str, Any] = {}
    a_by_key = a_by_key or {}
    for key in sorted(runs):
        ka, z, band = runs[key]
        ka = np.asarray(ka, dtype=np.float64)
        z = np.asarray(z, dtype=np.complex128)
        band = np.asarray(band, dtype=bool)
        m_fin = (band & np.isfinite(z)) if restrict_to_band else np.isfinite(z)
        n_fin = int(np.sum(m_fin))
        n_band = int(np.sum(band & np.isfinite(z)))
        a_m = float(a_by_key.get(key, float("nan")))
        if n_fin == 0:
            raise RuntimeError(f"{label}: run {key_label}={key!r} has no samples")
        ka_fin = ka[m_fin]
        ka_span = (float(np.min(ka_fin)), float(np.max(ka_fin)))
        meta_runs[str(key)] = {
            "n_finite": n_fin,
            "n_band_mask": n_band,
            "ka_min": ka_span[0],
            "ka_max": ka_span[1],
            "a_m": a_m,
            "restrict_to_band": restrict_to_band,
        }
        print(
            f"    {key_label}={key!r}: samples={n_fin}  "
            f"(design-band∩finite={n_band})  "
            f"ka∈[{ka_span[0]:.4g}, {ka_span[1]:.4g}]"
            + (f"  a={a_m:.6g} m" if math.isfinite(a_m) else ""),
            flush=True,
        )
        valid[key] = (ka_fin, z[m_fin])
        spans.append(ka_span)

    ka_min = max(s[0] for s in spans)
    ka_max = min(s[1] for s in spans)
    print(
        f"    overlap ka ∈ [{ka_min:.4g}, {ka_max:.4g}]  (ref {key_label}={ref_key!r})",
        flush=True,
    )
    if ka_min >= ka_max - 1e-12:
        raise RuntimeError(
            f"{label}: empty ka overlap [{ka_min:.4g}, {ka_max:.4g}] across runs"
        )

    if ref_key not in valid:
        raise RuntimeError(f"{label}: reference {key_label}={ref_key!r} missing")
    ka_u, z_u = valid[ref_key]
    m_grid = (ka_u >= ka_min) & (ka_u <= ka_max)
    if not np.any(m_grid):
        raise RuntimeError(f"{label}: reference run has no ka inside overlap")
    ka_grid = ka_u[m_grid]
    aligned: dict[_RunKey, np.ndarray] = {ref_key: z_u[m_grid]}
    for key, (ka_v, z_v) in valid.items():
        if key == ref_key:
            continue
        aligned[key] = interp_vs_ka(ka_grid, ka_v, z_v)
    meta: dict[str, Any] = {
        "runs": meta_runs,
        "ka_overlap": [ka_min, ka_max],
        "n_ka_common": int(ka_grid.size),
        "ref_key": ref_key,
        "restrict_to_band": restrict_to_band,
    }
    return ka_grid, aligned, meta


def h_transfer_err(h_ref: complex, h: complex) -> tuple[float, float]:
    """Compare transfer phasors H (dB and phase deg relative to *h_ref*)."""
    db = 20.0 * math.log10(abs(h) / max(abs(h_ref), 1e-30))
    ph = float(np.rad2deg(np.angle(h) - np.angle(h_ref)))
    while ph > 180.0:
        ph -= 360.0
    while ph < -180.0:
        ph += 360.0
    return db, ph


def richardson_extrap_complex(
    z_coarse: complex,
    z_fine: complex,
    *,
    r: float,
    p: float,
) -> complex:
    """Richardson extrapolation to zero mesh spacing from two mesh levels."""
    denom = float(r) ** float(p) - 1.0
    if abs(denom) < 1e-30:
        return z_fine
    return z_fine + (z_fine - z_coarse) / denom


def box_mode_probe_targets(
    cfg: Config,
    grid: Grid,
    derived: object,
) -> list[tuple[float, float, float]]:
    """Mid-gap probe locations used in C8/C9 (Huygens–CPML), snapped to the grid."""
    from boxrad.recorder import resolve_huygens_offset

    offset = resolve_huygens_offset(cfg)
    gap = int(derived.gap_cells)  # type: ignore[attr-defined]
    g_cells = 0.5 * (offset + gap)
    g = g_cells * grid.h
    W, D = grid.realized_W, grid.realized_D
    nominal = [
        (0.0, 0.0, g),
        (0.5 * W + g, 0.0, -0.5 * D),
        (0.0, 0.0, -D - g),
    ]
    snapped: list[tuple[float, float, float]] = []
    for x, y, z in nominal:
        i, j, k = grid.index_of(x, y, z)
        snapped.append(grid.coords_of(i, j, k))
    return snapped


def load_huygens_spectra(
    path: Path,
    derived: object,
    f_hz: float,
) -> tuple[Any, complex, np.ndarray, np.ndarray]:
    """Huygens face spectra and radiator ``V(f)`` (C6/C8 transfer-function convention)."""
    import h5py

    from boxrad.post.kirchhoff import SurfacePatches
    from boxrad.post.spectra import spectrum, spectrum_rows
    from boxrad.post.zrad import causal_end_taper
    from boxrad.source import velocity_pulse

    with h5py.File(path, "r") as f:
        centers = np.asarray(f["huygens/centers"][()], dtype=np.float64)
        normals = np.asarray(f["huygens/normals"][()], dtype=np.float64)
        areas = np.asarray(f["huygens/areas"][()], dtype=np.float64)
        patches = SurfacePatches(centers, normals, areas, float(derived.h))  # type: ignore[attr-defined]
        t_p = np.asarray(f["huygens/times_p"][()], dtype=np.float64)
        t_v = np.asarray(f["huygens/times_vn"][()], dtype=np.float64)
        p_ts = np.asarray(f["huygens/p_face"][()], dtype=np.float64)
        vn_ts = np.asarray(f["huygens/vn"][()], dtype=np.float64)
        t_rad = np.asarray(f["radiator/times_p"][()], dtype=np.float64)
        v_rad = np.array(
            [velocity_pulse(float(t), derived.pulse_sigma, derived.pulse_t0) for t in t_rad],  # type: ignore[attr-defined]
            dtype=np.float64,
        )
    n_samp = p_ts.shape[0]
    win = causal_end_taper(n_samp)
    p_win = p_ts * win[:, np.newaxis]
    vn_win = vn_ts * win[:, np.newaxis]
    with timed_section(f"huygens spectrum f={f_hz:.1f} Hz"):
        V = complex(spectrum(v_rad, t_rad, f_hz))
        p_face = np.asarray(spectrum_rows(p_win, t_p, f_hz), dtype=np.complex128)
        vn_face = np.asarray(spectrum_rows(vn_win, t_v, f_hz), dtype=np.complex128)
    return patches, V, p_face, vn_face


def load_radiator_velocity_spectrum(path: Path, derived: object, f_hz: float) -> complex:
    """Radiator velocity spectrum ``V(f)`` at pressure sample times."""
    import h5py

    from boxrad.post.spectra import spectrum
    from boxrad.source import velocity_pulse

    with h5py.File(path, "r") as f:
        t_rad = np.asarray(f["radiator/times_p"][()], dtype=np.float64)
        v_rad = np.array(
            [velocity_pulse(float(t), derived.pulse_sigma, derived.pulse_t0) for t in t_rad],  # type: ignore[attr-defined]
            dtype=np.float64,
        )
    return complex(spectrum(v_rad, t_rad, f_hz))


def load_box_surface_spectrum(path: Path, f_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Box-face pressure row spectrum at stored sample times."""
    import h5py

    from boxrad.post.spectra import spectrum_rows
    from boxrad.post.zrad import causal_end_taper

    with h5py.File(path, "r") as f:
        if "box_surface" not in f:
            raise RuntimeError("HDF5 missing box_surface group")
        centers = np.asarray(f["box_surface/centers"][()], dtype=np.float64)
        t = np.asarray(f["box_surface/times"][()], dtype=np.float64)
        p_ts = np.asarray(f["box_surface/p"][()], dtype=np.float64)
    win = causal_end_taper(p_ts.shape[0])
    p_win = p_ts * win[:, np.newaxis]
    p_spec = np.asarray(spectrum_rows(p_win, t, f_hz), dtype=np.complex128)
    return centers, p_spec.ravel()


def print_runtime_summary(run_timer: RunTimer, phases: tuple[str, ...]) -> None:
    total = run_timer.elapsed()
    print("\n=== Runtime (wall) ===", flush=True)
    for name in phases:
        t = run_timer.phases.get(name)
        if t is not None:
            print(f"  {name:<10} {t:7.2f} s", flush=True)
    print(f"  {'total':<10} {total:7.2f} s", flush=True)
