#!/usr/bin/env python3
"""C2-dispersion: free-field point source phase error vs points per wavelength (Stage A).

CPML-backed cubic domain, volume-velocity point source at center, receivers on
three rays. Compares measured two-receiver transfer ratios to the analytic
spherical wave and to the 3-D Yee numerical dispersion relation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# ``validation`` package import (install boxrad separately).
sys.path.insert(0, str(ROOT))

import validation._common as vc  # noqa: E402

from boxrad.config import (  # noqa: E402
    Air,
    Band,
    Config,
    DEFAULT_CPML_CELLS,
    Numerics,
    Output,
    Radiator,
    SQRT3,
    cfl_for_legacy_derive,
    derive,
    estimate_memory_bytes,
)
from boxrad.cpml import make_cpml_state  # noqa: E402
from boxrad.grid import Grid, allocate_fields, build_cpml_slabs  # noqa: E402
from boxrad.kernels import (  # noqa: E402
    UpdateCoeffs,
    inject_point_source,
    make_update_coeffs,
    point_source_pressure_increment,
    step,
)

RayName = Literal["axis", "face_diag", "body_diag"]
_CPML_CELLS = DEFAULT_CPML_CELLS
_DEFAULT_STABILITY_STEPS = 10_000
_STABILITY_TAIL_STEPS = 2_000
_GATE_PPW_MIN = 10.0
_GATE_PPW_MAX = 40.0
_GATE_REL_TOL = 0.20
# Per-frequency gate floor: max(20%·|theory|, this °/λ). 0.25 (not 0.05) because at
# low f (high ppw) theory lag is small (~0.26°/λ at ppw 40) while this analysis has a
# measurement floor ~0.06–0.2°/λ (Tukey windowing; Δr/λ ≈ 1 at the low-f end; the
# body-diagonal residual, where theory ≈ 0, shows the floor).
_GATE_ABS_DEG_PER_LAM = 0.25
# Informational only (does not gate): mean gated-band lag vs theory within 10% when
# mean theory lag exceeds this threshold (skip rays where dispersion is negligible).
_GATE_INFO_MEAN_THEORY_MIN = 0.2
_GATE_INFO_MEAN_REL_TOL = 0.10
_APP_PPW_VALUES = (15.0, 20.0, 25.0, 30.0)
_DEFAULT_BOX_MM = (100.0, 200.0, 300.0)
_MIN_SRC_CELLS = 15
_MIN_CPML_CLEARANCE_CELLS = 10
_MIN_PAIR_SEP_CELLS = 20
_SOURCE_BAND_DB = 30.0
_SIGN_CHECK_TOL_DEG = 10.0
_PLOT_DR_LAM_MIN = 0.5
_GATE_DR_LAM_MIN = 1.0
_THEORY_SMALL_K_TOL = 0.10
_APP_PHASE_THRESH_DEG = 10.0
_TUKEY_ALPHA = 0.25
_FFT_PAD_FACTOR = 8
_DEFAULT_DIST_CELLS = (15, 35, 55, 75)
_QUICK_DIST_CELLS = (15, 35)
# Interior cell counts per axis (plus 2× CPML on each face).
_QUICK_INTERIOR_CELLS = 96
_DEFAULT_INTERIOR_CELLS = 160
_PULSE_DELAY_SIGMA_FACTOR = 4.0
_WIN_SIGMA_SPAN = 5.0
_WIN_TAIL_CELLS = 10.0
_MIN_WINDOW_SAMPLES = 32


class _RunTimer:
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


def _fmt_bytes(b: float | int) -> str:
    b = float(b)
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b / 1e3:.1f} kB"


def _emit_sim_progress(
    line: str,
    *,
    tty: bool,
    final: bool,
) -> None:
    if tty and not final:
        print("\r" + line.ljust(96), end="", flush=True)
    else:
        print(line, flush=True)


def _sync_dt(cfg: Config, d: object) -> float:
    dt = cfl_for_legacy_derive(cfg.numerics) * d.h / (cfg.air.c * SQRT3)
    d.dt = dt
    return dt


def _pulse_q(t: float, sigma: float, t0: float) -> float:
    return float(np.exp(-0.5 * ((t - t0) / sigma) ** 2))


def _apply_h_from_ppw(d: object, cfg: Config, *, f_ref: float, ppw: float) -> None:
    """Set ``h = c/(f_ref·ppw)`` (C2 override; ignore radiator ``D/10`` cap)."""
    c = cfg.air.c
    h = c / (f_ref * ppw)
    d.h = h
    d.ppw_realized = ppw
    d.lambda_min = c / f_ref
    att = math.pow(10.0, -cfg.numerics.pulse_db / 20.0)
    d.pulse_sigma = math.sqrt(2.0 * math.log(att)) / (
        2.0 * math.pi * cfg.numerics.pulse_f_factor * f_ref
    )
    d.pulse_t0 = _PULSE_DELAY_SIGMA_FACTOR * d.pulse_sigma
    _sync_dt(cfg, d)


def build_dispersion_domain(
    *,
    f_ref: float,
    ppw: float,
    n: int,
    cpml_cells: int = _CPML_CELLS,
) -> tuple[Grid, object, Config, object, UpdateCoeffs, object]:
    """Uniform ``n³`` free-field grid with *h* from ``f_ref`` and target *ppw*."""
    num = Numerics(
        cpml_cells=cpml_cells,
        gap_cells=0,
        ppw=ppw,
        cfl_safety=0.99,
        cell_budget=10**12,
        cells_across_radiator=10,
    )
    cfg = Config(
        box=None,
        air=Air(c=343.0, rho=1.204),
        radiator=Radiator("circle", dims=[0.01]),
        band=Band(f_min=20.0, f_max=f_ref),
        numerics=num,
        output=Output(path=""),
    )
    d = derive(cfg)
    _apply_h_from_ppw(d, cfg, f_ref=f_ref, ppw=ppw)
    d.nx = d.ny = d.nz = n
    d.cpml_cells = cpml_cells
    d.total_cells = n**3
    g = Grid(cfg, d)
    g.cpml = cpml_cells
    g.cpml_slabs = build_cpml_slabs(n, n, n, cpml_cells)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    return g, d, cfg, cpml_st, coeffs, g.cpml_slabs


def _domain_half_extent_m(n: int, h: float) -> float:
    return 0.5 * n * h


def _center_index(n: int) -> int:
    return n // 2


def _interior_cells(n: int, cpml: int) -> int:
    return n - 2 * cpml


def _ray_indices(
    n: int,
    *,
    cpml: int,
    dist_cells: tuple[int, ...],
) -> dict[RayName, list[tuple[int, int, int, int]]]:
    """Receivers on three rays: ``(i, j, k, offset_cells along ray)``."""
    ic = jc = kc = _center_index(n)
    inset = cpml + _MIN_CPML_CLEARANCE_CELLS
    max_offset = min(ic - inset, n - 1 - ic - inset)

    def _offsets() -> tuple[int, ...]:
        extra = max_offset
        offs = [
            d
            for d in dist_cells
            if d >= _MIN_SRC_CELLS and d <= max_offset
        ]
        if extra >= _MIN_SRC_CELLS and (not offs or offs[-1] < extra):
            if extra not in offs:
                offs = tuple(sorted(set((*offs, extra))))
        return tuple(offs)

    offs = _offsets()
    out: dict[RayName, list[tuple[int, int, int, int]]] = {
        "axis": [],
        "face_diag": [],
        "body_diag": [],
    }
    for d in offs:
        out["axis"].append((ic + d, jc, kc, d))
        out["face_diag"].append((ic + d, jc + d, kc, d))
        out["body_diag"].append((ic + d, jc + d, kc + d, d))
    return out


def _path_length_m(ray: RayName, d_cells: int, h: float) -> float:
    if ray == "axis":
        return d_cells * h
    if ray == "face_diag":
        return d_cells * h * math.sqrt(2.0)
    return d_cells * h * math.sqrt(3.0)


def attach_physical_radii(
    rays: dict[RayName, list[tuple[int, int, int, int]]], h: float
) -> dict[RayName, list[dict[str, Any]]]:
    fixed: dict[RayName, list[dict[str, Any]]] = {}
    for name, pts in rays.items():
        fixed[name] = []
        for i, j, k, d_cells in pts:
            fixed[name].append(
                {
                    "i": i,
                    "j": j,
                    "k": k,
                    "r_cells": d_cells,
                    "r_m": _path_length_m(name, d_cells, h),  # type: ignore[arg-type]
                }
            )
    return fixed


def _coords_m(g: Grid, i: int, j: int, k: int) -> tuple[float, float, float]:
    return g.coords_of(i, j, k)


def _ppw_frequency_bounds(c: float, h: float) -> tuple[float, float]:
    """ppw ∈ [ppw_min, ppw_max] ⟺ f ∈ [c/(ppw_max·h), c/(ppw_min·h)]."""
    f_lo = c / (_GATE_PPW_MAX * h)
    f_hi = c / (_GATE_PPW_MIN * h)
    return f_lo, f_hi


def _arrival_time_s(pulse_t0: float, r_m: float, c: float) -> float:
    return pulse_t0 + r_m / c


def receiver_window_samples(
    g: Grid,
    cfg: Config,
    d: object,
    pt: dict[str, Any],
    *,
    ic: int,
    jc: int,
    kc: int,
    trace_len: int,
) -> tuple[int, int, str] | tuple[None, None, str]:
    """Absolute sample window ``[i_start, i_end)`` centered on the direct arrival."""
    c = cfg.air.c
    sigma = d.pulse_sigma
    pulse_t0 = d.pulse_t0
    h, dt = d.h, d.dt
    r_m = pt["r_m"]
    t_arr = _arrival_time_s(pulse_t0, r_m, c)
    t_start = t_arr - _WIN_SIGMA_SPAN * sigma
    t_end = t_arr + _WIN_SIGMA_SPAN * sigma + _WIN_TAIL_CELLS * h / c
    t_refl = first_reflection_time_s(
        g,
        cfg,
        i_src=ic,
        j_src=jc,
        k_src=kc,
        i_rec=pt["i"],
        j_rec=pt["j"],
        k_rec=pt["k"],
    )
    if t_end >= t_refl:
        return (
            None,
            None,
            f"window end {t_end * 1e3:.3f} ms >= first reflection {t_refl * 1e3:.3f} ms",
        )
    if t_start < 0.0:
        return None, None, f"window start {t_start * 1e3:.3f} ms < 0"
    i_start = int(math.ceil(t_start / dt))
    i_end = int(math.floor(t_end / dt))
    i_end = min(i_end, trace_len)
    if i_end - i_start < _MIN_WINDOW_SAMPLES:
        return (
            None,
            None,
            f"window length {i_end - i_start} samples < {_MIN_WINDOW_SAMPLES}",
        )
    return i_start, i_end, ""


def prepare_usable_receivers(
    g: Grid,
    cfg: Config,
    d: object,
    receivers: dict[RayName, list[dict[str, Any]]],
    *,
    ic: int,
    jc: int,
    kc: int,
    trace_len: int,
    verbose: bool = False,
) -> dict[RayName, list[dict[str, Any]]]:
    out: dict[RayName, list[dict[str, Any]]] = {}
    for ray, pts in receivers.items():
        good: list[dict[str, Any]] = []
        for pt in pts:
            i_start, i_end, reason = receiver_window_samples(
                g, cfg, d, pt, ic=ic, jc=jc, kc=kc, trace_len=trace_len
            )
            if i_start is None:
                if verbose:
                    print(f"  skip {ray} receiver {pt['r_cells']}c: {reason}")
                continue
            good.append({**pt, "i_start": i_start, "i_end": i_end})
        out[ray] = good
    return out


def quick_geometry_feasible(
    *,
    f_ref: float,
    ppw: float,
    cpml: int,
) -> tuple[bool, str]:
    n = pick_grid_n(quick=True, cpml=cpml)
    dist = _QUICK_DIST_CELLS
    g, d, cfg, *_ = build_dispersion_domain(
        f_ref=f_ref, ppw=ppw, n=n, cpml_cells=cpml
    )
    ic = jc = kc = _center_index(n)
    raw = _ray_indices(n, cpml=cpml, dist_cells=dist)
    rec = attach_physical_radii(raw, g.h)
    n_analysis, _ = _choose_n_steps(
        d, cfg, g, rec, ic=ic, jc=jc, kc=kc, usable=None, stability_steps=_DEFAULT_STABILITY_STEPS
    )
    usable = prepare_usable_receivers(
        g, cfg, d, rec, ic=ic, jc=jc, kc=kc, trace_len=n_analysis
    )
    bad = [ray for ray, pts in usable.items() if len(pts) < 2]
    if bad:
        return False, (
            f"quick domain (interior {_QUICK_INTERIOR_CELLS}³, offsets {dist}) "
            f"fits only {len(bad)} ray(s) with <2 windowed receivers ({', '.join(bad)}). "
            "Run without --quick for the full grid."
        )
    return True, ""


def plan_geometry(
    *,
    quick: bool,
) -> tuple[int, tuple[int, ...], str]:
    if quick:
        n = pick_grid_n(quick=True, cpml=_CPML_CELLS)
        return n, _QUICK_DIST_CELLS, "quick"
    n = pick_grid_n(quick=False, cpml=_CPML_CELLS)
    return n, _DEFAULT_DIST_CELLS, "default"


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
    x_s, y_s, z_s = _coords_m(g, i_src, j_src, k_src)
    x_r, y_r, z_r = _coords_m(g, i_rec, j_rec, k_rec)
    hx = _domain_half_extent_m(g.nx, g.h)
    hy = _domain_half_extent_m(g.ny, g.h)
    hz = _domain_half_extent_m(g.nz, g.h)
    times = [
        (x_s + hx + x_r + hx) / c,
        (hx - x_s + hx - x_r) / c,
        (y_s + hy + y_r + hy) / c,
        (hy - y_s + hy - y_r) / c,
        (z_s + hz + z_r + hz) / c,
        (hz - z_s + hz - z_r) / c,
    ]
    return margin * min(times)


def _choose_n_steps(
    d: object,
    cfg: Config,
    g: Grid,
    receivers: dict[RayName, list[dict[str, Any]]],
    *,
    ic: int,
    jc: int,
    kc: int,
    usable: dict[RayName, list[dict[str, Any]]] | None,
    stability_steps: int,
) -> tuple[int, int]:
    """Return (steps for spectral analysis, total steps including stability segment)."""
    c = cfg.air.c
    h = d.h
    sigma = d.pulse_sigma
    pulse_t0 = d.pulse_t0
    t_need = pulse_t0 + 8.0 * sigma
    pool = receivers if usable is None else usable
    for pts in pool.values():
        for pt in pts:
            t_arr = _arrival_time_s(pulse_t0, pt["r_m"], c)
            t_end = t_arr + _WIN_SIGMA_SPAN * sigma + _WIN_TAIL_CELLS * h / c
            t_need = max(t_need, t_end + 2.0 * d.dt)
            if "i_end" in pt:
                t_need = max(t_need, pt["i_end"] * d.dt)
            else:
                tr = first_reflection_time_s(
                    g,
                    cfg,
                    i_src=ic,
                    j_src=jc,
                    k_src=kc,
                    i_rec=pt["i"],
                    j_rec=pt["j"],
                    k_rec=pt["k"],
                )
                t_need = min(t_need, tr)
    n_analysis = max(64, int(math.ceil(t_need / d.dt)) + 1)
    n_total = max(stability_steps, n_analysis)
    return n_analysis, n_total


def warmup_compile(
    g: Grid,
    d: object,
    cfg: Config,
    coeffs: UpdateCoeffs,
    slabs: object,
    cpml_st: object,
    *,
    ic: int,
    jc: int,
    kc: int,
) -> float:
    """One FDTD step to trigger Numba JIT; return cold step time (ms)."""
    fields = allocate_fields(g)
    t0 = time.perf_counter()
    q = _pulse_q(0.0, d.pulse_sigma, d.pulse_t0)
    val = point_source_pressure_increment(
        cfg.air.rho, cfg.air.c, d.dt, d.h, q
    )
    inject_point_source(fields.p, ic, jc, kc, val)
    step(fields, coeffs, slabs, cpml=cpml_st)
    return (time.perf_counter() - t0) * 1e3


def run_simulation(
    g: Grid,
    d: object,
    cfg: Config,
    coeffs: UpdateCoeffs,
    slabs: object,
    cpml_st: object,
    receivers: dict[RayName, list[dict[str, Any]]],
    *,
    n_total: int,
    ic: int,
    jc: int,
    kc: int,
    run_timer: _RunTimer | None = None,
) -> tuple[dict[RayName, np.ndarray], dict[str, Any]]:
    """Advance FDTD and record pressure traces at receivers (float64)."""
    fields = allocate_fields(g)
    idx_map: dict[tuple[int, int, int], int] = {}
    flat: list[tuple[int, int, int]] = []
    for pts in receivers.values():
        for pt in pts:
            key = (pt["i"], pt["j"], pt["k"])
            if key not in idx_map:
                idx_map[key] = len(flat)
                flat.append(key)

    n_rec = len(flat)
    traces = np.zeros((n_rec, n_total), dtype=np.float64)
    peak_abs = 0.0
    t0_run = time.perf_counter()
    tty = sys.stdout.isatty()
    script_t0 = run_timer.t0 if run_timer is not None else t0_run
    milestones = {
        max(1, math.ceil(n_total * p / 100.0)) for p in range(5, 101, 5)
    }
    milestones.add(n_total)

    for step_i in range(n_total):
        t = step_i * d.dt
        q = _pulse_q(t, d.pulse_sigma, d.pulse_t0)
        val = point_source_pressure_increment(
            cfg.air.rho, cfg.air.c, d.dt, d.h, q
        )
        inject_point_source(fields.p, ic, jc, kc, val)
        step(fields, coeffs, slabs, cpml=cpml_st)
        peak_abs = max(peak_abs, float(np.max(np.abs(fields.p))))
        for ri, (ii, jj, kk) in enumerate(flat):
            traces[ri, step_i] = float(fields.p[ii, jj, kk])

        done = step_i + 1
        if done not in milestones:
            continue
        last_pct = (100 * done) // n_total
        sim_elapsed = time.perf_counter() - t0_run
        rate = done / sim_elapsed if sim_elapsed > 0 else 0.0
        eta = (n_total - done) / rate if rate > 0 else 0.0
        wall_elapsed = time.perf_counter() - script_t0
        line = (
            f"[sim] step {done}/{n_total} ({last_pct:3d}%) | "
            f"sim t = {t * 1e3:.2f} ms | "
            f"elapsed {wall_elapsed:.1f} s | "
            f"rate {rate:.1f} steps/s | "
            f"ETA {eta:.1f} s | "
            f"max|p| = {peak_abs:.4e}"
        )
        _emit_sim_progress(line, tty=tty, final=(done == n_total))
    if tty:
        print(flush=True)

    wall_s = time.perf_counter() - t0_run
    tail = traces[:, -_STABILITY_TAIL_STEPS:]
    tail_max = float(np.max(np.abs(tail))) if n_total >= _STABILITY_TAIL_STEPS else peak_abs
    meta = {
        "n_steps": n_total,
        "wall_s": wall_s,
        "peak_abs_p": peak_abs,
        "tail_max_abs_p": tail_max,
        "stability_ratio": tail_max / (peak_abs + 1e-30),
    }

    by_ray: dict[RayName, np.ndarray] = {}
    for name, pts in receivers.items():
        rows: list[np.ndarray] = []
        for pt in pts:
            key = (pt["i"], pt["j"], pt["k"])
            rows.append(traces[idx_map[key]])
        by_ray[name] = np.stack(rows, axis=0)
    return by_ray, meta


def _tukey(n: int, alpha: float = _TUKEY_ALPHA) -> np.ndarray:
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


def _fft_len(trace_len: int, i_end_max: int, pad_factor: int = _FFT_PAD_FACTOR) -> int:
    return int(
        2 ** math.ceil(math.log2(max(trace_len, i_end_max * pad_factor, 64)))
    )


def _spectrum_absolute(
    trace: np.ndarray,
    dt: float,
    *,
    i_start: int,
    i_end: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Tukey on ``[i_start, i_end)`` embedded at absolute indices (t=0 at step 0)."""
    n_win = i_end - i_start
    buf = np.zeros(n_fft, dtype=np.float64)
    seg = np.asarray(trace[i_start:i_end], dtype=np.float64) * _tukey(n_win)
    buf[i_start:i_end] = seg
    spec = np.fft.fft(buf, n=n_fft)
    freqs = np.fft.fftfreq(n_fft, dt)
    return freqs, spec, n_win


def _source_spectrum_absolute(
    dt: float,
    *,
    sigma: float,
    t0: float,
    i_start: int,
    i_end: int,
    n_fft: int,
) -> np.ndarray:
    n = n_fft
    t = np.arange(n, dtype=np.float64) * dt
    q = np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    if i_end > i_start:
        win = np.zeros(n, dtype=np.float64)
        win[i_start:i_end] = _tukey(i_end - i_start)
        q *= win
    return np.fft.fft(q, n=n_fft)


def _analysis_mask(
    freqs: np.ndarray,
    q_spec: np.ndarray,
    *,
    c: float,
    h: float,
    f_max: float,
    floor_db: float,
) -> np.ndarray:
    mag = np.abs(q_spec)
    pos = freqs > 0.0
    pk = float(np.max(mag[pos])) if np.any(pos) else float(np.max(mag))
    floor = pk * 10.0 ** (-floor_db / 20.0)
    f_lo, f_hi = _ppw_frequency_bounds(c, h)
    f_hi = min(f_hi, f_max)
    return (
        pos
        & (freqs >= f_lo)
        & (freqs <= f_hi)
        & (mag >= floor)
    )


def verify_sign_convention(
    freqs: np.ndarray,
    Sa: np.ndarray,
    Sb: np.ndarray,
    ppw: np.ndarray,
    *,
    ra: float,
    rb: float,
    dr_m: float,
    c: float,
    mask: np.ndarray,
) -> None:
    """At the lowest in-band frequency, ``wrap(arg(H)−arg(H₀))`` within tolerance."""
    if not np.any(mask):
        raise RuntimeError(
            "sign-convention self-check: no frequency in analysis mask"
        )
    idxs = np.where(mask)[0]
    idx = int(idxs[np.argmin(freqs[idxs])])
    f = float(freqs[idx])
    ppw_val = float(ppw[idx])
    H = Sb[idx] / (Sa[idx] + 1e-30)
    k = 2.0 * math.pi * f / c
    H0 = (ra / rb) * complex(math.cos(-k * dr_m), math.sin(-k * dr_m))
    dphi = math.degrees(math.atan2((H * np.conj(H0)).imag, (H * np.conj(H0)).real))
    if abs(dphi) > _SIGN_CHECK_TOL_DEG:
        raise RuntimeError(
            "sign-convention error: at lowest in-band frequency "
            f"f={f:.1f} Hz (ppw={ppw_val:.2f}), "
            f"wrap(arg(H)−arg(H₀))={dphi:.2f}° "
            f"(tolerance {_SIGN_CHECK_TOL_DEG}°)."
        )
    print(
        f"  sign check: f={f / 1e3:.3f} kHz  ppw={ppw_val:.2f}  "
        f"Δφ={dphi:.2f}°",
        flush=True,
    )


def _ray_n_components(ray: RayName) -> tuple[float, float, float]:
    if ray == "axis":
        return (1.0, 0.0, 0.0)
    if ray == "face_diag":
        u = 1.0 / math.sqrt(2.0)
        return (u, u, 0.0)
    u = 1.0 / math.sqrt(3.0)
    return (u, u, u)


def _sum_n_fourth(ray: RayName) -> float:
    return sum(n**4 for n in _ray_n_components(ray))


def _spatial_dispersion_sum(k: float, h: float, ray: RayName) -> float:
    acc = 0.0
    for n_i in _ray_n_components(ray):
        acc += (math.sin(0.5 * k * n_i * h) / h) ** 2
    return acc


def _k_numeric_from_omega(
    omega: float,
    *,
    c: float,
    dt: float,
    h: float,
    ray: RayName,
) -> float:
    """Root-find *k* in ``[sin(ωΔt/2)/(cΔt)]² = Σᵢ [sin(k nᵢ h/2)/h]²``."""
    if omega <= 0.0:
        return 0.0
    lhs = (math.sin(0.5 * omega * dt) / (c * dt)) ** 2

    def sum_k(k: float) -> float:
        return _spatial_dispersion_sum(abs(k), h, ray)

    k_hi = omega / c * 1.5 + 1.0
    for _ in range(80):
        if sum_k(k_hi) >= lhs:
            break
        k_hi *= 2.0
    else:
        return omega / c

    lo, hi = 0.0, k_hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sum_k(mid) < lhs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _phase_lag_deg_per_lambda(k_num: float, k0: float) -> float:
    """Phase lag per λ: ``360°·(k_num − k)/k`` (+ = numerical wave slower than *c*)."""
    if k0 <= 0.0:
        return 0.0
    return 360.0 * (k_num - k0) / k0


def _small_k_theory_deg_per_lambda(kh: float, *, S: float, ray: RayName) -> float:
    """Δv/c ≈ (kh)²/24 · (Σ nᵢ⁴ − S²); phase ≈ 360° × that at small *kh*."""
    dv_over_c = (kh**2) / 24.0 * (_sum_n_fourth(ray) - S * S)
    return 360.0 * dv_over_c


def verify_theory_small_k(
    *,
    c: float,
    dt: float,
    h: float,
    verbose: bool = False,
) -> dict[RayName, dict[str, float]]:
    S = c * dt / h
    omega = 2.0 * math.pi * c / (20.0 * h)
    kh = omega * h / c
    rows: dict[RayName, dict[str, float]] = {}
    if verbose:
        print(f"  theory ppw=20 (S={S:.3f}, kh={kh:.4f})")
    for ray in ("axis", "face_diag", "body_diag"):
        k_num = _k_numeric_from_omega(omega, c=c, dt=dt, h=h, ray=ray)  # type: ignore[arg-type]
        k0 = omega / c
        exact = _phase_lag_deg_per_lambda(k_num, k0)
        approx = _small_k_theory_deg_per_lambda(kh, S=S, ray=ray)  # type: ignore[arg-type]
        rel = abs(abs(exact) - abs(approx)) / max(abs(exact), 1e-12)
        rows[ray] = {"exact_deg_per_lam": exact, "small_k_deg_per_lam": approx, "rel_err": rel}
        if verbose:
            print(
                f"    {ray:<12} exact={exact:7.3f}°/λ  small-k={approx:7.3f}°/λ  "
                f"|.| rel err={rel:.1%}"
            )
        if rel > _THEORY_SMALL_K_TOL:
            raise RuntimeError(
                f"Yee theory vs small-k disagree by {rel:.1%} on {ray} "
                f"(limit {_THEORY_SMALL_K_TOL:.0%})"
            )
    return rows


def theory_phase_deg_per_lambda(
    freqs: np.ndarray,
    *,
    c: float,
    dt: float,
    h: float,
    ray: RayName,
) -> np.ndarray:
    out = np.zeros_like(freqs, dtype=np.float64)
    for i, f in enumerate(freqs):
        if f <= 0.0:
            continue
        omega = 2.0 * math.pi * f
        k_num = _k_numeric_from_omega(omega, c=c, dt=dt, h=h, ray=ray)
        k0 = omega / c
        out[i] = _phase_lag_deg_per_lambda(k_num, k0)
    return out


def _meas_lag_deg_per_lambda(phase_err_deg: np.ndarray, dr_lam: np.ndarray) -> np.ndarray:
    """Lag per λ: ``−arg(H/H₀) / ((r₂−r₁)/λ)`` (matches theory lag sign)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            dr_lam > 0.0,
            -phase_err_deg / dr_lam,
            np.nan,
        )


def _gate_tolerance(theory: np.ndarray) -> np.ndarray:
    return np.maximum(
        _GATE_ABS_DEG_PER_LAM,
        _GATE_REL_TOL * np.maximum(np.abs(theory), _GATE_ABS_DEG_PER_LAM),
    )


def _longest_pair(pair_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    good = [p for p in pair_results if p.get("n_plot_freq", 0) > 0]
    if not good:
        return None
    return max(good, key=lambda p: int(p.get("dr_cells", 0)))


def analyze_ray(
    g: Grid,
    d: object,
    cfg: Config,
    receivers: list[dict[str, Any]],
    trace_block: np.ndarray,
    *,
    ray: RayName,
    ic: int,
    jc: int,
    kc: int,
    sign_check_done: bool = False,
    verbose: bool = False,
    run_timer: _RunTimer | None = None,
) -> tuple[dict[str, Any], bool]:
    c = cfg.air.c
    h, dt = d.h, d.dt
    pairs: list[dict[str, Any]] = []
    f_lo, f_hi = _ppw_frequency_bounds(c, h)

    if run_timer is not None:
        run_timer.banner("analysis", f"ray={ray}")

    if len(receivers) < 2:
        print(f"  {ray}: domain too small for pulse (< 2 usable receivers)", flush=True)
        return (
            {
                "ray": ray,
                "pairs": [],
                "status": "domain too small for pulse",
                "window_samples": None,
                "ppw_band_hz": [f_lo, min(f_hi, d.f_max)],
            },
            sign_check_done,
        )

    trace_len = len(trace_block[0])
    i_end_max = max(pt["i_end"] for pt in receivers)
    n_fft = _fft_len(trace_len, i_end_max)
    freqs: np.ndarray | None = None
    spectra: list[np.ndarray] = []
    window_lengths: list[int] = []

    for ti, trace in enumerate(trace_block):
        pt = receivers[ti]
        freqs, spec, n_win = _spectrum_absolute(
            trace,
            dt,
            i_start=pt["i_start"],
            i_end=pt["i_end"],
            n_fft=n_fft,
        )
        spectra.append(spec)
        window_lengths.append(n_win)

    sigma = d.pulse_sigma
    pulse_t0 = d.pulse_t0
    i_q0 = int(math.ceil((pulse_t0 - _WIN_SIGMA_SPAN * sigma) / dt))
    i_q1 = int(
        math.floor(
            (pulse_t0 + _WIN_SIGMA_SPAN * sigma + _WIN_TAIL_CELLS * h / c) / dt
        )
    )
    i_q0 = max(0, i_q0)
    i_q1 = min(trace_len, max(i_q1, i_q0 + _MIN_WINDOW_SAMPLES))
    q_spec = _source_spectrum_absolute(
        dt,
        sigma=sigma,
        t0=pulse_t0,
        i_start=i_q0,
        i_end=i_q1,
        n_fft=n_fft,
    )
    assert freqs is not None
    band_mask = _analysis_mask(
        freqs, q_spec, c=c, h=h, f_max=d.f_max, floor_db=_SOURCE_BAND_DB
    )
    ppw = c / (np.maximum(freqs, 1e-30) * h)

    for a in range(len(receivers) - 1):
        for b in range(a + 1, len(receivers)):
            ra = receivers[a]["r_m"]
            rb = receivers[b]["r_m"]
            r1_cells = receivers[a]["r_cells"]
            r2_cells = receivers[b]["r_cells"]
            dr_cells = r2_cells - r1_cells
            if dr_cells < _MIN_PAIR_SEP_CELLS:
                continue
            if ra <= 0.0 or rb <= ra:
                continue
            dr = rb - ra
            Sa = spectra[a]
            Sb = spectra[b]
            H = Sb / (Sa + 1e-30)
            k = 2.0 * math.pi * freqs / c
            H0 = (ra / rb) * np.exp(-1j * k * dr)
            ratio = H / (H0 + 1e-30)
            phase_err_deg = np.degrees(np.angle(ratio))
            lam = c / np.maximum(freqs, 1e-30)
            dr_lam = dr / lam
            mag_db = 20.0 * np.log10(np.maximum(np.abs(ratio), 1e-30))
            theory_lag = theory_phase_deg_per_lambda(
                freqs, c=c, dt=dt, h=h, ray=ray
            )
            meas_lag = _meas_lag_deg_per_lambda(phase_err_deg, dr_lam)
            ppw_gate = (ppw >= _GATE_PPW_MIN) & (ppw <= _GATE_PPW_MAX)
            plot_mask = (
                band_mask
                & ppw_gate
                & (dr_lam >= _PLOT_DR_LAM_MIN)
                & np.isfinite(meas_lag)
                & np.isfinite(theory_lag)
            )
            gate_mask = plot_mask & (dr_lam >= _GATE_DR_LAM_MIN)
            if not np.any(plot_mask):
                pairs.append(
                    {
                        "r1_cells": r1_cells,
                        "r2_cells": r2_cells,
                        "dr_cells": dr_cells,
                        "r1_m": ra,
                        "r2_m": rb,
                        "n_plot_freq": 0,
                        "n_gate_freq": 0,
                        "gate_ok": False,
                        "is_longest_pair": False,
                    }
                )
                continue
            idxs = np.where(plot_mask)[0]
            order = idxs[np.argsort(freqs[idxs])]
            f_sel = freqs[idxs]
            ppw_sel = ppw[idxs]
            summary = (
                f"  {ray} r1={r1_cells}c r2={r2_cells}c: "
                f"plot {int(np.sum(plot_mask))} bins "
                f"(gate {int(np.sum(gate_mask))}), "
                f"f {f_sel.min() / 1e3:.2f}–{f_sel.max() / 1e3:.2f} kHz, "
                f"ppw {ppw_sel.min():.1f}–{ppw_sel.max():.1f}"
            )
            print(summary, flush=True)
            if verbose:
                dbg = [
                    f"{freqs[i] / 1e3:.3f}kHz(ppw={ppw[i]:.2f})"
                    for i in order
                ]
                print("    " + ", ".join(dbg), flush=True)
            if run_timer is not None:
                run_timer.banner("analysis", f"pair {ray} {r1_cells}→{r2_cells}c")

            pairs.append(
                {
                    "r1_cells": r1_cells,
                    "r2_cells": r2_cells,
                    "dr_cells": dr_cells,
                    "r1_m": ra,
                    "r2_m": rb,
                    "dr_m": dr,
                    "window_samples_r1": receivers[a]["i_end"] - receivers[a]["i_start"],
                    "window_samples_r2": receivers[b]["i_end"] - receivers[b]["i_start"],
                    "n_fft": n_fft,
                    "n_plot_freq": int(np.sum(plot_mask)),
                    "n_gate_freq": int(np.sum(gate_mask)),
                    "mean_mag_db": float(np.mean(mag_db[plot_mask])),
                    "gate_ok": False,
                    "is_longest_pair": False,
                    "freqs_hz": freqs[plot_mask].tolist(),
                    "ppw": ppw[plot_mask].tolist(),
                    "meas_lag_deg_per_lam": meas_lag[plot_mask].tolist(),
                    "theory_lag_deg_per_lam": theory_lag[plot_mask].tolist(),
                    "bin_in_gate": gate_mask[plot_mask].tolist(),
                    "meas_deg_per_lam": meas_lag[plot_mask].tolist(),
                    "theory_deg_per_lam": theory_lag[plot_mask].tolist(),
                    "gate_freqs_hz": freqs[gate_mask].tolist(),
                    "gate_ppw": ppw[gate_mask].tolist(),
                    "gate_meas_lag": meas_lag[gate_mask].tolist(),
                    "gate_theory_lag": theory_lag[gate_mask].tolist(),
                }
            )

    longest = _longest_pair(pairs)
    for pr in pairs:
        pr["is_longest_pair"] = longest is not None and pr is longest
        if not pr.get("is_longest_pair"):
            continue
        gm = np.asarray(pr["gate_meas_lag"], dtype=np.float64)
        gt = np.asarray(pr["gate_theory_lag"], dtype=np.float64)
        if gm.size == 0:
            pr["gate_ok"] = False
            continue
        err = np.abs(gm - gt)
        tol = _gate_tolerance(gt)
        pr["gate_ok"] = bool(np.all(err <= tol))
        pr["mean_meas_lag"] = float(np.mean(gm))
        pr["max_meas_lag"] = float(np.max(gm))
        pr["mean_theory_lag"] = float(np.mean(gt))
        pr["max_theory_lag"] = float(np.max(gt))
        pr["max_abs_lag_err"] = float(np.max(err))
        pr["n_freq"] = int(gm.size)
        mean_thr = pr["mean_theory_lag"]
        if mean_thr > _GATE_INFO_MEAN_THEORY_MIN:
            mean_rel = abs(pr["mean_meas_lag"] - mean_thr) / abs(mean_thr)
            pr["info_mean_agree_ok"] = mean_rel <= _GATE_INFO_MEAN_REL_TOL
            pr["info_mean_rel_err"] = float(mean_rel)
        else:
            pr["info_mean_agree_ok"] = None
            pr["info_mean_rel_err"] = None

    if (
        ray == "axis"
        and longest is not None
        and not sign_check_done
        and freqs is not None
    ):
        a_idx = next(
            i
            for i, p in enumerate(receivers)
            if p["r_cells"] == longest["r1_cells"]
        )
        b_idx = next(
            i
            for i, p in enumerate(receivers)
            if p["r_cells"] == longest["r2_cells"]
        )
        dr = float(longest["dr_m"])
        lam = c / np.maximum(freqs, 1e-30)
        dr_lam_long = dr / lam
        sign_mask = (
            band_mask
            & (ppw >= _GATE_PPW_MIN)
            & (ppw <= _GATE_PPW_MAX)
            & (dr_lam_long >= _GATE_DR_LAM_MIN)
        )
        verify_sign_convention(
            freqs,
            spectra[a_idx],
            spectra[b_idx],
            ppw,
            ra=float(longest["r1_m"]),
            rb=float(longest["r2_m"]),
            dr_m=dr,
            c=c,
            mask=sign_mask,
        )
        sign_check_done = True

    return (
        {
            "ray": ray,
            "pairs": pairs,
            "status": "ok",
            "window_samples": window_lengths,
            "ppw_band_hz": [f_lo, min(f_hi, d.f_max)],
        },
        sign_check_done,
    )


def _collect_meas_ppw_samples(
    pair_results: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    pr = _longest_pair(pair_results)
    if pr is None:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    ppw_all: list[float] = []
    val_all: list[float] = []
    for p, m in zip(pr["ppw"], pr["meas_lag_deg_per_lam"]):
        ppw_all.append(float(p))
        val_all.append(float(m))
    order = np.argsort(ppw_all)
    ppw = np.asarray(ppw_all, dtype=np.float64)[order]
    vals = np.asarray(val_all, dtype=np.float64)[order]
    ppw_u, idx = np.unique(ppw, return_index=True)
    vals_u = np.array([np.median(vals[ppw == p]) for p in ppw_u])
    return ppw_u, vals_u


def _interp_ppl_kh2(
    ppw_u: np.ndarray, vals_u: np.ndarray, target_ppw: float
) -> float:
    if ppw_u.size == 0:
        return 0.0
    if target_ppw <= float(ppw_u[0]):
        base = float(vals_u[0])
        return base * (target_ppw / float(ppw_u[0])) ** 2
    if target_ppw >= float(ppw_u[-1]):
        base = float(vals_u[-1])
        return base * (float(ppw_u[-1]) / target_ppw) ** 2
    return float(np.interp(target_ppw, ppw_u, vals_u))


def measured_deg_per_lambda_at_ppw(
    analysis: dict[str, Any], target_ppw: float
) -> dict[str, float]:
    out: dict[str, float] = {}
    for ray, block in analysis.items():
        if block.get("status") != "ok":
            out[ray] = 0.0
            continue
        ppw_u, vals_u = _collect_meas_ppw_samples(block["pairs"])
        out[ray] = _interp_ppl_kh2(ppw_u, vals_u, target_ppw)
    return out


def theory_deg_per_lambda_at_ppw(
    f_max: float,
    target_ppw: float,
    *,
    c: float,
    cfl: float,
    ray: RayName,
) -> float:
    h = c / (f_max * target_ppw)
    dt = cfl * h / (c * SQRT3)
    ppl = theory_phase_deg_per_lambda(
        np.array([f_max], dtype=np.float64), c=c, dt=dt, h=h, ray=ray
    )
    return float(ppl[0])


def _interp_at_ppw(
    freqs: np.ndarray,
    ppw: np.ndarray,
    values: np.ndarray,
    target_ppw: float,
) -> float:
    order = np.argsort(ppw)
    ppw_s = ppw[order]
    val_s = values[order]
    if target_ppw <= ppw_s[0]:
        return float(val_s[0]) * (target_ppw / ppw_s[0]) ** 2
    if target_ppw >= ppw_s[-1]:
        return float(val_s[-1]) * (ppw_s[-1] / target_ppw) ** 2
    return float(np.interp(target_ppw, ppw_s, val_s))


def summarize_ray_for_projection(
    pair_results: list[dict[str, Any]], target_ppw: float = 20.0
) -> float:
    """Lag (deg/λ) at *target_ppw* on the longest pair (kh² extrapolation)."""
    ppw_u, vals_u = _collect_meas_ppw_samples(pair_results)
    if ppw_u.size == 0:
        return 0.0
    return _interp_ppl_kh2(ppw_u, vals_u, target_ppw)


def theory_at_fmax(
    f_max: float,
    *,
    c: float,
    dt: float,
    h: float,
    ray: RayName,
) -> float:
    ppl = theory_phase_deg_per_lambda(
        np.array([f_max], dtype=np.float64), c=c, dt=dt, h=h, ray=ray
    )
    return float(ppl[0])


def _ppw_for_phase_budget(
    *,
    c: float,
    f_max: float,
    path_m: float,
    phase_deg_max: float,
    cfl: float,
    analysis: dict[str, Any],
) -> tuple[float | None, float | None]:
    """Minimum ppw so worst-ray measured phase ≤ *phase_deg_max* on *path_m*."""
    n_lam = path_m * f_max / c
    if n_lam <= 0.0:
        return None, None
    ppl_budget = phase_deg_max / n_lam
    ppw_grid = np.linspace(8.0, 80.0, 200)
    worst_meas = [
        max(abs(v) for v in measured_deg_per_lambda_at_ppw(analysis, float(p)).values())
        for p in ppw_grid
    ]
    ok = np.where(np.asarray(worst_meas) <= ppl_budget)[0]
    if ok.size == 0:
        return None, ppl_budget
    ppw_meas = float(ppw_grid[int(ok[0])])
    worst_theory = [
        max(
            theory_deg_per_lambda_at_ppw(
                f_max, float(p), c=c, cfl=cfl, ray=r  # type: ignore[arg-type]
            )
            for r in ("axis", "face_diag", "body_diag")
        )
        for p in ppw_grid
    ]
    ok_t = np.where(np.asarray(worst_theory) <= ppl_budget)[0]
    ppw_theory = float(ppw_grid[int(ok_t[0])]) if ok_t.size else None
    return ppw_meas, ppw_theory


def application_projection(
    *,
    c: float,
    f_max: float,
    cfl: float,
    box_mm: tuple[float, ...],
    analysis: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Worst-direction phase (deg) for path lengths 1× and 2× box size."""
    lam = c / f_max
    rows: list[dict[str, Any]] = []
    for box_m in box_mm:
        box_m = box_m * 1e-3
        for mult in (1.0, 2.0):
            path_m = mult * box_m
            n_lam = path_m / lam
            for ppw in _APP_PPW_VALUES:
                meas_by_ray = measured_deg_per_lambda_at_ppw(analysis, ppw)
                theory_by_ray = {
                    ray: theory_deg_per_lambda_at_ppw(
                        f_max, ppw, c=c, cfl=cfl, ray=ray  # type: ignore[arg-type]
                    )
                    for ray in meas_by_ray
                }
                theory_worst = max(theory_by_ray.values())
                meas_worst = max(meas_by_ray.values())
                phase_meas = meas_worst * n_lam
                phase_theory = theory_worst * n_lam
                rows.append(
                    {
                        "box_mm": box_m * 1e3,
                        "path_mult": mult,
                        "path_m": path_m,
                        "ppw": ppw,
                        "f_hz": f_max,
                        "n_wavelengths": n_lam,
                        "meas_deg_per_lam_by_ray": meas_by_ray,
                        "theory_deg_per_lam_by_ray": theory_by_ray,
                        "phase_deg_meas_worst_ray": phase_meas,
                        "phase_deg_theory_worst_ray": phase_theory,
                    }
                )
    ppw_meas, ppw_theory = _ppw_for_phase_budget(
        c=c,
        f_max=f_max,
        path_m=0.6,
        phase_deg_max=_APP_PHASE_THRESH_DEG,
        cfl=cfl,
        analysis=analysis,
    )
    meta = {
        "path_m_2x_300mm": 0.6,
        "phase_thresh_deg": _APP_PHASE_THRESH_DEG,
        "ppw_meas_meets_thresh": ppw_meas,
        "ppw_theory_meets_thresh": ppw_theory,
    }
    return rows, meta


def pick_grid_n(*, quick: bool, cpml: int) -> int:
    interior = _QUICK_INTERIOR_CELLS if quick else _DEFAULT_INTERIOR_CELLS
    return interior + 2 * cpml


def _traces_for_usable(
    trace_block: np.ndarray,
    all_pts: list[dict[str, Any]],
    usable: list[dict[str, Any]],
) -> np.ndarray:
    row_by_cells = {pt["r_cells"]: i for i, pt in enumerate(all_pts)}
    rows = [trace_block[row_by_cells[pt["r_cells"]]] for pt in usable]
    return np.stack(rows, axis=0)


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def print_tables(
    d: object,
    cfg: Config,
    analysis: dict[str, Any],
    stability: dict[str, Any],
    projection_rows: list[dict[str, Any]],
    projection_meta: dict[str, Any],
    gate_pass: bool,
    app_flag: str,
    *,
    ppw_cli: float,
    f_ref: float,
) -> None:
    print("\nC2-dispersion — boxed_piston Stage A")
    interior = _interior_cells(d.nx, _CPML_CELLS)
    print(
        f"  h={d.h * 1e3:.4f} mm (c/({f_ref/1e3:.0f} kHz·{ppw_cli:g} ppw))  "
        f"dt={d.dt * 1e6:.3f} µs  S={cfg.air.c * d.dt / d.h:.3f}  "
        f"ppw@f_ref={d.ppw_realized:.2f}  grid {d.nx}³ (interior {interior}³)"
    )

    print("\n=== Phase lag per λ (longest pair; gate dr/λ ≥ 1) ===")
    print(
        "  Gate tolerance per frequency: "
        f"max({_GATE_REL_TOL:.0%}·|theory|, {_GATE_ABS_DEG_PER_LAM:.2f}°/λ). "
        "Absolute floor raised from 0.05°/λ: at low f (high ppw) theory lag is small "
        "(~0.26°/λ at ppw 40) but measured lag has a floor ~0.06–0.2°/λ from Tukey "
        "windowing, Δr/λ ≈ 1 at the low-f end, and body-diagonal noise where theory ≈ 0."
    )
    hdr = (
        f"{'ray':<12} {'r1→r2':>12} "
        f"{'⟨meas⟩':>8} {'max m':>8} {'⟨thr⟩':>8} {'max t':>8} "
        f"{'max|Δ|':>8} {'gate':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for ray, block in analysis.items():
        if block.get("status") != "ok":
            print(f"{ray:<12}  — {block.get('status')}")
            continue
        pr = _longest_pair(block["pairs"])
        if pr is None or pr.get("n_gate_freq", 0) == 0:
            print(f"{ray:<12}  — insufficient band")
            continue
        gok = "PASS" if pr.get("gate_ok") else "FAIL"
        span = f"{pr['r1_cells']}→{pr['r2_cells']}c"
        print(
            f"{ray:<12} {span:>12} "
            f"{pr.get('mean_meas_lag', 0.0):8.3f} "
            f"{pr.get('max_meas_lag', 0.0):8.3f} "
            f"{pr.get('mean_theory_lag', 0.0):8.3f} "
            f"{pr.get('max_theory_lag', 0.0):8.3f} "
            f"{pr.get('max_abs_lag_err', 0.0):8.3f} {gok:>6}"
        )

    print(
        f"\n=== Mean gated-band lag vs theory (informational; ±"
        f"{_GATE_INFO_MEAN_REL_TOL:.0%}, only if ⟨thr⟩ > "
        f"{_GATE_INFO_MEAN_THEORY_MIN:.1f}°/λ; does not gate) ==="
    )
    for ray, block in analysis.items():
        if block.get("status") != "ok":
            print(f"  {ray:<12} — {block.get('status')}")
            continue
        pr = _longest_pair(block["pairs"])
        if pr is None or pr.get("n_gate_freq", 0) == 0:
            print(f"  {ray:<12} — insufficient band")
            continue
        info = pr.get("info_mean_agree_ok")
        if info is None:
            print(
                f"  {ray:<12} N/A (⟨thr⟩={pr.get('mean_theory_lag', 0.0):.3f}°/λ "
                f"≤ {_GATE_INFO_MEAN_THEORY_MIN:.1f}°/λ)"
            )
        else:
            flag = "agrees" if info else "differs"
            rel = pr.get("info_mean_rel_err", 0.0)
            print(
                f"  {ray:<12} {flag}: ⟨meas⟩={pr.get('mean_meas_lag', 0.0):.3f} "
                f"⟨thr⟩={pr.get('mean_theory_lag', 0.0):.3f}°/λ "
                f"(|Δ|/⟨thr⟩={rel:.1%})"
            )

    print("\n=== Long-run stability (last 2000 steps) ===")
    print(
        f"  steps={stability['n_steps']}  peak|p|={stability['peak_abs_p']:.6e}  "
        f"tail max|p|={stability['tail_max_abs_p']:.6e}  "
        f"ratio={stability['stability_ratio']:.6e}"
    )

    print("\n=== Application projection (worst ray, deg phase at f_max) ===")
    phdr = (
        f"{'box mm':>8} {'×path':>6} {'ppw':>5} "
        f"{'meas °':>10} {'theory °':>10}"
    )
    print(phdr)
    print("-" * len(phdr))
    for row in projection_rows:
        print(
            f"{row['box_mm']:8.0f} {row['path_mult']:6.1f} {row['ppw']:5.0f} "
            f"{row['phase_deg_meas_worst_ray']:10.3f} "
            f"{row['phase_deg_theory_worst_ray']:10.3f}"
        )
    print(f"  2×300 mm @ ppw=20 @ f_max: {app_flag}")
    pm = projection_meta.get("ppw_meas_meets_thresh")
    pt = projection_meta.get("ppw_theory_meets_thresh")
    if pm is not None:
        print(
            f"  ppw to meet {projection_meta['phase_thresh_deg']:.0f}° on "
            f"2×300 mm @ f_max: meas ≥{pm:.1f}"
            + (f", theory ≥{pt:.1f}" if pt is not None else "")
        )
    else:
        print(
            f"  ppw to meet {projection_meta['phase_thresh_deg']:.0f}° on "
            "2×300 mm @ f_max: not reached in ppw∈[8,80] (meas extrapolation)"
        )

    print("\n=== GATE (C2-dispersion) ===")
    stab_ok = stability["stability_ratio"] < 1e-3
    print(
        f"  dispersion vs theory (ppw { _GATE_PPW_MIN:.0f}–{_GATE_PPW_MAX:.0f}): "
        f"{'PASS' if gate_pass else 'FAIL'}"
    )
    print(f"  stability ratio < 1e-3: {'PASS' if stab_ok else 'FAIL'}")
    print(
        f"  Overall: {'PASS' if (gate_pass and stab_ok) else 'FAIL'}"
    )


def maybe_plot(
    analysis: dict[str, Any],
    out_path: Path | None,
) -> None:
    if out_path is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping PNG", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ray, block in analysis.items():
        for pr in block["pairs"]:
            if pr.get("n_plot_freq", 0) == 0:
                continue
            ppw = np.asarray(pr["ppw"], dtype=np.float64)
            meas = np.asarray(pr["meas_lag_deg_per_lam"], dtype=np.float64)
            theory = np.asarray(pr["theory_lag_deg_per_lam"], dtype=np.float64)
            longest = bool(pr.get("is_longest_pair"))
            in_gate = np.asarray(pr.get("bin_in_gate", []), dtype=bool)
            alpha_meas = np.where(
                in_gate,
                0.85 if longest else 0.35,
                0.25 if longest else 0.12,
            )
            sizes = np.where(in_gate, 14 if longest else 8, 6)
            ax.scatter(
                ppw,
                meas,
                s=sizes,
                alpha=alpha_meas,
                c="C0" if longest else "C0",
                label=f"{ray} meas" if longest else None,
            )
            if longest:
                order = np.argsort(ppw)
                ax.plot(
                    ppw[order],
                    theory[order],
                    "C3-",
                    linewidth=1.2,
                    label=f"{ray} theory",
                )
    ax.set_xlabel("points per wavelength")
    ax.set_ylabel("phase lag (deg/λ), + = numerical wave slower")
    ax.set_title("C2 dispersion: measured lag vs Yee theory")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="C2-dispersion Stage A: Yee phase error vs ppw"
    )
    parser.add_argument(
        "--f-ref",
        type=float,
        default=20_000.0,
        help="Reference frequency for grid spacing (Hz, default 20 kHz)",
    )
    parser.add_argument(
        "--ppw",
        type=float,
        default=8.0,
        help="PPW at f-ref used to set h (coarse grid; default 8)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller domain and shorter receiver offsets",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "validation" / "results" / "c02_dispersion",
        help="Output directory for JSON (and optional PNG)",
    )
    parser.add_argument(
        "--box-mm",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_BOX_MM),
        help="Box sizes (mm) for application projection (default 100,200,300)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib PNG",
    )
    parser.add_argument(
        "--stability-steps",
        type=int,
        default=_DEFAULT_STABILITY_STEPS,
        help=f"Total FDTD steps (default {_DEFAULT_STABILITY_STEPS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-frequency analysis bins and skipped receivers",
    )
    parser.add_argument(
        "--keep-h5",
        action="store_true",
        help="No FDTD HDF5 in this script; reserved for validation CLI consistency",
    )
    args = parser.parse_args()
    with vc.keep_awake("c02_dispersion"):
        return _c02_body(args)


def _c02_body(args: argparse.Namespace) -> int:
    vc.require_h5_disk(
        0,
        vc.workdir_disk_parent(keep=args.keep_h5),
        label="c02_dispersion",
    )
    run_timer = _RunTimer()
    box_mm = tuple(float(x.strip()) for x in args.box_mm.split(",") if x.strip())
    if args.quick:
        ok, msg = quick_geometry_feasible(
            f_ref=args.f_ref, ppw=args.ppw, cpml=_CPML_CELLS
        )
        if not ok:
            print(f"ERROR: {msg}", file=sys.stderr)
            return 2
    run_timer.begin("setup")
    n, dist_cells, geo_label = plan_geometry(quick=args.quick)
    g, d, cfg, cpml_st, coeffs, slabs = build_dispersion_domain(
        f_ref=args.f_ref, ppw=args.ppw, n=n, cpml_cells=_CPML_CELLS
    )
    ic = jc = kc = _center_index(n)
    raw_rays = _ray_indices(n, cpml=_CPML_CELLS, dist_cells=dist_cells)
    receivers = attach_physical_radii(raw_rays, g.h)
    n_pre, _ = _choose_n_steps(
        d,
        cfg,
        g,
        receivers,
        ic=ic,
        jc=jc,
        kc=kc,
        usable=None,
        stability_steps=args.stability_steps,
    )
    usable = prepare_usable_receivers(
        g,
        cfg,
        d,
        receivers,
        ic=ic,
        jc=jc,
        kc=kc,
        trace_len=n_pre,
        verbose=args.verbose,
    )
    n_analysis, n_total = _choose_n_steps(
        d,
        cfg,
        g,
        receivers,
        ic=ic,
        jc=jc,
        kc=kc,
        usable=usable,
        stability_steps=args.stability_steps,
    )
    f_lo, f_hi = _ppw_frequency_bounds(cfg.air.c, d.h)
    est_ram = estimate_memory_bytes(n, n, n, _CPML_CELLS)
    n_rec = sum(len(v) for v in receivers.values())
    S = cfg.air.c * d.dt / d.h
    setup_detail = (
        f"grid {n}³ h={d.h * 1e3:.3f} mm Δt={d.dt * 1e6:.2f} µs S={S:.3f} "
        f"ppw@f_ref={d.ppw_realized:g} pulse t₀={d.pulse_t0 * 1e3:.2f} ms "
        f"receivers={n_rec} steps={n_total:,} est RAM {_fmt_bytes(est_ram)} "
        f"band {f_lo / 1e3:.1f}–{min(f_hi, d.f_max) / 1e3:.1f} kHz "
        f"({geo_label})"
    )
    run_timer.end("setup")
    run_timer.banner("setup", setup_detail)

    run_timer.begin("compile")
    warm_ms = warmup_compile(
        g, d, cfg, coeffs, slabs, cpml_st, ic=ic, jc=jc, kc=kc
    )
    run_timer.end("compile")
    run_timer.banner("compile", f"Numba warm-up {warm_ms:.1f} ms (1 step)")

    run_timer.begin("sim")
    run_timer.banner("sim", f"{n_total:,} steps, ~{n_total * d.dt * 1e3:.1f} ms sim time")
    traces, stability = run_simulation(
        g,
        d,
        cfg,
        coeffs,
        slabs,
        cpml_st,
        receivers,
        n_total=n_total,
        ic=ic,
        jc=jc,
        kc=kc,
        run_timer=run_timer,
    )
    run_timer.end("sim")

    run_timer.begin("analysis")
    run_timer.banner("analysis", "Yee small-k check + transfer ratios")
    theory_check = verify_theory_small_k(
        c=cfg.air.c, dt=d.dt, h=d.h, verbose=args.verbose
    )
    analysis: dict[str, Any] = {}
    gate_pass = True
    sign_check_done = False
    trace_len = traces["axis"].shape[1]
    usable = prepare_usable_receivers(
        g,
        cfg,
        d,
        receivers,
        ic=ic,
        jc=jc,
        kc=kc,
        trace_len=trace_len,
        verbose=args.verbose,
    )
    for ray in ("axis", "face_diag", "body_diag"):
        block, sign_check_done = analyze_ray(
            g,
            d,
            cfg,
            usable[ray],
            _traces_for_usable(traces[ray], receivers[ray], usable[ray]),
            ray=ray,
            ic=ic,
            jc=jc,
            kc=kc,
            sign_check_done=sign_check_done,
            verbose=args.verbose,
            run_timer=run_timer,
        )
        analysis[ray] = block
        if block.get("status") != "ok":
            gate_pass = False
        lp = _longest_pair(block["pairs"])
        if lp is None or lp.get("n_gate_freq", 0) == 0:
            gate_pass = False
        elif not lp.get("gate_ok", False):
            gate_pass = False

    rays_summary: dict[RayName, float] = {
        ray: summarize_ray_for_projection(analysis[ray]["pairs"], target_ppw=20.0)
        for ray in analysis
    }
    theory_fmax: dict[RayName, float] = {
        ray: theory_at_fmax(d.f_max, c=cfg.air.c, dt=d.dt, h=d.h, ray=ray)
        for ray in analysis
    }
    projection_rows, projection_meta = application_projection(
        c=cfg.air.c,
        f_max=d.f_max,
        cfl=cfl_for_legacy_derive(cfg.numerics),
        box_mm=box_mm,
        analysis=analysis,
    )
    run_timer.end("analysis")

    run_timer.begin("output")
    key_row = next(
        (
            r
            for r in projection_rows
            if r["box_mm"] == 300.0 and r["path_mult"] == 2.0 and r["ppw"] == 20.0
        ),
        None,
    )
    if key_row is None:
        app_flag = "REVIEW"
    else:
        phase = key_row["phase_deg_meas_worst_ray"]
        app_flag = "OK" if phase <= 10.0 else "REVIEW"

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"c02_dispersion_{ts}.json"
    png_path = None if args.no_plot else out_dir / f"c02_dispersion_{ts}.png"

    payload: dict[str, Any] = {
        "timestamp_utc": ts,
        "f_ref_hz": args.f_ref,
        "ppw_target": args.ppw,
        "quick": args.quick,
        "grid_n": n,
        "cpml_cells": _CPML_CELLS,
        "dist_cells": list(dist_cells),
        "geometry_plan": geo_label,
        "pulse_t0_s": d.pulse_t0,
        "interior_cells": _interior_cells(n, _CPML_CELLS),
        "usable_receivers": usable,
        "analysis_band_db": _SOURCE_BAND_DB,
        "gate_ppw_range": [_GATE_PPW_MIN, _GATE_PPW_MAX],
        "derived": {
            "h_m": d.h,
            "dt_s": d.dt,
            "f_max_hz": d.f_max,
            "ppw_at_f_ref": d.ppw_realized,
            "n_analysis_steps": n_analysis,
            "n_total_steps": n_total,
        },
        "receivers": receivers,
        "stability": stability,
        "analysis": analysis,
        "application_projection": projection_rows,
        "rays_summary_deg_per_lam_at_ppw20": rays_summary,
        "theory_deg_per_lam_at_fmax": theory_fmax,
        "theory_small_k_check_ppw20": theory_check,
        "application_meta": projection_meta,
        "stability_steps": args.stability_steps,
        "gate": {
            "dispersion_pass": gate_pass,
            "stability_pass": stability["stability_ratio"] < 1e-3,
            "application_2x_300mm_ppw20": app_flag,
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    plot_msg = "no plot" if args.no_plot else str(png_path)
    run_timer.end("output")
    run_timer.banner("output", f"JSON {json_path.name}; {plot_msg}")

    maybe_plot(analysis, png_path)

    run_timer.begin("gate")
    run_timer.banner("gate", "tables and PASS/FAIL")
    print_tables(
        d,
        cfg,
        analysis,
        stability,
        projection_rows,
        projection_meta,
        gate_pass,
        app_flag,
        ppw_cli=args.ppw,
        f_ref=args.f_ref,
    )
    run_timer.end("gate")
    total = run_timer.elapsed()
    print("\n=== Runtime (wall) ===", flush=True)
    for name in ("setup", "compile", "sim", "analysis", "output", "gate"):
        t = run_timer.phases.get(name)
        if t is not None:
            print(f"  {name:<10} {t:7.2f} s", flush=True)
    print(f"  {'total':<10} {total:7.2f} s", flush=True)
    return 0 if (gate_pass and stability["stability_ratio"] < 1e-3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
