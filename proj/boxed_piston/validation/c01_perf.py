#!/usr/bin/env python3
"""C1-perf: Numba timing, memory, threading, and envelope projection (Stage A).

Free-field cubic domain with CPML and a Gaussian volume-velocity point source.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdtd.config import (  # noqa: E402
    Air,
    Band,
    Box,
    Config,
    DEFAULT_CPML_CELLS,
    DEFAULT_RAM_LIMIT_BYTES,
    Numerics,
    Output,
    Radiator,
    SQRT3,
    derive,
    estimate_memory_bytes,
)
from fdtd.cpml import build_cpml_coeffs, make_cpml_state  # noqa: E402
from fdtd.grid import Grid, allocate_fields, build_cpml_slabs, memory_bytes  # noqa: E402
from fdtd.kernels import (  # noqa: E402
    UpdateCoeffs,
    inject_point_source,
    make_update_coeffs,
    point_source_pressure_increment,
    step,
)

_FLOAT32 = np.dtype(np.float32)
_GATE_WALL_S = 2 * 3600.0
_DEFAULT_SIZES = (64, 96, 128, 192, 256)
_QUICK_SIZES = (32, 64, 96)
_DEFAULT_THREADS = (1, 2, 4, "max")
CPML_CELLS = DEFAULT_CPML_CELLS
WARMUP_STEPS = 3


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_thread_list(text: str) -> list[int | str]:
    out: list[int | str] = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "max":
            out.append("max")
        else:
            out.append(int(part))
    return out


def peak_rss_bytes() -> int:
    """``ru_maxrss``: bytes on macOS, kilobytes on Linux and most Unix."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def machine_info() -> dict[str, Any]:
    try:
        import numba

        numba_ver = numba.__version__
        num_threads = numba.get_num_threads()
    except Exception:
        numba_ver = "unknown"
        num_threads = None

    ram_bytes: int | None = None
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
            ram_bytes = int(out)
        except Exception:
            ram_bytes = None
    elif Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                ram_bytes = kb * 1024
                break

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "numba_version": numba_ver,
        "numpy_version": np.__version__,
        "numba_num_threads": num_threads,
        "python_version": sys.version.split()[0],
    }


def _sync_dt(cfg: Config, d: object) -> float:
    dt = cfg.numerics.cfl_safety * d.h / (cfg.air.c * SQRT3)
    d.dt = dt
    return dt


def build_cube(
    n: int,
    *,
    cpml_cells: int = CPML_CELLS,
    f_max: float = 4_000.0,
) -> tuple[Grid, object, Config, object, UpdateCoeffs, object]:
    """Uniform ``n³`` free-field grid for perf (``gap_cells=0``, fixed *h* from *f_max*)."""
    num = Numerics(
        cpml_cells=cpml_cells,
        gap_cells=0,
        ppw=12.0,
        cfl_safety=0.99,
        cell_budget=10**12,
    )
    cfg = Config(
        box=None,
        air=Air(c=343.0, rho=1.204),
        radiator=Radiator("circle", dims=[0.01]),
        band=Band(f_min=20.0, f_max=f_max),
        numerics=num,
        output=Output(path=""),
    )
    d = derive(cfg)
    d.nx = d.ny = d.nz = n
    d.cpml_cells = cpml_cells
    d.total_cells = n**3
    d.memory_bytes = float(estimate_memory_bytes(n, n, n, cpml_cells))
    _sync_dt(cfg, d)
    g = Grid(cfg, d)
    g.cpml = cpml_cells
    g.cpml_slabs = build_cpml_slabs(n, n, n, cpml_cells)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    return g, d, cfg, cpml_st, coeffs, g.cpml_slabs


def _pulse_q(t: float, sigma: float, t0: float) -> float:
    return float(np.exp(-0.5 * ((t - t0) / sigma) ** 2))


def advance_one_step(
    fields: object,
    coeffs: UpdateCoeffs,
    slabs: object,
    cpml_st: object,
    cfg: Config,
    d: object,
    step_index: int,
    *,
    ic: int,
    jc: int,
    kc: int,
) -> None:
    t = step_index * d.dt
    q = _pulse_q(t, d.pulse_sigma, d.pulse_t0)
    val = point_source_pressure_increment(
        cfg.air.rho, cfg.air.c, d.dt, d.h, q
    )
    inject_point_source(fields.p, ic, jc, kc, val)
    step(fields, coeffs, slabs, cpml=cpml_st)


def _median_step_ms(
    g: Grid,
    d: object,
    cfg: Config,
    coeffs: UpdateCoeffs,
    slabs: object,
    cpml_st: object,
    *,
    warmup: int,
    n_timed: int,
) -> tuple[float, int, int]:
    n = d.nx
    ic = jc = kc = n // 2
    fields = allocate_fields(g)
    rss_after_alloc = peak_rss_bytes()

    step_i = 0
    for _ in range(warmup):
        advance_one_step(
            fields, coeffs, slabs, cpml_st, cfg, d, step_i, ic=ic, jc=jc, kc=kc
        )
        step_i += 1

    times_s: list[float] = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        advance_one_step(
            fields, coeffs, slabs, cpml_st, cfg, d, step_i, ic=ic, jc=jc, kc=kc
        )
        step_i += 1
        times_s.append(time.perf_counter() - t0)

    rss_after_steps = peak_rss_bytes()
    median_s = statistics.median(times_s)
    return median_s * 1e3, rss_after_alloc, rss_after_steps


def measure_compile_cold(n: int = 64) -> dict[str, Any]:
    g, d, cfg, cpml_st, coeffs, slabs = build_cube(n)
    fields = allocate_fields(g)
    ic = jc = kc = n // 2
    t0 = time.perf_counter()
    advance_one_step(fields, coeffs, slabs, cpml_st, cfg, d, 0, ic=ic, jc=jc, kc=kc)
    cold_ms = (time.perf_counter() - t0) * 1e3
    hint = (
        "Numba uses cache=True on kernels; re-run this script in a second "
        "process to see warm (cached) first-step time — compile cost may "
        "then be mostly absent from step 1."
    )
    return {
        "grid_n": n,
        "first_step_cold_ms": cold_ms,
        "cache_warm_hint": hint,
    }


def _array_dtypes_ok(fields: object) -> list[str]:
    issues: list[str] = []
    for name in fields.__dataclass_fields__:
        arr = getattr(fields, name)
        if arr.dtype != _FLOAT32:
            issues.append(f"{name} dtype={arr.dtype}")
    return issues


def _coeffs_dtypes_ok(
    coeffs: UpdateCoeffs, cpml_coeffs: object
) -> list[str]:
    issues: list[str] = []
    if type(coeffs.cv) is not np.float32:
        issues.append(f"UpdateCoeffs.cv type={type(coeffs.cv)}")
    if type(coeffs.cp) is not np.float32:
        issues.append(f"UpdateCoeffs.cp type={type(coeffs.cp)}")
    for fname in (
        "b_p",
        "a_p",
        "inv_kappa_m1_p",
        "b_v",
        "a_v",
        "inv_kappa_m1_v",
    ):
        arr = getattr(cpml_coeffs, fname)
        if arr.dtype != _FLOAT32:
            issues.append(f"CpmlCoeffs.{fname} dtype={arr.dtype}")
    return issues


def check_dtypes(n: int = 128) -> dict[str, Any]:
    g, d, cfg, cpml_st, coeffs, slabs = build_cube(n)
    fields = allocate_fields(g)
    ic = jc = kc = n // 2
    for step_i in range(3):
        advance_one_step(
            fields, coeffs, slabs, cpml_st, cfg, d, step_i, ic=ic, jc=jc, kc=kc
        )
    field_issues = _array_dtypes_ok(fields)
    cpml_coeffs = build_cpml_coeffs(g, cfg, d, d.dt)
    coeff_issues = _coeffs_dtypes_ok(coeffs, cpml_coeffs)
    issues = field_issues + coeff_issues
    ok = len(issues) == 0
    if not ok:
        for msg in issues:
            print(f"  dtype issue: {msg}", file=sys.stderr)
    return {"grid_n": n, "ok": ok, "issues": issues}


def measure_grid_sweep(
    sizes: list[int],
    *,
    ram_limit: float,
    warmup: int,
    n_timed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in sizes:
        est = estimate_memory_bytes(n, n, n, CPML_CELLS)
        if est > ram_limit:
            rows.append(
                {
                    "n": n,
                    "skipped": True,
                    "reason": f"estimate {est / 1e9:.2f} GB > RAM limit {ram_limit / 1e9:.2f} GB",
                    "memory_estimate_bytes": est,
                }
            )
            continue
        g, d, cfg, cpml_st, coeffs, slabs = build_cube(n)
        fields = allocate_fields(g)
        allocated = memory_bytes(fields)
        median_ms, rss_alloc, rss_step = _median_step_ms(
            g,
            d,
            cfg,
            coeffs,
            slabs,
            cpml_st,
            warmup=warmup,
            n_timed=n_timed,
        )
        cells = n**3
        cups = cells / (median_ms * 1e-3) if median_ms > 0 else float("inf")
        rows.append(
            {
                "n": n,
                "skipped": False,
                "nx": n,
                "ny": n,
                "nz": n,
                "total_cells": cells,
                "memory_estimate_bytes": int(d.memory_bytes),
                "memory_allocated_bytes": allocated,
                "memory_allocated_vs_estimate_ratio": allocated / d.memory_bytes,
                "peak_rss_after_alloc_bytes": rss_alloc,
                "peak_rss_after_steps_bytes": rss_step,
                "median_ms_per_step": median_ms,
                "cell_updates_per_s": cups,
            }
        )
    return rows


def _resolve_threads(spec: int | str) -> int:
    if spec == "max":
        return os.cpu_count() or 1
    return int(spec)


def measure_thread_scaling(
    n: int,
    thread_specs: list[int | str],
    *,
    warmup: int,
    n_timed: int,
) -> list[dict[str, Any]]:
    import numba

    g, d, cfg, cpml_st, coeffs, slabs = build_cube(n)
    results: list[dict[str, Any]] = []
    t1_ms: float | None = None
    for spec in thread_specs:
        nthr = _resolve_threads(spec)
        numba.set_num_threads(nthr)
        median_ms, _, _ = _median_step_ms(
            g,
            d,
            cfg,
            coeffs,
            slabs,
            cpml_st,
            warmup=warmup,
            n_timed=n_timed,
        )
        if nthr == 1 and t1_ms is None:
            t1_ms = median_ms
        baseline = t1_ms if t1_ms is not None else median_ms
        speedup = baseline / median_ms if median_ms > 0 else float("nan")
        eff = speedup / nthr if nthr > 0 else float("nan")
        results.append(
            {
                "spec": spec,
                "num_threads": nthr,
                "median_ms_per_step": median_ms,
                "speedup_vs_1_thread": speedup,
                "parallel_efficiency": eff,
            }
        )
    return results


def projection_configs() -> list[tuple[str, Config]]:
    return [
        (
            "a_200mm_cube_25mm_piston_20khz",
            Config(
                box=Box(W=0.2, H=0.2, D=0.2),
                radiator=Radiator("circle", dims=[0.05], x_c=0.0, y_c=0.0),
                band=Band(f_min=20.0, f_max=20_000.0),
                output=Output(path=""),
            ),
        ),
        (
            "b_300mm_cube_centred_17khz",
            Config(
                box=Box(W=0.3, H=0.3, D=0.3),
                radiator=Radiator("circle", dims=[0.05], x_c=0.0, y_c=0.0),
                band=Band(f_min=20.0, f_max=17_000.0),
                output=Output(path=""),
            ),
        ),
        (
            "c_300mm_cube_20khz",
            Config(
                box=Box(W=0.3, H=0.3, D=0.3),
                radiator=Radiator("circle", dims=[0.05], x_c=0.0, y_c=0.0),
                band=Band(f_min=20.0, f_max=20_000.0),
                output=Output(path=""),
            ),
        ),
    ]


def _sweep_timing_points(sweep: list[dict[str, Any]]) -> list[tuple[int, float, float]]:
    """(total_cells, median_ms_per_step, cell_updates_per_s) for each non-skipped run."""
    out: list[tuple[int, float, float]] = []
    for r in sweep:
        if r.get("skipped") or "median_ms_per_step" not in r:
            continue
        cells = int(r["total_cells"])
        ms = float(r["median_ms_per_step"])
        cups = float(r.get("cell_updates_per_s", cells / (ms * 1e-3)))
        out.append((cells, ms, cups))
    out.sort(key=lambda x: x[0])
    return out


def project_ms_per_step(
    total_cells: int, sweep: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Map *total_cells* to ms/step: linear interp in-range, throughput extrap outside."""
    pts = _sweep_timing_points(sweep)
    if not pts:
        return None

    cells_min, ms_min, cups_min = pts[0]
    cells_max, ms_max, cups_max = pts[-1]
    tput_small = cups_min if cups_min > 0 else cells_min / (ms_min * 1e-3)
    tput_large = cups_max if cups_max > 0 else cells_max / (ms_max * 1e-3)

    def _ms_from_throughput(cells: float, throughput: float) -> float:
        """seconds/step = cells / throughput; convert to ms/step."""
        return cells / throughput * 1e3

    tc = float(total_cells)
    if len(pts) == 1:
        ms = _ms_from_throughput(tc, tput_large)
        return {
            "ms_per_step": ms,
            "timing_method": "extrap",
            "measured_cells_min": cells_min,
            "measured_cells_max": cells_max,
            "throughput_cells_per_s": tput_large,
            "extrap_factor_vs_largest": tc / cells_max if cells_max > 0 else None,
            "extrap_warn": tc > 4.0 * cells_max,
        }

    if tc < cells_min:
        ms = _ms_from_throughput(tc, tput_small)
        return {
            "ms_per_step": ms,
            "timing_method": "extrap",
            "measured_cells_min": cells_min,
            "measured_cells_max": cells_max,
            "throughput_cells_per_s": tput_small,
            "extrap_factor_vs_largest": None,
            "extrap_warn": False,
        }
    if tc > cells_max:
        ms = _ms_from_throughput(tc, tput_large)
        factor = tc / cells_max
        return {
            "ms_per_step": ms,
            "timing_method": "extrap",
            "measured_cells_min": cells_min,
            "measured_cells_max": cells_max,
            "throughput_cells_per_s": tput_large,
            "extrap_factor_vs_largest": factor,
            "extrap_warn": factor > 4.0,
        }

    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    ms = float(np.interp(tc, xs, ys))
    return {
        "ms_per_step": ms,
        "timing_method": "interp",
        "measured_cells_min": cells_min,
        "measured_cells_max": cells_max,
        "throughput_cells_per_s": None,
        "extrap_factor_vs_largest": None,
        "extrap_warn": False,
    }


def assert_sweep_ms_per_step_consistency(sweep: list[dict[str, Any]]) -> None:
    """Each measured grid must round-trip through project_ms_per_step within 1 %."""
    for r in sweep:
        if r.get("skipped"):
            continue
        cells = int(r["total_cells"])
        measured_ms = float(r["median_ms_per_step"])
        est = project_ms_per_step(cells, sweep)
        assert est is not None, f"no timing model for measured N yielding {cells} cells"
        got_ms = float(est["ms_per_step"])
        if measured_ms <= 0:
            assert got_ms <= 0, f"cells={cells}: got {got_ms} ms/step"
            continue
        rel_err = abs(got_ms - measured_ms) / measured_ms
        assert rel_err <= 0.01, (
            f"cells={cells}: project_ms_per_step={got_ms:.4f} ms/step vs "
            f"measured {measured_ms:.4f} ms/step (rel err {rel_err:.2%})"
        )


def build_projections(
    sweep: list[dict[str, Any]],
    *,
    ram_limit: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, cfg in projection_configs():
        d = derive(cfg)
        fits = d.memory_bytes <= ram_limit
        est = project_ms_per_step(d.total_cells, sweep)
        if est is None:
            ms = None
            wall_s = None
        else:
            ms = est["ms_per_step"]
            wall_s = d.n_steps * ms * 1e-3
        sim_duration_s = d.n_steps * d.dt
        sim_duration_ms = sim_duration_s * 1e3
        max_box_m = max(d.realised_W, d.realised_H, d.realised_D)
        box_transits = (
            sim_duration_s * cfg.air.c / max_box_m if max_box_m > 0 else None
        )
        row: dict[str, Any] = {
            "id": key,
            "total_cells": d.total_cells,
            "nx": d.nx,
            "ny": d.ny,
            "nz": d.nz,
            "memory_estimate_bytes": d.memory_bytes,
            "n_steps": d.n_steps,
            "ms_per_step": ms,
            "projected_wall_s": wall_s,
            "projected_wall_h": wall_s / 3600.0 if wall_s is not None else None,
            "sim_duration_ms": sim_duration_ms,
            "sim_duration_s": sim_duration_s,
            "box_transits": box_transits,
            "max_box_dimension_m": max_box_m,
            "fits_ram": fits,
            "h_m": d.h,
            "dt_s": d.dt,
            "f_max_hz": d.f_max,
        }
        if est is not None:
            row.update(
                {
                    "timing_method": est["timing_method"],
                    "measured_cells_min": est["measured_cells_min"],
                    "measured_cells_max": est["measured_cells_max"],
                    "throughput_cells_per_s": est["throughput_cells_per_s"],
                    "extrap_factor_vs_largest": est["extrap_factor_vs_largest"],
                    "extrap_warn": est["extrap_warn"],
                }
            )
        out.append(row)
    return out


def _fmt_bytes(b: float | int | None) -> str:
    if b is None:
        return "—"
    b = float(b)
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b / 1e3:.1f} kB"


def print_tables(
    compile_info: dict[str, Any],
    sweep: list[dict[str, Any]],
    threads: list[dict[str, Any]],
    dtype_info: dict[str, Any],
    projections: list[dict[str, Any]],
    thread_n: int,
) -> None:
    print("\n=== C1-perf: Numba compile (cold first step) ===")
    print(f"  grid N={compile_info['grid_n']}")
    print(f"  first step: {compile_info['first_step_cold_ms']:.1f} ms")
    print(f"  hint: {compile_info['cache_warm_hint']}")

    print("\n=== Per-step timing vs grid size ===")
    hdr = (
        f"{'N':>5} {'cells':>12} {'est RAM':>10} {'alloc':>10} "
        f"{'RSS alloc':>10} {'RSS step':>10} {'ms/step':>10} {'Mcupds/s':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sweep:
        if r.get("skipped"):
            print(f"{r['n']:>5}  SKIPPED — {r['reason']}")
            continue
        print(
            f"{r['n']:>5} {r['total_cells']:>12} "
            f"{_fmt_bytes(r['memory_estimate_bytes']):>10} "
            f"{_fmt_bytes(r['memory_allocated_bytes']):>10} "
            f"{_fmt_bytes(r['peak_rss_after_alloc_bytes']):>10} "
            f"{_fmt_bytes(r['peak_rss_after_steps_bytes']):>10} "
            f"{r['median_ms_per_step']:>10.3f} "
            f"{r['cell_updates_per_s'] / 1e6:>12.2f}"
        )

    print(f"\n=== Thread scaling (N={thread_n}) ===")
    print(
        f"{'threads':>8} {'ms/step':>10} {'speedup':>10} {'efficiency':>12}"
    )
    for r in threads:
        print(
            f"{r['num_threads']:>8} "
            f"{r['median_ms_per_step']:>10.3f} "
            f"{r['speedup_vs_1_thread']:>10.2f} "
            f"{r['parallel_efficiency']:>12.2f}"
        )

    print("\n=== Dtype check (float32 fields, coeffs, CPML profiles) ===")
    status = "PASS" if dtype_info["ok"] else "FAIL"
    print(f"  {status}" + ("" if dtype_info["ok"] else f": {dtype_info['issues']}"))

    print("\n=== Projections (derive; box as empty domain of same size) ===")
    for p in projections:
        ram = "fits RAM" if p["fits_ram"] else "DOES NOT fit RAM"
        if p["projected_wall_h"] is not None:
            tstr = f"{p['projected_wall_h']:.2f} h ({p['projected_wall_s']:.0f} s)"
        else:
            tstr = "n/a (no timing sweep)"
        method = p.get("timing_method", "n/a")
        ms = p.get("ms_per_step")
        ms_str = f"{ms:.3f}" if ms is not None else "n/a"
        sim_ms = p["sim_duration_ms"]
        transits = p.get("box_transits")
        tr_str = f"{transits:.2f}" if transits is not None else "n/a"
        line = (
            f"  [{p['id']}] cells={p['total_cells']:,} "
            f"est={_fmt_bytes(p['memory_estimate_bytes'])} "
            f"n_steps={p['n_steps']:,} ms/step={ms_str} ({method}) "
            f"sim_duration={sim_ms:.1f} ms box_transits={tr_str} "
            f"proj_wall={tstr} — {ram}"
        )
        print(line)
        if p.get("extrap_warn"):
            fac = p.get("extrap_factor_vs_largest")
            cmax = p.get("measured_cells_max")
            print(
                f"    WARNING: extrap >4× past largest measured "
                f"({fac:.2f}× vs {cmax:,} cells)"
            )

    print("\n=== TODO ===")
    print("  Surface-recording cost: not measurable until Stage C (recorder).")

    print("\n=== GATE (C1-perf) ===")
    pa = next((p for p in projections if p["id"].startswith("a_")), None)
    for p in projections:
        ram = "fits" if p["fits_ram"] else "NO"
        wt = (
            f"{p['projected_wall_h']:.2f} h"
            if p["projected_wall_h"] is not None
            else "n/a"
        )
        print(f"  {p['id']}: RAM {ram}; projected {wt}")
    if pa and pa["fits_ram"] and pa["projected_wall_s"] is not None:
        gate_ok = pa["projected_wall_s"] <= _GATE_WALL_S
        print(
            f"  Overall: {'PASS' if gate_ok else 'REVIEW'} "
            f"(case (a) fits RAM and ≤ 2 h wall time)"
        )
    else:
        print("  Overall: REVIEW (case (a) missing timing or exceeds RAM limit)")


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="C1-perf Stage A performance check")
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(s) for s in _DEFAULT_SIZES),
        help="Comma-separated cube sizes N (default: 64,96,128,192,256)",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default=",".join(str(t) for t in _DEFAULT_THREADS),
        help='Thread counts for scaling test (default: 1,2,4,max)',
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Timed steps per grid size / thread count (default: 20)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: sizes 32,64,96 only",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "validation" / "results" / "c01_perf",
        help="Output directory for JSON results",
    )
    args = parser.parse_args()

    sizes = list(_QUICK_SIZES if args.quick else _parse_int_list(args.sizes))
    thread_specs = _parse_thread_list(args.threads)
    ram_limit = float(DEFAULT_RAM_LIMIT_BYTES)
    thread_grid_n = 128
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"c01_perf_{ts}.json"

    mi = machine_info()
    print("C1-perf — boxed_piston Stage A")
    print(
        f"Machine: {mi['processor']}, {mi['cpu_count']} cores, "
        f"RAM {_fmt_bytes(mi.get('ram_bytes'))}, "
        f"numba {mi['numba_version']}, numpy {mi['numpy_version']}"
    )

    compile_info = measure_compile_cold(n=min(sizes))
    sweep = measure_grid_sweep(
        sizes, ram_limit=ram_limit, warmup=WARMUP_STEPS, n_timed=args.steps
    )
    assert_sweep_ms_per_step_consistency(sweep)
    dtype_info = check_dtypes(n=thread_grid_n)
    if not dtype_info["ok"]:
        raise AssertionError(f"dtype check failed: {dtype_info['issues']}")
    threads = measure_thread_scaling(
        thread_grid_n,
        thread_specs,
        warmup=WARMUP_STEPS,
        n_timed=args.steps,
    )
    projections = build_projections(sweep, ram_limit=ram_limit)

    payload = {
        "timestamp_utc": ts,
        "machine": mi,
        "ram_limit_bytes": ram_limit,
        "sizes_requested": sizes,
        "timed_steps": args.steps,
        "warmup_steps": WARMUP_STEPS,
        "cpml_cells": CPML_CELLS,
        "compile": compile_info,
        "grid_sweep": sweep,
        "dtype_check": dtype_info,
        "thread_scaling": {
            "grid_n": thread_grid_n,
            "results": threads,
        },
        "projections": projections,
        "gate_wall_s_limit": _GATE_WALL_S,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\nResults written to {out_path}")

    print_tables(
        compile_info, sweep, threads, dtype_info, projections, thread_grid_n
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
