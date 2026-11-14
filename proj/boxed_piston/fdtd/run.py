# Run length and n_steps come only from :func:`fdtd.config.derive` (``Derived.n_steps``).
# This module does not invent a step count or end time when Derived lacks them.

"""FDTD run driver: config validation, time stepping, HDF5 output."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from fdtd.config import (
    DEFAULT_RAM_LIMIT_BYTES,
    Config,
    Derived,
    derive,
    estimate_memory_bytes,
    validate,
)
from fdtd.cpml import make_cpml_state
from fdtd.grid import Grid, allocate_fields
from fdtd.kernels import make_update_coeffs, step
from fdtd.post.kirchhoff import box_surface_patches, huygens_box_patches
from fdtd.recorder import Recorder, decimation_factor, resolve_huygens_offset
from fdtd.source import BoxBoundary

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None  # type: ignore[assignment]

_PKG_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = _PKG_ROOT / "outputs"

# C1-perf envelope (brief §4): ~375 M cell-updates/s, no surface recording.
_C1_CELL_UPDATES_PER_S = 375.0e6

STABILITY_CHECK_INTERVAL = 100
STABILITY_ABS_P_MAX = 1.0e9


def _load_config(cfg: Config | str | Path) -> Config:
    if isinstance(cfg, Config):
        return cfg
    path = Path(cfg)
    return Config.from_json(path.read_text(encoding="utf-8"))


def _resolve_out_path(cfg: Config, cfg_source: Config | str | Path, out_path: Path | None) -> Path:
    if out_path is not None:
        return Path(out_path)
    if cfg.output.path:
        return Path(cfg.output.path)
    if isinstance(cfg_source, (str, Path)):
        stem = Path(cfg_source).stem or "run"
    else:
        stem = "run"
    return _DEFAULT_OUTPUT_DIR / f"{stem}.h5"


def _box_origin(cfg: Config, derived: Derived) -> tuple[float, float, float]:
    xc = float(cfg.radiator.x_c)
    yc = float(cfg.radiator.y_c)
    return (
        -xc - 0.5 * derived.realized_W,
        -yc - 0.5 * derived.realized_H,
        -derived.realized_D,
    )


def _cells_across_radiator(cfg: Config, derived: Derived) -> float:
    rad = cfg.radiator
    if rad.shape == "circle":
        d = rad.dims[0]
    else:
        d = min(rad.dims[0], rad.dims[1])
    return d / derived.h


def _estimate_output_bytes(cfg: Config, derived: Derived) -> int:
    """Rough uncompressed payload (datasets only; excludes HDF5 overhead)."""
    m = decimation_factor(derived.f_max, derived.dt)
    n_full = derived.n_steps
    n_dec = (n_full + m - 1) // m if n_full else 0
    origin = _box_origin(cfg, derived)
    offset = resolve_huygens_offset(cfg)
    nh = huygens_box_patches(
        derived.realized_W,
        derived.realized_H,
        derived.realized_D,
        derived.h,
        origin,
        offset_cells=offset,
    ).centers.shape[0]
    nb = box_surface_patches(
        derived.realized_W,
        derived.realized_H,
        derived.realized_D,
        derived.h,
        origin,
    ).centers.shape[0]
    nbytes = n_full * 3 * 8  # v_int, v_half, p_avg (float64)
    nbytes += n_dec * (nh * 4 + nh * 4 + nb * 4)  # huygens + box p (float32)
    nbytes += n_dec * 4 * 8  # decimated time stamps (float64)
    return int(nbytes)


def _estimate_wall_time_s(derived: Derived) -> float:
    updates = derived.n_steps * derived.total_cells
    return updates / _C1_CELL_UPDATES_PER_S


def _git_commit_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=_PKG_ROOT,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _numba_thread_count() -> int:
    if numba is None:
        return 0
    return int(numba.get_num_threads())


def _check_memory(derived: Derived, ram_limit_bytes: float) -> None:
    need = estimate_memory_bytes(derived.nx, derived.ny, derived.nz, derived.cpml_cells)
    if need > ram_limit_bytes:
        raise ValueError(
            f"estimated memory {need / 1e9:.2f} GB exceeds "
            f"limit {ram_limit_bytes / 1e9:.2f} GB"
        )


def _print_warnings(warnings: list[str]) -> None:
    for msg in warnings:
        print(f"warning: {msg}", file=sys.stderr)


def _dry_run_summary(cfg: Config, derived: Derived) -> dict[str, Any]:
    m = decimation_factor(derived.f_max, derived.dt)
    mem = estimate_memory_bytes(derived.nx, derived.ny, derived.nz, derived.cpml_cells)
    summary: dict[str, Any] = {
        "grid_shape": (derived.nx, derived.ny, derived.nz),
        "h": derived.h,
        "dt": derived.dt,
        "n_steps": derived.n_steps,
        "f_min_hz": cfg.band.f_min,
        "f_max_hz": derived.f_max,
        "ppw_realized": derived.ppw_realized,
        "cells_across_radiator": _cells_across_radiator(cfg, derived),
        "memory_bytes": mem,
        "decimation_m": m,
        "output_bytes_est": _estimate_output_bytes(cfg, derived),
        "wall_time_s_est": _estimate_wall_time_s(derived),
        "zrad_source": "analytic_baffled" if derived.zrad_fallback else "fdtd",
    }
    return summary


def _print_dry_run_summary(summary: dict[str, Any]) -> None:
    print("dry run summary:")
    for key, val in summary.items():
        if key == "grid_shape":
            print(f"  {key}: {'×'.join(str(v) for v in val)}")
        elif key in ("h", "dt"):
            print(f"  {key}: {val:.6g} m" if key == "h" else f"  {key}: {val:.6g} s")
        elif key == "memory_bytes":
            print(f"  {key}: {val / 1e9:.3f} GB")
        elif key == "output_bytes_est":
            print(f"  {key}: {val / 1e6:.2f} MB (approx.)")
        elif key == "wall_time_s_est":
            print(f"  {key}: {val:.2f} s (C1 envelope)")
        else:
            print(f"  {key}: {val}")


def _stability_failed(p: np.ndarray, p_max: float) -> bool:
    if not np.all(np.isfinite(p)):
        return True
    p_abs = float(np.max(np.abs(p)))
    return p_abs > p_max


def _progress_tick(
    n: int,
    n_steps: int,
    dt: float,
    t_start: float,
    *,
    last_pct: int,
) -> int:
    if n_steps <= 0:
        return last_pct
    pct = int(100 * (n + 1) / n_steps)
    if pct < last_pct + 2 and n + 1 < n_steps:
        return last_pct
    elapsed = time.perf_counter() - t_start
    sim_t = (n + 1) * dt
    rate = (n + 1) / elapsed if elapsed > 0 else 0.0
    eta = (n_steps - n - 1) / rate if rate > 0 else float("inf")
    print(
        f"step {n + 1}/{n_steps}  t={sim_t:.4e} s  "
        f"elapsed={elapsed:.1f} s  ETA={eta:.1f} s  {rate:.1f} steps/s"
    )
    return pct


def run(
    cfg: Config | str | Path,
    out_path: str | Path | None = None,
    *,
    dry_run: bool = False,
    progress: bool = True,
    ram_limit_bytes: float = DEFAULT_RAM_LIMIT_BYTES,
    stability_interval: int = STABILITY_CHECK_INTERVAL,
    stability_p_max: float = STABILITY_ABS_P_MAX,
) -> Path | dict[str, Any]:
    """Validate config, run FDTD (or dry-run summary), write HDF5 unless ``dry_run``."""
    cfg_source = cfg
    config = _load_config(cfg)
    if config.box is None:
        raise ValueError("run requires cfg.box (boxed-piston geometry)")

    derived = derive(config)
    errors, warnings = validate(config, derived, ram_limit_bytes=ram_limit_bytes)
    _print_warnings(warnings)
    if errors:
        raise ValueError("; ".join(errors))

    _check_memory(derived, ram_limit_bytes)

    if dry_run:
        summary = _dry_run_summary(config, derived)
        _print_dry_run_summary(summary)
        return summary

    dest = _resolve_out_path(config, cfg_source, Path(out_path) if out_path else None)
    dest.parent.mkdir(parents=True, exist_ok=True)

    grid = Grid(config, derived)
    fields = allocate_fields(grid)
    coeffs = make_update_coeffs(config.air.rho, config.air.c, derived.h, derived.dt)
    slabs = grid.cpml_slabs
    cpml = make_cpml_state(grid, config, derived, derived.dt)
    boundary = BoxBoundary.from_grid(grid, derived)

    rec: Recorder | None = None
    aborted_step: int | None = None
    t_wall_start = time.perf_counter()
    last_pct = -2
    try:
        rec = Recorder(config, derived, grid, boundary, dest)
        n_steps = derived.n_steps
        for n in range(n_steps):
            step(fields, coeffs, slabs, cpml, boundary, n)
            rec.sample(fields, n)
            if progress:
                last_pct = _progress_tick(
                    n, n_steps, derived.dt, t_wall_start, last_pct=last_pct
                )
            if (n + 1) % stability_interval == 0 or n + 1 == n_steps:
                if _stability_failed(fields.p, stability_p_max):
                    aborted_step = n
                    raise RuntimeError(
                        f"stability guard failed at step {n} "
                        f"(non-finite p or |p| > {stability_p_max:.3g})"
                    )
    except RuntimeError:
        if rec is not None and rec._file is not None and aborted_step is not None:
            meta = rec._file["meta"]
            meta.attrs["aborted"] = True
            meta.attrs["aborted_at_step"] = aborted_step
        raise
    finally:
        if rec is not None:
            wall_s = time.perf_counter() - t_wall_start
            if rec._file is not None:
                meta = rec._file["meta"]
                meta.attrs["wall_time_s"] = wall_s
                meta.attrs["numba_num_threads"] = _numba_thread_count()
                meta.attrs["git_commit"] = _git_commit_hash()
            rec.close()
    return dest


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Boxed-piston FDTD run driver")
    parser.add_argument("config", type=Path, help="JSON config path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="HDF5 output path")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print summary only")
    parser.add_argument("--quiet", action="store_true", help="Disable progress lines")
    args = parser.parse_args(argv)
    try:
        result = run(
            args.config,
            out_path=args.output,
            dry_run=args.dry_run,
            progress=not args.quiet,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if isinstance(result, dict):
        return 0
    print(f"wrote {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())