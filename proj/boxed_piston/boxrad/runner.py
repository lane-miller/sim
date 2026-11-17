# Run length and n_steps come only from :func:`boxrad.config.derive` (``Derived.n_steps``).
# This module does not invent a step count or end time when Derived lacks them.

"""FDTD run driver: config validation, time stepping, HDF5 output."""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from boxrad.config import (
    DEFAULT_RAM_LIMIT_BYTES,
    Config,
    Derived,
    derive,
    estimate_memory_bytes,
    format_realized_box_dims,
    validate,
)
from boxrad.cpml import make_cpml_state
from boxrad.grid import Grid, allocate_fields
from boxrad.kernels import make_update_coeffs, step
from boxrad.post.kirchhoff import (
    box_surface_patches,
    huygens_box_patches,
    huygens_open_patches,
)
from boxrad.recorder import (
    Recorder,
    decimation_factor,
    resolve_decimation_factor,
    resolve_huygens_offset,
)
from boxrad.source import BoxBoundary

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
    if cfg.geometry == "baffle":
        xc = float(cfg.radiator.x_c)
        yc = float(cfg.radiator.y_c)
        return (
            -xc - 0.5 * derived.realized_W,
            -yc - 0.5 * derived.realized_H,
            0.0,
        )
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


def estimate_output_bytes(cfg: Config, derived: Derived) -> int:
    """Rough uncompressed payload (datasets only; excludes HDF5 overhead)."""
    m = resolve_decimation_factor(cfg, derived)
    n_full = derived.n_steps
    n_dec = (n_full + m - 1) // m if n_full else 0
    offset = resolve_huygens_offset(cfg)
    if cfg.geometry == "baffle":
        origin = _box_origin(cfg, derived)
        depth = 0.0
        nh = huygens_open_patches(
            derived.realized_W,
            derived.realized_H,
            depth,
            derived.h,
            origin,
            offset_cells=offset,
        ).centers.shape[0]
        nb = 0
    else:
        origin = _box_origin(cfg, derived)
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
    m = resolve_decimation_factor(cfg, derived)
    dt_dec = m * derived.dt
    f_nyq = 0.5 / dt_dec if dt_dec > 0.0 else float("nan")
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
        "f_nyquist_decimated_hz": f_nyq,
        "f_nyquist_over_f_max": f_nyq / derived.f_max if derived.f_max > 0 else float("nan"),
        "alias_level_db": derived.alias_level_db,
        "output_bytes_est": estimate_output_bytes(cfg, derived),
        "wall_time_s_est": _estimate_wall_time_s(derived),
        "zrad_source": "fdtd",
        "geometry": cfg.geometry,
    }
    if cfg.geometry == "box" and cfg.box is not None:
        summary["box_dims"] = format_realized_box_dims(cfg, derived)
    return summary


def _sigfig4(x: float) -> str:
    return f"{float(x):.4g}"


def _print_dry_run_summary(summary: dict[str, Any]) -> None:
    print("dry run summary:")
    for key, val in summary.items():
        if key == "grid_shape":
            print(f"  {key}: {'×'.join(str(v) for v in val)}")
        elif key == "h":
            print(f"  {key}: {_sigfig4(val)} m")
        elif key == "dt":
            print(f"  {key}: {_sigfig4(val)} s")
        elif key == "memory_bytes":
            print(f"  {key}: {_sigfig4(val / 1e9)} GB")
        elif key == "output_bytes_est":
            print(f"  {key}: {_sigfig4(val / 1e6)} MB (approx.)")
        elif key == "wall_time_s_est":
            print(f"  {key}: {_sigfig4(val)} s (C1 envelope)")
        elif key == "alias_level_db":
            print(f"  {key}: {_sigfig4(val)} dB re |S(f_max)|")
        elif key == "f_nyquist_over_f_max":
            print(f"  {key}: {_sigfig4(val)}")
        elif isinstance(val, float):
            print(f"  {key}: {_sigfig4(val)}")
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
    plot: bool = False,
    ram_limit_bytes: float = DEFAULT_RAM_LIMIT_BYTES,
    stability_interval: int = STABILITY_CHECK_INTERVAL,
    stability_p_max: float = STABILITY_ABS_P_MAX,
) -> Path | dict[str, Any]:
    """Validate a config, then run the FDTD and write one HDF5 file (or dry-run).

    Warnings from :func:`boxrad.config.validate` are printed to stderr as
    ``warning: …`` and do not stop the run. After the run, a warning is printed
    if the radiator-pressure tail is above −60 dB re peak.

    Args:
        cfg: :class:`~boxrad.config.Config` or path to a JSON config.
        out_path: HDF5 destination. Default: ``cfg.output.path`` if non-empty,
            else ``<package root>/outputs/<config stem>.h5`` (``run.h5`` when
            ``cfg`` is a ``Config`` object). Parent directories are created.
        dry_run: Only validate and print the derived-parameter summary.
        progress: Print a progress line about every 2 % of steps.
        plot: After a full run, open the geometry window (blocks until closed).
        ram_limit_bytes: Estimated field memory above this is an error
            (default 5 GiB).
        stability_interval: Check the pressure field every this many steps.
        stability_p_max: Abort if ``|p|`` exceeds this (Pa) or is non-finite.

    Returns:
        Path of the written HDF5 file, or with ``dry_run`` the summary ``dict``
        (keys ``grid_shape``, ``h`` [m], ``dt`` [s], ``n_steps``, ``f_min_hz``,
        ``f_max_hz``, ``ppw_realized``, ``cells_across_radiator``,
        ``memory_bytes``, ``decimation_m``, ``output_bytes_est``,
        ``wall_time_s_est`` [s], ``zrad_source``, ``geometry``, and in box mode
        ``box_dims``).

    Raises:
        ValueError: Config errors, also for ``dry_run`` (missing box, radiator
            off the face or within ``min_edge_cells`` of an edge, domain above
            ``numerics.cell_budget``, baffle-mode radiator clipped by more than
            one cell, memory above the limit, Huygens surface not fitting in the
            gap, invalid band).
        RuntimeError: Stability guard tripped; ``/meta`` gets ``aborted=True``
            and ``aborted_at_step``.
    """
    cfg_source = cfg
    config = _load_config(cfg)
    if config.geometry == "box" and config.box is None:
        raise ValueError("run requires cfg.box when geometry=box")

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
    tail_db: float | None = None
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
            tail_db = rec._tail_db
    if tail_db is not None and tail_db > -60.0:
        print(
            f"warning: p_avg tail {tail_db:.1f} dB re peak (> −60 dB); "
            "increase numerics.n_transits",
            file=sys.stderr,
        )
    if plot:
        from boxrad.viz import plot_geometry

        plot_geometry(dest, show=True)
    return dest


@contextmanager
def keep_awake(label: str) -> Generator[None, None, None]:
    """On macOS, run ``caffeinate -i -w <pid>`` for the wrapped block; no-op elsewhere.

    Prevents idle sleep while the block runs. ``caffeinate`` is stopped on exit,
    on SIGINT/SIGTERM (then ``SystemExit(128 + signum)``), and at interpreter
    exit. Start/stop lines are printed with the ``caffeinate`` pid.

    Args:
        label: Informational only (unused).
    """
    _ = label
    proc: subprocess.Popen[bytes] | None = None
    pid_printed: int | None = None

    def _stop() -> None:
        nonlocal proc, pid_printed
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        if pid_printed is not None:
            print(
                f"[keep_awake] caffeinate stopped (pid {pid_printed})",
                flush=True,
            )
        proc = None
        pid_printed = None

    def _signal_handler(signum: int, _frame: object) -> None:
        _stop()
        raise SystemExit(128 + signum)

    if sys.platform == "darwin" and shutil.which("caffeinate"):
        proc = subprocess.Popen(
            ["caffeinate", "-i", "-w", str(os.getpid())],
        )
        pid_printed = proc.pid
        print(f"[keep_awake] caffeinate started (pid {proc.pid})", flush=True)
        atexit.register(_stop)
        prev_int = signal.signal(signal.SIGINT, _signal_handler)
        prev_term = signal.signal(signal.SIGTERM, _signal_handler)
    else:
        prev_int = prev_term = None
    try:
        yield
    finally:
        if prev_int is not None:
            signal.signal(signal.SIGINT, prev_int)
        if prev_term is not None:
            signal.signal(signal.SIGTERM, prev_term)
        try:
            atexit.unregister(_stop)
        except Exception:
            pass
        _stop()


def main(argv: list[str] | None = None) -> int:
    """``boxrad-run CONFIG [-o OUT] [--dry-run] [--no-show] [--plot] [--quiet]``.

    A full run is wrapped in :func:`keep_awake` (``caffeinate`` on macOS). A dry
    run prints the summary and opens the geometry window unless ``--no-show``.
    ``--plot`` opens the geometry window from the HDF5 file after a full run.

    Args:
        argv: Argument list without the program name (default ``sys.argv[1:]``).

    Returns:
        Exit code: 0 on success, 1 on ``ValueError``/``RuntimeError`` (message
        printed to stderr as ``error: …``).
    """
    parser = argparse.ArgumentParser(description="Boxed-piston FDTD run driver")
    parser.add_argument("config", type=Path, help="JSON config path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="HDF5 output path")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print summary only")
    parser.add_argument("--quiet", action="store_true", help="Disable progress lines")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After a full run, open the geometry window (config + result HDF5)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Dry run only: print summary without opening the geometry window",
    )
    args = parser.parse_args(argv)

    result: Path | dict[str, Any] | None = None
    try:
        if args.dry_run:
            result = run(
                args.config,
                out_path=args.output,
                dry_run=True,
                progress=not args.quiet,
                plot=False,
            )
        else:
            with keep_awake("boxrad.runner"):
                result = run(
                    args.config,
                    out_path=args.output,
                    dry_run=False,
                    progress=not args.quiet,
                    plot=False,
                )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if isinstance(result, dict):
        pass
    elif result is not None:
        out_p = Path(result)
        if out_p.is_file():
            print(f"wrote {result}  ({out_p.stat().st_size / 1e6:.2f} MB)")
        else:
            print(f"wrote {result}")

    from boxrad.viz import plot_geometry

    if args.dry_run:
        if not args.no_show:
            plot_geometry(args.config, show=True)
    elif args.plot and result is not None and not isinstance(result, dict):
        plot_geometry(result, show=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())