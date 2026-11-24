"""Parametric geometry FDTD stepping (v0.3 step 6b1; recording in 6b2)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from boxrad.config import (
    Config,
    Derived,
    estimate_memory_bytes,
    provisional_geom_cfl,
    resolve_parametric_cfl,
    test_source_active,
)
from boxrad.cutcell import Fractions, compute_fractions
from boxrad.cpml import make_cpml_state
from boxrad.cutkernel import CutOperator, step_cut
from boxrad.geom.geometry import Geometry, build_geometry, radiator_clearance
from boxrad.geomderive import GeomGrid, cut_grid, derive_geom, validate_geom
from boxrad.geomop import (
    GeomOpCache,
    assert_test_source_cell_fluid,
    build_geom_op_cache,
    build_run_operator,
)
from boxrad.grid import Fields, allocate_fields
from boxrad.kernels import UpdateCoeffs, make_update_coeffs
from boxrad.runner import (
    STABILITY_ABS_P_MAX,
    STABILITY_CHECK_INTERVAL,
    _cells_across_radiator,
    _estimate_wall_time_s,
    estimate_output_bytes,
    resolve_decimation_factor,
)
from boxrad.source import velocity_at_half_step

_F32 = np.float32


@dataclass
class GeomRun:
    cfg: Config
    derived: Derived
    gg: GeomGrid
    geometry: Geometry
    fields: Fields
    coeffs: UpdateCoeffs
    cpml: object
    op: CutOperator
    info: dict[str, Any]
    sigma: float
    t0: float
    dt: float
    warnings: list[str]
    test_source: tuple[int, np.float32, np.float32] | None = None


def _stability_failed(p: np.ndarray, p_max: float) -> bool:
    if not np.all(np.isfinite(p)):
        return True
    return float(np.max(np.abs(p))) > p_max


def _print_warnings(warnings: list[str]) -> None:
    for msg in warnings:
        print(f"warning: {msg}", file=sys.stderr)


def _test_source_injection(
    cfg: Config,
    derived: Derived,
    coeffs: UpdateCoeffs,
    fractions,
) -> tuple[int, np.float32, np.float32] | None:
    if not test_source_active(cfg):
        return None
    idx = derived.test_source_cell_index
    if idx is None:
        raise ValueError("derived.test_source_cell_index missing for numerics.test_source run")
    assert_test_source_cell_fluid(fractions, idx)
    h = float(derived.h)
    flat = int(np.ravel_multi_index(idx, (derived.nx, derived.ny, derived.nz)))
    coeff = _F32(float(coeffs.cp) / (h * h))
    return flat, coeff, _F32(0.0)


def _derive_geom_with_cfl_resolution(
    cfg: Config,
    geometry: Geometry,
) -> tuple[Derived, GeomGrid, Fractions, bool]:
    """``derive_geom`` with cut-cell CFL auto; returns whether a second derive pass ran."""
    provisional = provisional_geom_cfl(cfg.numerics)
    derived, gg = derive_geom(cfg, geometry, cfl_safety=provisional)
    cgrid = cut_grid(gg)
    fractions = compute_fractions(geometry, cgrid)
    n_cut = int(fractions.stats.get("n_cut_cells", 0))
    resolved, source = resolve_parametric_cfl(cfg.numerics, n_cut)
    second_pass = resolved != provisional
    if second_pass:
        derived, gg = derive_geom(cfg, geometry, cfl_safety=resolved)
    derived = replace(
        derived,
        cfl_safety_resolved=resolved,
        cfl_safety_source=source,
    )
    return derived, gg, fractions, second_pass


def resolve_drive_indices(cfg: Config) -> list[int]:
    """Radiator indices to drive, in config order; radiator 0 is always included."""
    if test_source_active(cfg):
        return [0]
    spec = cfg.geometry_spec
    if spec is None or not spec.radiators:
        return [0]
    names = [r.name for r in spec.radiators]
    n_rad = len(names)
    requested = cfg.numerics.drive_radiators
    if requested is None:
        return list(range(n_rad))
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    chosen: set[int] = set()
    for nm in requested:
        if nm not in name_to_idx:
            raise ValueError(f"unknown drive_radiators name {nm!r}")
        chosen.add(name_to_idx[nm])
    if 0 not in chosen:
        print(
            "note: radiator 0 added to numerics.drive_radiators",
            file=sys.stderr,
        )
        chosen.add(0)
    return [i for i in range(n_rad) if i in chosen]


def prepare_geom_run(
    cfg: Config,
    *,
    driven: int = 0,
    cache: GeomOpCache | None = None,
) -> GeomRun:
    geometry = build_geometry(cfg)
    if geometry is None:
        raise ValueError("prepare_geom_run requires parametric geometry")
    derived, gg, fractions, _ = _derive_geom_with_cfl_resolution(cfg, geometry)
    errors, warnings = validate_geom(cfg, derived, geometry)
    _print_warnings(warnings)
    if errors:
        raise ValueError("; ".join(errors))
    fields = allocate_fields(gg)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, derived.h, derived.dt)
    cpml = make_cpml_state(gg, cfg, derived, derived.dt)
    if cache is None:
        cache = build_geom_op_cache(
            cfg, derived, gg, geometry, fractions=fractions
        )
    op, info = build_run_operator(
        cfg, derived, gg, geometry, driven=driven, cache=cache
    )
    inj = _test_source_injection(cfg, derived, coeffs, info["fractions"])
    sigma = derived.pulse_sigma
    t0 = derived.pulse_t0
    return GeomRun(
        cfg=cfg,
        derived=derived,
        gg=gg,
        geometry=geometry,
        fields=fields,
        coeffs=coeffs,
        cpml=cpml,
        op=op,
        info=info,
        sigma=sigma,
        t0=t0,
        dt=derived.dt,
        warnings=warnings,
        test_source=(inj[0], inj[1], inj[2]) if inj is not None else None,
    )


def step_geom(run: GeomRun, n: int) -> None:
    v_now = _F32(
        velocity_at_half_step(n, float(run.dt), float(run.sigma), float(run.t0))
    )
    nf = len(run.op.radiators_fast)
    npatch = len(run.op.radiators_patch)
    if test_source_active(run.cfg):
        v_list: list[np.float32] = [_F32(0.0)] * (nf + npatch)
    else:
        v_list = [v_now]
    test_src = None
    if run.test_source is not None:
        flat, coeff, _ = run.test_source
        test_src = (flat, coeff, v_now)
    step_cut(
        run.fields,
        run.coeffs,
        run.op,
        run.cpml,
        v_list,
        test_source=test_src,
    )


def simulate_geom(
    cfg: Config,
    n_steps: int | None = None,
    stability_interval: int = STABILITY_CHECK_INTERVAL,
    stability_p_max: float = STABILITY_ABS_P_MAX,
) -> GeomRun:
    run = prepare_geom_run(cfg)
    n_max = n_steps if n_steps is not None else run.derived.n_steps
    for n in range(n_max):
        step_geom(run, n)
        if (n + 1) % stability_interval == 0 or n + 1 == n_max:
            if _stability_failed(run.fields.p, stability_p_max):
                raise RuntimeError(
                    f"stability guard failed at step {n} "
                    f"(non-finite p or |p| > {stability_p_max:.3g})"
                )
    return run


def multirad_output_basename(base: Path, j: int, rad_name: str) -> str | None:
    """Sibling HDF5 basename for radiator *j*; ``None`` if not a driven run in the set."""
    import re

    if j == 0:
        return base.name
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", rad_name)
    return f"{base.stem}.r{j}_{safe}{base.suffix}"


def multirad_output_path(base: Path, j: int, rad_name: str) -> Path:
    if j == 0:
        return base
    name = multirad_output_basename(base, j, rad_name)
    assert name is not None
    return base.parent / name


def dry_run_geom(cfg: Config, *, out_path: Path | None = None) -> dict[str, Any]:
    geometry = build_geometry(cfg)
    if geometry is None:
        raise ValueError("dry_run_geom requires parametric geometry")
    derived, gg, fractions, _ = _derive_geom_with_cfl_resolution(cfg, geometry)
    errors, warnings = validate_geom(cfg, derived, geometry)
    if errors:
        raise ValueError("; ".join(errors))
    cache = build_geom_op_cache(cfg, derived, gg, geometry, fractions=fractions)
    drive = resolve_drive_indices(cfg)
    _, info = build_run_operator(cfg, derived, gg, geometry, driven=drive[0], cache=cache)
    _print_warnings(warnings)

    m = resolve_decimation_factor(cfg, derived)
    dt_dec = m * derived.dt
    f_nyq = 0.5 / dt_dec if dt_dec > 0.0 else float("nan")
    mem = estimate_memory_bytes(derived.nx, derived.ny, derived.nz, derived.cpml_cells)
    floor = info["floor_stats"]
    fr_stats = info["fractions_stats"]

    clearances: dict[str, float] = {}
    for rd in geometry.radiators:
        cmin, _ = radiator_clearance(geometry, rd, n=64)
        if np.isfinite(cmin):
            clearances[rd.name] = float(cmin)

    n_runs = len(drive)
    per_wall = _estimate_wall_time_s(derived)
    per_bytes = estimate_output_bytes(cfg, derived)
    names = [r.name for r in geometry.radiators]
    n_rad = len(names)
    base = out_path or Path("out.h5")
    run_files: list[str | None] = [None] * max(n_rad, 1)
    planned: list[str] = []
    for j in drive:
        bn = (
            base.name
            if test_source_active(cfg) or j == 0
            else multirad_output_basename(base, j, names[j] if j < n_rad else "radiator")
        )
        if bn is not None:
            planned.append(bn)
        if j < len(run_files):
            run_files[j] = bn
    return {
        "grid_shape": (derived.nx, derived.ny, derived.nz),
        "h": derived.h,
        "dt": derived.dt,
        "n_steps": derived.n_steps,
        "f_min_hz": cfg.band.f_min,
        "f_max_hz": derived.f_max,
        "ppw_realized": derived.ppw_realized,
        "cfl_safety_resolved": derived.cfl_safety_resolved,
        "cfl_safety_source": derived.cfl_safety_source,
        "cells_across_radiator": _cells_across_radiator(cfg, derived),
        "memory_bytes": mem,
        "decimation_m": m,
        "f_nyquist_decimated_hz": f_nyq,
        "alias_level_db": derived.alias_level_db,
        "output_bytes_est": per_bytes,
        "wall_time_s_est": per_wall,
        "zrad_source": "fdtd",
        "geometry": cfg.geometry,
        "S_grid": info["S_grid"],
        "radiator_path": info["path"],
        "n_cut": fr_stats.get("n_cut_cells", 0),
        "n_floored": floor.get("n_floored", 0),
        "delta_eq_over_h": floor.get("delta_eq_over_h", 0.0),
        "clearances_m": clearances,
        "warnings": list(warnings),
        "n_special_cells": info["n_special_cells"],
        "n_closed_faces": info["n_closed_faces"],
        "n_runs": n_runs,
        "wall_time_s_est_per_run": per_wall,
        "wall_time_s_est_total": per_wall * n_runs,
        "output_bytes_est_per_run": per_bytes,
        "output_bytes_est_total": per_bytes * n_runs,
        "planned_output_files": planned,
        "drive_indices": drive,
        "run_files_json_preview": run_files,
    }