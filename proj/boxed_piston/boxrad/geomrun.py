"""Parametric geometry FDTD stepping (v0.3 step 6b1; recording in 6b2)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from boxrad.config import Config, Derived, estimate_memory_bytes
from boxrad.cpml import make_cpml_state
from boxrad.cutkernel import CutOperator, step_cut
from boxrad.geom.geometry import Geometry, build_geometry, radiator_clearance
from boxrad.geomderive import GeomGrid, derive_geom, validate_geom
from boxrad.geomop import build_run_operator
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


def _stability_failed(p: np.ndarray, p_max: float) -> bool:
    if not np.all(np.isfinite(p)):
        return True
    return float(np.max(np.abs(p))) > p_max


def _print_warnings(warnings: list[str]) -> None:
    for msg in warnings:
        print(f"warning: {msg}", file=sys.stderr)


def prepare_geom_run(cfg: Config) -> GeomRun:
    geometry = build_geometry(cfg)
    if geometry is None:
        raise ValueError("prepare_geom_run requires parametric geometry")
    derived, gg = derive_geom(cfg, geometry)
    errors, warnings = validate_geom(cfg, derived, geometry)
    _print_warnings(warnings)
    if errors:
        raise ValueError("; ".join(errors))
    fields = allocate_fields(gg)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, derived.h, derived.dt)
    cpml = make_cpml_state(gg, cfg, derived, derived.dt)
    op, info = build_run_operator(cfg, derived, gg, geometry)
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
    )


def step_geom(run: GeomRun, n: int) -> None:
    v_now = _F32(
        velocity_at_half_step(n, float(run.dt), float(run.sigma), float(run.t0))
    )
    v_list = [v_now]
    step_cut(run.fields, run.coeffs, run.op, run.cpml, v_list)


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


def dry_run_geom(cfg: Config) -> dict[str, Any]:
    geometry = build_geometry(cfg)
    if geometry is None:
        raise ValueError("dry_run_geom requires parametric geometry")
    derived, gg = derive_geom(cfg, geometry)
    errors, warnings = validate_geom(cfg, derived, geometry)
    if errors:
        raise ValueError("; ".join(errors))
    _, info = build_run_operator(cfg, derived, gg, geometry)
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

    return {
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
        "alias_level_db": derived.alias_level_db,
        "output_bytes_est": estimate_output_bytes(cfg, derived),
        "wall_time_s_est": _estimate_wall_time_s(derived),
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
    }