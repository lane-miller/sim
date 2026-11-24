"""Parametric / STEP dry-run report (v0.3 step 12)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from boxrad.config import (
    Config,
    DEFAULT_RAM_LIMIT_BYTES,
    test_source_active,
)
from boxrad.cutcell import wetted_wall_arrays
from boxrad.geom.geometry import (
    Geometry,
    build_geometry,
    radiator_clearance,
)
from boxrad.geomrun import (
    _derive_geom_with_cfl_resolution,
    multirad_output_basename,
    resolve_drive_indices,
)
from boxrad.geomop import build_geom_op_cache, build_run_operator
from boxrad.geomderive import validate_geom
from boxrad.recorder import (
    _cpml_free_center_bounds,
    resolve_decimation_factor,
    resolve_huygens_offset,
)
from boxrad.geomrec import validate_huygens_clearance_geom
from boxrad.runner import _estimate_wall_time_s, estimate_output_bytes

_F64 = np.float64


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _sigfig4(x: float) -> str:
    return f"{float(x):.4g}"


def _p_index_from_xyz(gg, x: float, y: float, z: float) -> tuple[int, int, int]:
    h = gg.h
    x0 = gg.x0
    i = int(math.floor((float(x) - float(x0[0])) / h))
    j = int(math.floor((float(y) - float(x0[1])) / h))
    k = int(math.floor((float(z) - float(x0[2])) / h))
    return i, j, k


def _probe_in_cpml(
    q: int,
    x: float,
    y: float,
    z: float,
    i: int,
    j: int,
    k: int,
    slabs,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
) -> bool:
    i0, i1 = slabs.i_inner.lo, slabs.i_inner.hi
    j0, j1 = slabs.j_inner.lo, slabs.j_inner.hi
    k0, k1 = slabs.k_inner.lo, slabs.k_inner.hi
    if i0 <= i < i1 and j0 <= j < j1 and k0 <= k < k1:
        return False
    _ = q, x, y, z, bounds_lo, bounds_hi
    return True


def probe_statuses(
    cfg: Config,
    gg,
    geometry: Geometry,
    beta_tilde: np.ndarray,
) -> list[dict[str, Any]]:
    """Per-probe OK / in CPML / other issue (no exceptions)."""
    targets: list[tuple[float, float, float]] = []
    if cfg.output.probes:
        for row in cfg.output.probes:
            if len(row) != 3:
                targets.append((float("nan"), float("nan"), float("nan")))
            else:
                targets.append((float(row[0]), float(row[1]), float(row[2])))
    if not targets:
        return []
    n = len(targets)
    beta_flat = beta_tilde.reshape(-1)
    shape = (gg.nx, gg.ny, gg.nz)
    x0 = gg.x0.astype(_F64)
    h = gg.h
    domain_hi = x0 + np.array([gg.nx, gg.ny, gg.nz], dtype=_F64) * h
    slabs = gg.cpml_slabs
    bounds_lo, bounds_hi = _cpml_free_center_bounds(slabs, x0=x0, h=h)
    planes = list(geometry.planes)
    out: list[dict[str, Any]] = []

    def _behind_plane(xv: float, yv: float, zv: float) -> str | None:
        pxyz = (xv, yv, zv)
        for pl in planes:
            ax = {"x": 0, "y": 1, "z": 2}[pl.axis]
            if pl.side == "-" and pxyz[ax] < float(pl.offset) - 0.5 * h:
                return f"{pl.axis}{pl.side}"
            if pl.side == "+" and pxyz[ax] > float(pl.offset) + 0.5 * h:
                return f"{pl.axis}{pl.side}"
        return None

    for q, (x, y, z) in enumerate(targets):
        row: dict[str, Any] = {"index": q, "xyz_m": [x, y, z]}
        if not all(np.isfinite([x, y, z])):
            row["status"] = "invalid probe entry"
            out.append(row)
            continue
        behind = _behind_plane(x, y, z)
        if behind is not None:
            row["status"] = f"behind rigid plane {behind}"
            out.append(row)
            continue
        if (
            x < float(x0[0])
            or y < float(x0[1])
            or z < float(x0[2])
            or x >= float(domain_hi[0])
            or y >= float(domain_hi[1])
            or z >= float(domain_hi[2])
        ):
            row["status"] = "outside domain"
            out.append(row)
            continue
        i, j, k = _p_index_from_xyz(gg, x, y, z)
        if not (0 <= i < gg.nx and 0 <= j < gg.ny and 0 <= k < gg.nz):
            row["status"] = "out of bounds"
            out.append(row)
            continue
        if _probe_in_cpml(q, x, y, z, i, j, k, slabs, bounds_lo, bounds_hi):
            row["status"] = "in CPML"
            row["ijk"] = [i, j, k]
            out.append(row)
            continue
        flat = int(np.ravel_multi_index((i, j, k), shape))
        if beta_flat[flat] == 0.0:
            row["status"] = "solid cell"
            row["ijk"] = [i, j, k]
            out.append(row)
            continue
        row["status"] = "OK"
        row["ijk"] = [i, j, k]
        out.append(row)
    return out


def _delta_status(floor_status: str) -> str:
    s = floor_status.lower()
    if s == "ok":
        return "OK"
    if s == "check":
        return "CHECK"
    return "WARN"


def _bbox_world(geometry: Geometry) -> list[float] | None:
    if geometry.solid.vertices.shape[0] == 0:
        return None
    xlo, ylo, zlo, xhi, yhi, zhi = geometry.solid.bbox
    return [xlo, ylo, zlo, xhi, yhi, zhi]


def _config_stem(cfg: Config, cfg_path: Path | None) -> str:
    if cfg_path is not None:
        return cfg_path.stem
    if cfg.output.path:
        return Path(cfg.output.path).stem
    return "config"


def geom_report_exit_code(report: dict[str, Any]) -> int:
    errors = report.get("errors") or []
    return 2 if errors else 0


def geom_report(
    cfg: Config,
    *,
    out_path: Path | None = None,
    ram_limit_bytes: float = DEFAULT_RAM_LIMIT_BYTES,
    cfg_path: Path | None = None,
) -> dict[str, Any]:
    """Full pre-run report for parametric / STEP (JSON-serializable)."""
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "config_name": _config_stem(cfg, cfg_path),
        "geometry": {"tag": cfg.geometry},
        "radiators": [],
        "grid": {},
        "time": {},
        "fractions": {},
        "recording": {},
        "errors": errors,
        "warnings": warnings,
    }

    tag = cfg.geometry
    if tag not in ("parametric", "step"):
        errors.append(f"geom_report requires parametric or step geometry, got {tag!r}")
        return report

    geometry: Geometry | None = None
    try:
        geometry = build_geometry(cfg)
    except (ValueError, NotImplementedError, AssertionError) as exc:
        errors.append(str(exc))
        prov = report["geometry"]
        if tag == "step":
            spec = cfg.geometry_spec
            if spec is not None:
                step_meta = getattr(spec, "step", None)
                if step_meta is not None:
                    prov["step_file"] = getattr(step_meta, "path", None)
                    prov["step_units"] = getattr(step_meta, "units", None)
                    prov["step_sha256"] = getattr(step_meta, "sha256", None)
        report["errors"] = _dedupe(errors)
        report["warnings"] = _dedupe(warnings)
        return report

    assert geometry is not None
    gsec: dict[str, Any] = {
        "tag": tag,
        "part_count": int(geometry.provenance.get("part_count", 0)),
        "n_tri": int(geometry.solid.faces.shape[0]),
        "mesh_volume_m3": float(geometry.solid.volume),
        "bbox_m": _bbox_world(geometry),
        "planes": [
            {"axis": pl.axis, "side": pl.side, "offset_m": float(pl.offset)}
            for pl in geometry.planes
        ],
        "provenance": dict(geometry.provenance),
    }
    if tag == "step":
        spec = cfg.geometry_spec
        step_meta = getattr(spec, "step", None) if spec is not None else None
        if step_meta is not None:
            gsec["file"] = getattr(step_meta, "path", None)
            gsec["units"] = getattr(step_meta, "units", None)
            gsec["sha256"] = getattr(step_meta, "sha256", None)
    report["geometry"] = gsec

    derived = None
    gg = None
    try:
        derived, gg, _fractions, _ = _derive_geom_with_cfl_resolution(cfg, geometry)
    except (ValueError, AssertionError) as exc:
        errors.append(str(exc))
        report["errors"] = _dedupe(errors)
        report["warnings"] = _dedupe(warnings)
        return report

    assert derived is not None and gg is not None
    val_errors, val_warnings = validate_geom(
        cfg, derived, geometry, ram_limit_bytes=ram_limit_bytes
    )
    errors.extend(val_errors)
    warnings.extend(val_warnings)

    cache = None
    info: dict[str, Any] = {}
    beta_tilde = None
    try:
        cache = build_geom_op_cache(cfg, derived, gg, geometry)
        beta_tilde = cache.beta_tilde
        drive = resolve_drive_indices(cfg)
        _, info = build_run_operator(
            cfg, derived, gg, geometry, driven=drive[0], cache=cache
        )
    except ValueError as exc:
        errors.append(str(exc))

    if beta_tilde is None and cache is not None:
        beta_tilde = cache.beta_tilde

    fr_stats = info.get("fractions_stats") or (
        cache.fractions.stats if cache is not None else {}
    )
    floor_stats = info.get("floor_stats") or (
        cache.floor_stats if cache is not None else {}
    )
    thin_stats = info.get("thin_wall_stats") or (
        cache.thin_stats if cache is not None else {}
    )

    rad_rows: list[dict[str, Any]] = []
    spec = cfg.geometry_spec
    h = float(derived.h)
    rad_s_list = fr_stats.get("radiator_S") or []
    for j, rd in enumerate(geometry.radiators):
        cmin, cmed = radiator_clearance(geometry, rd, n=64)
        s_exact = float(rd.area)
        s_grid = float(rad_s_list[j]) if j < len(rad_s_list) else float("nan")
        ratio = s_grid / s_exact if s_exact > 0 else float("nan")
        n_hat = rd.pose.R[:, 2]
        n_hat = n_hat / max(np.linalg.norm(n_hat), 1e-30)
        shape_kind = "unknown"
        if spec is not None and j < len(spec.radiators):
            sh = spec.radiators[j].shape
            if sh is not None:
                shape_kind = sh.kind
        row: dict[str, Any] = {
            "name": rd.name,
            "shape": shape_kind,
            "S_exact_m2": s_exact,
            "S_grid_m2": s_grid,
            "S_ratio": ratio,
            "n_hat": [float(n_hat[0]), float(n_hat[1]), float(n_hat[2])],
            "clearance_min_m": float(cmin) if np.isfinite(cmin) else None,
            "clearance_median_m": float(cmed) if np.isfinite(cmed) else None,
            "boss_margin_m": None,
        }
        if np.isfinite(cmin):
            cells = cmin / h
            if cells < 5.0 - 1e-9:
                h_need = cmin / 5.0
                warnings.append(
                    f"{rd.name}: clearance {cmin:g} m, {cells:.1f} cells, "
                    f"h needed {h_need:g} m"
                )
        rad_rows.append(row)
    report["radiators"] = rad_rows

    h_nom = float(gg.h_nominal) if gg.h_nominal > 0 else h
    shrink = h / h_nom if h_nom > 0 else 1.0
    cpml_side_names = [
        f"{ax}{sign}"
        for ax, sign, on in zip(
            ("x", "x", "y", "y", "z", "z"),
            ("-", "+", "-", "+", "-", "+"),
            derived.cpml_sides,
            strict=True,
        )
        if on
    ]
    offset = resolve_huygens_offset(cfg)
    rigid = dict(gg.rigid_sides or {})
    try:
        validate_huygens_clearance_geom(derived, offset, rigid)
    except ValueError as exc:
        errors.append(str(exc))

    report["grid"] = {
        "h_nominal_m": h_nom,
        "h_m": h,
        "h_shrink_factor": shrink if abs(shrink - 1.0) > 1e-12 else None,
        "ppw_at_f_max": float(derived.ppw_realized),
        "shape": [int(derived.nx), int(derived.ny), int(derived.nz)],
        "total_cells": int(derived.total_cells),
        "memory_bytes": float(derived.memory_bytes),
        "memory_limit_bytes": float(ram_limit_bytes),
        "cell_budget": int(cfg.numerics.cell_budget),
        "cell_budget_exceeded": bool(derived.cell_budget_exceeded),
        "cpml_sides": cpml_side_names,
        "gap_cells": int(derived.gap_cells),
        "cpml_cells": int(derived.cpml_cells),
        "huygens_offset_cells": offset,
        "test_source_cell": list(derived.test_source_cell_index)
        if derived.test_source_cell_index is not None
        else None,
    }

    drive = resolve_drive_indices(cfg)
    n_runs = len(drive)
    per_wall = _estimate_wall_time_s(derived)
    report["time"] = {
        "cfl_safety_resolved": float(derived.cfl_safety_resolved),
        "cfl_safety_source": str(derived.cfl_safety_source),
        "dt_s": float(derived.dt),
        "n_steps": int(derived.n_steps),
        "n_runs": n_runs,
        "wall_time_s_est_per_run": per_wall,
        "wall_time_s_est_total": per_wall * n_runs,
    }

    floor_st = _delta_status(str(floor_stats.get("floor_status", "ok")))
    report["fractions"] = {
        "n_cut_cells": int(fr_stats.get("n_cut_cells", 0)),
        "n_cut_faces": int(fr_stats.get("n_cut_faces", 0)),
        "n_floored": int(floor_stats.get("n_floored", 0)),
        "min_raw_beta": float(floor_stats.get("min_raw_beta_cut", fr_stats.get("min_beta_positive", 0.0))),
        "delta_V_m3": float(floor_stats.get("delta_V", 0.0)),
        "delta_eq_over_h": float(floor_stats.get("delta_eq_over_h", 0.0)),
        "delta_eq_over_h_status": floor_st,
        "max_closure_residual": float(fr_stats.get("max_line_closure_residual", 0.0)),
        "voids_removed_count": int(fr_stats.get("voids_removed_count", 0)),
        "voids_removed_volume_m3": float(fr_stats.get("voids_removed_volume", 0.0)),
        "thin_wall_cells": int(thin_stats.get("n_confirmed", 0)),
    }

    m = resolve_decimation_factor(cfg, derived)
    n_wet = 0
    if cfg.output.surface_p and beta_tilde is not None and cache is not None:
        n_wet = int(wetted_wall_arrays(cache.fractions, beta_tilde, derived.h)[0].size)
    per_bytes = estimate_output_bytes(cfg, derived, n_wet=n_wet)
    per_bytes_base = estimate_output_bytes(cfg, derived, n_wet=0)
    surface_extra = max(0, per_bytes - per_bytes_base) if cfg.output.surface_p else 0
    probes = (
        probe_statuses(cfg, gg, geometry, beta_tilde)
        if beta_tilde is not None
        else []
    )
    for pr in probes:
        if pr.get("status") == "in CPML":
            errors.append(
                f"probe {pr['index']} at {tuple(pr['xyz_m'])} is in CPML"
            )
    report["recording"] = {
        "probes": probes,
        "probe_count": len(probes),
        "surface_p_enabled": bool(cfg.output.surface_p),
        "surface_p_n_wet": n_wet,
        "surface_p_bytes_est": int(surface_extra),
        "decimation_m": int(m),
        "output_bytes_est_per_run": int(per_bytes),
        "output_bytes_est_total": int(per_bytes * n_runs),
    }

    base = out_path or Path("out.h5")
    names = [r.name for r in geometry.radiators]
    planned: list[str] = []
    for j in drive:
        bn = (
            base.name
            if test_source_active(cfg) or j == 0
            else multirad_output_basename(base, j, names[j] if j < len(names) else "radiator")
        )
        if bn is not None:
            planned.append(bn)
    report["recording"]["planned_output_files"] = planned

    report["errors"] = _dedupe(errors)
    report["warnings"] = _dedupe(warnings)
    return report


def format_geom_report(d: dict[str, Any]) -> str:
    """Human-readable dry-run report."""
    lines: list[str] = []
    name = d.get("config_name", "config")
    lines.append(f"geometry report: {name}")
    lines.append("")

    g = d.get("geometry") or {}
    lines.append("=== Geometry ===")
    lines.append(f"  tag: {g.get('tag')}")
    if "part_count" in g:
        lines.append(f"  part_count: {g.get('part_count')}")
    if "n_tri" in g:
        lines.append(f"  n_tri: {g.get('n_tri')}")
    if g.get("mesh_volume_m3") is not None:
        lines.append(f"  mesh_volume: {_sigfig4(g['mesh_volume_m3'])} m³")
    if g.get("bbox_m"):
        b = g["bbox_m"]
        lines.append(
            f"  bbox: [{_sigfig4(b[0])}, {_sigfig4(b[1])}, {_sigfig4(b[2])}] – "
            f"[{_sigfig4(b[3])}, {_sigfig4(b[4])}, {_sigfig4(b[5])}] m"
        )
    for pl in g.get("planes") or []:
        lines.append(
            f"  plane {pl['axis']}{pl['side']}: offset {_sigfig4(pl['offset_m'])} m"
        )
    for key in ("file", "units", "sha256"):
        if g.get(key) is not None:
            lines.append(f"  step_{key}: {g[key]}")
    lines.append("")

    lines.append("=== Radiators ===")
    for rd in d.get("radiators") or []:
        lines.append(f"  {rd['name']} ({rd.get('shape')})")
        lines.append(f"    S_exact: {_sigfig4(rd['S_exact_m2'])} m²  S_grid: {_sigfig4(rd['S_grid_m2'])} m²  ratio: {_sigfig4(rd['S_ratio'])}")
        nh = rd.get("n_hat") or [0, 0, 1]
        lines.append(f"    n̂: ({_sigfig4(nh[0])}, {_sigfig4(nh[1])}, {_sigfig4(nh[2])})")
        if rd.get("clearance_min_m") is not None:
            lines.append(
                f"    clearance: min {_sigfig4(rd['clearance_min_m'])} m, "
                f"median {_sigfig4(rd['clearance_median_m'])} m"
            )
    lines.append("")

    gr = d.get("grid") or {}
    lines.append("=== Grid ===")
    lines.append(f"  h_nominal: {_sigfig4(gr.get('h_nominal_m', 0))} m  h: {_sigfig4(gr.get('h_m', 0))} m")
    if gr.get("h_shrink_factor") is not None:
        lines.append(f"  h shrink factor: {_sigfig4(gr['h_shrink_factor'])}")
    sh = gr.get("shape") or [0, 0, 0]
    lines.append(f"  shape: {sh[0]}×{sh[1]}×{sh[2]}  cells: {gr.get('total_cells', 0):,}")
    lines.append(f"  ppw @ f_max: {_sigfig4(gr.get('ppw_at_f_max', 0))}")
    lines.append(
        f"  memory: {_sigfig4(gr.get('memory_bytes', 0) / 1e9)} GB "
        f"(limit {_sigfig4(gr.get('memory_limit_bytes', 0) / 1e9)} GB)"
    )
    lines.append(f"  cell_budget: {gr.get('cell_budget', 0):,}  exceeded: {gr.get('cell_budget_exceeded')}")
    lines.append(f"  CPML sides: {gr.get('cpml_sides')}")
    lines.append(
        f"  gap_cells: {gr.get('gap_cells')}  cpml_cells: {gr.get('cpml_cells')}  "
        f"Huygens offset: {gr.get('huygens_offset_cells')}"
    )
    if gr.get("test_source_cell") is not None:
        lines.append(f"  test_source_cell: {gr['test_source_cell']}")
    lines.append("")

    tm = d.get("time") or {}
    lines.append("=== Time ===")
    lines.append(
        f"  cfl_safety: {_sigfig4(tm.get('cfl_safety_resolved', 0))} "
        f"({tm.get('cfl_safety_source')})"
    )
    lines.append(f"  dt: {_sigfig4(tm.get('dt_s', 0))} s  n_steps: {tm.get('n_steps')}")
    lines.append(f"  runs: {tm.get('n_runs')}")
    lines.append(
        f"  wall_time est: {_sigfig4(tm.get('wall_time_s_est_per_run', 0))} s/run, "
        f"{_sigfig4(tm.get('wall_time_s_est_total', 0))} s total"
    )
    lines.append("")

    fr = d.get("fractions") or {}
    lines.append("=== Fractions ===")
    lines.append(
        f"  n_cut_cells: {fr.get('n_cut_cells')}  n_cut_faces: {fr.get('n_cut_faces')}  "
        f"n_floored: {fr.get('n_floored')}"
    )
    lines.append(f"  min raw β: {_sigfig4(fr.get('min_raw_beta', 0))}")
    lines.append(
        f"  ΔV: {_sigfig4(fr.get('delta_V_m3', 0))} m³  "
        f"δ_eq/h: {_sigfig4(fr.get('delta_eq_over_h', 0))} ({fr.get('delta_eq_over_h_status')})"
    )
    lines.append(f"  max closure residual: {_sigfig4(fr.get('max_closure_residual', 0))}")
    lines.append(
        f"  voids removed: {fr.get('voids_removed_count')} "
        f"(vol {_sigfig4(fr.get('voids_removed_volume_m3', 0))} m³)"
    )
    lines.append(f"  thin-wall cells: {fr.get('thin_wall_cells')}")
    lines.append("")

    rec = d.get("recording") or {}
    lines.append("=== Recording ===")
    lines.append(f"  probes: {rec.get('probe_count', 0)}")
    for pr in rec.get("probes") or []:
        xyz = pr.get("xyz_m") or [0, 0, 0]
        lines.append(
            f"    [{pr.get('index')}] ({_sigfig4(xyz[0])}, {_sigfig4(xyz[1])}, {_sigfig4(xyz[2])}) m: {pr.get('status')}"
        )
    if rec.get("surface_p_enabled"):
        lines.append(
            f"  surface_p n_wet: {rec.get('surface_p_n_wet')}  "
            f"bytes est: {_sigfig4(rec.get('surface_p_bytes_est', 0))}"
        )
    lines.append(f"  decimation: {rec.get('decimation_m')}")
    lines.append(
        f"  output est: {_sigfig4(rec.get('output_bytes_est_per_run', 0) / 1e6)} MB/run "
        f"({_sigfig4(rec.get('output_bytes_est_total', 0) / 1e6)} MB total)"
    )
    if rec.get("planned_output_files"):
        lines.append(f"  planned files: {rec['planned_output_files']}")
    lines.append("")

    wmsg = d.get("warnings") or []
    emsg = d.get("errors") or []
    if wmsg:
        lines.append("=== Warnings ===")
        for w in wmsg:
            lines.append(f"  warning: {w}")
        lines.append("")
    if emsg:
        lines.append("=== Errors ===")
        for e in emsg:
            lines.append(f"  error: {e}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"