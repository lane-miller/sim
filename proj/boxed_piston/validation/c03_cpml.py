#!/usr/bin/env python3
"""C3-cpml: CPML absorber vs large reference (Stage A, free field).

Distributed s³ volume-velocity source, impedance Z(f)=P̄/Q and reflection vs
reference, parameter sweep with provisional gates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdtd.config import (  # noqa: E402
    Air,
    Band,
    Config,
    DEFAULT_RAM_LIMIT_BYTES,
    Numerics,
    Output,
    Radiator,
    derive,
)
from fdtd.cpml import make_cpml_state  # noqa: E402
from fdtd.grid import Grid, allocate_fields, build_cpml_slabs  # noqa: E402
from fdtd.kernels import (  # noqa: E402
    UpdateCoeffs,
    make_update_coeffs,
    point_source_pressure_increment,
    step,
)
from validation import _common as vc  # noqa: E402

F_MIN_HZ = 20.0
_DEFAULT_F_MAX = 4_000.0
_DEFAULT_PPW = 20.0
_DEFAULT_S = 6
_BASELINE_CPML = 16
_BASELINE_R = 1e-4
_BASELINE_M = 3.0
_BASELINE_KAPPA = 1.0
_BASELINE_ALPHA = 1.0
_BASELINE_GAP = 10
_STABILITY_TAIL_FRAC = 0.20
_DECAY_THRESH = 1e-6
_INTERIOR_INSET_CELLS = 2
_KA_EFF_MAX = 0.3
_GATE_RE_Z_ANALYTIC = 0.02  # informational only (not in overall pass)
_GATE_GAP_RE_Z_REL = 0.02
_GATE_GAP_Z_REL = 0.005
_GAP_LF_REF_CELLS = 40
_GAP_LF_F_MAX_HZ = 500.0
_GATE_DZ_REF = 0.01
_GATE_REFL_NORMAL_DB = -40.0
_GATE_REFL_OBLIQUE_DB = -30.0
_REF_CPML = _BASELINE_CPML
_DEFAULT_MAX_STEPS = 50_000
_DEFAULT_TAIL_CELLS = vc.DEFAULT_TAIL_CELLS
_DEFAULT_ECHO_MARGIN = vc.ECHO_CLEARANCE_MARGIN
_MAX_GRID_N = 2048
_CAL_FREQS_HZ = (1_000.0, 2_000.0, 3_000.0)
_CAL_MAG_TOL = 0.05
_CAL_PHASE_TOL_DEG = 5.0
_Z_SUMMARY_FREQS_HZ = (50.0, 100.0, 200.0, 500.0)


@dataclass(frozen=True)
class CpmlSweep:
    name: str
    cpml_cells: int = _BASELINE_CPML
    cpml_R: float = _BASELINE_R
    cpml_m: float = _BASELINE_M
    cpml_kappa_max: float = _BASELINE_KAPPA
    cpml_alpha_factor: float = _BASELINE_ALPHA
    gap_cells: int = _BASELINE_GAP

    def key(self) -> tuple[Any, ...]:
        return (
            self.cpml_cells,
            self.cpml_R,
            self.cpml_m,
            self.cpml_kappa_max,
            self.cpml_alpha_factor,
            self.gap_cells,
        )


@dataclass
class SourceGeom:
    i0: int
    j0: int
    k0: int
    s: int
    ic: int
    jc: int
    kc: int


@dataclass(frozen=True)
class ReceiverVectors:
    """Offset vectors from source-cube centre (cells), tied to each domain's CPML geometry."""

    offset_normal: tuple[int, int, int]
    offset_oblique: tuple[int, int, int]


@dataclass
class ReceiverSet:
    offset_normal: tuple[int, int, int]
    offset_oblique: tuple[int, int, int]
    normal: tuple[int, int, int]
    oblique: tuple[int, int, int]


@dataclass(frozen=True)
class LayoutSpec:
    """Calibration offsets from source centre (reference domain)."""

    offset_cal: tuple[tuple[int, int, int], ...]


@dataclass
class VectorCmpSpectra:
    offset_normal: tuple[int, int, int]
    offset_oblique: tuple[int, int, int]
    p_normal_spec: np.ndarray
    p_oblique_spec: np.ndarray


@dataclass(frozen=True)
class WindowPlan:
    t_window_s: float
    t_pulse_s: float
    t_prop_s: float
    t_tail_s: float
    f_res_hz: float
    i_start: int
    i_end_nominal: int


@dataclass
class DomainPlanRow:
    label: str
    n: int
    cpml_cells: int
    est_bytes: int
    n_steps: int
    t_echo_s: float | None = None
    t_cpml_refl_s: float | None = None
    t_window_s: float | None = None
    testable: bool = True
    notes: str = ""


@dataclass(frozen=True)
class ConfigWindowMeta:
    t_cpml_refl_arrival_s: float
    t_window_end_s: float
    testable: bool
    plan: WindowPlan


# Conservative throughput for pre-run wall projection (cells updated per second).
_WALL_PROJ_CELLS_PER_S = 1.5e6


def _baseline() -> CpmlSweep:
    return CpmlSweep("baseline")


def _from_baseline(**overrides: Any) -> CpmlSweep:
    name = overrides.pop("name", "variant")
    b = _baseline()
    fields = {
        "cpml_cells": b.cpml_cells,
        "cpml_R": b.cpml_R,
        "cpml_m": b.cpml_m,
        "cpml_kappa_max": b.cpml_kappa_max,
        "cpml_alpha_factor": b.cpml_alpha_factor,
        "gap_cells": b.gap_cells,
    }
    fields.update(overrides)
    return CpmlSweep(name, **fields)


def iter_sweep_configs(*, quick: bool) -> list[CpmlSweep]:
    out: list[CpmlSweep] = []
    seen: set[tuple[Any, ...]] = set()

    def add(sw: CpmlSweep) -> None:
        k = sw.key()
        if k in seen:
            return
        seen.add(k)
        out.append(sw)

    add(_baseline())
    if quick:
        add(_from_baseline(name="cpml_8", cpml_cells=8))
        add(_from_baseline(name="alpha_100", cpml_alpha_factor=100.0))
        add(_from_baseline(name="R_1e-3", cpml_R=1e-3))
        add(_from_baseline(name="gap_10", gap_cells=10))
        add(_from_baseline(name="gap_40", gap_cells=40))
        return out

    for c in (8, 12, 16, 24):
        add(_from_baseline(name=f"cpml_{c}", cpml_cells=c))
    for r in (1e-3, 1e-4, 1e-6):
        add(_from_baseline(name=f"R_{r:.0e}".replace("+", ""), cpml_R=r))
    for kappa in (1.0, 3.0, 7.0):
        add(_from_baseline(name=f"kappa_{int(kappa)}", cpml_kappa_max=kappa))
    for af in (0.0, 1.0, 10.0, 100.0):
        add(
            _from_baseline(
                name=f"alpha_{int(af) if af >= 1 else 0}",
                cpml_alpha_factor=af,
            )
        )
    for gap in (5, 10, 20, 40):
        add(_from_baseline(name=f"gap_{gap}", gap_cells=gap))
    for c in (12, 16):
        for kappa in (1.0, 3.0):
            for af in (1.0, 10.0):
                add(
                    _from_baseline(
                        name=f"combo_c{c}_k{int(kappa)}_a{int(af)}",
                        cpml_cells=c,
                        cpml_kappa_max=kappa,
                        cpml_alpha_factor=af,
                    )
                )
    return out


def build_cfg(
    sw: CpmlSweep,
    *,
    f_max: float,
    ppw: float,
    f_min: float = F_MIN_HZ,
) -> tuple[Config, object]:
    num = Numerics(
        cpml_cells=sw.cpml_cells,
        gap_cells=sw.gap_cells,
        ppw=ppw,
        cfl_safety=0.99,
        cell_budget=10**12,
        cells_across_radiator=10,
        cpml_R=sw.cpml_R,
        cpml_m=sw.cpml_m,
        cpml_kappa_max=sw.cpml_kappa_max,
        cpml_alpha_factor=sw.cpml_alpha_factor,
    )
    cfg = Config(
        box=None,
        air=Air(c=343.0, rho=1.204),
        radiator=Radiator("circle", dims=[0.01]),
        band=Band(f_min=f_min, f_max=f_max),
        numerics=num,
        output=Output(path=""),
    )
    d = derive(cfg)
    vc.apply_h_from_ppw(d, cfg, f_ref=f_max, ppw=ppw)
    return cfg, d


def grid_n(sw: CpmlSweep, s: int) -> int:
    interior = s + 2 * sw.gap_cells
    return interior + 2 * sw.cpml_cells


def source_geom(sw: CpmlSweep, n: int, s: int) -> SourceGeom:
    """Gap-anchored source (small swept domains)."""
    i0 = sw.cpml_cells + sw.gap_cells
    j0 = k0 = i0
    ic = i0 + (s - 1) // 2
    jc = j0 + (s - 1) // 2
    kc = k0 + (s - 1) // 2
    return SourceGeom(i0=i0, j0=j0, k0=k0, s=s, ic=ic, jc=jc, kc=kc)


def source_geom_centered(n: int, s: int, cpml: int) -> SourceGeom:
    """Source cube centred in the interior (reference domain only)."""
    ic = jc = kc = cpml + (n - 2 * cpml) // 2
    i0 = ic - (s - 1) // 2
    j0 = jc - (s - 1) // 2
    k0 = kc - (s - 1) // 2
    return SourceGeom(i0=i0, j0=j0, k0=k0, s=s, ic=ic, jc=jc, kc=kc)


def make_layout_spec(s: int) -> LayoutSpec:
    """Calibration probe offsets from source centre (+y, on reference domain)."""
    return LayoutSpec(
        offset_cal=(
            (0, 4 * s, 0),
            (0, 5 * s, 0),
            (0, 6 * s, 0),
        ),
    )


def cells_to_plus_x_interface(*, ic: int, n: int, cpml: int) -> int:
    """Cells from source-cube centre to the +x interior/CPML interface (last interior index)."""
    return (n - cpml - 1) - ic


def receiver_vectors_for_geom(
    *,
    n: int,
    cpml: int,
    geom: SourceGeom,
    label: str = "",
) -> ReceiverVectors:
    """Normal at (d−2,0,0); oblique 45° at (d−2,d−2,0) against +x and +y interfaces."""
    d = cells_to_plus_x_interface(ic=geom.ic, n=n, cpml=cpml)
    need = _INTERIOR_INSET_CELLS + 2
    if d < need:
        tag = f" ({label})" if label else ""
        raise ValueError(
            f"{tag}d={d} cells to +x CPML interface too small at n={n} cpml={cpml} "
            f"(need d≥{need} for {_INTERIOR_INSET_CELLS}-cell inset receivers)"
        )
    return ReceiverVectors((d - 2, 0, 0), (d - 2, d - 2, 0))


def collect_config_receiver_vectors(
    configs: list[CpmlSweep],
    *,
    s: int,
    f_max: float,
    ppw: float,
) -> dict[str, ReceiverVectors]:
    out: dict[str, ReceiverVectors] = {}
    for sw in configs:
        n = grid_n(sw, s)
        geom = source_geom(sw, n, s)
        out[sw.name] = receiver_vectors_for_geom(
            n=n, cpml=sw.cpml_cells, geom=geom, label=sw.name
        )
    return out


def union_receiver_vectors(
    by_name: dict[str, ReceiverVectors],
) -> list[ReceiverVectors]:
    seen: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    out: list[ReceiverVectors] = []
    for v in by_name.values():
        key = (v.offset_normal, v.offset_oblique)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _receiver_at(
    geom: SourceGeom,
    n: int,
    cpml: int,
    offset: tuple[int, int, int],
    *,
    label: str,
) -> tuple[int, int, int]:
    ic, jc, kc = geom.ic, geom.jc, geom.kc
    i = ic + offset[0]
    j = jc + offset[1]
    k = kc + offset[2]
    lo = cpml + _INTERIOR_INSET_CELLS
    hi = n - cpml - 1 - _INTERIOR_INSET_CELLS
    if not (lo <= i <= hi and lo <= j <= hi and lo <= k <= hi):
        raise ValueError(
            f"{label}: receiver ({i},{j},{k}) not ≥{_INTERIOR_INSET_CELLS} cells inside "
            f"interior [{cpml}, {n - cpml}) (allowed [{lo}, {hi}])"
        )
    return i, j, k


def build_receiver_set(
    geom: SourceGeom,
    n: int,
    cpml: int,
    vectors: ReceiverVectors,
    *,
    label: str,
) -> ReceiverSet:
    normal = _receiver_at(geom, n, cpml, vectors.offset_normal, label=f"{label} normal")
    oblique = _receiver_at(geom, n, cpml, vectors.offset_oblique, label=f"{label} oblique")
    return ReceiverSet(
        offset_normal=vectors.offset_normal,
        offset_oblique=vectors.offset_oblique,
        normal=normal,
        oblique=oblique,
    )


def build_receivers_and_cal(
    geom: SourceGeom,
    n: int,
    cpml: int,
    vectors: ReceiverVectors,
    layout: LayoutSpec,
    *,
    label: str,
    include_cal: bool = False,
) -> tuple[ReceiverSet, list[tuple[int, int, int]]]:
    rec = build_receiver_set(geom, n, cpml, vectors, label=label)
    cal: list[tuple[int, int, int]] = []
    if include_cal:
        cal = [
            _receiver_at(geom, n, cpml, off, label=f"{label} cal")
            for off in layout.offset_cal
        ]
    return rec, cal


def vector_spectra_key(vectors: ReceiverVectors) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (vectors.offset_normal, vectors.offset_oblique)


def monitor_trace_key(vectors: ReceiverVectors, role: str) -> str:
    on = vectors.offset_normal
    oo = vectors.offset_oblique
    return f"n{on[0]}_{on[1]}_{on[2]}_o{oo[0]}_{oo[1]}_{oo[2]}_{role}"


def cpml_params_summary(
    cfg: Config, d: object, cpml_st: object
) -> dict[str, float]:
    coeff = cpml_st.coeffs  # type: ignore[attr-defined]
    return {
        "sigma_max": float(coeff.sigma_max),
        "kappa_max": float(cfg.numerics.cpml_kappa_max),
        "alpha_max": float(d.cpml_alpha),
        "alpha_factor": float(cfg.numerics.cpml_alpha_factor),
        "cpml_R": float(cfg.numerics.cpml_R),
        "cpml_cells": int(cfg.numerics.cpml_cells),
    }


def assert_cpml_params_change(
    a: dict[str, float],
    b: dict[str, float],
    *,
    ctx: str,
) -> None:
    """Different ``R`` or ``α_factor`` must change σ_max or α_max in the state."""
    if a["cpml_R"] != b["cpml_R"] and a["sigma_max"] == b["sigma_max"]:
        raise AssertionError(
            f"{ctx}: cpml_R {a['cpml_R']:.2e} vs {b['cpml_R']:.2e} "
            f"but identical σ_max={a['sigma_max']:.6e}"
        )
    if a.get("alpha_factor") != b.get("alpha_factor") and abs(
        a["alpha_max"] - b["alpha_max"]
    ) < 1e-12:
        raise AssertionError(
            f"{ctx}: alpha_factor changed but alpha_max={a['alpha_max']:.6e} unchanged"
        )
    if a["cpml_cells"] != b["cpml_cells"] and a["sigma_max"] == b["sigma_max"]:
        raise AssertionError(
            f"{ctx}: cpml_cells {a['cpml_cells']} vs {b['cpml_cells']} "
            f"but identical σ_max"
        )


def print_geometry_parity(
    *,
    domain: str,
    geom: SourceGeom,
    rec: ReceiverSet,
    g: Grid,
) -> dict[str, Any]:
    ic, jc, kc = geom.ic, geom.jc, geom.kc
    rows: dict[str, Any] = {}
    for name, pt, off in (
        ("normal", rec.normal, rec.offset_normal),
        ("oblique", rec.oblique, rec.offset_oblique),
    ):
        got = (pt[0] - ic, pt[1] - jc, pt[2] - kc)
        if got != off:
            raise AssertionError(
                f"{domain} {name}: offset {got} != layout {off}"
            )
        r_m = vc.path_length_m(g, ic, jc, kc, pt[0], pt[1], pt[2])
        rows[name] = {"offset_cells": off, "r_m": r_m, "indices": pt}
    rows["source_centre"] = (ic, jc, kc)
    return rows


def _set_cube_n(d: object, n: int, cpml_cells: int) -> None:
    d.nx = d.ny = d.nz = n
    d.cpml_cells = cpml_cells
    d.total_cells = n**3


def build_grid(
    cfg: Config,
    d: object,
    n: int,
    *,
    ram_limit: float,
    label: str,
) -> tuple[Grid, UpdateCoeffs, object, object]:
    cpml = cfg.numerics.cpml_cells
    vc.require_ram(n, cpml, ram_limit, label=label)
    _set_cube_n(d, n, cpml)
    g = Grid(cfg, d)
    g.cpml = cfg.numerics.cpml_cells
    g.cpml_slabs = build_cpml_slabs(n, n, n, g.cpml)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    cpml_st = make_cpml_state(g, cfg, d, d.dt)
    return g, coeffs, g.cpml_slabs, cpml_st


@njit(cache=True)
def _inject_cube_uniform(
    p: np.ndarray,
    i0: int,
    j0: int,
    k0: int,
    s: int,
    dP: float,
) -> None:
    for i in range(i0, i0 + s):
        for j in range(j0, j0 + s):
            for k in range(k0, k0 + s):
                p[i, j, k] += dP


@njit(cache=True)
def _cube_mean_p(
    p: np.ndarray, i0: int, j0: int, k0: int, s: int
) -> float:
    acc = 0.0
    n = float(s * s * s)
    for i in range(i0, i0 + s):
        for j in range(j0, j0 + s):
            for k in range(k0, k0 + s):
                acc += p[i, j, k]
    return acc / n


def calibration_window_seconds(
    cfg: Config,
    d: object,
    g: Grid,
    *,
    i_src: int,
    j_src: int,
    k_src: int,
    receiver: tuple[int, int, int],
    tail_cells: float,
) -> float:
    """Per calibration receiver: direct arrival + pulse span + tail (does not widen cmp win)."""
    c = cfg.air.c
    h = d.h
    t_pulse = d.pulse_t0 + 6.0 * d.pulse_sigma
    r_m = vc.path_length_m(g, i_src, j_src, k_src, *receiver)
    t_prop = r_m / c
    t_tail = tail_cells * h / c
    return t_prop + t_pulse + t_tail


def cpml_interface_planes_m(n: int, cpml: int, h: float) -> tuple[float, float, float, float, float, float]:
    """Interior/CPML boundary face positions (m, domain-centred)."""
    x_lo = (cpml - 0.5 * n) * h
    x_hi = (n - cpml - 0.5 * n) * h
    y_lo = (cpml - 0.5 * n) * h
    y_hi = (n - cpml - 0.5 * n) * h
    z_lo = (cpml - 0.5 * n) * h
    z_hi = (n - cpml - 0.5 * n) * h
    return x_lo, x_hi, y_lo, y_hi, z_lo, z_hi


def cpml_interface_reflection_travel_s(
    xs: float,
    ys: float,
    zs: float,
    xr: float,
    yr: float,
    zr: float,
    *,
    c: float,
    planes: tuple[float, float, float, float, float, float],
) -> float:
    """Shortest source→CPML interface→receiver path (seconds, no pulse offset)."""
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = planes
    times = [
        (abs(xs - x_lo) + abs(xr - x_lo)) / c,
        (abs(xs - x_hi) + abs(xr - x_hi)) / c,
        (abs(ys - y_lo) + abs(yr - y_lo)) / c,
        (abs(ys - y_hi) + abs(yr - y_hi)) / c,
        (abs(zs - z_lo) + abs(zr - z_lo)) / c,
        (abs(zs - z_hi) + abs(zr - z_hi)) / c,
    ]
    return min(times)


def reflection_monitor_points(geom: SourceGeom, rec: ReceiverSet) -> list[tuple[int, int, int]]:
    """Receivers, index centre, and every source-cube cell."""
    pts: list[tuple[int, int, int]] = [rec.normal, rec.oblique, (geom.ic, geom.jc, geom.kc)]
    for di in range(geom.s):
        for dj in range(geom.s):
            for dk in range(geom.s):
                pt = (geom.i0 + di, geom.j0 + dj, geom.k0 + dk)
                if pt not in pts:
                    pts.append(pt)
    return pts


def earliest_cpml_reflection_arrival_s(
    g: Grid,
    cfg: Config,
    d: object,
    geom: SourceGeom,
    rec: ReceiverSet,
) -> float:
    """Earliest pulse-aligned arrival of a first reflection off this domain's CPML interfaces."""
    c = cfg.air.c
    pulse_t0 = d.pulse_t0
    xs, ys, zs = vc.coords_m(g, geom.ic, geom.jc, geom.kc)
    planes = cpml_interface_planes_m(g.nx, g.cpml, g.h)
    t_min = math.inf
    for pt in reflection_monitor_points(geom, rec):
        xr, yr, zr = vc.coords_m(g, pt[0], pt[1], pt[2])
        travel = cpml_interface_reflection_travel_s(
            xs, ys, zs, xr, yr, zr, c=c, planes=planes
        )
        t_min = min(t_min, pulse_t0 + travel)
    return t_min


def build_reflection_measurable_window(
    cfg: Config,
    d: object,
    g: Grid,
    geom: SourceGeom,
    rec: ReceiverSet,
    *,
    tail_cells: float,
) -> tuple[WindowPlan, float]:
    """Comparison window ≥ direct (pulse+prop+tail) and ≥ CPML-reflection arrival + pulse + tail."""
    c = cfg.air.c
    h = d.h
    sigma, t0 = d.pulse_sigma, d.pulse_t0
    t_pulse = t0 + 6.0 * sigma
    t_prop = 0.0
    for pt in (rec.normal, rec.oblique):
        r_m = vc.path_length_m(g, geom.ic, geom.jc, geom.kc, pt[0], pt[1], pt[2])
        t_prop = max(t_prop, r_m / c)
    t_tail = tail_cells * h / c
    t_direct = t_pulse + t_prop + t_tail
    t_cpml_refl = earliest_cpml_reflection_arrival_s(g, cfg, d, geom, rec)
    t_cpml_window = t_cpml_refl + t_pulse + t_tail
    t_window = max(t_direct, t_cpml_window)
    i0, i1 = vc.window_sample_range(d, t_window_s=t_window)
    plan = WindowPlan(
        t_window_s=t_window,
        t_pulse_s=t_pulse,
        t_prop_s=t_prop,
        t_tail_s=t_tail,
        f_res_hz=vc.f_resolution_hz(t_window),
        i_start=i0,
        i_end_nominal=i1,
    )
    return plan, t_cpml_refl


def config_window_testable(
    plan: WindowPlan,
    *,
    t_cpml_refl_s: float,
    max_steps: int,
    dt: float,
) -> bool:
    """False if the record cannot reach the required window end or omit the CPML reflection."""
    need_s = t_cpml_refl_s + plan.t_pulse_s + plan.t_tail_s
    if plan.t_window_s + 1e-12 < need_s:
        return False
    if plan.i_end_nominal > max_steps:
        return False
    if max_steps * dt + 1e-12 < plan.t_window_s:
        return False
    return True


def build_window_plan(
    cfg: Config,
    d: object,
    geom: SourceGeom,
    rec: ReceiverSet,
    *,
    n: int,
    tail_cells: float,
) -> WindowPlan:
    _set_cube_n(d, n, cfg.numerics.cpml_cells)
    g = Grid(cfg, d)
    t_window, t_pulse, t_prop = vc.comparison_window_seconds(
        cfg,
        d,
        g,
        i_src=geom.ic,
        j_src=geom.jc,
        k_src=geom.kc,
        receivers=(rec.normal, rec.oblique),
        tail_cells=tail_cells,
    )
    t_tail = tail_cells * d.h / cfg.air.c
    i0, i1 = vc.window_sample_range(d, t_window_s=t_window)
    return WindowPlan(
        t_window_s=t_window,
        t_pulse_s=t_pulse,
        t_prop_s=t_prop,
        t_tail_s=t_tail,
        f_res_hz=vc.f_resolution_hz(t_window),
        i_start=i0,
        i_end_nominal=i1,
    )


def comparison_sample_window(
    win: WindowPlan,
    d: object,
    *,
    trace_len: int,
    t_echo_s: float | None = None,
    echo_cap: bool = True,
) -> tuple[int, int]:
    """``[i0, i1)`` for ref vs small cmp FFTs; echo cap applies to reference sizing only."""
    i0 = win.i_start
    i1 = win.i_end_nominal
    if echo_cap and t_echo_s is not None:
        i_echo = int(math.floor(t_echo_s / d.dt))
        i1 = min(i1, i_echo)
    i1 = min(i1, trace_len)
    i1 = max(i1, i0 + vc.MIN_WINDOW_SAMPLES)
    return i0, i1


def small_run_step_limits(
    cmp_i_end: int,
    cmp_win_len: int,
    *,
    max_steps: int,
) -> tuple[int, int, int]:
    """``(min_steps, decay_margin, n_steps_budget)`` for small-domain FDTD early stop."""
    min_steps = cmp_i_end
    margin = max(2000, 2 * cmp_win_len)
    return min_steps, margin, max_steps


def q_spectrum_from_injection(
    q_trace: np.ndarray,
    dt: float,
    *,
    i_start: int,
    i_end: int,
    n_fft: int,
    win: np.ndarray | None = None,
) -> np.ndarray:
    """FFT of the actual per-step total volume velocity ``Q(t)`` injected."""
    _, q_spec, _ = vc.spectrum_absolute(
        q_trace,
        dt,
        i_start=i_start,
        i_end=i_end,
        n_fft=n_fft,
        win=win,
        index_shift=0,
    )
    return q_spec


def extract_comparison_spectra(
    d: object,
    *,
    avg_p: np.ndarray,
    p_normal: np.ndarray,
    p_oblique: np.ndarray,
    q_trace: np.ndarray,
    i_start: int,
    i_end: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal end-taper spectra; identical window on Q and every p trace."""
    dt = d.dt
    n_win = i_end - i_start
    win = vc.causal_end_taper(n_win)
    freqs, q_spec, _ = vc.spectrum_absolute(
        q_trace,
        dt,
        i_start=i_start,
        i_end=i_end,
        n_fft=n_fft,
        win=win,
        index_shift=0,
    )
    _, p_spec, _ = vc.spectrum_absolute(
        avg_p,
        dt,
        i_start=i_start,
        i_end=i_end,
        n_fft=n_fft,
        win=win,
        index_shift=0,
    )
    _, pn_spec, _ = vc.spectrum_absolute(
        p_normal,
        dt,
        i_start=i_start,
        i_end=i_end,
        n_fft=n_fft,
        win=win,
        index_shift=vc.PRESSURE_TRACE_INDEX_SHIFT,
    )
    _, pg_spec, _ = vc.spectrum_absolute(
        p_oblique,
        dt,
        i_start=i_start,
        i_end=i_end,
        n_fft=n_fft,
        win=win,
        index_shift=vc.PRESSURE_TRACE_INDEX_SHIFT,
    )
    vc.assert_same_length(
        (freqs, "freqs"),
        (p_spec, "p_avg_spec"),
        (q_spec, "q_spec"),
        (pn_spec, "p_normal_spec"),
        (pg_spec, "p_oblique_spec"),
        context="comparison spectra",
    )
    return freqs, p_spec, q_spec, pn_spec, pg_spec


def analytic_lf_n_fft(dt: float, trace_len: int, f_min: float = F_MIN_HZ) -> int:
    """FFT length for LF Z: zero-pad to ≥ 2^k samples covering at least 4/f_min (0.2 s at 20 Hz)."""
    n_duration = int(math.ceil((4.0 / f_min) / dt))
    return int(2 ** math.ceil(math.log2(max(trace_len, n_duration, 64))))


def extract_analytic_spectra(
    d: object,
    *,
    avg_p: np.ndarray,
    q_trace: np.ndarray,
    trace_len: int,
    f_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Full small-domain record FFT for Re Z vs analytic (end taper only; not cmp window)."""
    dt = d.dt
    i_start = 0
    i_end = trace_len
    n_fft = analytic_lf_n_fft(dt, trace_len)
    win = vc.causal_end_taper(trace_len)
    freqs, q_spec, p_spec, _ = vc.paired_spectra_absolute(
        q_trace,
        avg_p,
        dt,
        i_start=i_start,
        i_end=i_end,
        n_fft=n_fft,
        win=win,
        p_index_shift=0,
    )
    vc.assert_same_length(
        (freqs, "freqs_analytic"),
        (p_spec, "p_avg_spec_analytic"),
        (q_spec, "q_spec_analytic"),
        context="analytic Z spectra",
    )
    band = vc.analysis_mask(freqs, q_spec, f_min=F_MIN_HZ, f_max=f_max)
    vc.assert_same_length(
        (freqs, "freqs_analytic"),
        (band, "analytic_band_mask"),
        context="analytic band mask",
    )
    return freqs, p_spec, q_spec, band, n_fft


def plan_reference_n(
    cfg: Config,
    d: object,
    *,
    s: int,
    cpml: int,
    layout: LayoutSpec,
    vector_union: list[ReceiverVectors],
    window: WindowPlan,
    tail_cells: float,
    echo_margin: float,
    ram_limit: float,
) -> tuple[int, float, str]:
    """Centred reference; grow ``n`` until every union receiver vector fits and clears echo."""
    h, c = d.h, cfg.air.c
    n_guess = max(grid_n(_baseline(), s), 128)
    if n_guess % 2:
        n_guess += 1
    n = n_guess
    t_echo_min = 0.0
    binding = ""
    while n <= _MAX_GRID_N:
        vc.require_ram(n, cpml, ram_limit, label="reference domain (plan)")
        _set_cube_n(d, n, cpml)
        g = Grid(cfg, d)
        geom = source_geom_centered(n, s, cpml)
        layout_ok = True
        receivers: list[tuple[str, tuple[int, int, int], float]] = []
        try:
            _, cal_pts = build_receivers_and_cal(
                geom,
                n,
                cpml,
                vector_union[0],
                layout,
                label="reference plan",
                include_cal=True,
            )
            for vec in vector_union:
                rec = build_receiver_set(
                    geom, n, cpml, vec, label="reference plan"
                )
                tag = f"vec n{vec.offset_normal}/o{vec.offset_oblique}"
                receivers.append((f"{tag} normal", rec.normal, window.t_window_s))
                receivers.append((f"{tag} oblique", rec.oblique, window.t_window_s))
        except ValueError:
            layout_ok = False
            cal_pts = []
        if not layout_ok:
            n += 8
            continue
        for ci, (pt, off) in enumerate(zip(cal_pts, layout.offset_cal)):
            t_cal = calibration_window_seconds(
                cfg,
                d,
                g,
                i_src=geom.ic,
                j_src=geom.jc,
                k_src=geom.kc,
                receiver=pt,
                tail_cells=tail_cells,
            )
            receivers.append((f"cal[{ci}] +y{off[1]}c", pt, t_cal))

        all_ok = True
        binding = ""
        worst_slack = -math.inf
        t_echo_min = math.inf
        for name, pt, t_win in receivers:
            t_need = t_win * echo_margin
            t_echo_pt = vc.first_reflection_time_s(
                g,
                cfg,
                i_src=geom.ic,
                j_src=geom.jc,
                k_src=geom.kc,
                i_rec=pt[0],
                j_rec=pt[1],
                k_rec=pt[2],
            )
            t_echo_min = min(t_echo_min, t_echo_pt)
            slack = t_echo_pt - t_need
            if slack < worst_slack:
                worst_slack = slack
                binding = name
            if t_echo_pt < t_need:
                all_ok = False

        if all_ok:
            print(
                f"[setup] reference domain planned n={n}³ "
                f"(source centred, binding receiver={binding!r}, "
                f"slack={worst_slack * 1e3:.3f} ms)",
                flush=True,
            )
            return n, t_echo_min, binding
        n += 8
    raise RuntimeError(
        "Reference domain sizing failed: "
        f"at n≤{_MAX_GRID_N} binding={binding!r} slack={worst_slack * 1e3:.3f} ms. "
        "Reduce tail/window or raise RAM limit."
    )


def run_fdtd(
    g: Grid,
    d: object,
    cfg: Config,
    coeffs: UpdateCoeffs,
    slabs: object,
    cpml_st: object | None,
    geom: SourceGeom,
    receivers: ReceiverSet,
    cal_indices: list[tuple[int, int, int]],
    n_steps: int,
    *,
    run_timer: vc.RunTimer | None = None,
    label: str = "",
    min_steps: int = 0,
    decay_margin_steps: int | None = None,
    extra_p_monitors: dict[str, tuple[int, int, int]] | None = None,
) -> dict[str, Any]:
    fields = allocate_fields(g)
    s3 = float(geom.s**3)
    rho, c, h, dt = cfg.air.rho, cfg.air.c, d.h, d.dt
    avg_trace = np.zeros(n_steps, dtype=np.float64)
    p_norm = np.zeros(n_steps, dtype=np.float64)
    p_oblique = np.zeros(n_steps, dtype=np.float64)
    q_trace = np.zeros(n_steps, dtype=np.float64)
    n_cal = len(cal_indices)
    p_cal = np.zeros((n_cal, n_steps), dtype=np.float64)
    extra = dict(extra_p_monitors or {})
    extra_traces: dict[str, np.ndarray] = {
        k: np.zeros(n_steps, dtype=np.float64) for k in extra
    }
    ir, jr, kr = receivers.normal
    io, jo, ko = receivers.oblique
    peak = 0.0
    tty = sys.stdout.isatty()
    t0_run = __import__("time").perf_counter()
    script_t0 = run_timer.t0 if run_timer is not None else t0_run
    milestones = {max(1, math.ceil(n_steps * p / 100.0)) for p in range(5, 101, 5)}
    milestones.add(n_steps)
    tail_start = max(0, int(n_steps * (1.0 - _STABILITY_TAIL_FRAC)))
    decay_first: int | None = None
    n_actual = n_steps

    for step_i in range(n_steps):
        t = step_i * dt
        q = vc.pulse_q(t, d.pulse_sigma, d.pulse_t0)
        dP = float(
            point_source_pressure_increment(rho, c, dt, h, q / s3)
        )
        _inject_cube_uniform(
            fields.p, geom.i0, geom.j0, geom.k0, geom.s, dP
        )
        q_trace[step_i] = q
        avg_post = _cube_mean_p(fields.p, geom.i0, geom.j0, geom.k0, geom.s)
        avg_trace[step_i] = avg_post
        step(fields, coeffs, slabs, cpml=cpml_st)
        avg = _cube_mean_p(fields.p, geom.i0, geom.j0, geom.k0, geom.s)
        p_norm[step_i] = float(fields.p[ir, jr, kr])
        p_oblique[step_i] = float(fields.p[io, jo, ko])
        for key, (ii, jj, kk) in extra.items():
            extra_traces[key][step_i] = float(fields.p[ii, jj, kk])
        for ci, (ii, jj, kk) in enumerate(cal_indices):
            p_cal[ci, step_i] = float(fields.p[ii, jj, kk])
        peak = max(peak, abs(avg_trace[step_i]), abs(avg))
        if peak > 0.0 and abs(avg_trace[step_i]) < _DECAY_THRESH * peak:
            if decay_first is None:
                decay_first = step_i + 1

        done = step_i + 1
        if decay_margin_steps is not None and decay_first is not None:
            stop_at = decay_first + decay_margin_steps
            if done >= max(min_steps, stop_at):
                n_actual = done
                sim_elapsed = __import__("time").perf_counter() - t0_run
                rate = done / sim_elapsed if sim_elapsed > 0 else 0.0
                wall = __import__("time").perf_counter() - script_t0
                line = (
                    f"[sim{label}] step {done}/{n_steps} ({(100 * done) // n_steps:3d}%) | "
                    f"t={t * 1e3:.2f} ms | wall {wall:.1f}s | "
                    f"rate {rate:.1f} st/s | early stop | |⟨p⟩|_pk={peak:.3e}"
                )
                vc.emit_sim_progress(line, tty=tty, final=True)
                break
        if done not in milestones:
            continue
        sim_elapsed = __import__("time").perf_counter() - t0_run
        rate = done / sim_elapsed if sim_elapsed > 0 else 0.0
        eta = (n_steps - done) / rate if rate > 0 else 0.0
        wall = __import__("time").perf_counter() - script_t0
        line = (
            f"[sim{label}] step {done}/{n_steps} ({(100 * done) // n_steps:3d}%) | "
            f"t={t * 1e3:.2f} ms | wall {wall:.1f}s | "
            f"rate {rate:.1f} st/s | ETA {eta:.1f}s | |⟨p⟩|_pk={peak:.3e}"
        )
        vc.emit_sim_progress(line, tty=tty, final=(done == n_steps))
    if tty:
        print(flush=True)

    avg_trace = avg_trace[:n_actual]
    p_norm = p_norm[:n_actual]
    p_oblique = p_oblique[:n_actual]
    q_trace = q_trace[:n_actual]
    for key in extra_traces:
        extra_traces[key] = extra_traces[key][:n_actual]
    if n_cal:
        p_cal = p_cal[:, :n_actual]

    tail_start = max(0, int(n_actual * (1.0 - _STABILITY_TAIL_FRAC)))
    tail = avg_trace[tail_start:]
    tail_max = float(np.max(np.abs(tail))) if tail.size else 0.0
    stab_ratio = tail_max / (peak + 1e-30)
    decay_idx = n_actual
    thresh = _DECAY_THRESH * peak
    for i in range(n_actual - 1, -1, -1):
        if abs(avg_trace[i]) >= thresh:
            decay_idx = min(i + 1, n_actual)
            break
    decay_reached = decay_first is not None
    tail_level_ratio = stab_ratio
    return {
        "avg_p": avg_trace,
        "p_normal": p_norm,
        "p_oblique": p_oblique,
        "p_extra": extra_traces,
        "n_steps": n_actual,
        "n_steps_budget": n_steps,
        "peak_avg_p": peak,
        "tail_max": tail_max,
        "tail_level_ratio": tail_level_ratio,
        "stability_ratio": stab_ratio,
        "stable": stab_ratio < 1e-3,
        "decay_first_step": decay_first,
        "decay_steps_reported": decay_idx,
        "decay_reached": decay_reached,
        "wall_s": __import__("time").perf_counter() - t0_run,
        "q_trace": q_trace,
        "p_cal": p_cal,
        "cal_indices": list(cal_indices),
    }


def a_eff_m(s: int, h: float) -> float:
    vol = (s * h) ** 3
    r = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    return r


def analytic_re_z(freqs: np.ndarray, *, rho: float, c: float) -> np.ndarray:
    """Low-ka radiation resistance for Z = ⟨P⟩/Q (compact monopole, any geometry).

    Re Z = ρ c k² / (4π)  with units Pa·s/m³ (acoustic impedance).
    ``a_eff`` is not in this formula; use ka_eff only for the analytic-band gate.
    """
    k = 2.0 * math.pi * np.maximum(freqs, 0.0) / c
    return rho * c * (k**2) / (4.0 * math.pi)


def _assert_analytic_re_z_100hz(*, rho: float, c: float) -> None:
    f_hz = 100.0
    k = 2.0 * math.pi * f_hz / c
    expected = rho * c * (k**2) / (4.0 * math.pi)
    got = float(analytic_re_z(np.array([f_hz], dtype=np.float64), rho=rho, c=c)[0])
    rtol = 1e-12 * max(1.0, abs(expected))
    if abs(got - expected) > rtol:
        raise AssertionError(
            f"analytic_re_z self-check @ {f_hz} Hz: got {got:.6e}, "
            f"expected ρck²/(4π)={expected:.6e}"
        )


def injection_self_term_re_z(*, rho: float, c: float, dt: float, h: float, s: int) -> float:
    """Spurious Re Z offset scale ρc²Δt/V (pre-injection ⟨p⟩); post-injection removes it."""
    vol = (float(s) * h) ** 3
    return rho * c * c * dt / vol


def print_re_z_discrepancy_diagnostics(
    freqs: np.ndarray,
    p_spec: np.ndarray,
    q_spec: np.ndarray,
    *,
    rho: float,
    c: float,
    a_eff: float,
    h: float,
    dt: float,
    s: int,
    f_hz_list: tuple[float, ...] = _Z_SUMMARY_FREQS_HZ,
    domain: str = "baseline",
) -> None:
    """Re Z = ⟨P⟩/Q from full-record FFT; post-injection ⟨p⟩ paired with Qⁿ."""
    artefact = injection_self_term_re_z(rho=rho, c=c, dt=dt, h=h, s=s)
    artefact_half = 0.5 * artefact
    print(
        f"\n=== Re Z vs analytic ({domain}, post-injection ⟨p⟩/Qⁿ, full record) ===",
        flush=True,
    )
    print(
        "  INFORMATIONAL (not gated): Re Z/|Z| is only ~0.5–2% at 50–500 Hz; "
        "sub-degree phase errors dominate.",
        flush=True,
    )
    print(
        f"  legacy centred offset ±ρc²Δt/(2V) = ±{artefact_half:.6e}; "
        f"pre-injection −ρc²Δt/V = −{artefact:.6e} Pa·s/m³ "
        f"(V=({s}·h)³) — ΔRe should NOT be constant at these scales",
        flush=True,
    )
    hdr = (
        f"{'f Hz':>6} {'f_bin':>8} {'Re Z meas':>12} {'Re Z ana':>12} "
        f"{'ΔRe':>12} {'rel err':>9} {'|Im Z|':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    delta_res: list[float] = []
    for f_hz in f_hz_list:
        idx = int(np.argmin(np.abs(freqs - f_hz)))
        f_act = float(freqs[idx])
        p_bar = p_spec[idx]
        q = q_spec[idx]
        z = p_bar / (q + 1e-30)
        re_z = float(np.real(z))
        im_z = float(np.imag(z))
        re_ana = float(analytic_re_z(np.array([f_act]), rho=rho, c=c)[0])
        delta_re = re_z - re_ana
        delta_res.append(delta_re)
        rel = abs(re_z - re_ana) / (abs(re_ana) + 1e-30)
        print(
            f"{f_hz:6.0f} {f_act:8.2f} {re_z:12.4e} {re_ana:12.4e} "
            f"{delta_re:12.4e} {rel:9.3f} {abs(im_z):12.4e}"
        )
    if delta_res:
        mean_delta = float(np.mean(delta_res))
        targets = (artefact_half, -artefact_half, -artefact)
        offset_present = any(
            abs(mean_delta - t) <= 0.2 * max(abs(t), 1e-30) for t in targets
        )
        if offset_present:
            print(
                f"  WARN: mean ΔRe over 50–500 Hz = {mean_delta:.4e} within 20% of a "
                f"sampling self-term (±ρc²Δt/(2V) or −ρc²Δt/V) — "
                f"constant offset may still be present",
                flush=True,
            )
        else:
            print(
                "  OK: no constant sampling offset; residual ΔRe grows ∝ f "
                "(phase-limited)",
                flush=True,
            )
    omega_ref = 2.0 * math.pi * 100.0
    im_mass_scale = omega_ref * rho / (4.0 * math.pi * max(a_eff, 1e-12))
    print(
        f"  sanity @100 Hz: |Im Z| vs ωρ/(4π a_eff) scale ≈ {im_mass_scale:.4e} Pa·s/m³",
        flush=True,
    )


def low_freq_gap_mask(
    freqs: np.ndarray,
    *,
    a_eff: float,
    c: float,
    f_min: float = F_MIN_HZ,
    f_max: float = _GAP_LF_F_MAX_HZ,
    ka_max: float = _KA_EFF_MAX,
) -> np.ndarray:
    ka = 2.0 * math.pi * np.maximum(freqs, 0.0) * a_eff / c
    return (freqs >= f_min) & (freqs <= f_max) & (ka < ka_max)


def gap_convergence_maxima(
    freqs: np.ndarray,
    z: np.ndarray,
    z_ref: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    vc.assert_same_length(
        (freqs, "gap freqs"),
        (z, "gap z"),
        (z_ref, "gap z_ref"),
        (mask, "gap mask"),
        context="gap convergence",
    )
    if not np.any(mask):
        return float("nan"), float("nan")
    re_rel = np.abs(np.real(z) - np.real(z_ref)) / (np.abs(np.real(z_ref)) + 1e-30)
    z_rel = np.abs(z - z_ref) / (np.abs(z_ref) + 1e-30)
    return float(np.max(re_rel[mask])), float(np.max(z_rel[mask]))


def apply_gap_convergence_gates(
    results: list[dict[str, Any]],
    *,
    s: int,
    c: float,
) -> dict[str, Any]:
    """Compare each run's Z(f) to gap_40 over 20–500 Hz, ka_eff<0.3 (all configs except gap_40)."""
    ref_name = f"gap_{_GAP_LF_REF_CELLS}"
    ref = next((r for r in results if r["name"] == ref_name), None)
    summary: dict[str, Any] = {
        "reference": ref_name,
        "gap_cells_ref": _GAP_LF_REF_CELLS,
        "f_hz": (F_MIN_HZ, _GAP_LF_F_MAX_HZ),
        "gates": {
            "max_re_rel": _GATE_GAP_RE_Z_REL,
            "max_z_rel": _GATE_GAP_Z_REL,
        },
        "missing_reference": ref is None,
    }
    if ref is None:
        for r in results:
            if not r.get("testable", True):
                continue
            g = r.setdefault("gate", {})
            g["gap_conv"] = False
            g["pass"] = _row_passes_gates(r)
        return summary

    h_ref = float(ref.get("metrics", {}).get("h_m", 0.0))
    a_eff_ref = a_eff_m(s, h_ref) if h_ref > 0 else float("nan")

    sp_ref = ref.get("spectra", {}).get("lf_gap", {}) or {}
    f_ref = np.asarray(sp_ref.get("freqs_hz", []), dtype=np.float64)
    z_ref = np.asarray(sp_ref.get("z_re", []), dtype=np.float64) + 1j * np.asarray(
        sp_ref.get("z_im", []), dtype=np.float64
    )
    if f_ref.size == 0 or z_ref.size != f_ref.size:
        summary["missing_reference"] = True
        return summary

    mask_ref = (
        low_freq_gap_mask(f_ref, a_eff=a_eff_ref, c=c)
        if np.isfinite(a_eff_ref)
        else np.ones(f_ref.shape, dtype=bool)
    )

    for r in results:
        g = r.setdefault("gate", {})
        if not r.get("testable", True):
            g["gap_conv"] = False
            g["pass"] = False
            continue
        if r["name"] == ref_name:
            g["gap_conv"] = True
            r.setdefault("metrics", {})["gap_conv"] = {
                "max_re_rel": 0.0,
                "max_z_rel": 0.0,
                "applies": False,
                "reference": True,
            }
        else:
            sp = r.get("spectra", {}).get("lf_gap", {}) or {}
            f = np.asarray(sp.get("freqs_hz", []), dtype=np.float64)
            z = np.asarray(sp.get("z_re", []), dtype=np.float64) + 1j * np.asarray(
                sp.get("z_im", []), dtype=np.float64
            )
            gap_cells = int((r.get("sweep") or {}).get("gap_cells", _BASELINE_GAP))
            if f.size != f_ref.size or not np.allclose(f, f_ref):
                max_re, max_z = float("nan"), float("nan")
                ok = False
            else:
                h_m = float(r.get("metrics", {}).get("h_m", h_ref))
                a_eff = a_eff_m(s, h_m)
                mask = low_freq_gap_mask(f, a_eff=a_eff, c=c)
                if not np.any(mask):
                    max_re, max_z = float("nan"), float("nan")
                    ok = False
                else:
                    max_re, max_z = gap_convergence_maxima(f, z, z_ref, mask & mask_ref)
                    ok = max_re <= _GATE_GAP_RE_Z_REL and max_z <= _GATE_GAP_Z_REL
            g["gap_conv"] = bool(ok)
            r.setdefault("metrics", {})["gap_conv"] = {
                "max_re_rel": max_re,
                "max_z_rel": max_z,
                "applies": True,
                "gap_cells": gap_cells,
            }
        g["pass"] = _row_passes_gates(r)
    return summary


def _row_passes_gates(r: dict[str, Any]) -> bool:
    if not r.get("testable", True):
        return False
    g = r.get("gate") or {}
    return bool(
        g.get("dz_ref")
        and g.get("refl_normal")
        and g.get("refl_oblique")
        and g.get("stable")
        and g.get("gap_conv", False)
    )


def analytic_monopole_p(
    freqs: np.ndarray,
    q_spec: np.ndarray,
    *,
    r_m: float,
    rho: float,
    c: float,
) -> np.ndarray:
    """``P = jωρQ/(4πr) · e^{−jkr}`` with e^{+jωt} convention."""
    omega = 2.0 * math.pi * freqs
    k = omega / c
    return (
        1j
        * omega
        * rho
        * q_spec
        / (4.0 * math.pi * max(r_m, 1e-12))
        * np.exp(-1j * k * r_m)
    )


def _sinc_phys(x: np.ndarray) -> np.ndarray:
    """sinc(x) = sin(x)/x (physics convention, not NumPy's π-normalized sinc)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x, dtype=np.float64)
    nz = np.abs(x) > 1e-30
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def uniform_cube_form_factor(
    freqs: np.ndarray,
    n_hat: tuple[float, float, float],
    *,
    L_m: float,
    c: float,
) -> np.ndarray:
    """Far-field directivity of a uniform cubic source (side L): F(k, n̂) = Π_i sinc(k n_i L/2)."""
    k = 2.0 * math.pi * np.maximum(freqs, 0.0) / c
    nx, ny, nz = n_hat
    return (
        _sinc_phys(k * nx * L_m * 0.5)
        * _sinc_phys(k * ny * L_m * 0.5)
        * _sinc_phys(k * nz * L_m * 0.5)
    )


def fit_calibration_phase_model(
    rows: list[dict[str, Any]], *, dt: float
) -> dict[str, float]:
    """Fit Δφ(f) ≈ φ₀ + 360°·f·τ with τ in units of Δt (linear in f)."""
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if "skip" in row:
            continue
        xs.append(float(row["f_hz"]))
        ys.append(float(row["phase_err_deg"]))
    if len(xs) < 2:
        return {
            "phi0_deg": float("nan"),
            "tau_steps": float("nan"),
            "n_points": float(len(xs)),
        }
    fx = np.asarray(xs, dtype=np.float64)
    fy = np.asarray(ys, dtype=np.float64)
    slope, intercept = np.polyfit(fx, fy, 1)
    tau_steps = float(slope / (360.0 * dt))
    return {
        "phi0_deg": float(intercept),
        "tau_steps": tau_steps,
        "n_points": float(len(xs)),
    }


def run_monopole_calibration(
    g: Grid,
    cfg: Config,
    d: object,
    geom: SourceGeom,
    layout: LayoutSpec,
    sim: dict[str, Any],
    *,
    s: int,
    tail_cells: float,
    label: str = "",
) -> dict[str, Any]:
    """Compare calibration receiver spectra to F·jωρQ/(4πr)·e^{−jkr} (uniform cube + monopole)."""
    trace_len = len(sim["avg_p"])
    rho, c, h = cfg.air.rho, cfg.air.c, d.h
    L_m = float(s) * h
    ic, jc, kc = geom.ic, geom.jc, geom.kc
    xc, yc, zc = vc.cube_geometric_centre_m(g, geom)
    xic, yic, zic = vc.coords_m(g, ic, jc, kc)
    dr_centre = math.hypot(xc - xic, yc - yic, zc - zic)
    rows: list[dict[str, Any]] = []
    ok = True
    win_meta: list[dict[str, Any]] = []
    for ci, off in enumerate(layout.offset_cal):
        if off[1] < 4 * s or off[0] != 0 or off[2] != 0:
            raise ValueError(f"calibration offset {off} must be (0, ≥4s, 0) along +y")
        pt = sim["cal_indices"][ci]
        xr, yr, zr = vc.coords_m(g, pt[0], pt[1], pt[2])
        dx, dy, dz = xr - xc, yr - yc, zr - zc
        r_m = math.hypot(dx, dy, dz)
        if r_m < 1e-12:
            raise ValueError(f"calibration receiver {pt} coincident with source centroid")
        n_hat = (dx / r_m, dy / r_m, dz / r_m)
        t_win = calibration_window_seconds(
            cfg,
            d,
            g,
            i_src=ic,
            j_src=jc,
            k_src=kc,
            receiver=pt,
            tail_cells=tail_cells,
        )
        i0, i1 = vc.window_sample_range(d, t_window_s=t_win, trace_len=trace_len)
        n_fft = vc.comparison_n_fft(i0, i1)
        freqs, q_spec, p_spec, win = vc.paired_spectra_absolute(
            sim["q_trace"],
            sim["p_cal"][ci],
            d.dt,
            i_start=i0,
            i_end=i1,
            n_fft=n_fft,
        )
        p_mono = analytic_monopole_p(freqs, q_spec, r_m=r_m, rho=rho, c=c)
        form_f = uniform_cube_form_factor(freqs, n_hat, L_m=L_m, c=c)
        p_ana = form_f * p_mono
        win_meta.append(
            {
                "receiver": ci,
                "t_window_s": t_win,
                "i_win": (i0, i1),
                "n_fft": n_fft,
            }
        )
        for f0 in _CAL_FREQS_HZ:
            idx = int(np.argmin(np.abs(freqs - f0)))
            f = float(freqs[idx])
            k = 2.0 * math.pi * f / c
            if k * r_m <= 1.0:
                rows.append({"receiver": ci, "f_hz": f0, "skip": f"kr≤1"})
                continue
            h_meas = p_spec[idx] / (q_spec[idx] + 1e-30)
            h_ana = p_ana[idx] / (q_spec[idx] + 1e-30)
            mag_ratio = abs(h_meas) / (abs(h_ana) + 1e-30)
            dphi = math.degrees(
                math.atan2(
                    (h_meas * np.conj(h_ana)).imag,
                    (h_meas * np.conj(h_ana)).real,
                )
            )
            F_val = float(np.real(form_f[idx]))
            mag_ok = abs(mag_ratio - 1.0) <= _CAL_MAG_TOL
            phase_ok = abs(dphi) <= _CAL_PHASE_TOL_DEG
            ok = ok and mag_ok and phase_ok
            rows.append(
                {
                    "receiver": ci,
                    "offset_cells": off,
                    "r_m": r_m,
                    "n_hat": n_hat,
                    "L_m": L_m,
                    "f_hz": f0,
                    "kr": k * r_m,
                    "form_factor": F_val,
                    "mag_ratio": mag_ratio,
                    "phase_err_deg": dphi,
                    "mag_ok": mag_ok,
                    "phase_ok": phase_ok,
                }
            )
    phase_fit = fit_calibration_phase_model(rows, dt=d.dt)
    return {
        "pass": ok,
        "rows": rows,
        "windows": win_meta,
        "phase_fit": phase_fit,
        "source_centroid_m": (xc, yc, zc),
        "index_centre_m": (xic, yic, zic),
        "centre_offset_m": dr_centre,
        "label": label,
        "s_cube": s,
    }


def print_calibration_table(cal: dict[str, Any], *, dt: float) -> None:
    tag = cal.get("label") or "reference domain"
    print(f"\n=== Monopole calibration ({tag}, s={cal.get('s_cube', '?')}) ===")
    if cal.get("centre_offset_m") is not None:
        print(
            f"  source centroid vs (ic,jc,kc): Δr={cal['centre_offset_m']:.3e} m",
            flush=True,
        )
    hdr = (
        f"{'rcv':>3} {'f Hz':>6} {'kr':>5} {'F':>7} {'|H/H0|':>8} "
        f"{'Δφ°':>7} {'mag':>4} {'phs':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in cal["rows"]:
        if "skip" in row:
            print(f"{row['receiver']:3d} {row['f_hz']:6.0f}  — skip {row['skip']}")
            continue
        print(
            f"{row['receiver']:3d} {row['f_hz']:6.0f} {row['kr']:5.2f} "
            f"{row['form_factor']:7.4f} {row['mag_ratio']:8.4f} "
            f"{row['phase_err_deg']:7.2f} "
            f"{'ok' if row['mag_ok'] else 'NO':>4} "
            f"{'ok' if row['phase_ok'] else 'NO':>4}"
        )
    print(f"  GATE: {'PASS' if cal['pass'] else 'FAIL'} (|H/H₀|±{_CAL_MAG_TOL:.0%}, |Δφ|≤{_CAL_PHASE_TOL_DEG:.0f}°)")
    pf = cal.get("phase_fit") or {}
    if pf.get("n_points", 0) >= 2:
        print(
            f"  fitted τ = {pf['tau_steps']:.4f} Δt  "
            f"(φ₀={pf['phi0_deg']:.2f}°, {pf['n_points']:.0f} points)",
            flush=True,
        )
        print(
            "  residual source/sample time offset ≈ τ·Δt. Expected ~0.5Δt from "
            "volume-source vs p staggering. Revisit in Stage C (Z_rad timing).",
            flush=True,
        )


def run_cal_point_source_isolation(
    *,
    f_max: float,
    ppw: float,
    ref_n: int,
    tail_cells: float,
    ram_limit: float,
    run_timer: vc.RunTimer,
) -> dict[str, Any]:
    """FDTD calibration with s=1 point source (form-factor isolation)."""
    s = 1
    layout = make_layout_spec(s)
    sw = CpmlSweep("cal_s1", cpml_cells=_REF_CPML, gap_cells=_BASELINE_GAP)
    cfg, d = build_cfg(sw, f_max=f_max, ppw=ppw)
    n = ref_n
    geom = source_geom_centered(n, s, _REF_CPML)
    vec = receiver_vectors_for_geom(n=n, cpml=_REF_CPML, geom=geom, label="cal s=1")
    rec, cal_pts = build_receivers_and_cal(
        geom, n, _REF_CPML, vec, layout, label="cal s=1", include_cal=True
    )
    g, coeffs, slabs, cpml_st = build_grid(
        cfg, d, n, ram_limit=ram_limit, label="cal point source"
    )
    win_probe = build_window_plan(
        cfg, d, geom, rec, n=n, tail_cells=tail_cells
    )
    t_sim = reference_sim_horizon_seconds(
        cfg, d, g, geom, cal_pts, win_probe, tail_cells
    )
    n_steps = int(math.ceil(t_sim / d.dt)) + 64
    run_timer.banner("cal-s1", f"s=1 isolation on n={n}³ steps={n_steps:,}")
    sim = run_fdtd(
        g,
        d,
        cfg,
        coeffs,
        slabs,
        cpml_st,
        geom,
        rec,
        cal_pts,
        n_steps,
        label=":cal-s1",
    )
    return run_monopole_calibration(
        g,
        cfg,
        d,
        geom,
        layout,
        sim,
        s=s,
        tail_cells=tail_cells,
        label="point source s=1",
    )


@dataclass
class RefCache:
    """One large-domain reference simulation (baseline CPML numerics)."""

    n: int
    t_echo_min: float
    cmp_n_fft: int
    cmp_freqs: np.ndarray
    cmp_p_avg_spec: np.ndarray
    cmp_q_spec: np.ndarray
    cmp_band_ref: np.ndarray
    cmp_by_vectors: dict[tuple[tuple[int, int, int], tuple[int, int, int]], VectorCmpSpectra]
    window: WindowPlan
    cmp_i_win: tuple[int, int]
    layout: LayoutSpec
    geom: SourceGeom
    vector_union: list[ReceiverVectors]
    geom_parity_by_vector: dict[tuple[tuple[int, int, int], tuple[int, int, int]], dict[str, Any]]
    meta: dict[str, Any]


def reference_sim_horizon_seconds(
    cfg: Config,
    d: object,
    g: Grid,
    geom: SourceGeom,
    cal_pts: list[tuple[int, int, int]],
    window: WindowPlan,
    tail_cells: float,
) -> float:
    t_cal = [
        calibration_window_seconds(
            cfg,
            d,
            g,
            i_src=geom.ic,
            j_src=geom.jc,
            k_src=geom.kc,
            receiver=pt,
            tail_cells=tail_cells,
        )
        for pt in cal_pts
    ]
    return max([window.t_window_s, *t_cal])


def run_reference(
    *,
    f_max: float,
    ppw: float,
    s: int,
    window: WindowPlan,
    layout: LayoutSpec,
    vector_union: list[ReceiverVectors],
    tail_cells: float,
    ram_limit: float,
    echo_margin: float,
    run_timer: vc.RunTimer,
) -> tuple[RefCache, dict[str, Any]]:
    sw = CpmlSweep("reference", cpml_cells=_REF_CPML, gap_cells=_BASELINE_GAP)
    cfg, d = build_cfg(sw, f_max=f_max, ppw=ppw)
    n, t_echo, binding = plan_reference_n(
        cfg,
        d,
        s=s,
        cpml=_REF_CPML,
        layout=layout,
        vector_union=vector_union,
        window=window,
        tail_cells=tail_cells,
        echo_margin=echo_margin,
        ram_limit=ram_limit,
    )
    geom = source_geom_centered(n, s, _REF_CPML)
    rec, cal_pts = build_receivers_and_cal(
        geom,
        n,
        _REF_CPML,
        vector_union[0],
        layout,
        label="reference",
        include_cal=True,
    )
    extra_p: dict[str, tuple[int, int, int]] = {}
    for vec in vector_union[1:]:
        rset = build_receiver_set(geom, n, _REF_CPML, vec, label="reference")
        extra_p[monitor_trace_key(vec, "normal")] = rset.normal
        extra_p[monitor_trace_key(vec, "oblique")] = rset.oblique
    g, coeffs, slabs, cpml_st = build_grid(
        cfg, d, n, ram_limit=ram_limit, label="reference domain"
    )
    t_sim = reference_sim_horizon_seconds(
        cfg, d, g, geom, cal_pts, window, tail_cells
    )
    n_steps_ref = min(
        int(math.ceil(t_sim / d.dt)) + 64,
        int(math.floor(t_echo / d.dt)),
    )
    n_steps_ref = max(n_steps_ref, window.i_end_nominal + 8)
    parity_map: dict[tuple[tuple[int, int, int], tuple[int, int, int]], dict[str, Any]] = {}
    for vec in vector_union:
        rset = build_receiver_set(geom, n, _REF_CPML, vec, label="reference")
        parity_map[vector_spectra_key(vec)] = print_geometry_parity(
            domain="reference",
            geom=geom,
            rec=rset,
            g=g,
        )
    p0 = parity_map[vector_spectra_key(vector_union[0])]
    print(
        f"[ref] centred source @ {p0['source_centre']} "
        f"union vectors={len(vector_union)} "
        f"r_norm@0={p0['normal']['r_m']:.4f} m binding={binding!r}",
        flush=True,
    )
    est = vc.memory_estimate_cube(n, _REF_CPML)
    run_timer.banner(
        "ref",
        f"grid {n}³ (interior {n - 2 * _REF_CPML}) h={d.h * 1e3:.3f} mm "
        f"steps={n_steps_ref:,} T_win={window.t_window_s * 1e3:.3f} ms "
        f"echo≥{t_echo * 1e3:.3f} ms est {vc.fmt_bytes(est)}",
    )
    sim = run_fdtd(
        g,
        d,
        cfg,
        coeffs,
        slabs,
        cpml_st,
        geom,
        rec,
        cal_pts,
        n_steps_ref,
        run_timer=run_timer,
        label=":ref",
        extra_p_monitors=extra_p,
    )
    calibration = run_monopole_calibration(
        g, cfg, d, geom, layout, sim, s=s, tail_cells=tail_cells
    )
    trace_len = len(sim["avg_p"])
    i_start, i_end = comparison_sample_window(
        window, d, trace_len=trace_len, t_echo_s=t_echo, echo_cap=True
    )
    n_fft_cmp = vc.comparison_n_fft(window.i_start, window.i_end_nominal)
    freqs, p_spec, q_spec, _, _ = extract_comparison_spectra(
        d,
        avg_p=sim["avg_p"],
        p_normal=sim["p_normal"],
        p_oblique=sim["p_oblique"],
        q_trace=sim["q_trace"],
        i_start=i_start,
        i_end=i_end,
        n_fft=n_fft_cmp,
    )
    band_cmp = vc.analysis_mask(freqs, q_spec, f_min=F_MIN_HZ, f_max=f_max)
    band_ref = band_cmp & (freqs >= window.f_res_hz)
    vc.assert_same_length(
        (freqs, "cmp_freqs"),
        (band_cmp, "cmp_band"),
        (band_ref, "cmp_band_ref"),
        context="reference comparison band masks",
    )

    def _p_trace_for(vec: ReceiverVectors, role: str) -> np.ndarray:
        key = monitor_trace_key(vec, role)
        if vec.offset_normal == rec.offset_normal and vec.offset_oblique == rec.offset_oblique:
            return sim["p_normal"] if role == "normal" else sim["p_oblique"]
        return sim["p_extra"][key]

    cmp_by_vectors: dict[tuple[tuple[int, int, int], tuple[int, int, int]], VectorCmpSpectra] = {}
    for vec in vector_union:
        _, _, _, pn_spec, po_spec = extract_comparison_spectra(
            d,
            avg_p=sim["avg_p"],
            p_normal=_p_trace_for(vec, "normal"),
            p_oblique=_p_trace_for(vec, "oblique"),
            q_trace=sim["q_trace"],
            i_start=i_start,
            i_end=i_end,
            n_fft=n_fft_cmp,
        )
        cmp_by_vectors[vector_spectra_key(vec)] = VectorCmpSpectra(
            offset_normal=vec.offset_normal,
            offset_oblique=vec.offset_oblique,
            p_normal_spec=pn_spec,
            p_oblique_spec=po_spec,
        )

    ref = RefCache(
        n=n,
        t_echo_min=t_echo,
        cmp_n_fft=n_fft_cmp,
        cmp_freqs=freqs,
        cmp_p_avg_spec=p_spec,
        cmp_q_spec=q_spec,
        cmp_band_ref=band_ref,
        cmp_by_vectors=cmp_by_vectors,
        window=window,
        cmp_i_win=(i_start, i_end),
        layout=layout,
        geom=geom,
        vector_union=vector_union,
        geom_parity_by_vector=parity_map,
        meta={
            "n_steps": n_steps_ref,
            "h_m": d.h,
            "dt_s": d.dt,
            "t_echo_min_s": t_echo,
            "binding_receiver": binding,
            "t_sim_horizon_s": t_sim,
            "wall_s": sim["wall_s"],
            "f_res_hz": window.f_res_hz,
            "calibration": calibration,
            "cpml_params": cpml_params_summary(cfg, d, cpml_st),
            "low_freq_authority": (
                "Re Z vs analytic is informational (50–500 Hz); "
                "|ΔZ|/|Z_ref| and reflection gates apply for f ≥ f_res; "
                f"low-f gap convergence vs gap_{_GAP_LF_REF_CELLS} for all configs "
                f"20–{_GAP_LF_F_MAX_HZ:.0f} Hz (ka_eff<{_KA_EFF_MAX})."
            ),
        },
    )
    return ref, calibration


def untestable_result_row(
    sw: CpmlSweep,
    *,
    n: int,
    t_cpml_refl_s: float,
    t_window_s: float,
    reason: str,
    h_m: float,
) -> dict[str, Any]:
    return {
        "name": sw.name,
        "sweep": asdict(sw),
        "testable": False,
        "untestable_reason": reason,
        "grid_n": n,
        "total_cells": n**3,
        "n_steps": 0,
        "n_steps_budget": 0,
        "metrics": {
            "h_m": h_m,
            "t_cpml_refl_arrival_s": t_cpml_refl_s,
            "t_window_end_s": t_window_s,
            "max_dz_rel_ref": float("nan"),
            "max_refl_normal_db": float("nan"),
            "max_refl_oblique_db": float("nan"),
            "max_re_z_err_analytic_ka": float("nan"),
        },
        "gate": {
            "pass": False,
            "untestable": True,
            "dz_ref": False,
            "refl_normal": False,
            "refl_oblique": False,
            "stable": False,
            "gap_conv": False,
        },
        "spectra": {},
    }


def metrics_for_config(
    sw: CpmlSweep,
    *,
    f_max: float,
    ppw: float,
    s: int,
    ref: RefCache,
    win_meta: ConfigWindowMeta,
    vectors: ReceiverVectors,
    max_steps: int,
    ram_limit: float,
    run_timer: vc.RunTimer,
    verbose: bool,
) -> dict[str, Any]:
    cfg, d = build_cfg(sw, f_max=f_max, ppw=ppw)
    n = grid_n(sw, s)
    geom = source_geom(sw, n, s)
    rec = build_receiver_set(geom, n, sw.cpml_cells, vectors, label=sw.name)
    vkey = vector_spectra_key(vectors)
    ref_spec = ref.cmp_by_vectors[vkey]
    ref_parity = ref.geom_parity_by_vector[vkey]
    t_cpml_refl = win_meta.t_cpml_refl_arrival_s
    t_window_end = win_meta.t_window_end_s
    print(
        f"  {sw.name}: CPML-reflection arrival {t_cpml_refl * 1e3:.3f} ms; "
        f"comparison window end {t_window_end * 1e3:.3f} ms "
        f"({'TESTABLE' if win_meta.testable else 'UNTESTABLE'})",
        flush=True,
    )
    if not win_meta.testable:
        reason = (
            "CPML reflection cannot be captured within --max-steps "
            f"at window end {t_window_end * 1e3:.3f} ms"
        )
        return untestable_result_row(
            sw,
            n=n,
            t_cpml_refl_s=t_cpml_refl,
            t_window_s=t_window_end,
            reason=reason,
            h_m=d.h,
        )
    g, coeffs, slabs, cpml_st = build_grid(
        cfg, d, n, ram_limit=ram_limit, label=f"small domain ({sw.name})"
    )
    cpml_p = cpml_params_summary(cfg, d, cpml_st)
    print(
        f"  {sw.name} CPML σ_max={cpml_p['sigma_max']:.6e} "
        f"κ_max={cpml_p['kappa_max']:.3g} α_max={cpml_p['alpha_max']:.6e} "
        f"R={cpml_p['cpml_R']:.2e}",
        flush=True,
    )
    parity_s = print_geometry_parity(
        domain=sw.name,
        geom=geom,
        rec=rec,
        g=g,
    )
    if rec.offset_normal != ref_spec.offset_normal or rec.offset_oblique != ref_spec.offset_oblique:
        raise AssertionError(
            f"{sw.name}: receiver vectors {rec.offset_normal}/{rec.offset_oblique} "
            f"!= reference {ref_spec.offset_normal}/{ref_spec.offset_oblique}"
        )
    for key in ("normal", "oblique"):
        r_s = parity_s[key]["r_m"]
        r_r = ref_parity[key]["r_m"]
        off_s = parity_s[key]["offset_cells"]
        off_r = ref_parity[key]["offset_cells"]
        if off_s != off_r:
            raise AssertionError(
                f"{sw.name}: {key} offset cells {off_s} != reference {off_r}"
            )
        if not vc.radii_match_m(r_s, r_r, h=d.h):
            raise AssertionError(
                f"{sw.name}: {key} radius {r_s:.9f} m != reference {r_r:.9f} m "
                f"(h={d.h:.6e} m)"
            )
    print(
        f"  {sw.name} vectors normal={rec.offset_normal} "
        f"oblique={rec.offset_oblique} (integer offsets & r match ref)",
        flush=True,
    )
    ref_i0, ref_i1 = ref.cmp_i_win
    cmp_len = ref_i1 - ref_i0
    min_steps, decay_margin, n_steps_budget = small_run_step_limits(
        ref_i1, cmp_len, max_steps=max_steps
    )
    if verbose:
        points = (
            rec.normal,
            rec.oblique,
            (geom.ic, geom.jc, geom.kc),
        )
        t_echo = vc.min_echo_time_s(
            g,
            cfg,
            i_src=geom.ic,
            j_src=geom.jc,
            k_src=geom.kc,
            points=points,
        )
        print(
            f"  {sw.name}: n={n}³ steps≤{n_steps_budget:,} (early stop after decay) "
            f"cmp_win [{ref_i0},{ref_i1}) echo~{t_echo * 1e3:.2f} ms "
            f"cells={d.total_cells:,}",
            flush=True,
        )
    sim = run_fdtd(
        g,
        d,
        cfg,
        coeffs,
        slabs,
        cpml_st,
        geom,
        rec,
        [],
        n_steps_budget,
        run_timer=run_timer,
        label=f":{sw.name[:12]}",
        min_steps=min_steps,
        decay_margin_steps=decay_margin,
    )
    n_used = int(sim["n_steps"])
    print(
        f"  {sw.name}: steps used {n_used:,}/{n_steps_budget:,} "
        f"(min cmp end {min_steps:,}, decay margin {decay_margin:,})",
        flush=True,
    )
    if sim["decay_reached"]:
        d_step = sim.get("decay_first_step") or sim["decay_steps_reported"]
        decay_msg = (
            f"  {sw.name}: decay REACHED (⟨p⟩ < {_DECAY_THRESH:.0e}×peak) "
            f"first @ step {d_step:,}; ran {n_used:,}/{n_steps_budget:,}"
        )
    else:
        decay_msg = (
            f"  {sw.name}: NO DECAY — ran full {n_used:,}/{n_steps_budget:,} steps "
            f"(⟨p⟩ never < {_DECAY_THRESH:.0e}×peak)"
        )
    print(decay_msg, flush=True)
    print(
        f"  {sw.name}: tail level (last 10% max |⟨p⟩| / peak) = "
        f"{sim['tail_level_ratio']:.3e}",
        flush=True,
    )

    trace_len = len(sim["avg_p"])
    if trace_len < ref_i1:
        raise AssertionError(
            f"{sw.name}: trace length {trace_len} < reference cmp end {ref_i1} "
            f"— increase --max-steps"
        )
    i_cmp0, i_cmp1 = ref_i0, ref_i1
    if (i_cmp0, i_cmp1) != ref.cmp_i_win:
        raise AssertionError(
            f"{sw.name}: FFT window steps [{i_cmp0},{i_cmp1}) != reference "
            f"{ref.cmp_i_win}"
        )
    if rec.offset_normal != ref_spec.offset_normal or rec.offset_oblique != ref_spec.offset_oblique:
        raise AssertionError(
            f"{sw.name}: receiver vectors != reference spectra key "
            f"{ref_spec.offset_normal}/{ref_spec.offset_oblique}"
        )
    print(
        f"  {sw.name} cmp window steps [{i_cmp0},{i_cmp1}) "
        f"Δt·steps={(i_cmp1 - i_cmp0) * d.dt * 1e3:.3f} ms (matches ref)",
        flush=True,
    )
    n_fft_cmp = ref.cmp_n_fft
    f_cmp, p_cmp, q_cmp, pn_cmp, po_cmp = extract_comparison_spectra(
        d,
        avg_p=sim["avg_p"],
        p_normal=sim["p_normal"],
        p_oblique=sim["p_oblique"],
        q_trace=sim["q_trace"],
        i_start=i_cmp0,
        i_end=i_cmp1,
        n_fft=n_fft_cmp,
    )
    vc.assert_same_length(
        (f_cmp, "small cmp_freqs"),
        (ref.cmp_freqs, "ref cmp_freqs"),
        context=f"{sw.name} comparison frequency grid",
    )
    z_cmp = p_cmp / (q_cmp + 1e-30)
    z_ref = ref.cmp_p_avg_spec / (ref.cmp_q_spec + 1e-30)
    dz_rel = np.abs(z_cmp - z_ref) / (np.abs(z_ref) + 1e-30)
    vc.assert_same_length(
        (f_cmp, "cmp_freqs"),
        (dz_rel, "dz_rel"),
        (ref.cmp_band_ref, "cmp_band_ref"),
        context=f"{sw.name} |ΔZ|/|Z_ref|",
    )

    def refl_db(p_small: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
        vc.assert_same_length(
            (p_small, "p_small"),
            (p_ref, "p_ref"),
            context="reflection spectra",
        )
        num = np.abs(p_small - p_ref)
        den = np.abs(p_ref) + 1e-30
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(num / den)

    refl_n = refl_db(pn_cmp, ref_spec.p_normal_spec)
    refl_o = refl_db(po_cmp, ref_spec.p_oblique_spec)
    band_ref = ref.cmp_band_ref.copy()

    f_az, p_az, q_az, band_az, n_fft_analytic = extract_analytic_spectra(
        d,
        avg_p=sim["avg_p"],
        q_trace=sim["q_trace"],
        trace_len=trace_len,
        f_max=f_max,
    )
    lf_df_hz = 1.0 / (n_fft_analytic * d.dt)
    print(
        f"  {sw.name}: LF analysis n_fft={n_fft_analytic:,} "
        f"Δf={lf_df_hz:.4f} Hz (trace {trace_len:,} samples, "
        f"zero-pad ≥4/f_min={4.0 / F_MIN_HZ:.2f} s)",
        flush=True,
    )
    rho, c, h = cfg.air.rho, cfg.air.c, d.h
    a_eff = a_eff_m(s, h)
    re_analytic = analytic_re_z(f_az, rho=rho, c=c)
    ka_eff = 2.0 * math.pi * f_az * a_eff / c
    analytic_band = band_az & (ka_eff < _KA_EFF_MAX)
    vc.assert_same_length(
        (f_az, "freqs_analytic"),
        (analytic_band, "analytic_band"),
        context=f"{sw.name} analytic Re Z",
    )
    z_az = p_az / (q_az + 1e-30)
    re_z = np.real(z_az)
    re_err_analytic = np.abs(re_z - re_analytic) / (np.abs(re_analytic) + 1e-30)
    vc.assert_same_length(
        (f_az, "freqs_analytic"),
        (re_err_analytic, "re_z_err_analytic"),
        context=f"{sw.name} analytic Re Z error",
    )

    def band_max(arr: np.ndarray, mask: np.ndarray, *, ctx: str) -> float:
        vc.assert_same_length(
            (arr, "metric"),
            (mask, "mask"),
            context=ctx,
        )
        if not np.any(mask):
            return float("nan")
        return float(np.max(arr[mask]))

    def at_f(freqs: np.ndarray, arr: np.ndarray, f0: float) -> float:
        idx = int(np.argmin(np.abs(freqs - f0)))
        return float(arr[idx])

    gate_analytic_info = (
        band_max(
            re_err_analytic,
            analytic_band,
            ctx=f"{sw.name} informational analytic",
        )
        if np.any(analytic_band)
        else float("nan")
    )
    gate_dz = (
        band_max(dz_rel, band_ref, ctx=f"{sw.name} gate |ΔZ|") <= _GATE_DZ_REF
        if np.any(band_ref)
        else False
    )
    gate_rn = (
        band_max(refl_n, band_ref, ctx=f"{sw.name} gate refl normal")
        <= _GATE_REFL_NORMAL_DB
        if np.any(band_ref)
        else False
    )
    gate_ro = (
        band_max(refl_o, band_ref, ctx=f"{sw.name} gate refl oblique")
        <= _GATE_REFL_OBLIQUE_DB
        if np.any(band_ref)
        else False
    )
    stable = bool(sim["stable"])
    lf_mask = low_freq_gap_mask(f_az, a_eff=a_eff, c=c)
    vc.assert_same_length(
        (f_az, "freqs_lf_gap"),
        (lf_mask, "lf_gap_mask"),
        (z_az, "z_az"),
        context=f"{sw.name} lf gap spectra",
    )
    passes = gate_dz and gate_rn and gate_ro and stable  # gap_conv applied later
    cost = n**3
    z_check: dict[str, dict[str, float]] = {}
    for f0 in _Z_SUMMARY_FREQS_HZ:
        idx = int(np.argmin(np.abs(f_az - f0)))
        rz_m = float(re_z[idx])
        rz_a = float(re_analytic[idx])
        z_check[f"{f0:g}"] = {
            "re_z_meas": rz_m,
            "re_z_analytic": rz_a,
            "rel_err": abs(rz_m - rz_a) / (abs(rz_a) + 1e-30),
        }

    if sw.name == "baseline":
        print_re_z_discrepancy_diagnostics(
            f_az,
            p_az,
            q_az,
            rho=rho,
            c=c,
            a_eff=a_eff,
            h=h,
            dt=d.dt,
            s=s,
            f_hz_list=_Z_SUMMARY_FREQS_HZ,
            domain=f"small {sw.name} domain",
        )

    return {
        "name": sw.name,
        "testable": True,
        "sweep": asdict(sw),
        "cpml_params": cpml_p,
        "geometry_parity": parity_s,
        "grid_n": n,
        "total_cells": cost,
        "n_steps": n_used,
        "n_steps_budget": n_steps_budget,
        "n_steps_cap": max_steps,
        "n_steps_capped": n_used >= n_steps_budget,
        "decay_steps": sim["decay_steps_reported"],
        "decay_first_step": sim.get("decay_first_step"),
        "decay_reached": sim["decay_reached"],
        "decay_status": "OK" if sim["decay_reached"] else "NO DECAY",
        "tail_level_ratio": sim["tail_level_ratio"],
        "stability_ratio": sim["stability_ratio"],
        "stable": stable,
        "wall_s": sim["wall_s"],
        "receivers": {
            "offset_normal": list(rec.offset_normal),
            "offset_oblique": list(rec.offset_oblique),
            "normal": list(rec.normal),
            "oblique": list(rec.oblique),
        },
        "metrics": {
            "h_m": float(h),
            "t_cpml_refl_arrival_s": float(t_cpml_refl),
            "t_window_end_s": float(t_window_end),
            "max_re_z_err_analytic_ka": float(gate_analytic_info),
            "re_z_analytic_informational": True,
            "max_dz_rel_ref": band_max(
                dz_rel, band_ref, ctx=f"{sw.name} max |ΔZ|"
            ),
            "max_refl_normal_db": band_max(
                refl_n, band_ref, ctx=f"{sw.name} max refl N"
            ),
            "max_refl_oblique_db": band_max(
                refl_o, band_ref, ctx=f"{sw.name} max refl oblique"
            ),
            "f_res_hz": ref.window.f_res_hz,
            "cmp_n_fft": n_fft_cmp,
            "analytic_n_fft": int(n_fft_analytic),
            "lf_df_hz": float(lf_df_hz),
            "z_check_hz": z_check,
            "at_20_hz": {
                "re_z_err_analytic": at_f(f_az, re_err_analytic, 20.0),
                "dz_rel_ref": at_f(f_cmp, dz_rel, 20.0)
                if ref.window.f_res_hz <= 20.0
                else None,
                "refl_normal_db": at_f(f_cmp, refl_n, 20.0)
                if ref.window.f_res_hz <= 20.0
                else None,
                "refl_oblique_db": at_f(f_cmp, refl_o, 20.0)
                if ref.window.f_res_hz <= 20.0
                else None,
            },
            "at_100_hz": {
                "re_z_err_analytic": at_f(f_az, re_err_analytic, 100.0),
                "dz_rel_ref": at_f(f_cmp, dz_rel, 100.0),
                "refl_normal_db": at_f(f_cmp, refl_n, 100.0),
                "refl_oblique_db": at_f(f_cmp, refl_o, 100.0),
            },
            "at_f_max": {
                "re_z_err_analytic": at_f(f_az, re_err_analytic, f_max),
                "dz_rel_ref": at_f(f_cmp, dz_rel, f_max),
                "refl_normal_db": at_f(f_cmp, refl_n, f_max),
                "refl_oblique_db": at_f(f_cmp, refl_o, f_max),
            },
        },
        "gate": {
            "re_z_analytic_info": gate_analytic_info,
            "re_z_analytic": gate_analytic_info <= _GATE_RE_Z_ANALYTIC
            if isinstance(gate_analytic_info, float)
            and not math.isnan(gate_analytic_info)
            else False,
            "dz_ref": gate_dz,
            "refl_normal": gate_rn,
            "refl_oblique": gate_ro,
            "stable": stable,
            "gap_conv": None,
            "pass": passes,
        },
        "spectra": {
            "lf_gap": {
                "freqs_hz": f_az[lf_mask].tolist(),
                "z_re": np.real(z_az[lf_mask]).tolist(),
                "z_im": np.imag(z_az[lf_mask]).tolist(),
            },
            "analytic": {
                "freqs_hz": f_az[analytic_band].tolist(),
                "re_z_err_analytic": re_err_analytic[analytic_band].tolist(),
            },
            "comparison": {
                "freqs_hz": f_cmp[band_ref].tolist(),
                "dz_rel_ref": dz_rel[band_ref].tolist(),
                "refl_normal_db": refl_n[band_ref].tolist(),
                "refl_oblique_db": refl_o[band_ref].tolist(),
            },
        },
    }


def merge_window_plans(plans: list[WindowPlan]) -> WindowPlan:
    """Binding comparison window = longest ``t_window_s`` across configs."""
    best = max(plans, key=lambda p: p.t_window_s)
    return best


def plan_config_windows(
    configs: list[CpmlSweep],
    *,
    f_max: float,
    ppw: float,
    s: int,
    vectors_by_name: dict[str, ReceiverVectors],
    tail_cells: float,
    max_steps: int,
) -> dict[str, ConfigWindowMeta]:
    cfg0, d0 = build_cfg(_baseline(), f_max=f_max, ppw=ppw)
    dt = d0.dt
    out: dict[str, ConfigWindowMeta] = {}
    for sw in configs:
        cfg, d = build_cfg(sw, f_max=f_max, ppw=ppw)
        n = grid_n(sw, s)
        geom = source_geom(sw, n, s)
        vec = vectors_by_name[sw.name]
        rec = build_receiver_set(geom, n, sw.cpml_cells, vec, label=f"{sw.name} plan")
        _set_cube_n(d, n, sw.cpml_cells)
        g = Grid(cfg, d)
        plan, t_refl = build_reflection_measurable_window(
            cfg, d, g, geom, rec, tail_cells=tail_cells
        )
        testable = config_window_testable(
            plan, t_cpml_refl_s=t_refl, max_steps=max_steps, dt=dt
        )
        out[sw.name] = ConfigWindowMeta(
            t_cpml_refl_arrival_s=t_refl,
            t_window_end_s=plan.t_window_s,
            testable=testable,
            plan=plan,
        )
    return out


def project_total_wall_seconds(
    rows: list[DomainPlanRow],
    *,
    max_steps: int,
    win_i_end_nominal: int,
) -> float:
    """Pre-run wall estimate: Σ (n³·steps) / conservative cell-update throughput."""
    small_steps = min(max_steps, win_i_end_nominal + 4000)
    cell_steps = 0.0
    for row in rows:
        steps = row.n_steps if row.label == "reference" else small_steps
        cell_steps += float(row.n**3) * float(steps)
    return cell_steps / _WALL_PROJ_CELLS_PER_S


def plan_all_domains(
    configs: list[CpmlSweep],
    *,
    f_max: float,
    ppw: float,
    s: int,
    tail_cells: float,
    echo_margin: float,
    ram_limit: float,
    max_steps: int,
) -> tuple[
    WindowPlan,
    LayoutSpec,
    list[DomainPlanRow],
    dict[str, ConfigWindowMeta],
    dict[str, ReceiverVectors],
    list[ReceiverVectors],
]:
    layout = make_layout_spec(s)
    vectors_by_name = collect_config_receiver_vectors(
        configs, s=s, f_max=f_max, ppw=ppw
    )
    vector_union = union_receiver_vectors(vectors_by_name)
    config_windows = plan_config_windows(
        configs,
        f_max=f_max,
        ppw=ppw,
        s=s,
        vectors_by_name=vectors_by_name,
        tail_cells=tail_cells,
        max_steps=max_steps,
    )
    window = merge_window_plans([m.plan for m in config_windows.values()])
    cfg_b, d_b = build_cfg(_baseline(), f_max=f_max, ppw=ppw)
    n_ref, t_echo_ref, binding_ref = plan_reference_n(
        cfg_b,
        d_b,
        s=s,
        cpml=_REF_CPML,
        layout=layout,
        vector_union=vector_union,
        window=window,
        tail_cells=tail_cells,
        echo_margin=echo_margin,
        ram_limit=ram_limit,
    )
    geom_ref = source_geom_centered(n_ref, s, _REF_CPML)
    _set_cube_n(d_b, n_ref, _REF_CPML)
    g_ref = Grid(cfg_b, d_b)
    _, cal_ref = build_receivers_and_cal(
        geom_ref,
        n_ref,
        _REF_CPML,
        vector_union[0],
        layout,
        label="reference plan",
        include_cal=True,
    )
    t_sim_ref = reference_sim_horizon_seconds(
        cfg_b, d_b, g_ref, geom_ref, cal_ref, window, tail_cells
    )
    est_ref = vc.require_ram(
        n_ref, _REF_CPML, ram_limit, label="reference domain (planned)"
    )
    ref_steps = min(
        max_steps,
        max(
            int(math.ceil(t_sim_ref / d_b.dt)) + 64,
            int(math.floor(t_echo_ref / d_b.dt)),
        ),
    )
    rows: list[DomainPlanRow] = [
        DomainPlanRow(
            label="reference",
            n=n_ref,
            cpml_cells=_REF_CPML,
            est_bytes=est_ref,
            n_steps=ref_steps,
            t_echo_s=t_echo_ref,
            t_window_s=window.t_window_s,
            notes=f"source centred; binding={binding_ref}; sweep-max window",
        )
    ]
    for sw in configs:
        n = grid_n(sw, s)
        est = vc.require_ram(
            n, sw.cpml_cells, ram_limit, label=f"small domain planned ({sw.name})"
        )
        meta = config_windows[sw.name]
        rows.append(
            DomainPlanRow(
                label=sw.name,
                n=n,
                cpml_cells=sw.cpml_cells,
                est_bytes=est,
                n_steps=max_steps,
                t_cpml_refl_s=meta.t_cpml_refl_arrival_s,
                t_window_s=meta.t_window_end_s,
                testable=meta.testable,
                notes="gap-anchored source",
            )
        )
    print(
        f"[setup] reference domain planned n={n_ref}³ "
        f"({len(vector_union)} unique receiver vector pairs in union)",
        flush=True,
    )
    return window, layout, rows, config_windows, vectors_by_name, vector_union


def recommended_gap_cells(ranked: list[dict[str, Any]]) -> int | None:
    """Smallest gap among passing, testable gap-sweep rows."""
    gaps = [
        int(r["sweep"]["gap_cells"])
        for r in ranked
        if str(r.get("name", "")).startswith("gap_")
    ]
    return min(gaps) if gaps else None


def rank_passing(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    good = [
        r
        for r in results
        if r.get("testable", True)
        and r.get("gate", {}).get("pass")
        and r["sweep"]["cpml_cells"] <= 24
    ]
    return sorted(good, key=lambda r: r["total_cells"])


def _gap_sweep_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [r for r in results if str(r.get("name", "")).startswith("gap_")],
        key=lambda r: int(r["sweep"]["gap_cells"]),
    )


def _interp_gap_cells(
    gaps: list[float],
    vals: list[float],
    *,
    threshold: float,
    lower_is_better: bool,
) -> float | None:
    if len(gaps) < 2 or len(vals) != len(gaps):
        return None
    for i in range(len(gaps)):
        if lower_is_better:
            if vals[i] <= threshold:
                return float(gaps[i])
        elif vals[i] >= threshold:
            return float(gaps[i])
    if lower_is_better and vals[-1] > threshold and vals[0] < vals[-1]:
        for i in range(len(gaps) - 1):
            v0, v1 = vals[i], vals[i + 1]
            if v0 <= threshold or v1 <= threshold:
                continue
            if v0 > threshold and v1 > threshold and v1 != v0:
                g0, g1 = gaps[i], gaps[i + 1]
                frac = (threshold - v0) / (v1 - v0)
                return float(g0 + frac * (g1 - g0))
    return None


def print_no_pass_diagnostics(results: list[dict[str, Any]], *, h_m: float) -> None:
    """When no config passes all gates: best per metric + gap interpolation."""
    testable = [r for r in results if r.get("testable", True)]
    if not testable:
        print("\n=== No testable configs — cannot rank ===")
        return
    print("\n=== No full PASS — best per gate (testable configs) ===")
    metrics = [
        ("|ΔZ|/|Z_ref|", "max_dz_rel_ref", _GATE_DZ_REF, True),
        ("refl normal dB", "max_refl_normal_db", _GATE_REFL_NORMAL_DB, True),
        ("refl oblique dB", "max_refl_oblique_db", _GATE_REFL_OBLIQUE_DB, True),
        ("gap |ΔZ|/|Z_ref|", "gap_max_z_rel", _GATE_GAP_Z_REL, True),
        ("gap |ΔRe Z|/Re", "gap_max_re_rel", _GATE_GAP_RE_Z_REL, True),
    ]
    for label, key, thr, lower in metrics:
        best: dict[str, Any] | None = None
        best_val = math.inf if lower else -math.inf
        for r in testable:
            m = r.get("metrics") or {}
            if key == "gap_max_z_rel":
                val = (m.get("gap_conv") or {}).get("max_z_rel", float("nan"))
            elif key == "gap_max_re_rel":
                val = (m.get("gap_conv") or {}).get("max_re_rel", float("nan"))
            else:
                val = m.get(key, float("nan"))
            if not isinstance(val, (int, float)) or math.isnan(float(val)):
                continue
            val_f = float(val)
            if lower and val_f < best_val:
                best_val = val_f
                best = r
            elif not lower and val_f > best_val:
                best_val = val_f
                best = r
        if best is None:
            print(f"  {label}: (no data)")
            continue
        ok = best_val <= thr if lower else best_val >= thr
        print(
            f"  {label}: {best['name']} = {best_val:.4g} "
            f"(gate {thr:.4g}) {'PASS' if ok else 'FAIL'}"
        )

    gap_rows = _gap_sweep_rows(testable)
    if len(gap_rows) < 2:
        return
    gaps = [float(r["sweep"]["gap_cells"]) for r in gap_rows]
    print("\n=== Gap sweep interpolation (cells and mm @ h) ===")
    interp_defs = [
        ("|ΔZ|/|Z_ref|", lambda r: float(r["metrics"].get("max_dz_rel_ref", float("nan"))), _GATE_DZ_REF),
        ("refl normal dB", lambda r: float(r["metrics"].get("max_refl_normal_db", float("nan"))), _GATE_REFL_NORMAL_DB),
        ("refl oblique dB", lambda r: float(r["metrics"].get("max_refl_oblique_db", float("nan"))), _GATE_REFL_OBLIQUE_DB),
        ("gap |ΔZ|/|Z_ref|", lambda r: float((r.get("metrics", {}).get("gap_conv") or {}).get("max_z_rel", float("nan"))), _GATE_GAP_Z_REL),
        ("gap |ΔRe Z|/Re", lambda r: float((r.get("metrics", {}).get("gap_conv") or {}).get("max_re_rel", float("nan"))), _GATE_GAP_RE_Z_REL),
    ]
    for label, fn, thr in interp_defs:
        vals = [fn(r) for r in gap_rows]
        if any(math.isnan(v) for v in vals):
            print(f"  {label}: insufficient gap-sweep data")
            continue
        g_need = _interp_gap_cells(gaps, vals, threshold=thr, lower_is_better=True)
        if g_need is None:
            print(f"  {label}: no interpolated crossing within gap∈{gaps}")
        else:
            print(
                f"  {label}: ~{g_need:.1f} cells "
                f"({g_need * h_m * 1e3:.2f} mm @ h={h_m * 1e3:.3f} mm) for gate {thr}"
            )


def numerics_recommendation(sw: CpmlSweep) -> dict[str, Any]:
    return {
        "cpml_cells": sw.cpml_cells,
        "cpml_R": sw.cpml_R,
        "cpml_m": sw.cpml_m,
        "cpml_kappa_max": sw.cpml_kappa_max,
        "cpml_alpha_factor": sw.cpml_alpha_factor,
        "gap_cells": sw.gap_cells,
        "note": (
            "Set these fields on fdtd.config.Numerics (defaults in config.py); "
            "cpml_alpha = cpml_alpha_factor * pi * f_min is applied in derive()."
        ),
    }


def print_report(
    results: list[dict[str, Any]],
    ref: RefCache,
    *,
    f_max: float,
    ppw: float,
    h_m: float,
    gate_pass: bool,
    recommended: CpmlSweep | None,
    recommended_gap: int | None,
    sweep_estimate_s: float | None,
    ranked: list[dict[str, Any]],
) -> None:
    win = ref.window
    print("\nC3-cpml — boxed_piston Stage A")
    print(
        f"  f={F_MIN_HZ:.0f}–{f_max / 1e3:.1f} kHz  ppw={ppw:g}  "
        f"reference n={ref.n}³ (interior {ref.n - 2 * _REF_CPML})  "
        f"echo≥{ref.t_echo_min * 1e3:.3f} ms"
    )
    print(
        f"  T_window={win.t_window_s * 1e3:.3f} ms "
        f"(pulse {win.t_pulse_s * 1e3:.3f} + prop {win.t_prop_s * 1e3:.3f} "
        f"+ tail {win.t_tail_s * 1e3:.3f})  f_res={win.f_res_hz:.1f} Hz"
    )
    print(f"  {ref.meta['low_freq_authority']}")
    if sweep_estimate_s is not None:
        print(
            f"  projected full-sweep wall (after 1st config): ~{sweep_estimate_s:.0f} s"
        )
    hdr = (
        f"{'name':<22} {'cells':>9} {'ReZ*':>9} {'|ΔZ|/Zr':>9} "
        f"{'gap Z':>9} {'norm dB':>9} {'obl dB':>9} {'decay':>8} "
        f"{'tail':>8} {'stab':>5} {'gate':>10}"
    )
    print("\n=== Sweep summary (band maxima) ===")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        m = r["metrics"]
        g = r["gate"]
        if not r.get("testable", True):
            decay_col = "—"
            tail_col = "—"
            gate_col = "UNTESTABLE"
            gap_col = "        —"
            re_z = dz = rn = rg = float("nan")
            stab = "—"
        else:
            gc = m.get("gap_conv") or {}
            gap_col = f"{gc.get('max_z_rel', float('nan')):>9.3f}"
            re_z = m.get("max_re_z_err_analytic_ka", float("nan"))
            dz = m.get("max_dz_rel_ref", float("nan"))
            rn = m.get("max_refl_normal_db", float("nan"))
            rg = m.get("max_refl_oblique_db", float("nan"))
            decay_col = (
                "OK"
                if r.get("decay_reached")
                else (r.get("decay_status") or "NO DECAY")
            )
            tail_r = r.get("tail_level_ratio")
            tail_col = f"{tail_r:.2e}" if tail_r is not None else "—"
            stab = "ok" if g.get("stable") else "NO"
            gate_col = "PASS" if g.get("pass") else "FAIL"
        print(
            f"{r['name']:<22} {r['total_cells']:>9,} "
            f"{re_z:>9.3f} "
            f"{dz:>9.3f} "
            f"{gap_col} "
            f"{rn:>9.2f} "
            f"{rg:>9.2f} "
            f"{decay_col:>8} "
            f"{tail_col:>8} "
            f"{stab:>5} "
            f"{gate_col:>10}"
        )
    print("  *ReZ informational only (not gated)")

    base = next((r for r in results if r["name"] == "baseline"), results[0] if results else None)
    if base is not None and "z_check_hz" in base.get("metrics", {}):
        print("\n=== Re Z vs analytic (baseline, post-injection ⟨p⟩/Qⁿ) — informational ===")
        zhdr = f"{'f Hz':>6} {'Re Z meas':>12} {'Re Z ana':>12} {'rel err':>10}"
        print(zhdr)
        print("-" * len(zhdr))
        for fk, row in sorted(
            base["metrics"]["z_check_hz"].items(),
            key=lambda x: float(x[0]),
        ):
            print(
                f"{float(fk):6.0f} {row['re_z_meas']:12.4e} "
                f"{row['re_z_analytic']:12.4e} {row['rel_err']:10.3f}"
            )

    print("\n=== Ranked PASS (testable, all gates; cheapest cells first) ===")
    if not ranked:
        print("  (none)")
        print_no_pass_diagnostics(results, h_m=h_m)
    else:
        for i, r in enumerate(ranked[:10], 1):
            print(
                f"  {i:2d}. {r['name']:<20} cells={r['total_cells']:,} "
                f"cpml={r['sweep']['cpml_cells']} gap={r['sweep']['gap_cells']}"
            )

    print("\n=== GATE (provisional C3-cpml) ===")
    print(
        f"  Re Z vs analytic (ka_eff<{_KA_EFF_MAX}): INFORMATIONAL only "
        f"(typical |Re Z|/|Z| ~0.5–2% at 50–500 Hz; phase dominates)"
    )
    print(
        f"  Low-f gap convergence vs gap_{_GAP_LF_REF_CELLS} "
        f"({F_MIN_HZ:.0f}–{_GAP_LF_F_MAX_HZ:.0f} Hz, ka_eff<{_KA_EFF_MAX}): "
        f"|ΔRe Z|/Re Z_ref ≤{_GATE_GAP_RE_Z_REL:.0%}, "
        f"|ΔZ|/|Z_ref| ≤{_GATE_GAP_Z_REL:.1%} (every config except gap_40)"
    )
    if recommended_gap is not None:
        print(
            f"  Recommended gap (smallest passing in ranked set): "
            f"{recommended_gap} cells ({recommended_gap * h_m * 1e3:.2f} mm @ h)"
        )
    print(
        f"  |ΔZ|/|Z_ref| for f≥f_res: ≤{_GATE_DZ_REF:.0%}"
    )
    print(
        f"  normal reflection (f≥f_res): ≤{_GATE_REFL_NORMAL_DB:.0f} dB; "
        f"oblique: ≤{_GATE_REFL_OBLIQUE_DB:.0f} dB"
    )
    print("  late-time stability: tail/peak < 1e-3")
    print(f"  Overall PASS: {'PASS' if gate_pass else 'FAIL'}")
    if recommended is not None:
        print("\n=== RECOMMENDED Numerics defaults ===")
        for k, v in numerics_recommendation(recommended).items():
            if k != "note":
                print(f"  {k} = {v!r}")
        print(f"  ({numerics_recommendation(recommended)['note']})")


def calibration_for_json(cal: dict[str, Any] | None) -> dict[str, Any] | None:
    """Plain calibration summary for JSON (no nested live sim references)."""
    if cal is None:
        return None
    out: dict[str, Any] = {
        "pass": bool(cal.get("pass")),
        "label": cal.get("label"),
        "s_cube": cal.get("s_cube"),
        "phase_fit": dict(cal.get("phase_fit") or {}),
        "centre_offset_m": cal.get("centre_offset_m"),
        "source_centroid_m": list(cal.get("source_centroid_m") or ()),
        "index_centre_m": list(cal.get("index_centre_m") or ()),
    }
    rows: list[dict[str, Any]] = []
    for row in cal.get("rows") or []:
        if not isinstance(row, dict):
            continue
        r = {k: row[k] for k in row if k not in ("n_hat",)}
        if "n_hat" in row:
            r["n_hat"] = [float(x) for x in row["n_hat"]]
        if "offset_cells" in row:
            r["offset_cells"] = list(row["offset_cells"])
        rows.append(r)
    out["rows"] = rows
    out["windows"] = [
        {
            "receiver": w.get("receiver"),
            "t_window_s": w.get("t_window_s"),
            "i_win": list(w.get("i_win") or ()),
            "n_fft": w.get("n_fft"),
        }
        for w in cal.get("windows") or []
    ]
    return out


def result_row_for_json(r: dict[str, Any]) -> dict[str, Any]:
    """One sweep result: metrics/gates + decimated spectra only."""
    gp = r.get("geometry_parity") or {}
    geom_p: dict[str, Any] = {}
    for key, val in gp.items():
        if key == "source_centre":
            geom_p[key] = list(val) if val is not None else None
        elif isinstance(val, dict):
            geom_p[key] = {
                "offset_cells": list(val.get("offset_cells") or ()),
                "r_m": val.get("r_m"),
                "indices": list(val.get("indices") or ()),
            }
        else:
            geom_p[key] = val
    sp = r.get("spectra") or {}
    sp_out: dict[str, Any] = {}
    for band_name in ("analytic", "comparison", "lf_gap"):
        block = sp.get(band_name)
        if not block:
            continue
        sp_out[band_name] = {
            k: (v if not isinstance(v, list) else v)
            for k, v in block.items()
        }
    return {
        "name": r.get("name"),
        "testable": r.get("testable", True),
        "untestable_reason": r.get("untestable_reason"),
        "sweep": dict(r.get("sweep") or {}),
        "cpml_params": dict(r.get("cpml_params") or {}),
        "geometry_parity": geom_p,
        "grid_n": r.get("grid_n"),
        "total_cells": r.get("total_cells"),
        "n_steps": r.get("n_steps"),
        "n_steps_budget": r.get("n_steps_budget"),
        "n_steps_cap": r.get("n_steps_cap"),
        "n_steps_capped": r.get("n_steps_capped"),
        "decay_steps": r.get("decay_steps"),
        "decay_first_step": r.get("decay_first_step"),
        "decay_reached": r.get("decay_reached"),
        "decay_status": r.get("decay_status"),
        "tail_level_ratio": r.get("tail_level_ratio"),
        "stability_ratio": r.get("stability_ratio"),
        "stable": r.get("stable"),
        "wall_s": r.get("wall_s"),
        "receivers": r.get("receivers"),
        "metrics": r.get("metrics"),
        "gate": r.get("gate"),
        "spectra": sp_out,
    }


def build_json_payload(
    *,
    ts: str,
    args: argparse.Namespace,
    ref: RefCache,
    results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    recommended: CpmlSweep | None,
    gate_pass: bool,
    calibration: dict[str, Any],
    gap_conv_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    win = ref.window
    cal_j = calibration_for_json(calibration)
    ref_meta = {
        k: v
        for k, v in ref.meta.items()
        if k != "calibration"
    }
    ref_meta["cpml_params"] = dict(ref.meta.get("cpml_params") or {})
    return {
        "timestamp_utc": ts,
        "f_min_hz": F_MIN_HZ,
        "f_max_hz": args.f_max,
        "ppw": args.ppw,
        "source_cube_s": args.source_size,
        "quick": args.quick,
        "gates": {
            "re_z_analytic_max": _GATE_RE_Z_ANALYTIC,
            "re_z_analytic_gated": False,
            "gap_lf_ref_cells": _GAP_LF_REF_CELLS,
            "gap_lf_f_hz": [F_MIN_HZ, _GAP_LF_F_MAX_HZ],
            "gap_re_z_max": _GATE_GAP_RE_Z_REL,
            "gap_z_max": _GATE_GAP_Z_REL,
            "dz_ref_max": _GATE_DZ_REF,
            "refl_normal_db": _GATE_REFL_NORMAL_DB,
            "refl_oblique_db": _GATE_REFL_OBLIQUE_DB,
            "ka_eff_max_analytic": _KA_EFF_MAX,
        },
        "window": asdict(win),
        "f_res_hz": win.f_res_hz,
        "low_freq_authority": ref.meta.get("low_freq_authority"),
        "layout": asdict(ref.layout),
        "calibration": cal_j,
        "reference": {
            "grid_n": ref.n,
            "t_echo_min_s": ref.t_echo_min,
            "cmp_i_win": list(ref.cmp_i_win),
            "cmp_n_fft": ref.cmp_n_fft,
            "meta": ref_meta,
            "receiver_vector_union": [
                {
                    "offset_normal": list(v.offset_normal),
                    "offset_oblique": list(v.offset_oblique),
                }
                for v in ref.vector_union
            ],
            "geom_parity_by_vector": [
                {
                    "offset_normal": list(k[0]),
                    "offset_oblique": list(k[1]),
                    **{kk: vv for kk, vv in val.items()},
                }
                for k, val in ref.geom_parity_by_vector.items()
            ],
        },
        "results": [result_row_for_json(r) for r in results],
        "ranked_pass": [r["name"] for r in ranked],
        "recommended": asdict(recommended) if recommended else None,
        "recommended_numerics": (
            numerics_recommendation(recommended) if recommended else None
        ),
        "gate_pass": gate_pass,
        "gap_convergence": gap_conv_summary,
    }


def maybe_plot(
    results: list[dict[str, Any]],
    out_path: Path | None,
    *,
    names: tuple[str, ...] = ("baseline",),
    recommend_name: str | None = None,
) -> None:
    if out_path is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping PNG", file=sys.stderr)
        return
    by_name = {r["name"]: r for r in results}
    plot_names = list(names)
    if recommend_name and recommend_name not in plot_names:
        plot_names.append(recommend_name)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    for nm in plot_names:
        r = by_name.get(nm)
        if r is None:
            continue
        sp = r["spectra"]
        sp_a = sp.get("analytic", {})
        sp_c = sp.get("comparison", {})
        if sp_a:
            f = np.asarray(sp_a["freqs_hz"], dtype=np.float64)
            axes[0].semilogy(
                f, sp_a["re_z_err_analytic"], label=nm, alpha=0.85
            )
        if sp_c:
            fc = np.asarray(sp_c["freqs_hz"], dtype=np.float64)
            axes[1].plot(fc, sp_c["refl_normal_db"], label=f"{nm} normal", alpha=0.85)
            axes[1].plot(
                fc,
                sp_c["refl_oblique_db"],
                linestyle="--",
                label=f"{nm} oblique",
                alpha=0.65,
            )
    axes[0].axhline(_GATE_RE_Z_ANALYTIC, color="k", ls=":", lw=0.8)
    axes[0].set_ylabel("Re Z rel. err vs analytic")
    axes[0].set_title("C3-cpml impedance error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].axhline(_GATE_REFL_NORMAL_DB, color="C0", ls=":", lw=0.8)
    axes[1].axhline(_GATE_REFL_OBLIQUE_DB, color="C1", ls=":", lw=0.8)
    axes[1].set_ylabel("reflection (dB)")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_title("Reflection vs large reference")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  plot: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="C3-cpml Stage A: CPML vs reference")
    parser.add_argument("--f-max", type=float, default=_DEFAULT_F_MAX, help="Band top (Hz)")
    parser.add_argument("--ppw", type=float, default=_DEFAULT_PPW, help="PPW at f-max")
    parser.add_argument("--source-size", type=int, default=_DEFAULT_S, help="Cube side s")
    parser.add_argument("--quick", action="store_true", help="Baseline + 3 key variants")
    parser.add_argument("--only", type=str, default="", help="Run a single named config")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "validation" / "results" / "c03_cpml",
        help="Output directory",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG")
    parser.add_argument("--verbose", action="store_true", help="Per-config geometry detail")
    parser.add_argument(
        "--ram-limit-gib",
        type=float,
        default=DEFAULT_RAM_LIMIT_BYTES / 1024**3,
        help="Max estimated grid RAM (GiB, default 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_DEFAULT_MAX_STEPS,
        help="Small-domain FDTD steps (decay cap)",
    )
    parser.add_argument(
        "--tail-cells",
        type=float,
        default=_DEFAULT_TAIL_CELLS,
        help="Tail margin in cells·h/c added to comparison window",
    )
    parser.add_argument(
        "--echo-margin",
        type=float,
        default=_DEFAULT_ECHO_MARGIN,
        help="Reference echo must exceed T_window by this factor",
    )
    parser.add_argument(
        "--cal-point-source",
        action="store_true",
        help="After calibration, rerun monopole cal with s=1 point source",
    )
    args = parser.parse_args()

    run_timer = vc.RunTimer()
    s = args.source_size
    ram_limit = float(args.ram_limit_gib) * 1024**3
    configs = iter_sweep_configs(quick=args.quick)
    if args.only:
        configs = [c for c in configs if c.name == args.only]
        if not configs:
            print(f"ERROR: unknown --only {args.only!r}", file=sys.stderr)
            return 2

    run_timer.begin("setup")
    try:
        window, layout, domain_rows, config_windows, vectors_by_name, vector_union = (
            plan_all_domains(
            configs,
            f_max=args.f_max,
            ppw=args.ppw,
            s=s,
            tail_cells=args.tail_cells,
            echo_margin=args.echo_margin,
            ram_limit=ram_limit,
            max_steps=args.max_steps,
            )
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except vc.RamLimitExceeded as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    proj_wall_s = project_total_wall_seconds(
        domain_rows,
        max_steps=args.max_steps,
        win_i_end_nominal=window.i_end_nominal,
    )
    setup_lines = [
        (
            f"f_max={args.f_max / 1e3:.1f} kHz ppw={args.ppw:g} s={s} "
            f"configs={len(configs)} RAM≤{vc.fmt_bytes(ram_limit)}"
        ),
        (
            f"T_window(max)={window.t_window_s * 1e3:.3f} ms "
            f"f_res={window.f_res_hz:.1f} Hz tail={args.tail_cells:g} cells"
        ),
        f"projected total wall (ref + sweep, pre-run): ~{proj_wall_s:.0f} s ({proj_wall_s / 60:.1f} min)",
        "Planned domains (estimate before any allocation):",
    ]
    for row in domain_rows:
        echo = (
            f" echo≥{row.t_echo_s * 1e3:.3f} ms"
            if row.t_echo_s is not None
            else ""
        )
        refl = (
            f" CPML.refl@{row.t_cpml_refl_s * 1e3:.2f}ms"
            if row.t_cpml_refl_s is not None
            else ""
        )
        win_end = (
            f" T_win={row.t_window_s * 1e3:.2f}ms"
            if row.t_window_s is not None
            else ""
        )
        test = "" if row.label == "reference" else (
            " TESTABLE" if row.testable else " UNTESTABLE"
        )
        setup_lines.append(
            f"  {row.label:<22} n={row.n:>4} cpml={row.cpml_cells:>2} "
            f"steps={row.n_steps:>6,} est {vc.fmt_bytes(row.est_bytes):>9}{echo}"
            f"{refl}{win_end}{test} {row.notes}"
        )
    run_timer.end("setup")
    run_timer.banner("setup", " | ".join(setup_lines[:2]))
    for line in setup_lines[2:]:
        print(f"[setup] {line}", flush=True)

    run_timer.begin("compile")
    sw_w = _baseline()
    cfg_w, d_w = build_cfg(sw_w, f_max=args.f_max, ppw=args.ppw)
    n_w = grid_n(sw_w, s)
    geom_w = source_geom(sw_w, n_w, s)
    g_w, coeffs_w, slabs_w, cpml_w = build_grid(
        cfg_w, d_w, n_w, ram_limit=ram_limit, label="Numba warm-up"
    )
    vec_w = vectors_by_name[sw_w.name]
    build_receiver_set(geom_w, n_w, sw_w.cpml_cells, vec_w, label="warm-up")
    fields_w = allocate_fields(g_w)
    s3 = float(s**3)
    dP = float(
        point_source_pressure_increment(
            cfg_w.air.rho, cfg_w.air.c, d_w.dt, d_w.h, 1.0 / s3
        )
    )
    _inject_cube_uniform(fields_w.p, geom_w.i0, geom_w.j0, geom_w.k0, s, dP)
    t0 = __import__("time").perf_counter()
    step(fields_w, coeffs_w, slabs_w, cpml=cpml_w)
    warm_ms = (__import__("time").perf_counter() - t0) * 1e3
    run_timer.end("compile")
    run_timer.banner("compile", f"Numba warm-up {warm_ms:.1f} ms (1 step)")
    vc.print_leapfrog_timing_convention(dt=float(d_w.dt))

    run_timer.begin("ref")
    try:
        ref, calibration = run_reference(
            f_max=args.f_max,
            ppw=args.ppw,
            s=s,
            window=window,
            layout=layout,
            vector_union=vector_union,
            tail_cells=args.tail_cells,
            ram_limit=ram_limit,
            echo_margin=args.echo_margin,
            run_timer=run_timer,
        )
    except vc.RamLimitExceeded as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    run_timer.end("ref")

    run_timer.begin("cal")
    dt_ref = float(ref.meta["dt_s"])
    print_calibration_table(calibration, dt=dt_ref)
    if args.cal_point_source:
        cal_s1 = run_cal_point_source_isolation(
            f_max=args.f_max,
            ppw=args.ppw,
            ref_n=ref.n,
            tail_cells=args.tail_cells,
            ram_limit=ram_limit,
            run_timer=run_timer,
        )
        print_calibration_table(cal_s1, dt=dt_ref)
        pf_c = calibration.get("phase_fit") or {}
        pf_s = cal_s1.get("phase_fit") or {}
        print(
            "\n=== Point-source isolation (s=1 vs s={}) ===".format(s),
            flush=True,
        )
        print(
            f"  cube cal:  φ₀={pf_c.get('phi0_deg', float('nan')):.2f}° "
            f"τ={pf_c.get('tau_steps', float('nan')):.4f}·Δt",
            flush=True,
        )
        print(
            f"  point cal: φ₀={pf_s.get('phi0_deg', float('nan')):.2f}° "
            f"τ={pf_s.get('tau_steps', float('nan')):.4f}·Δt",
            flush=True,
        )
        if cal_s1["pass"] and not calibration["pass"]:
            print(
                "  → point source PASS, cube FAIL: implicate cube form factor / centre",
                flush=True,
            )
        elif abs(pf_s.get("phi0_deg", 0.0)) < abs(pf_c.get("phi0_deg", 0.0)):
            print(
                "  → smaller phase error at s=1: check geometric centre vs (ic,jc,kc)",
                flush=True,
            )
    if not calibration["pass"]:
        bad = [
            r
            for r in calibration["rows"]
            if not r.get("mag_ok", True) or not r.get("phase_ok", True)
        ]
        print(
            "ERROR: monopole calibration failed — check Q injection "
            f"(ρc²ΔtQ/(s³h³) per cell), q_trace FFT, leapfrog sample time, "
            f"and P/Q sign. Failed bins: {len(bad)}",
            file=sys.stderr,
        )
        run_timer.end("cal")
        return 2
    run_timer.end("cal")

    cfg_chk, _ = build_cfg(_baseline(), f_max=args.f_max, ppw=args.ppw)
    rho_chk, c_chk = cfg_chk.air.rho, cfg_chk.air.c
    _assert_analytic_re_z_100hz(rho=rho_chk, c=c_chk)

    run_timer.begin("sweep")
    results: list[dict[str, Any]] = []
    sweep_estimate_s: float | None = None
    baseline_cpml: dict[str, float] | None = None
    for i, sw in enumerate(configs, 1):
        run_timer.banner("sweep", f"{i}/{len(configs)} {sw.name}")
        try:
            row = metrics_for_config(
                sw,
                f_max=args.f_max,
                ppw=args.ppw,
                s=s,
                ref=ref,
                win_meta=config_windows[sw.name],
                vectors=vectors_by_name[sw.name],
                max_steps=args.max_steps,
                ram_limit=ram_limit,
                run_timer=run_timer,
                verbose=args.verbose,
            )
        except vc.RamLimitExceeded as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        results.append(row)
        cp = row.get("cpml_params")
        if cp is not None:
            if baseline_cpml is None:
                baseline_cpml = cp
            elif sw.key() != _baseline().key():
                bl = _baseline()
                if (
                    sw.cpml_R != bl.cpml_R
                    or sw.cpml_alpha_factor != bl.cpml_alpha_factor
                    or sw.cpml_cells != bl.cpml_cells
                ):
                    assert_cpml_params_change(cp, baseline_cpml, ctx=sw.name)
        if i == 1:
            per_cfg = row["wall_s"]
            n_used = row.get("n_steps", 0)
            n_budget = row.get("n_steps_budget", row.get("n_steps_cap", 0))
            ref_wall = ref.meta.get("wall_s", 0.0)
            sweep_estimate_s = ref_wall + per_cfg * len(configs)
            print(
                f"[sweep] first config wall {per_cfg:.1f} s "
                f"({n_used:,}/{n_budget:,} steps, early stop) → "
                f"estimate ~{sweep_estimate_s:.0f} s for "
                f"{len(configs)} small runs (+ ref already done)",
                flush=True,
            )
    run_timer.end("sweep")

    _cfg_end, _ = build_cfg(_baseline(), f_max=args.f_max, ppw=args.ppw)
    gap_conv_summary = apply_gap_convergence_gates(
        results, s=s, c=float(_cfg_end.air.c)
    )
    ranked = rank_passing(results)
    rec_gap = recommended_gap_cells(ranked)
    gap_conv_summary["recommended_gap_cells"] = rec_gap
    gate_pass = len(ranked) > 0
    recommended: CpmlSweep | None = None
    if ranked:
        r0 = ranked[0]
        sw_fields = dict(r0["sweep"])
        sw_fields.pop("name", None)
        recommended = CpmlSweep(r0["name"], **sw_fields)

    run_timer.begin("output")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"c03_cpml_{ts}.json"
    png_path = None if args.no_plot else out_dir / f"c03_cpml_{ts}.png"
    rec_name = ranked[0]["name"] if ranked else None
    payload = build_json_payload(
        ts=ts,
        args=args,
        ref=ref,
        results=results,
        ranked=ranked,
        recommended=recommended,
        gate_pass=gate_pass,
        calibration=calibration,
        gap_conv_summary=gap_conv_summary,
    )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(vc.to_json_safe(payload), f, indent=2)
    run_timer.end("output")
    run_timer.banner("output", f"JSON {json_path.name}")

    maybe_plot(
        results,
        png_path,
        names=("baseline",),
        recommend_name=rec_name,
    )

    run_timer.begin("gate")
    print_report(
        results,
        ref,
        f_max=args.f_max,
        ppw=args.ppw,
        h_m=float(ref.meta["h_m"]),
        gate_pass=gate_pass,
        recommended=recommended,
        recommended_gap=rec_gap,
        sweep_estimate_s=sweep_estimate_s,
        ranked=ranked,
    )
    run_timer.end("gate")
    phases = ("setup", "compile", "ref", "cal", "sweep", "output", "gate")
    vc.print_runtime_summary(run_timer, phases)
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
