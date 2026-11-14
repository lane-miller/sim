#!/usr/bin/env python3
"""C6-baffle tail diagnostics: p_avg and interior energy vs time (three ablations)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdtd.config import (  # noqa: E402
    Config,
    DEFAULT_RAM_LIMIT_BYTES,
    Derived,
    derive,
    estimate_memory_bytes,
    validate,
)
from fdtd.cpml import make_cpml_state  # noqa: E402
from fdtd.grid import Grid, allocate_fields  # noqa: E402
from fdtd.kernels import make_update_coeffs, step  # noqa: E402
from fdtd.source import velocity_pulse_derivative  # noqa: E402
from fdtd.source import BoxBoundary  # noqa: E402
import validation._common as vc  # noqa: E402

_F64 = np.float64


def _interior_energy_half_step(
    p_n: np.ndarray,
    p_np1: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    grid: Grid,
    *,
    rho: float,
    c: float,
) -> float:
    """Yee energy in the non-CPML interior at time level n+½ (kernel test definition)."""
    s = grid.cpml_slabs
    i0, i1 = s.i_inner.lo, s.i_inner.hi
    j0, j1 = s.j_inner.lo, s.j_inner.hi
    k0, k1 = s.k_inner.lo, s.k_inner.hi
    h = grid.h
    vol = h**3
    pn = p_n[i0:i1, j0:j1, k0:k1]
    pp = p_np1[i0:i1, j0:j1, k0:k1]
    pe = 0.5 * vol / (rho * c**2) * np.sum(pn.astype(_F64) * pp.astype(_F64))
    vxi = vx[i0 : i1 + 1, j0:j1, k0:k1]
    vyi = vy[i0:i1, j0 : j1 + 1, k0:k1]
    vzi = vz[i0:i1, j0:j1, k0 : k1 + 1]
    ke = 0.5 * rho * vol * (
        np.sum(vxi.astype(_F64) ** 2)
        + np.sum(vyi.astype(_F64) ** 2)
        + np.sum(vzi.astype(_F64) ** 2)
    )
    return float(pe + ke)


def _radiator_p_avg(
    fields,
    grid: Grid,
    boundary: BoxBoundary,
    derived: Derived,
    n: int,
) -> float:
    if boundary.idx_radiator.size == 0:
        return 0.0
    dt = float(derived.dt)
    t_p = (float(n) + 1.0) * dt
    dvdt = velocity_pulse_derivative(t_p, float(derived.pulse_sigma), float(derived.pulse_t0))
    vz_shape = (grid.nx, grid.ny, grid.nz + 1)
    ir, jr, _ = np.unravel_index(boundary.idx_radiator, vz_shape)
    kf = int(grid.k_face_z)
    p_flat = fields.p.reshape(-1)
    rad_p = np.ravel_multi_index((ir, jr, np.full_like(ir, kf)), (grid.nx, grid.ny, grid.nz))
    w = boundary.w_radiator.astype(_F64)
    wsum = float(np.sum(w)) if w.size else 1.0
    pc = p_flat[rad_p].astype(_F64)
    corr = 0.5 * grid.h * grid.cfg.air.rho * w * dvdt
    p_face = pc + corr
    return float(np.sum(w * p_face) / wsum)


def _tail_stats(p: np.ndarray) -> dict[str, float]:
    if p.size == 0:
        return {"tail_db": float("nan"), "tail_mean": float("nan"), "tail_rms": float("nan")}
    pk = float(np.max(np.abs(p)))
    n_tail = max(1, int(math.floor(0.05 * p.size)))
    seg = p[-n_tail:]
    tail_pk = float(np.max(np.abs(seg)))
    tail_db = 20.0 * math.log10(tail_pk / pk) if pk > 0 else float("-inf")
    return {
        "tail_db": tail_db,
        "tail_mean": float(np.mean(seg)),
        "tail_rms": float(np.sqrt(np.mean(seg**2))),
    }


def _energy_milestones(energy: np.ndarray) -> dict[str, Any]:
    if energy.size == 0:
        return {}
    peak = float(np.max(energy))
    if peak <= 0:
        rel = np.zeros_like(energy)
    else:
        rel = energy / peak
    n = energy.size
    idx = {pct: int(round(pct * (n - 1))) for pct in (0.25, 0.5, 0.75, 1.0)}
    mid = n // 2
    growing = bool(n > 2 and energy[-1] > energy[mid])
    return {
        "peak": peak,
        "rel_25": float(rel[idx[0.25]]),
        "rel_50": float(rel[idx[0.5]]),
        "rel_75": float(rel[idx[0.75]]),
        "rel_100": float(rel[idx[1.0]]),
        "growing_second_half": growing,
    }


def _run_diag_case(
    cfg: Config,
    *,
    label: str,
    step_mult: int = 4,
) -> dict[str, Any]:
    d = derive(cfg)
    errs, warns = validate(cfg, d)
    if errs:
        raise ValueError("; ".join(errs))
    for w in warns:
        print(f"  warn: {w}", flush=True)
    d = replace(d, n_steps=int(d.n_steps) * step_mult, run_time_s=d.run_time_s * step_mult)

    grid = Grid(cfg, d)
    fields = allocate_fields(grid)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, d.h, d.dt)
    slabs = grid.cpml_slabs
    cpml = make_cpml_state(grid, cfg, d, d.dt)
    boundary = BoxBoundary.from_grid(grid, d)

    n_steps = d.n_steps
    sample_every = max(1, n_steps // 400)
    times_p: list[float] = []
    p_avg: list[float] = []
    times_e: list[float] = []
    energy: list[float] = []

    rho, c = cfg.air.rho, cfg.air.c
    for n in range(n_steps):
        p_n = fields.p.copy()
        step(fields, coeffs, slabs, cpml, boundary, n)
        if n % sample_every == 0 or n + 1 == n_steps:
            t_p = (float(n) + 1.0) * d.dt
            t_e = (float(n) + 0.5) * d.dt
            times_p.append(t_p)
            p_avg.append(_radiator_p_avg(fields, grid, boundary, d, n))
            times_e.append(t_e)
            energy.append(
                _interior_energy_half_step(
                    p_n, fields.p, fields.vx, fields.vy, fields.vz, grid, rho=rho, c=c
                )
            )

    p_arr = np.asarray(p_avg, dtype=np.float64)
    e_arr = np.asarray(energy, dtype=np.float64)
    t_p = np.asarray(times_p, dtype=np.float64)
    t_e = np.asarray(times_e, dtype=np.float64)
    stats = _tail_stats(p_arr)
    e_stat = _energy_milestones(e_arr)

    print(f"\n=== {label}  steps={n_steps}  grid {d.nx}×{d.ny}×{d.nz} ===")
    print(
        f"  tail (last 5%): {stats['tail_db']:.1f} dB re peak; "
        f"mean={stats['tail_mean']:.4e}  RMS={stats['tail_rms']:.4e}"
    )
    print(
        f"  interior E / peak @ 25/50/75/100%: "
        f"{e_stat['rel_25']:.3f} / {e_stat['rel_50']:.3f} / "
        f"{e_stat['rel_75']:.3f} / {e_stat['rel_100']:.3f}; "
        f"growing in last half: {e_stat['growing_second_half']}"
    )

    return {
        "label": label,
        "n_steps": n_steps,
        "grid": [d.nx, d.ny, d.nz],
        "times_p": t_p,
        "p_avg": p_arr,
        "times_e": t_e,
        "energy": e_arr,
        **stats,
        **e_stat,
    }


def _maybe_plot(runs: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot", flush=True)
        return

    fig, (ax_p, ax_e) = plt.subplots(2, 1, sharex=False, figsize=(8, 6), layout="tight")
    for run in runs:
        p = np.maximum(np.abs(run["p_avg"]), 1e-30)
        ax_p.plot(run["times_p"], np.log10(p), label=run["label"])
        ax_e.plot(run["times_e"], run["energy"], label=run["label"])
    ax_p.set_ylabel(r"$\log_{10}|p_{\mathrm{avg}}|$")
    ax_p.legend(fontsize=8)
    ax_p.set_xlabel("t (s)")
    ax_e.set_ylabel("interior energy")
    ax_e.legend(fontsize=8)
    ax_e.set_xlabel("t (s)")
    fig.suptitle("C6-baffle tail diagnostics")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C6-baffle p_avg / energy diagnostics")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "c06_baffle.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "results")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--ram-gb", type=float, default=DEFAULT_RAM_LIMIT_BYTES / 1e9)
    args = parser.parse_args(argv)

    base_cfg = Config.from_json(args.config.read_text(encoding="utf-8"))
    ram_limit = float(args.ram_gb) * 1e9
    timer = vc.RunTimer()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, Config]] = [
        ("baseline×4 steps", copy.deepcopy(base_cfg)),
    ]
    cfg_cpml = copy.deepcopy(base_cfg)
    n_cpml = cfg_cpml.numerics
    cfg_cpml.numerics = replace(n_cpml, cpml_cells=2 * n_cpml.cpml_cells)
    variants.append(("×4 steps, 2× cpml_cells", cfg_cpml))
    cfg_gap = copy.deepcopy(base_cfg)
    n_gap = cfg_gap.numerics
    gap0 = n_gap.gap_cells if n_gap.gap_cells is not None else 10
    cfg_gap.numerics = replace(n_gap, gap_cells=2 * gap0)
    variants.append(("×4 steps, 2× gap_cells", cfg_gap))

    for label, cfg in variants:
        d_plan = derive(cfg)
        est = estimate_memory_bytes(d_plan.nx, d_plan.ny, d_plan.nz, d_plan.cpml_cells)
        if est > ram_limit:
            raise vc.RamLimitExceeded(label, d_plan.nx, float(est), float(ram_limit))

    runs: list[dict[str, Any]] = []
    for label, cfg in variants:
        timer.banner("c06_diag", label)
        timer.begin(label)
        runs.append(_run_diag_case(cfg, label=label, step_mult=4))
        timer.end(label)

    png = None if args.no_plot else args.out / f"c06_diag_{stamp}.png"
    if png is not None:
        _maybe_plot(runs, png)

    payload = {
        "config": str(args.config),
        "runs": [
            {k: v for k, v in r.items() if k not in ("times_p", "times_e", "p_avg", "energy")}
            for r in runs
        ],
        "wall_phases_s": timer.phases,
    }
    out_json = args.out / f"c06_diag_{stamp}.json"
    out_json.write_text(json.dumps(vc.to_json_safe(payload), indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())