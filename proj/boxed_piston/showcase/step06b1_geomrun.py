#!/usr/bin/env python3
"""Step 6b1 showcase: parametric derive, dry-run summaries, stepping parity."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boxrad.config import Config, derive
from boxrad.cpml import make_cpml_state
from boxrad.geomrun import dry_run_geom, prepare_geom_run, step_geom
from boxrad.grid import Grid, allocate_fields
from boxrad.kernels import make_update_coeffs, step
from boxrad.source import BoxBoundary

_PKG = Path(__file__).resolve().parents[1]
_SHOWCASE = _PKG / "configs" / "showcase"
_OUT = Path(__file__).resolve().parent / "out" / "step06b1_geomrun.png"


def _load(name: str) -> Config:
    return Config.from_json((_SHOWCASE / name).read_text(encoding="utf-8"))


def main() -> None:
    t0 = time.perf_counter()
    for name in (
        "s1_cylinder.json",
        "s3p_c09_parametric.json",
        "s4_tilted.json",
        "s5_platform.json",
        "s6_downfire.json",
    ):
        s = dry_run_geom(_load(name))
        print(
            f"{name}: {s['grid_shape']} h={s['h']:.4g} path={s['radiator_path']} "
            f"S={s['S_grid']:.5g} n_cut={s['n_cut']}"
        )

    c09 = Config.from_json((_PKG / "configs" / "c09_box.json").read_text(encoding="utf-8"))
    leg_d = derive(c09)
    grid = Grid(c09, leg_d)
    boundary = BoxBoundary.from_grid(grid, leg_d)
    coeffs = make_update_coeffs(c09.air.rho, c09.air.c, leg_d.h, leg_d.dt)
    cpml = make_cpml_state(grid, c09, leg_d, leg_d.dt)
    leg = allocate_fields(grid)

    s3p = _load("s3p_c09_parametric.json")
    run_p = prepare_geom_run(s3p)
    fr = run_p.info["fractions"]
    mask = fr.beta > 0
    xc, yc = 0.03, 0.02
    probe = np.array([xc, yc, 3.0 * leg_d.h], dtype=np.float64)
    ic = int(round((probe[0] - float(run_p.gg.x0[0])) / leg_d.h - 0.5))
    jc = int(round((probe[1] - float(run_p.gg.x0[1])) / leg_d.h - 0.5))
    kc = int(round((probe[2] - float(run_p.gg.x0[2])) / leg_d.h - 0.5))
    p_leg, p_cut, p_rel = [], [], []
    for n in range(400):
        step(leg, coeffs, grid.cpml_slabs, cpml, boundary, n)
        step_geom(run_p, n)
        p_leg.append(float(leg.p[ic, jc, kc]))
        p_cut.append(float(run_p.fields.p[ic, jc, kc]))
        if n > 0:
            d = abs(p_cut[-1] - p_leg[-1]) / max(abs(p_leg[-1]), 1e-30)
            p_rel.append(d)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.plot(p_leg, label="C9 legacy")
    ax.plot(p_cut, "--", label="S3p cut")
    ax.set_title("p(t) 3h above radiator center")
    ax.legend()
    ax = axes[0, 1]
    ax.semilogy(np.maximum(p_rel, 1e-30))
    ax.set_title("relative difference")

    traces = {}
    for name, n_steps in (
        ("s1_cylinder.json", 300),
        ("s4_tilted.json", 300),
        ("s6_downfire.json", 300),
    ):
        cfg = _load(name)
        run = prepare_geom_run(cfg)
        rd = run.geometry.radiators[0]
        n_hat = rd.pose.R[:, 2] / np.linalg.norm(rd.pose.R[:, 2])
        pos = rd.pose.position + 0.05 * n_hat
        h = run.derived.h
        x0 = run.gg.x0
        i = int(round((pos[0] - x0[0]) / h - 0.5))
        j = int(round((pos[1] - x0[1]) / h - 0.5))
        k = int(round((pos[2] - x0[2]) / h - 0.5))
        pt = []
        for n in range(n_steps):
            step_geom(run, n)
            pt.append(float(run.fields.p[i, j, k]))
        traces[name] = pt

    ax = axes[1, 0]
    for label, pt in traces.items():
        ax.plot(pt, label=label.split(".")[0])
    ax.set_title("p(t) 0.05 m in front of radiator 0")
    ax.legend(fontsize=8)
    axes[1, 1].axis("off")
    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=120)
    print(f"wrote {_OUT} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()