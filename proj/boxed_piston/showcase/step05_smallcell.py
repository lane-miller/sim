#!/usr/bin/env python3
"""Step 5 showcase: small-cell floor, stability, thin-wall, reference energy."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boxrad.config import Config, derive
from boxrad.cutcell import _STATE_CUT, compute_fractions, grid_for_geometry
from boxrad.geom.geometry import build_geometry
from boxrad.refstep import closed_alpha, init_velocity, lam_max, step
from boxrad.smallcell import apply_floor, courant_lambda, stability_ok, thin_wall_cells

_PKG = Path(__file__).resolve().parents[1]
_SHOWCASE = _PKG / "configs" / "showcase"
_OUT = Path(__file__).resolve().parent / "out" / "step05_smallcell.png"
H = 0.005

def _case_path(jname: str) -> Path:
    p = _SHOWCASE / jname
    if p.exists():
        return p
    return _PKG / "configs" / jname


CASES = [
    ("S1", "s1_cylinder.json"),
    ("S2", "s2_bookshelf.json"),
    ("S3", "c09_box.json"),
    ("S4", "s4_tilted.json"),
    ("S5", "s5_platform.json"),
    ("S6", "s6_downfire.json"),
]


def _rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _energy_setup():
    from boxrad.cutcell import Grid
    from boxrad.geom import geometry, mesh

    R = _rot_z(37.0) @ _rot_x(20.0)
    solid = mesh.box_mesh([6.0, 6.0, 6.0]).transformed(R, np.zeros(3))
    grid = Grid(
        x0=np.array([-7.0, -7.0, -7.0], dtype=np.float64),
        h=1.0,
        shape=(14, 14, 14),
        rigid_sides={},
    )
    geom = geometry.Geometry(solid=solid, radiators=[], planes=(), provenance={})
    fr = compute_fractions(geom, grid)
    return fr, grid


def main() -> None:
    cfg_c9 = Config.from_json((_PKG / "configs" / "c09_box.json").read_text(encoding="utf-8"))
    lam_ref = courant_lambda(cfg_c9, derive(cfg_c9))
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9])
    axes = np.empty((3, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])
    hist_floored = None
    hist_raw = None
    blow_step = None

    for idx, (name, jname) in enumerate(CASES):
        cfg = Config.from_json(_case_path(jname).read_text(encoding="utf-8"))
        geom = build_geometry(cfg)
        assert geom is not None
        grid = grid_for_geometry(geom, H)
        fr = compute_fractions(geom, grid)
        lam = lam_ref
        beta_tilde, fst = apply_floor(fr, lam, h=H)
        ok, _ = stability_ok(beta_tilde, fr.alpha, lam)
        tw_cand, tw_st = thin_wall_cells(fr, geom, grid, max_thin=10**9)
        tag = (
            "PASS"
            if ok and fst["delta_eq_over_h"] <= 0.05 and tw_st["n_confirmed"] == 0
            else "CHECK"
        )
        print(
            f"{name}: n_cut={fr.stats['n_cut_cells']} min_raw_beta={fst['min_raw_beta_cut']:.4g} "
            f"n_floored={fst['n_floored']} dV={fst['delta_V']:.4g} "
            f"deq/h={fst['delta_eq_over_h']:.4g}({fst['floor_status']}) "
            f"stable={ok} thin_cand={tw_st['n_candidates']} thin_conf={tw_st['n_confirmed']} {tag}"
        )
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        cut = fr.state == _STATE_CUT
        vals = fr.beta[cut]
        vals = vals[vals > 0.0]
        if vals.size:
            ax.hist(vals, bins=30, log=True)
        ax.set_xlim(1e-6, 1.0)
        fl = fst["floor_level_median"]
        if fl > 0:
            ax.axvline(fl, color="r", ls="--", lw=1)
        ax.set_title(name)

    fr_e, grid_e = _energy_setup()
    lam_e = 0.99 / math.sqrt(3.0)
    beta_tilde, _ = apply_floor(fr_e, lam_e, h=grid_e.h)
    alpha_e = closed_alpha(fr_e.alpha)
    rng = np.random.default_rng(0)
    p = np.zeros_like(fr_e.beta)
    p[fr_e.beta > 0.0] = rng.standard_normal(np.sum(fr_e.beta > 0.0))
    u = init_velocity(alpha_e)
    _, _, hist_floored = step(p, u, beta_tilde, alpha_e, lam_e, 4000)
    u2 = init_velocity(alpha_e)
    p2 = p.copy()
    hist_raw = []
    pmax0 = float(np.max(np.abs(p2)))
    for n in range(500):
        p2, u2, h1 = step(p2, u2, fr_e.beta, alpha_e, lam_e, 1)
        hist_raw.append(h1[0])
        if not np.all(np.isfinite(p2)) or float(np.max(np.abs(p2))) > 1e6 * pmax0:
            blow_step = n + 1
            break
    hist_raw = np.array(hist_raw, dtype=np.float64)
    h0 = hist_floored[0]
    rel = np.abs(hist_floored - h0) / max(abs(h0), 1e-30)
    ax = fig.add_subplot(gs[2, :])
    ax.semilogy(rel, label="floored")
    rel_u = np.abs(hist_raw - h0) / max(abs(h0), 1e-30)
    rel_u = np.minimum(rel_u, 1e6)
    ax.semilogy(rel_u, label="unfloored", alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("|H-H0|/H0")
    ax.legend()
    ax.set_title(f"energy drift (λ={lam_ref:.4g} showcase; λ_e={lam_e:.4g})")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(_OUT, dpi=120)
    print(f"wrote {_OUT} blow_step={blow_step}")


if __name__ == "__main__":
    main()