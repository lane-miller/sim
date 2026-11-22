#!/usr/bin/env python3
"""Step 4 showcase: cut-cell fractions for S1–S6."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boxrad.config import Config, derive
from boxrad.cutcell import _STATE_CUT, compute_fractions, grid_for_geometry
from boxrad.geom.geometry import build_geometry

_PKG = Path(__file__).resolve().parents[1]
_SHOWCASE = _PKG / "configs" / "showcase"
_OUT = Path(__file__).resolve().parent / "out" / "step04_fractions.png"
H = 0.005
def _case_path(jname: str) -> Path:
    p = _SHOWCASE / jname
    if p.exists():
        return p
    return _PKG / "configs" / jname


CASES = [
    ("S1", "s1_cylinder.json", 0, (0, 2)),
    ("S2", "s2_bookshelf.json", 0, (1, 2)),
    ("S3", "c09_box.json", 0, (0, 2)),
    ("S4", "s4_tilted.json", 0, None),
    ("S5", "s5_platform.json", 0, (1, 2)),
    ("S6", "s6_downfire.json", 0, (0, 2)),
]
RTOL_V = 1e-12
RTOL_S = 1e-12


def _analytic_vol_info(name: str, cfg: Config, geom) -> float | None:
    """Analytic reference volumes for info-only printout (not used for PASS)."""
    if name == "S1":
        part = cfg.geometry_spec.parts[0]
        d, L = part.params["d"], part.params["length"]
        r = 0.5 * d
        return math.pi * r * r * L
    if name == "S4":
        return 0.2 * 0.15 * 0.25
    if name == "S2":
        return 0.2 * 0.25 * 0.30
    if name == "S5":
        return 0.2 * 0.25 * 0.30 + 0.40 * 0.35 * 0.03
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    rows = []
    for idx, (name, jname, ridx, slab_axes) in enumerate(CASES):
        cfg = Config.from_json(_case_path(jname).read_text(encoding="utf-8"))
        geom = build_geometry(cfg)
        assert geom is not None
        grid = grid_for_geometry(geom, H)
        fr = compute_fractions(geom, grid)
        if name == "S3":
            d = derive(cfg)
            target_v = d.realized_W * d.realized_H * d.realized_D
        else:
            target_v = geom.solid.volume + fr.stats["voids_removed_volume"]
        rel_v = abs(fr.stats["fluid_volume"] - target_v) / max(abs(target_v), 1e-30)
        ok = rel_v <= RTOL_V and fr.stats["max_line_closure_residual"] <= 1e-9
        for s, rd in zip(fr.radiator_S, geom.radiators):
            rel_s = abs(s - rd.area) / max(rd.area, 1e-30)
            ok = ok and rel_s <= RTOL_S
        info_v = _analytic_vol_info(name, cfg, geom)
        info_s = ""
        if info_v is not None:
            info_s = f" analytic_vol={info_v:.6e}"
        hs = f"{grid.h * 1e3:.3f} mm"
        if grid.h != grid.h_nominal:
            hs += f" (h'={grid.h:.6f})"
        print(
            f"{name}: shape={grid.shape} h={hs} N={grid.n_cells} "
            f"cut={fr.stats['n_cut_cells']} "
            f"vol err={rel_v:.2e} res={fr.stats['max_line_closure_residual']:.1e} "
            f"time={fr.stats['wall_time_s']:.2f}s "
            f"{'PASS' if ok else 'CHECK'}"
            f"{info_s}"
        )
        rows.append(ok)
        ax = axes.flat[idx]
        beta = fr.beta
        if name == "S2":
            ix = int(grid.world_to_index(np.zeros(3))[0])
            ix = max(0, min(beta.shape[0] - 1, ix))
            sl = beta[ix, :, :].T
        elif slab_axes == (0, 2):
            sl = beta[beta.shape[0] // 2, :, :].T
        elif slab_axes == (1, 2):
            sl = beta[:, beta.shape[1] // 2, :].T
        else:
            sl = beta[:, :, beta.shape[2] // 2].T
        ax.imshow(sl, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(f"{name} cut={fr.stats['n_cut_cells']}")
    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()