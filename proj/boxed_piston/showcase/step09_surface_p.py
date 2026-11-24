#!/usr/bin/env python3
"""Step 9: sphere surface pressure vs analytic moments (wall patches)."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_OUT = _PKG / "outputs" / "step09"
_CFG = _PKG / "configs" / "c13_sphere_ps.json"
_FREQS = (300.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0)

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import Config  # noqa: E402
from boxrad.geomrun import dry_run_geom  # noqa: E402
from boxrad.post.transfer import h_transfer_err  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402
from boxrad.validation.analytic import sphere_point_source, sphere_surface_moments  # noqa: E402
from validation._common import to_json_safe  # noqa: E402

_F64 = np.float64
_GATE_MEAN_DB, _GATE_MEAN_DEG = 0.15, 1.5
_GATE_FORCE_DB, _GATE_FORCE_DEG = 0.3, 2.0
_GATE_PW_DB, _GATE_PW_DEG = 0.5, 3.0


def _cfg_n(n_cells: int) -> Config:
    raw = json.loads(_CFG.read_text(encoding="utf-8"))
    raw["numerics"]["h_override"] = 0.05 / float(n_cells)
    raw["numerics"].pop("cells_across_radiator", None)
    out = raw.setdefault("output", {})
    out["path"] = ""
    out["probes"] = []
    return Config.from_dict(raw)


def _h_target(n_cells: int) -> float:
    return 0.05 / float(n_cells)


def _reuse_h5_if_matching(h5: Path, h_target: float) -> bool:
    if not h5.is_file():
        return False
    try:
        import h5py

        with h5py.File(h5, "r") as f:
            if "surface" not in f or "patch_centroids" not in f["surface"]:
                return False
            derived = json.loads(bytes(f["meta/derived_json"][()]).decode("utf-8"))
            h_stored = float(derived["h"])
    except OSError:
        return False
    return abs(h_stored - h_target) <= 1e-12 * max(abs(h_target), 1e-30)


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=_F64), q))


def _analyze(h5_path: Path, cfg: Config, n_cells: int) -> dict[str, Any]:
    rho = float(cfg.air.rho)
    c = float(cfg.air.c)
    a = 0.05
    r0 = float(cfg.numerics.test_source.position[2])  # type: ignore[union-attr]
    src = np.array([0.0, 0.0, r0], dtype=_F64)
    ez = np.array([0.0, 0.0, 1.0], dtype=_F64)

    out: dict[str, Any] = {"n_cells_across_d": n_cells, "h5": str(h5_path), "frequencies_hz": list(_FREQS)}
    rows: list[dict[str, Any]] = []

    f_arr = np.asarray(_FREQS, dtype=_F64)
    with Result(h5_path, frame="world") as res:
        pts = res.surface_points(frame="world")
        n_w = pts.shape[0]
        areas = res.surface_areas()
        n_w_vec = res.surface_area_vectors(frame="world")
        out["n_wet"] = int(n_w)
        h_all = res.surface_p(f_arr)

        for fi, f_hz in enumerate(_FREQS):
            p_w = h_all[fi]
            area_sum = np.sum(areas)
            mean_fd = np.sum(areas * p_w) / area_sum
            c0, f_ref = sphere_surface_moments(f_hz, a, r0, rho, c)
            mean_db, mean_deg = h_transfer_err(c0, mean_fd)
            n_dot_ez = np.sum(n_w_vec * ez, axis=1)
            f_fd = -np.sum(p_w * n_dot_ez)
            f_db, f_deg = h_transfer_err(f_ref, f_fd)
            radii = np.linalg.norm(pts, axis=1)
            radii = np.maximum(radii, a)
            theta = np.arccos(np.clip(pts[:, 2] / radii, -1.0, 1.0))
            p_an = sphere_point_source(pts, f_hz, np.zeros(3), a, src, rho, c)
            pw_db: list[float] = []
            pw_deg: list[float] = []
            for k in range(n_w):
                db, ph = h_transfer_err(p_an[k], p_w[k])
                pw_db.append(db)
                pw_deg.append(abs(ph))
            row = {
                "f_hz": f_hz,
                "mean_db": mean_db,
                "mean_deg": mean_deg,
                "force_db": f_db,
                "force_deg": f_deg,
                "pointwise_max_db": max(pw_db),
                "pointwise_max_deg": max(pw_deg),
                "pointwise_p95_db": _percentile(pw_db, 95),
                "pointwise_p95_deg": _percentile(pw_deg, 95),
                "theta_rad": theta.tolist(),
                "p_w_abs": np.abs(p_w).tolist(),
                "p_an_abs": np.abs(p_an).tolist(),
            }
            rows.append(row)

    out["per_f"] = rows
    return out


def _plot_panel(results: dict[int, dict[str, Any]], png_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for n_cells, style in ((9, "o-"), (18, "s--")):
        data = results[n_cells]
        freqs = [r["f_hz"] for r in data["per_f"]]
        mean_db = [r["mean_db"] for r in data["per_f"]]
        force_db = [r["force_db"] for r in data["per_f"]]
        axes[0].plot(freqs, mean_db, style, label=f"n={n_cells}")
        axes[1].plot(freqs, force_db, style, label=f"n={n_cells}")
    axes[0].set_xlabel("f [Hz]")
    axes[0].set_ylabel("mean c₀ error [dB]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("f [Hz]")
    axes[1].set_ylabel("force error [dB]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    row_3k = next(r for r in results[9]["per_f"] if abs(r["f_hz"] - 3000.0) < 1.0)
    th = np.asarray(row_3k["theta_rad"], dtype=_F64)
    p_fd = np.asarray(row_3k["p_w_abs"], dtype=_F64)
    p_an = np.asarray(row_3k["p_an_abs"], dtype=_F64)
    order = np.argsort(th)
    axes[2].plot(np.rad2deg(th[order]), p_an[order], "-", label="series")
    axes[2].plot(np.rad2deg(th[order]), p_fd[order], ".", label="FDTD n=9")
    axes[2].set_xlabel("θ [deg]")
    axes[2].set_ylabel("|p_w| [Pa per Q]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _gate_n9(report: dict[str, Any]) -> dict[str, bool]:
    ok = True
    checks: dict[str, bool] = {}
    for row in report["per_f"]:
        checks[f"mean_{row['f_hz']}"] = (
            abs(row["mean_db"]) <= _GATE_MEAN_DB and abs(row["mean_deg"]) <= _GATE_MEAN_DEG
        )
        checks[f"force_{row['f_hz']}"] = (
            abs(row["force_db"]) <= _GATE_FORCE_DB and abs(row["force_deg"]) <= _GATE_FORCE_DEG
        )
        checks[f"pw_{row['f_hz']}"] = (
            row["pointwise_p95_db"] <= _GATE_PW_DB and row["pointwise_p95_deg"] <= _GATE_PW_DEG
        )
        ok = ok and checks[f"mean_{row['f_hz']}"] and checks[f"force_{row['f_hz']}"] and checks[f"pw_{row['f_hz']}"]
    checks["all"] = ok
    return checks


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    json_path = _OUT / "surface_p.json"
    png_path = _OUT / "surface_p.png"
    results: dict[str, Any] = {"runs": {}}
    analyzed: dict[int, dict[str, Any]] = {}

    for n_cells in (9, 18):
        cfg = _cfg_n(n_cells)
        h5 = _OUT / f"c13_sphere_n{n_cells}.h5"
        h_want = _h_target(n_cells)
        est = dry_run_geom(cfg)
        print(
            f"n={n_cells} grid={est['grid_shape']} h={est['h']:.4g} "
            f"n_wet={est['n_wet']} n_steps={est['n_steps']} wall_est={est['wall_time_s_est']:.2g} s"
        )
        if _reuse_h5_if_matching(h5, h_want):
            print(f"n={n_cells} reusing existing {h5} (h={h_want:.6g})")
        else:
            t0 = time.perf_counter()
            with keep_awake(f"step09 n={n_cells}"):
                run(cfg, out_path=h5, progress=True)
            print(f"n={n_cells} run wall {time.perf_counter() - t0:.1f} s")
        rep = _analyze(h5, cfg, n_cells)
        analyzed[n_cells] = rep
        results["runs"][str(n_cells)] = rep

    gates = _gate_n9(analyzed[9])
    results["gates_n9"] = gates
    ratio: dict[str, float] = {}
    for row9, row18 in zip(analyzed[9]["per_f"], analyzed[18]["per_f"]):
        f_hz = row9["f_hz"]
        for key in ("mean_db", "force_db", "pointwise_p95_db"):
            v9 = abs(row9[key])
            v18 = abs(row18[key])
            ratio[f"{key}_{f_hz}"] = v9 / max(v18, 1e-30)
    results["error_ratio_n9_over_n18"] = ratio
    results["gate_thresholds"] = {
        "mean": [_GATE_MEAN_DB, _GATE_MEAN_DEG],
        "force": [_GATE_FORCE_DB, _GATE_FORCE_DEG],
        "pointwise_p95": [_GATE_PW_DB, _GATE_PW_DEG],
    }

    json_path.write_text(json.dumps(to_json_safe(results), indent=2), encoding="utf-8")
    _plot_panel(analyzed, png_path)
    print(f"wrote {json_path} and {png_path}")
    print(f"n_wet n9={analyzed[9]['n_wet']} n18={analyzed[18]['n_wet']}")
    print(f"gates n9: {gates}")


if __name__ == "__main__":
    main()