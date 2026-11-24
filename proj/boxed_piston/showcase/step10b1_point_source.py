#!/usr/bin/env python3
"""Step 10b1: free-field monopole (numerics.test_source) vs analytic monopole_free."""

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
_OUT = _PKG / "outputs" / "step10b1"
_CFG = _PKG / "configs" / "c13_free_ps.json"

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import Config  # noqa: E402
from boxrad.geomrun import dry_run_geom  # noqa: E402
from boxrad.post.directivity import di_from_pattern  # noqa: E402
from boxrad.post.transfer import h_transfer_err, probe_fdtd_pressure_spectrum  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402
from boxrad.validation.analytic import monopole_free  # noqa: E402
from validation._common import to_json_safe  # noqa: E402

_F64 = np.float64
_FREQS = (300.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0)
_FAR_R = 2.0
_FAR_DIRS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0),
    np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0),
    np.array([0.0, 0.0, -1.0]),
)
_GATE_PROBE_DB = 0.15
_GATE_PROBE_DEG = 1.0
_GATE_FF_DB = 0.15
_GATE_FF_DEG = 1.5
_GATE_DI_DB = 0.05


def _load_cfg() -> Config:
    return Config.from_json(_CFG.read_text(encoding="utf-8"))


def _snap_probes_to_grid(cfg: Config) -> Config:
    est = dry_run_geom(cfg)
    from boxrad.geomrun import prepare_geom_run

    run = prepare_geom_run(cfg)
    cc = np.array(run.derived.test_source_cell_center or (0.0, 0.0, 0.0), dtype=_F64)
    h = float(run.derived.h)
    x0 = run.gg.x0.astype(_F64)
    d = cfg.to_dict()
    probes = []
    for row in cfg.output.probes or []:
        p = np.array(row, dtype=_F64)
        rel = np.round((p - x0) / h - 0.5)
        snapped = x0 + (rel + 0.5) * h
        probes.append(snapped.tolist())
    d.setdefault("output", {})["probes"] = probes
    del est, cc
    return Config.from_dict(d)


def main() -> None:
    cfg = _snap_probes_to_grid(_load_cfg())
    _OUT.mkdir(parents=True, exist_ok=True)
    h5_path = _OUT / "c13_free.h5"
    json_path = _OUT / "c13_free.json"
    png_path = _OUT / "c13_free.png"

    est = dry_run_geom(cfg)
    print(
        f"grid={est['grid_shape']}  h={est['h']:.4g}  n_steps={est['n_steps']}  "
        f"wall_est={est['wall_time_s_est']:.3g} s"
    )

    t0 = time.perf_counter()
    with keep_awake("step10b1 c13_free"):
        run(cfg, out_path=h5_path, progress=True)
    print(f"run wall {time.perf_counter() - t0:.1f} s")

    rho = float(cfg.air.rho)
    c = float(cfg.air.c)
    src = np.array(cfg.numerics.test_source.position, dtype=_F64)  # type: ignore[union-attr]
    results: dict[str, Any] = {"frequencies_hz": list(_FREQS), "probes": [], "far_field": [], "di_db": {}}

    probe_db: list[float] = []
    probe_ph: list[float] = []
    ff_db: list[float] = []

    with Result(h5_path, frame="world") as res:
        import h5py

        with h5py.File(h5_path, "r") as f:
            t_pr = np.asarray(f["probes/times_p"][()], dtype=_F64)
            p_ts = np.asarray(f["probes/p"][()], dtype=_F64)
            centers = np.asarray(f["probes/centers"][()], dtype=_F64)
        h = float(res.derived.h)
        lam0 = c / 1000.0
        for f_hz in _FREQS:
            lam = c / f_hz
            print(f"\n--- f = {f_hz:.0f} Hz  R/λ probe0 = {np.linalg.norm(centers[1] - src) / lam:.4g}")
            row_p: dict[str, Any] = {"f_hz": f_hz, "probes": []}
            for q, cen in enumerate(centers):
                p_fd = probe_fdtd_pressure_spectrum(p_ts, t_pr, f_hz, q)
                _, p_ref = monopole_free(cen[np.newaxis, :], f_hz, src, rho, c)
                h_fd = p_fd / res._radiator_v_at_f(f_hz)
                db, deg = h_transfer_err(p_ref[0, 0], h_fd)
                print(f"  probe {q} @{cen}  dB {db:+.4f}  deg {deg:+.2f}")
                row_p["probes"].append({"db": db, "deg": deg, "center": cen.tolist()})
                if q < 3:
                    probe_db.append(db)
                    probe_ph.append(deg)
            results["probes"].append(row_p)

            row_ff: dict[str, Any] = {"f_hz": f_hz, "directions": []}
            for dname, u in zip(("axis", "diag_xy", "diag_xyz", "down"), _FAR_DIRS):
                pt = (src + _FAR_R * u)[np.newaxis, :]
                p_fd = res.field(pt, f_hz, frame="world")[0, 0]
                _, p_ref = monopole_free(pt, f_hz, src, rho, c)
                db, deg = h_transfer_err(p_ref[0, 0], p_fd)
                print(f"  far {dname}  dB {db:+.4f}  deg {deg:+.2f}")
                ff_db.append(db)
                row_ff["directions"].append({"name": dname, "db": db, "deg": deg})
            results["far_field"].append(row_ff)

            th = np.linspace(0.0, math.pi, 24)
            ph = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
            th_g, ph_g = np.meshgrid(th, ph, indexing="ij")
            d_pat = res.directivity(th_g.ravel(), ph_g.ravel(), f_hz, frame="world")
            di = di_from_pattern(th_g, ph_g, d_pat)
            results["di_db"][str(int(f_hz))] = float(20.0 * math.log10(abs(di)))
            print(f"  DI {results['di_db'][str(int(f_hz))]:+.3f} dB")

    json_path.write_text(json.dumps(to_json_safe(results), indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].plot(_FREQS, probe_db[0::3], "o-", label="probe 0")
    axes[0].plot(_FREQS, probe_db[1::3], "s-", label="probe 1")
    axes[0].plot(_FREQS, probe_db[2::3], "^-", label="probe 2")
    axes[0].set_ylabel("dB err")
    axes[0].set_xlabel("f [Hz]")
    axes[0].legend(fontsize=8)
    axes[0].set_title("probes dB")
    axes[1].plot(_FREQS, probe_ph[0::3], "o-", label="probe 0")
    axes[1].plot(_FREQS, probe_ph[1::3], "s-", label="probe 1")
    axes[1].plot(_FREQS, probe_ph[2::3], "^-", label="probe 2")
    axes[1].set_ylabel("deg err")
    axes[1].set_xlabel("f [Hz]")
    axes[1].set_title("probes phase")
    mean_ff = [float(np.mean(ff_db[i * 4 : (i + 1) * 4])) for i in range(len(_FREQS))]
    axes[2].plot(_FREQS, mean_ff, "o-")
    axes[2].set_ylabel("mean dB err")
    axes[2].set_xlabel("f [Hz]")
    axes[2].set_title("far-field dB")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"wrote {json_path}  {png_path}")


if __name__ == "__main__":
    main()