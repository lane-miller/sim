#!/usr/bin/env python3
"""Step 6b2b: radiator-frame Result API — report-only showcase."""

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
_OUT = _PKG / "outputs" / "step06b2b"
_STEP06A = _PKG / "outputs" / "step06b2a"
_CONFIGS = _PKG / "configs"
_SHOWCASE = _CONFIGS / "showcase"

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import Config  # noqa: E402
from boxrad.geomrun import dry_run_geom  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402
from validation import _common as vc  # noqa: E402

_F64 = np.float64


def _load_json(path: Path) -> Config:
    return Config.from_json(path.read_text(encoding="utf-8"))


def _ensure_h5(name: str, cfg_path: Path) -> Path:
    cached = _STEP06A / f"{name}.h5"
    if cached.is_file():
        print(f"reuse {_STEP06A / name}.h5")
        return cached
    cfg = _load_json(cfg_path)
    if cfg.geometry == "parametric":
        est = dry_run_geom(cfg)
    else:
        est = run(cfg, dry_run=True, progress=False)
        assert isinstance(est, dict)
    print(
        f"run {name}: grid={est.get('grid_shape')}  n_steps={est.get('n_steps')}  "
        f"wall_est={est.get('wall_time_s_est', float('nan')):.3g} s"
    )
    out = _OUT / f"{name}.h5"
    with keep_awake(f"step06b2b {name}"):
        run(cfg, out_path=out, progress=True)
    return out


def _c18_preview(s3p_h5: Path, s4_h5: Path) -> dict[str, Any]:
    freqs = np.array([500.0, 1000.0, 2000.0, 3000.0], dtype=_F64)
    phi_list = (0.0, 0.5 * math.pi)
    theta = np.deg2rad(np.arange(0.0, 181.0, 5.0))
    rows: list[dict[str, Any]] = []
    print("\nC18(i) directivity (S4 local vs s3p world), normalize=True")
    with Result(s3p_h5) as r_s3, Result(s4_h5) as r_s4:
        for f_hz in freqs:
            for phi in phi_list:
                d_s3 = r_s3.directivity(theta, phi, f_hz, frame="world", normalize=True)
                d_s4 = r_s4.directivity(theta, phi, f_hz, frame="local", normalize=True)
                ref = float(np.max(np.abs(d_s3)))
                mask = 20.0 * np.log10(np.maximum(np.abs(d_s3) / max(ref, 1e-30), 1e-30)) > -20.0
                if np.any(mask):
                    ratio = np.where(mask, d_s4 / np.where(mask, d_s3, 1.0), 1.0)
                    db_max = float(np.max(np.abs(20.0 * np.log10(np.abs(ratio)))))
                else:
                    db_max = float("nan")
                ph_deg = float(np.rad2deg(phi))
                print(f"  f={f_hz:.0f} Hz  phi={ph_deg:.0f}°  max |ΔdB|={db_max:.4g}")
                rows.append({"f_hz": f_hz, "phi_deg": ph_deg, "max_db": db_max})
    return {"c18": rows}


def _frame_sanity(s4_h5: Path) -> dict[str, Any]:
    with Result(s4_h5) as res:
        r_mat, t = res.pose
        n_hat = np.asarray(res.meta["radiator0_normal"], dtype=_F64)
        # On-axis local +z → world normal from rotation column 2
        axis_world = r_mat[:, 2]
        err = float(np.linalg.norm(axis_world - n_hat))
        print("\n(b) S4 frame sanity")
        print(f"  t = {t}")
        print(f"  R =\n{r_mat}")
        print(f"  local θ=0 world dir = {axis_world}  |Δn| vs meta = {err:.3e}")
        return {
            "pose_t": t.tolist(),
            "pose_R": r_mat.tolist(),
            "axis_world": axis_world.tolist(),
            "n_hat_meta": n_hat.tolist(),
            "axis_err": err,
        }


def _s6_gap_table(s6_h5: Path) -> dict[str, Any]:
    gap = np.array([0.0, 0.0, 0.015], dtype=_F64)
    side = np.array([0.5, 0.0, 0.015], dtype=_F64)
    top = np.array([0.0, 0.0, 0.73], dtype=_F64)
    pts = np.vstack([gap, side, top])
    labels = ("gap probe", "side 0.5 m", "top 0.5 m")
    freqs_line = np.geomspace(80.0, 3000.0, 60)
    print_hz = [100.0, 300.0, 1000.0, 3000.0]
    with Result(s6_h5) as res:
        tf = res.field(pts, freqs_line, frame="world")
        mag_db = 20.0 * np.log10(np.maximum(np.abs(tf), 1e-30))
        print("\n(c) S6 |p/v| (dB re 1 Pa/(m/s))")
        table: list[dict[str, Any]] = []
        for f_pr in print_hz:
            fi = int(np.argmin(np.abs(freqs_line - f_pr)))
            row = {"f_hz": f_pr}
            for qi, lab in enumerate(labels):
                val = float(mag_db[fi, qi])
                print(f"  {f_pr:.0f} Hz  {lab:<14}  {val:7.2f} dB")
                row[lab] = val
            table.append(row)
    return {"s6_table_db": table, "freqs_line": freqs_line.tolist(), "mag_db": mag_db.tolist()}


def main() -> int:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    cfg_map = {
        "s3p": _SHOWCASE / "s3p_c09_parametric.json",
        "s4": _SHOWCASE / "s4_tilted.json",
        "s6": _SHOWCASE / "s6_downfire.json",
    }
    h5 = {k: _ensure_h5(k, p) for k, p in cfg_map.items()}
    payload: dict[str, Any] = {"elapsed_s": None, "h5": {k: str(v) for k, v in h5.items()}}
    payload.update(_c18_preview(h5["s3p"], h5["s4"]))
    payload["frame_sanity"] = _frame_sanity(h5["s4"])
    payload["s6"] = _s6_gap_table(h5["s6"])
    png_main = _OUT / "frames.png"
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0])
    theta = np.deg2rad(np.arange(0.0, 181.0, 5.0))
    with Result(h5["s3p"]) as r_s3, Result(h5["s4"]) as r_s4:
        for idx, (f_hz, phi) in enumerate(((1000.0, 0.0), (3000.0, 0.0), (1000.0, 0.5 * math.pi), (3000.0, 0.5 * math.pi))):
            ax = fig.add_subplot(gs[idx // 2, idx % 2], projection="polar")
            d_s3 = r_s3.directivity(theta, phi, f_hz, frame="world", normalize=True)
            d_s4 = r_s4.directivity(theta, phi, f_hz, frame="local", normalize=True)
            ax.plot(theta, 20 * np.log10(np.maximum(np.abs(d_s3[0]), 1e-30)), "-", label="s3p")
            ax.plot(theta, 20 * np.log10(np.maximum(np.abs(d_s4[0]), 1e-30)), "--", label="S4")
            ax.set_title(f"f={f_hz:.0f} Hz  phi={np.rad2deg(phi):.0f}°", fontsize=9)
    ax_s6 = fig.add_subplot(gs[2, :])
    freqs_line = np.asarray(payload["s6"]["freqs_line"], dtype=_F64)
    mag_db = np.asarray(payload["s6"]["mag_db"], dtype=_F64)
    for qi, lab in enumerate(("gap", "side", "top")):
        ax_s6.semilogx(freqs_line, mag_db[:, qi], label=lab)
    ax_s6.set_xlabel("f (Hz)")
    ax_s6.set_ylabel("|p/v| dB")
    ax_s6.legend()
    ax_s6.set_title("S6 gap probe + references")
    fig.tight_layout()
    fig.savefig(png_main, dpi=140)
    plt.close(fig)

    payload["elapsed_s"] = time.perf_counter() - t0
    (_OUT / "frames.json").write_text(json.dumps(vc.to_json_safe(payload), indent=2), encoding="utf-8")
    print(f"\nwrote {_OUT / 'frames.json'} and frames.png  ({payload['elapsed_s']:.1f} s wall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())