#!/usr/bin/env python3
"""Step 10a gate C18(i): h-refinement between s3p and S4."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_OUT = _PKG / "outputs" / "step10a"
_SHOWCASE = _PKG / "configs" / "showcase"

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import DEFAULT_GAP_CELLS, Config  # noqa: E402
from boxrad.geomrun import dry_run_geom  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402
from validation._common import to_json_safe  # noqa: E402

_F64 = np.float64
_C128 = np.complex128
_H_LEVELS = (0.005, 0.005 / 1.5, 0.0025)  # 5 mm, 5/1.5 mm, 2.5 mm
_GATE_Z = 0.01
_GATE_FF_DB = 0.2
_FAR_N = 12
_DIR_FREQS = (500.0, 1000.0, 2000.0, 3000.0)
_DIR_PHI = (0.0, 0.5 * math.pi)
_THETA_DEG = np.arange(0.0, 91.0, 5.0)


def _load_json(path: Path) -> Config:
    return Config.from_json(path.read_text(encoding="utf-8"))


def _h_cfg(base: Config, h: float, *, gap_cells: int | None = None) -> Config:
    raw = json.loads(base.to_json())
    raw["numerics"]["h_override"] = h
    if gap_cells is not None:
        raw["numerics"]["gap_cells"] = gap_cells
    return Config.from_dict(raw)


def _run_meta(cfg: Config) -> dict[str, float | int]:
    est = dry_run_geom(cfg)
    gap_cells = cfg.numerics.gap_cells if cfg.numerics.gap_cells is not None else DEFAULT_GAP_CELLS
    h_m = float(est["h"])
    return {
        "gap_cells": gap_cells,
        "gap_m": gap_cells * h_m,
        "record_length_s": float(est["n_steps"]) * float(est["dt"]),
    }


def _ensure_run(name: str, cfg: Config, out: Path) -> Path:
    if out.is_file():
        print(f"reuse {out}")
        return out
    est = dry_run_geom(cfg)
    ncell = est.get("grid_shape")
    if ncell:
        ncell = int(np.prod(ncell))
    print(
        f"{name}: cells≈{ncell}  n_steps={est.get('n_steps')}  "
        f"h={est.get('h', float('nan')) * 1e3:.4g} mm  "
        f"wall_est={est.get('wall_time_s_est', float('nan')):.3g} s"
    )
    with keep_awake(f"step10a C18 {name}"):
        run(cfg, out_path=out, progress=True)
    return out


def _z_metrics(s3p: Path, s4: Path) -> dict[str, float]:
    with Result(s3p) as r0, Result(s4) as r1:
        f, z0 = r0.z()
        _, z1 = r1.z()
        z0 = np.asarray(z0, dtype=_C128)
        z1 = np.asarray(z1, dtype=_C128)
    re_ratio = float(np.max(np.abs(z1.real - z0.real))) / max(float(np.max(np.abs(z0.real))), 1e-30)
    im_ratio = float(np.max(np.abs(z1.imag - z0.imag))) / max(float(np.max(np.abs(z0.imag))), 1e-30)
    return {"ez_re": re_ratio, "ez_im": im_ratio, "freqs_hz": f}


def _on_axis_ff_db(s3p: Path, s4: Path, f_min: float, f_max: float) -> float:
    freqs = np.geomspace(f_min, f_max, _FAR_N)
    with Result(s3p) as r0, Result(s4) as r1:
        db_max = 0.0
        for f_hz in freqs:
            d0 = r0.directivity(0.0, 0.0, float(f_hz), frame="local", normalize=False)
            d1 = r1.directivity(0.0, 0.0, float(f_hz), frame="local", normalize=False)
            ratio = complex(d1.flat[0]) / complex(d0.flat[0])
            db = abs(20.0 * math.log10(max(abs(ratio), 1e-30)))
            db_max = max(db_max, db)
    return float(db_max)


def _directivity_db(s3p: Path, s4: Path) -> float:
    theta = np.deg2rad(_THETA_DEG)
    db_worst = 0.0
    with Result(s3p) as r0, Result(s4) as r1:
        for f_hz in _DIR_FREQS:
            for phi in _DIR_PHI:
                d0 = r0.directivity(theta, phi, f_hz, frame="world", normalize=True)
                d1 = r1.directivity(theta, phi, f_hz, frame="local", normalize=True)
                ref = float(np.max(np.abs(d0)))
                mask = 20.0 * np.log10(np.maximum(np.abs(d0) / max(ref, 1e-30), 1e-30)) > -20.0
                if not np.any(mask):
                    continue
                ratio = np.where(mask, d1 / np.where(mask, d0, 1.0), 1.0)
                db = float(np.max(np.abs(20.0 * np.log10(np.maximum(np.abs(ratio), 1e-30)))))
                db_worst = max(db_worst, db)
    return db_worst


def _observed_order(e_coarse: float, e_fine: float, h_coarse: float, h_fine: float) -> float:
    if e_coarse <= 0 or e_fine <= 0 or h_coarse <= h_fine:
        return float("nan")
    return math.log(e_coarse / e_fine) / math.log(h_coarse / h_fine)


def _richardson_limit(e_mid: float, e_fine: float, p: float, *, r: float) -> float:
    p_c = min(2.0, max(1.0, p))
    denom = r**p_c - 1.0
    if denom <= 0:
        return e_fine
    return e_fine + (e_fine - e_mid) / denom


def _run_level(
    label: str,
    base_name: str,
    cfg_base: Config,
    h: float,
    idx: int,
    *,
    fixed_gap_m: float | None,
) -> dict[str, Any]:
    gap_cells: int | None = None
    if fixed_gap_m is not None:
        gap_cells = round(fixed_gap_m / h)
    cfg = _h_cfg(cfg_base, h, gap_cells=gap_cells)
    meta = _run_meta(cfg)
    reuse_default_h1 = (
        fixed_gap_m is not None
        and idx == 1
        and meta["gap_cells"] == DEFAULT_GAP_CELLS
    )
    suffix = ""
    if fixed_gap_m is not None and not reuse_default_h1:
        suffix = "_g"
    out = _OUT / f"{base_name}_h{idx}{suffix}.h5"
    _ensure_run(f"{label} h{idx}", cfg, out)
    return {"h5": out, "h_m": h, "cfg": cfg, "meta": meta, "reused_h1_default": reuse_default_h1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-gap-m",
        type=float,
        default=None,
        metavar="X",
        help="Fix physical air gap (m): set numerics.gap_cells=round(X/h) per level",
    )
    args = parser.parse_args(argv)
    fixed_gap_m = args.fixed_gap_m

    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    stem = "c18_fixedgap" if fixed_gap_m is not None else "c18"
    cfg_s3p = _load_json(_SHOWCASE / "s3p_c09_parametric.json")
    cfg_s4 = _load_json(_SHOWCASE / "s4_tilted.json")

    print("C18(i) dry-run summary (all six levels):")
    for i, h in enumerate(_H_LEVELS, start=1):
        for nm, cfg in (("s3p", cfg_s3p), ("s4", cfg_s4)):
            gap_cells = round(fixed_gap_m / h) if fixed_gap_m is not None else None
            est = dry_run_geom(_h_cfg(cfg, h, gap_cells=gap_cells))
            ncell = int(np.prod(est["grid_shape"]))
            print(
                f"  {nm} h{i}: cells={ncell}  h={h * 1e3:.4g} mm  "
                f"n_steps={est['n_steps']}  wall_est={est['wall_time_s_est']:.3g} s"
            )

    s3p_runs = [
        _run_level("s3p", "s3p", cfg_s3p, h, i, fixed_gap_m=fixed_gap_m)
        for i, h in enumerate(_H_LEVELS, 1)
    ]
    s4_runs = [
        _run_level("S4", "s4", cfg_s4, h, i, fixed_gap_m=fixed_gap_m)
        for i, h in enumerate(_H_LEVELS, 1)
    ]
    if fixed_gap_m is not None:
        reused = [str(r["h5"]) for r in (*s3p_runs, *s4_runs) if r.get("reused_h1_default")]
        if reused:
            print("fixed-gap: reused default h1 HDF5:", ", ".join(reused))

    metrics: dict[str, Any] = {"h_m": list(_H_LEVELS)}
    per_level: list[dict[str, Any]] = []
    for i in range(3):
        ff = _on_axis_ff_db(
            s3p_runs[i]["h5"],
            s4_runs[i]["h5"],
            cfg_s3p.band.f_min,
            cfg_s3p.band.f_max,
        )
        di = _directivity_db(s3p_runs[i]["h5"], s4_runs[i]["h5"])
        row = {"level": i + 1, "h_m": _H_LEVELS[i], **_z_metrics(s3p_runs[i]["h5"], s4_runs[i]["h5"])}
        row["farfield_on_axis_db"] = ff
        row["directivity_max_db"] = di
        row.pop("freqs_hz", None)
        row["s3p"] = s3p_runs[i]["meta"]
        row["s4"] = s4_runs[i]["meta"]
        per_level.append(row)

    orders: dict[str, list[float]] = {
        "ez_re": [],
        "ez_im": [],
        "farfield_on_axis_db": [],
        "directivity_max_db": [],
    }
    for key in orders:
        for j in range(2):
            e0 = per_level[j][key]
            e1 = per_level[j + 1][key]
            orders[key].append(_observed_order(e0, e1, _H_LEVELS[j], _H_LEVELS[j + 1]))

    rich: dict[str, Any] = {}
    fine = per_level[2]
    mid = per_level[1]
    r_fine = _H_LEVELS[1] / _H_LEVELS[2]
    for key in orders:
        p = orders[key][1]
        rich[f"{key}_h0_limit"] = _richardson_limit(mid[key], fine[key], p, r=r_fine)
        rich[f"{key}_p_21"] = p
        rich[f"{key}_h_ratio_21"] = r_fine

    gate = {
        "z_re_fine": fine["ez_re"] <= _GATE_Z,
        "z_im_fine": fine["ez_im"] <= _GATE_Z,
        "farfield_fine_db": fine["farfield_on_axis_db"] <= _GATE_FF_DB,
        "pass": bool(
            fine["ez_re"] <= _GATE_Z
            and fine["ez_im"] <= _GATE_Z
            and fine["farfield_on_axis_db"] <= _GATE_FF_DB
        ),
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    ax = axes[0]
    hs = np.asarray(_H_LEVELS, dtype=_F64) * 1e3
    for key, lbl in (
        ("ez_re", "Re z"),
        ("ez_im", "Im z"),
        ("farfield_on_axis_db", "on-axis FF"),
        ("directivity_max_db", "directivity"),
    ):
        vals = np.array([row[key] for row in per_level], dtype=_F64)
        ax.loglog(hs, vals, "o-", label=lbl)
    h0 = hs[0]
    v0 = per_level[0]["ez_re"]
    ax.loglog(hs, v0 * (hs / h0) ** 1.0, ":", color="gray", alpha=0.5, label="slope 1 (e.g.)")
    ax.loglog(hs, v0 * (hs / h0) ** 2.0, ":", color="0.5", alpha=0.5, label="slope 2 (e.g.)")
    ax.set_xlabel("h (mm)")
    ax.set_ylabel("metric")
    ax.legend(fontsize=8)
    ax.set_title("C18(i) metrics vs h")

    ax = axes[1]
    styles = ("-", "--", ":")
    for i, h in enumerate(_H_LEVELS):
        for run, lab, off in (
            (s3p_runs[i]["h5"], "s3p", 0),
            (s4_runs[i]["h5"], "S4", 2),
        ):
            with Result(run) as res:
                f, z = res.z()
            ls = styles[i]
            ax.semilogx(
                f,
                z.real,
                color=f"C{off}",
                ls=ls,
                lw=1.0,
                label=f"{lab} Re h={h * 1e3:.3g} mm",
            )
            ax.semilogx(f, z.imag, color=f"C{off + 1}", ls=ls, lw=0.9, alpha=0.85)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("z")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Re/Im z at three h (solid=Re color, paired Im)")

    fig.tight_layout()
    fig.savefig(_OUT / f"{stem}.png", dpi=140)

    z_curves = {}
    for i, runs in enumerate((s3p_runs, s4_runs)):
        tag = "s3p" if i == 0 else "s4"
        z_curves[tag] = []
        for run in runs:
            with Result(run["h5"]) as res:
                f, z = res.z()
                z_curves[tag].append(
                    {
                        "h_m": run["h_m"],
                        "freqs_hz": f.tolist(),
                        "z_real": z.real.tolist(),
                        "z_imag": z.imag.tolist(),
                        **run["meta"],
                    }
                )

    payload = to_json_safe(
        {
            "per_level": per_level,
            "orders": orders,
            "richardson_h0": rich,
            "gates": gate,
            "z_spectra": z_curves,
            "h5": {
                "s3p": [str(r["h5"]) for r in s3p_runs],
                "s4": [str(r["h5"]) for r in s4_runs],
            },
            "elapsed_s": time.perf_counter() - t0,
        }
    )
    json_path = _OUT / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {json_path} and {stem}.png  ({payload['elapsed_s']:.1f} s)")
    print(
        f"C18 gate (finest h): Re z={fine['ez_re']:.4g}  Im z={fine['ez_im']:.4g}  "
        f"FF={fine['farfield_on_axis_db']:.4g} dB  pass={gate['pass']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())