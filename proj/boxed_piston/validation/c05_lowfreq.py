#!/usr/bin/env python3
"""C5-lowfreq: small-box monopole limit of Z_rad (Stage C)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
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
    derive,
    estimate_memory_bytes,
)
from fdtd.grid import Grid  # noqa: E402
from fdtd.post.zrad import zrad_from_h5  # noqa: E402
from fdtd.run import run  # noqa: E402
from fdtd.source import BoxBoundary  # noqa: E402
import validation._common as vc  # noqa: E402

_GATE_KL = 0.2
_REPORT_KL = 1.0
_G1_TOL = 0.03
_G2_DELTA_A = (0.55, 0.90)
_G3_TOL = 0.02
_C1_UPDATES_PER_S = 375.0e6


def _geometry(cfg: Config, d: object) -> tuple[float, float, float, float]:
    grid = Grid(cfg, d)  # type: ignore[arg-type]
    bound = BoxBoundary.from_grid(grid, d)  # type: ignore[arg-type]
    S = float(bound.s_grid)
    a = math.sqrt(S / math.pi)
    L = max(float(d.realized_W), float(d.realized_H), float(d.realized_D))  # type: ignore[attr-defined]
    return S, a, L, float(d.h)  # type: ignore[attr-defined]


def _masks(
    freqs: np.ndarray, *, c: float, L: float, f_min: float, f_max: float
) -> tuple[np.ndarray, np.ndarray]:
    k = 2.0 * np.pi * freqs / c
    kl = k * L
    base = (freqs >= f_min) & (freqs <= f_max) & np.isfinite(freqs)
    return base & (kl <= _GATE_KL), base & (kl <= _REPORT_KL)


def _fit_coeffs(freqs: np.ndarray, z: np.ndarray, mask: np.ndarray, *, c: float) -> dict[str, float]:
    k = 2.0 * np.pi * freqs[mask] / c
    z_g = z[mask]
    X_re = np.column_stack([k**2, k**4])
    A, AB = np.linalg.lstsq(X_re, np.real(z_g), rcond=None)[0]
    B = AB / A if abs(A) > 1e-30 else 0.0
    X_im = np.column_stack([k, k**3])
    delta, dC = np.linalg.lstsq(X_im, np.imag(z_g), rcond=None)[0]
    C = dC / delta if abs(delta) > 1e-30 else 0.0
    return {"A": float(A), "B": float(B), "delta": float(delta), "C": float(C)}


def _p_avg_tail_db(path: Path) -> float:
    import h5py

    with h5py.File(path, "r") as f:
        p = np.asarray(f["radiator/p_avg"][()], dtype=np.float64)
    if p.size == 0:
        return float("nan")
    pk = float(np.max(np.abs(p)))
    if pk <= 0.0:
        return float("-inf")
    n_tail = max(1, int(math.floor(0.05 * p.size)))
    tail = float(np.max(np.abs(p[-n_tail:])))
    return 20.0 * math.log10(tail / pk)


def _wall_est(d: object, wall_measured: float | None = None) -> float:
    if wall_measured is not None:
        return wall_measured
    return float(d.n_steps * d.total_cells / _C1_UPDATES_PER_S)  # type: ignore[attr-defined]


def _run_case(
    cfg: Config,
    out_path: Path,
    *,
    ram_limit: float,
    run_timer: vc.RunTimer,
    label: str,
) -> dict[str, Any]:
    d = derive(cfg)
    est = estimate_memory_bytes(d.nx, d.ny, d.nz, d.cpml_cells)
    if est > ram_limit:
        raise vc.RamLimitExceeded(label, d.nx, float(est), float(ram_limit))
    run_timer.banner(label, f"grid {d.nx}×{d.ny}×{d.nz}  steps≈{d.n_steps:,}")
    run_timer.begin(label)
    run(cfg, out_path=out_path, progress=True, ram_limit_bytes=ram_limit)
    run_timer.end(label)
    return {"derived": d, "path": out_path}


def _analyze_h5(
    path: Path,
    cfg: Config,
    *,
    time_shift_s: float = 0.0,
) -> dict[str, Any]:
    d = derive(cfg)
    c = cfg.air.c
    f_min = cfg.band.f_min
    f_max = min(float(d.f_max), cfg.band.f_max or d.f_max)
    freqs, z = zrad_from_h5(path, df_hz=1.0, time_shift_s=time_shift_s)
    S, a, L, h = _geometry(cfg, d)
    gate_m, report_m = _masks(freqs, c=c, L=L, f_min=f_min, f_max=f_max)
    valid = gate_m & np.isfinite(z.real) & np.isfinite(z.imag)
    if not np.any(valid):
        raise RuntimeError(f"{path}: no samples in gate band")
    fit = _fit_coeffs(freqs, z, valid, c=c)
    k = 2.0 * np.pi * freqs / c
    ka = k * a
    return {
        "freqs": freqs,
        "z": z,
        "ka": ka,
        "gate_mask": gate_m,
        "report_mask": report_m,
        "S": S,
        "a": a,
        "L": L,
        "h": h,
        "fit": fit,
        "dt": float(d.dt),
    }


def _g1_signed(fit: dict[str, float], S: float) -> float:
    ref = S / (4.0 * math.pi)
    return float(fit["A"] / ref - 1.0)


def _gate_g1(fit: dict[str, float], S: float) -> tuple[bool, float]:
    signed = _g1_signed(fit, S)
    return abs(signed) <= _G1_TOL, signed


def _fmt_sci3(x: float) -> str:
    return f"{x:.3e}"


def _scan_g1_zero_shift(
    path: Path,
    cfg: Config,
    *,
    dt: float,
) -> dict[str, Any]:
    """Scan pressure time shift; return signed G1 curve and Δt that nulls G1."""
    span = 1.5
    step = 0.05
    units = np.arange(-span, span + 0.5 * step, step, dtype=np.float64)
    signed: list[float] = []
    for u in units:
        an = _analyze_h5(path, cfg, time_shift_s=float(u) * dt)
        signed.append(_g1_signed(an["fit"], an["S"]))
    signed_arr = np.asarray(signed, dtype=np.float64)
    zero_dt: float | None = None
    for i in range(len(units) - 1):
        a, b = signed_arr[i], signed_arr[i + 1]
        if a == 0.0:
            zero_dt = float(units[i])
            break
        if a * b < 0.0 or b == 0.0:
            t = abs(a) / (abs(a) + abs(b)) if (a * b < 0.0) else 1.0
            zero_dt = float(units[i] + t * (units[i + 1] - units[i]))
            break
    return {
        "shift_units": units.tolist(),
        "g1_signed": signed_arr.tolist(),
        "g1_zero_shift_dt_units": zero_dt,
    }


def _gate_g2(
    freqs: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    delta: float,
    a: float,
) -> tuple[bool, float]:
    im_ok = bool(np.all(np.imag(z[mask]) > 0.0))
    ratio = delta / a
    band_ok = _G2_DELTA_A[0] <= ratio <= _G2_DELTA_A[1]
    return im_ok and band_ok, ratio


def _gate_g3(z_ref: np.ndarray, z_alt: np.ndarray, mask: np.ndarray) -> tuple[bool, float]:
    num = np.abs(z_alt[mask] - z_ref[mask])
    den = np.maximum(np.abs(z_ref[mask]), 1e-30)
    err = float(np.max(num / den))
    return err <= _G3_TOL, err


def maybe_plot(
    analysis: dict[str, Any],
    path: Path | None,
    *,
    S: float,
    a: float,
) -> None:
    if path is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping PNG", file=sys.stderr)
        return
    ka = analysis["ka"]
    z = analysis["z"]
    m = analysis["report_mask"] & np.isfinite(z.real)
    ka_p = ka[m]
    re_z = np.real(z[m])
    im_z = np.imag(z[m])
    ka_line = np.linspace(0.0, max(float(np.max(ka_p)), 0.5), 200)
    ref_quad = (ka_line**2) / 2.0
    ref_mono = (ka_line**2) / 4.0
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), layout="tight")
    ax1.plot(ka_p, re_z, "b.", ms=3, label="Re z")
    ax1.plot(ka_line, ref_quad, "k--", lw=1, label=r"$(ka)^2/2$")
    ax1.plot(ka_line, ref_mono, "k:", lw=1, label=r"$k^2 S/(4\pi)$")
    ax1.set_ylabel("Re z")
    ax1.legend(loc="best", fontsize=8)
    ax1.axvline(0.61, color="gray", ls="--", lw=0.8, label=None)
    ax1.axvline(0.85, color="gray", ls=":", lw=0.8)
    ax2.plot(ka_p, im_z, "r.", ms=3, label="Im z")
    ax2.set_xlabel(r"$ka$")
    ax2.set_ylabel("Im z")
    ax2.legend(loc="best", fontsize=8)
    ax2.axvline(0.61, color="gray", ls="--", lw=0.8)
    ax2.axvline(0.85, color="gray", ls=":", lw=0.8)
    fig.suptitle(f"C5-lowfreq  S={S:.4g} m²  a={a:.4g} m")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_report(
    payload: dict[str, Any],
    *,
    gate_pass: bool,
) -> None:
    print("\n=== C5-lowfreq report ===", flush=True)
    g = payload["gates"]
    print(
        f"  G1 (A/(S/4π)−1): {_fmt_sci3(g['g1_signed'])}  "
        f"{'PASS' if g['g1_pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"  G2 Im z>0 & δ/a={g['delta_over_a']:.4f} "
        f"({'PASS' if g['g2_pass'] else 'FAIL'})",
        flush=True,
    )
    print(
        f"  G3 gap truncation: {_fmt_sci3(g['g3_rel'])}  "
        f"{'PASS' if g['g3_pass'] else 'FAIL'}",
        flush=True,
    )
    print(f"  p_avg tail: {payload['tail_db']:.1f} dB rel peak (last 5%)", flush=True)
    scan = payload.get("g1_shift_scan", {})
    z0 = scan.get("g1_zero_shift_dt_units")
    if z0 is not None:
        print(f"  G1 zero crossing: {z0:+.3f} Δt", flush=True)
    ts = payload.get("time_shift_g1", {})
    if ts:
        print(
            f"  G1 @ shift +0.5Δt: {_fmt_sci3(ts['plus']['signed'])}",
            flush=True,
        )
        print(
            f"  G1 @ shift −0.5Δt: {_fmt_sci3(ts['minus']['signed'])}",
            flush=True,
        )
    print(f"  Overall: {'PASS' if gate_pass else 'FAIL'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C5-lowfreq validation")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "c05_lowfreq.json",
        help="Base JSON config",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "validation" / "results",
        help="Results directory",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--ram-gb",
        type=float,
        default=DEFAULT_RAM_LIMIT_BYTES / 1e9,
        help="RAM limit (GB)",
    )
    args = parser.parse_args(argv)
    ram_limit = args.ram_gb * 1e9
    run_timer = vc.RunTimer()
    run_timer.banner("start", args.config.name)

    cfg_text = args.config.read_text(encoding="utf-8")
    cfg_base = Config.from_json(cfg_text)
    cfg_gap2 = Config.from_json(cfg_text)
    gap0 = cfg_base.numerics.gap_cells
    if gap0 is None:
        gap0 = 10
    cfg_gap2.numerics.gap_cells = 2 * gap0

    d_plan = derive(cfg_base)
    est = estimate_memory_bytes(d_plan.nx, d_plan.ny, d_plan.nz, d_plan.cpml_cells)
    if est > ram_limit:
        raise vc.RamLimitExceeded(
            "c05 baseline plan", d_plan.nx, float(est), float(ram_limit)
        )
    print(
        f"  baseline grid {d_plan.nx}×{d_plan.ny}×{d_plan.nz}  "
        f"mem≈{vc.fmt_bytes(est)}  steps={d_plan.n_steps}",
        flush=True,
    )
    d_plan2 = derive(cfg_gap2)
    est2 = estimate_memory_bytes(d_plan2.nx, d_plan2.ny, d_plan2.nz, d_plan2.cpml_cells)
    print(
        f"  gap×2 grid {d_plan2.nx}×{d_plan2.ny}×{d_plan2.nz}  "
        f"mem≈{vc.fmt_bytes(est2)}  steps={d_plan2.n_steps}",
        flush=True,
    )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with tempfile.TemporaryDirectory(prefix="c05_") as tmp:
        tmp_p = Path(tmp)
        h5_base = tmp_p / "baseline.h5"
        h5_gap2 = tmp_p / "gap2.h5"
        _run_case(cfg_base, h5_base, ram_limit=ram_limit, run_timer=run_timer, label="run_base")
        _run_case(cfg_gap2, h5_gap2, ram_limit=ram_limit, run_timer=run_timer, label="run_gap2")

        an_base = _analyze_h5(h5_base, cfg_base)
        an_gap2 = _analyze_h5(h5_gap2, cfg_gap2)

        g1_pass, g1_signed = _gate_g1(an_base["fit"], an_base["S"])
        g2_pass, delta_a = _gate_g2(
            an_base["freqs"],
            an_base["z"],
            an_base["gate_mask"] & np.isfinite(an_base["z"].real),
            an_base["fit"]["delta"],
            an_base["a"],
        )
        g3_pass, g3_rel = _gate_g3(
            an_base["z"],
            an_gap2["z"],
            an_base["gate_mask"] & np.isfinite(an_base["z"].real),
        )
        gate_pass = g1_pass and g2_pass and g3_pass

        tail_db = _p_avg_tail_db(h5_base)
        dt = an_base["dt"]
        an_shift_p = _analyze_h5(h5_base, cfg_base, time_shift_s=+0.5 * dt)
        an_shift_m = _analyze_h5(h5_base, cfg_base, time_shift_s=-0.5 * dt)
        _, g1_p_signed = _gate_g1(an_shift_p["fit"], an_base["S"])
        _, g1_m_signed = _gate_g1(an_shift_m["fit"], an_base["S"])
        g1_shift_scan = _scan_g1_zero_shift(h5_base, cfg_base, dt=dt)

        freqs = an_base["freqs"]
        z = an_base["z"]
        rep = an_base["report_mask"] & np.isfinite(z.real)
        k = 2.0 * np.pi * freqs[rep] / cfg_base.air.c
        mono_ratio = np.real(z[rep]) / (k**2 * an_base["S"] / (4.0 * math.pi))

        png_path = None if args.no_plot else out_dir / f"c05_lowfreq_{ts}.png"
        maybe_plot(an_base, png_path, S=an_base["S"], a=an_base["a"])

    payload: dict[str, Any] = {
        "check": "c05_lowfreq",
        "timestamp_utc": ts,
        "config": str(args.config),
        "grid_baseline": [d_plan.nx, d_plan.ny, d_plan.nz],
        "grid_gap2": [d_plan2.nx, d_plan2.ny, d_plan2.nz],
        "n_steps": d_plan.n_steps,
        "n_transits": cfg_base.numerics.n_transits,
        "S_grid_m2": an_base["S"],
        "a_m": an_base["a"],
        "L_m": an_base["L"],
        "fit": an_base["fit"],
        "tail_db": tail_db,
        "delta_over_a": delta_a,
        "mono_re_ratio_report_band": {
            "freqs_hz": freqs[rep].tolist(),
            "ratio": mono_ratio.tolist(),
        },
        "gates": {
            "g1_pass": g1_pass,
            "g1_signed": g1_signed,
            "g2_pass": g2_pass,
            "delta_over_a": delta_a,
            "g3_pass": g3_pass,
            "g3_rel": g3_rel,
            "gate_pass": gate_pass,
        },
        "g1_shift_scan": g1_shift_scan,
        "time_shift_g1": {
            "plus": {"shift_dt_units": 0.5, "signed": g1_p_signed},
            "minus": {"shift_dt_units": -0.5, "signed": g1_m_signed},
        },
        "wall_s_est_baseline": _wall_est(d_plan),
        "wall_s_est_gap2": _wall_est(d_plan2),
        "phases_s": run_timer.phases,
    }
    json_path = out_dir / f"c05_lowfreq_{ts}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(vc.to_json_safe(payload), f, indent=2)
    run_timer.banner("output", json_path.name)

    print_report(payload, gate_pass=gate_pass)
    vc.print_runtime_summary(run_timer, ("run_base", "run_gap2", "output"))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())