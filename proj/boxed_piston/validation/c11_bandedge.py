#!/usr/bin/env python3
"""C11-bandedge: radiator vs Huygens vs far-field power at band-edge frequencies."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boxrad.config import (  # noqa: E402
    Config,
    DEFAULT_RAM_LIMIT_BYTES,
    derive,
    estimate_memory_bytes,
)
from boxrad.post.kirchhoff import dpdn_from_vn, kh_pressure  # noqa: E402
from boxrad.post.transfer import load_huygens_spectra  # noqa: E402
from boxrad.post.zrad import zrad_at_frequencies  # noqa: E402
from boxrad.recorder import resolve_decimation_factor  # noqa: E402
from boxrad.result import Result  # noqa: E402
from boxrad.runner import estimate_output_bytes, run  # noqa: E402
from boxrad.source import pulse_spectrum  # noqa: E402
import validation._common as vc  # noqa: E402

_F_HZ = (600.0, 700.0, 750.0, 800.0)
_R_SPHERE_M = 30.0
_N_THETA = 32
_N_PHI = 64
_G_FAR_TOL = 0.002  # |P_far/P_face − 1| ≤ 0.2 %
_G_HUYG_TOL = 0.002  # |P_huyg/P_face − 1| ≤ 0.2 %
_C1_UPDATES_PER_S = 375.0e6

_VARIANTS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("V0", "base", {}),
    ("V1", "decimation m/2", {"decimation_key": "m_half"}),
    ("V2", "decimation m/4", {"decimation_key": "m_quarter"}),
    ("V3", "decimation 1 (full rate)", {"decimation_override": 1}),
    ("V4", "band 200–1000 Hz", {"band_f_max": 1000.0}),
    ("V5", "pulse_f_factor 1.5", {"pulse_f_factor": 1.5}),
    ("V6", "pulse_db −60", {"pulse_db": -60.0}),
    ("V7", "n_transits 6", {"n_transits": 6.0}),
    ("V8", "cpml_cells 32", {"cpml_cells": 32}),
)


def _sphere_quadrature(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    u_nodes, w_u = leggauss(int(n_theta))
    phi = (np.arange(int(n_phi), dtype=np.float64) + 0.5) * (2.0 * math.pi / float(n_phi))
    w_phi = 2.0 * math.pi / float(n_phi)
    u_grid, phi_grid = np.meshgrid(u_nodes, phi, indexing="ij")
    w_u_grid, _ = np.meshgrid(w_u, phi, indexing="ij")
    sin_theta = np.sqrt(np.maximum(1.0 - u_grid * u_grid, 0.0))
    x = sin_theta * np.cos(phi_grid)
    y = sin_theta * np.sin(phi_grid)
    z = u_grid
    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    weights = (w_u_grid * w_phi).reshape(-1)
    w_sum = float(np.sum(weights))
    if abs(w_sum - 4.0 * math.pi) > 1e-6 * max(1.0, 4.0 * math.pi):
        weights *= (4.0 * math.pi) / w_sum
    return dirs.astype(np.float64, copy=False), weights.astype(np.float64, copy=False)


def _huygens_tail_db(h5_path: Path) -> float:
    import h5py

    with h5py.File(h5_path, "r") as f:
        p = np.asarray(f["huygens/p_face"][()], dtype=np.float64)
    if p.size == 0:
        return float("nan")
    pk = float(np.max(np.abs(p)))
    if pk <= 0.0:
        return float("-inf")
    n_tail = max(1, int(math.floor(0.05 * p.shape[0])))
    tail = float(np.max(np.abs(p[-n_tail:, :])))
    return 20.0 * math.log10(tail / pk)


def _p_face_power(z_ac: complex, s_grid: float) -> float:
    u_mag_sq = float(s_grid) ** 2
    return 0.5 * float(np.real(z_ac)) * u_mag_sq


def _p_huygens(
    p_face: np.ndarray,
    vn_face: np.ndarray,
    areas: np.ndarray,
    v: complex,
) -> float:
    scale = abs(v)
    if scale < 1e-30:
        return float("nan")
    p_tf = p_face / v
    vn_tf = vn_face / v
    return float(0.5 * np.real(np.sum(p_tf * np.conj(vn_tf) * areas)))


def _p_far_sphere(
    h5_path: Path,
    cfg: Config,
    d: object,
    f_hz: float,
    *,
    s_grid: float,
) -> float:
    rho = float(cfg.air.rho)
    c = float(cfg.air.c)
    patches, v_ref, p_face, vn_face = load_huygens_spectra(h5_path, d, f_hz)
    areas = np.asarray(patches.areas, dtype=np.float64)
    dirs, w_omega = _sphere_quadrature(_N_THETA, _N_PHI)
    pts = _R_SPHERE_M * dirs
    omega = np.asarray([2.0 * math.pi * f_hz], dtype=np.float64)
    k_sel = np.asarray([2.0 * math.pi * f_hz / c], dtype=np.float64)
    dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
    p_row = p_face[np.newaxis, :]
    with vc.timed_section(f"far sphere f={f_hz:.0f} Hz"):
        p_obs = kh_pressure(
            pts,
            patches,
            p_row,
            dpdn,
            k_sel,
            image_plane_z=None,
            near_subdivide=False,
        )[0]
    h = p_obs / v_ref
    dens = (np.abs(h) ** 2) / (2.0 * rho * c)
    return float(np.sum(dens * (_R_SPHERE_M ** 2) * w_omega))


def _apply_variant(base: Config, spec: dict[str, Any], base_m: int) -> Config:
    cfg = copy.deepcopy(base)
    if "decimation_override" in spec:
        cfg.numerics.decimation_override = int(spec["decimation_override"])
    key = spec.get("decimation_key")
    if key == "m_half":
        cfg.numerics.decimation_override = max(1, base_m // 2)
    elif key == "m_quarter":
        cfg.numerics.decimation_override = max(1, base_m // 4)
    if "band_f_max" in spec:
        cfg.band.f_max = float(spec["band_f_max"])
    for attr in ("pulse_f_factor", "pulse_db", "n_transits", "cpml_cells"):
        if attr in spec:
            setattr(cfg.numerics, attr, spec[attr])
    return cfg


def _decim_diagnostics(
    cfg: Config,
    d: object,
    *,
    tail_face_db: float,
    tail_huyg_db: float,
) -> dict[str, Any]:
    m = resolve_decimation_factor(cfg, d)
    dt = float(d.dt)  # type: ignore[attr-defined]
    dt_dec = m * dt
    f_nyq = 0.5 / dt_dec if dt_dec > 0 else float("nan")
    sigma = float(d.pulse_sigma)  # type: ignore[attr-defined]
    t0 = float(d.pulse_t0)  # type: ignore[attr-defined]
    f_rows: list[dict[str, Any]] = []
    for f_hz in _F_HZ:
        f_a = 2.0 * f_nyq - f_hz
        s_f = pulse_spectrum(f_hz, sigma, t0)
        s_a = pulse_spectrum(f_a, sigma, t0)
        mag_f = abs(s_f)
        mag_a = abs(s_a)
        alias_db = (
            20.0 * math.log10(mag_a / mag_f)
            if mag_f > 0 and mag_a > 0
            else float("-inf")
        )
        f_rows.append(
            {
                "f_hz": f_hz,
                "f_alias_hz": f_a,
                "alias_spectrum_db": alias_db,
            }
        )
    return {
        "decimation_m": m,
        "dt_dec_s": dt_dec,
        "f_nyquist_decimated_hz": f_nyq,
        "f_over_f_nyquist_max": float(max(f / f_nyq for f in _F_HZ)) if f_nyq > 0 else None,
        "tail_face_db": tail_face_db,
        "tail_huygens_db": tail_huyg_db,
        "per_f": f_rows,
    }


def _metrics_at_f(
    h5_path: Path,
    cfg: Config,
    d: object,
    f_hz: float,
    *,
    s_grid: float,
) -> dict[str, float]:
    _, z_arr = zrad_at_frequencies(h5_path, np.asarray([f_hz], dtype=np.float64))
    z_ac = float(cfg.air.rho) * float(cfg.air.c) * complex(z_arr[0]) / s_grid
    p_face = _p_face_power(z_ac, s_grid)
    patches, v_ref, p_face_h, vn_face = load_huygens_spectra(h5_path, d, f_hz)
    areas = np.asarray(patches.areas, dtype=np.float64)
    p_huyg = _p_huygens(p_face_h, vn_face, areas, v_ref)
    p_far = _p_far_sphere(h5_path, cfg, d, f_hz, s_grid=s_grid)
    rh = p_huyg / max(p_face, 1e-30)
    rf = p_far / max(p_face, 1e-30)
    return {
        "f_hz": f_hz,
        "P_face_W": p_face,
        "P_huyg_W": p_huyg,
        "P_far_W": p_far,
        "P_huyg_over_face_pct": (rh - 1.0) * 100.0,
        "P_far_over_face_pct": (rf - 1.0) * 100.0,
        "Re_Z_acoustic": float(np.real(z_ac)),
        "Im_Z_acoustic": float(np.imag(z_ac)),
    }


def _v4_compare_v0(
    h5_v4: Path,
    cfg_v4: Config,
    ref_row: dict[str, Any],
) -> dict[str, Any]:
    with Result(h5_v4) as res:
        z_v4 = complex(res.Z_acoustic(800.0)[0])
        dirs, _ = _sphere_quadrature(_N_THETA, _N_PHI)
        pts = (_R_SPHERE_M * dirs).astype(np.float64)
        h_on = complex(res.field(np.array([[0.0, 0.0, _R_SPHERE_M]]), 800.0)[0, 0])
        h_ff = res.field(pts, 800.0)[0]
    z_ref = complex(ref_row["Re_Z_acoustic"] + 1j * ref_row["Im_Z_acoustic"])
    return {
        "f_hz": 800.0,
        "V0_Re_Z_acoustic": float(np.real(z_ref)),
        "V4_Re_Z_acoustic": float(np.real(z_v4)),
        "dRe_Z_pct": 100.0 * (np.real(z_v4) - np.real(z_ref)) / max(abs(np.real(z_ref)), 1e-30),
        "V0_field_on_axis_mag": abs(ref_row.get("field_on_axis_mag", float("nan"))),
        "V4_field_on_axis_mag": abs(h_on),
        "field_on_axis_rel_pct": 100.0
        * (abs(h_on) - ref_row.get("field_on_axis_mag", 0.0))
        / max(ref_row.get("field_on_axis_mag", 1e-30), 1e-30),
        "mean_far_field_mag_V4": float(np.mean(np.abs(h_ff))),
    }


def _gate_power_rows(rows: list[dict[str, Any]]) -> tuple[bool, bool, float, float]:
    max_far = max(abs(r["P_far_over_face_pct"]) / 100.0 for r in rows)
    max_huyg = max(abs(r["P_huyg_over_face_pct"]) / 100.0 for r in rows)
    return (
        max_far <= _G_FAR_TOL,
        max_huyg <= _G_HUYG_TOL,
        max_far,
        max_huyg,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C11-bandedge: decimation aliasing vs power balance")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "demo" / "bandedge_box.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "results")
    parser.add_argument("--ram-gb", type=float, default=DEFAULT_RAM_LIMIT_BYTES / 1e9)
    parser.add_argument(
        "--keep-h5",
        action="store_true",
        help="Keep FDTD HDF5 in a kept workdir under outputs/ (default: temporary)",
    )
    args = parser.parse_args(argv)
    with vc.keep_awake("c11_bandedge"):
        return _c11_body(args)


def _c11_body(args: argparse.Namespace) -> int:
    ram_limit = float(args.ram_gb) * 1e9
    timer = vc.RunTimer()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)

    base = Config.from_json(args.config.read_text(encoding="utf-8"))
    d_base = derive(base)
    base_m = int(d_base.decimation_m)

    variant_cfgs: list[tuple[str, str, Config]] = []
    for vid, label, spec in _VARIANTS:
        variant_cfgs.append((vid, label, _apply_variant(base, spec, base_m)))

    est_mem = max(estimate_memory_bytes(derive(c).nx, derive(c).ny, derive(c).nz, derive(c).cpml_cells) for _, _, c in variant_cfgs)
    print("\n=== C11-bandedge RAM check ===", flush=True)
    if est_mem > ram_limit:
        raise vc.RamLimitExceeded("c11_bandedge", derive(base).nx, float(est_mem), float(ram_limit))
    print(f"  peak grid mem ≈{vc.fmt_bytes(est_mem)}  variants={len(variant_cfgs)}", flush=True)

    planned = sum(estimate_output_bytes(c, derive(c)) for _, _, c in variant_cfgs)
    vc.require_h5_disk(
        planned,
        vc.workdir_disk_parent(keep=args.keep_h5),
        label="c11_bandedge",
    )

    results: list[dict[str, Any]] = []
    v0_ref_800: dict[str, Any] | None = None

    with vc.run_workdir("c11_bandedge", keep=args.keep_h5) as wd:
        n_var = len(variant_cfgs)
        t0 = time.perf_counter()
        for i, (vid, label, cfg) in enumerate(variant_cfgs):
            h5_path = wd / f"{vid}.h5"
            d = derive(cfg)
            wall_est = d.n_steps * d.total_cells / _C1_UPDATES_PER_S
            print(
                f"\n=== [{vid}] {label}  steps={d.n_steps:,}  "
                f"decim={resolve_decimation_factor(cfg, d)}  wall≈{wall_est:.0f} s ===",
                flush=True,
            )
            timer.begin(vid)
            run(cfg, out_path=h5_path, progress=True, ram_limit_bytes=ram_limit)
            timer.end(vid)

            with Result(h5_path) as res:
                s_grid = res.S_grid

            if vid == "V0":
                with Result(h5_path) as res:
                    tail_face = float(res.meta.get("tail_db", float("nan")))
                tail_huyg = _huygens_tail_db(h5_path)
                decim_info = _decim_diagnostics(cfg, d, tail_face_db=tail_face, tail_huyg_db=tail_huyg)
                print(
                    f"  V0 decimation: m={decim_info['decimation_m']}  "
                    f"dt_dec={decim_info['dt_dec_s']:.4e} s  "
                    f"f_N={decim_info['f_nyquist_decimated_hz']:.2f} Hz",
                    flush=True,
                )
                for row in decim_info["per_f"]:
                    print(
                        f"    f={row['f_hz']:.0f} Hz  f_alias={row['f_alias_hz']:.1f} Hz  "
                        f"|S(f_a)|/|S(f)|={row['alias_spectrum_db']:+.2f} dB",
                        flush=True,
                    )
                print(
                    f"    tail: face={decim_info['tail_face_db']:.1f} dB  "
                    f"huygens={decim_info['tail_huygens_db']:.1f} dB",
                    flush=True,
                )
            else:
                decim_info = None
                tail_face = float("nan")
                tail_huyg = float("nan")

            freq_rows: list[dict[str, Any]] = []
            for f_hz in _F_HZ:
                row = _metrics_at_f(h5_path, cfg, d, f_hz, s_grid=s_grid)
                freq_rows.append(row)
                print(
                    f"  f={f_hz:.0f} Hz  P_far/P_face−1={row['P_far_over_face_pct']:+.3f}%  "
                    f"P_huyg/P_face−1={row['P_huyg_over_face_pct']:+.3f}%",
                    flush=True,
                )
                if vid == "V0" and abs(f_hz - 800.0) < 0.5:
                    with Result(h5_path) as res:
                        h_ax = complex(res.field([[0.0, 0.0, _R_SPHERE_M]], 800.0)[0, 0])
                    v0_ref_800 = {**row, "field_on_axis_mag": abs(h_ax)}

            far_ok, huyg_ok, max_far, max_huyg = _gate_power_rows(freq_rows)
            entry: dict[str, Any] = {
                "id": vid,
                "label": label,
                "decimation_m": resolve_decimation_factor(cfg, d),
                "frequencies": freq_rows,
                "decimation_diagnostics": decim_info,
                "max_abs_P_far_over_face_err": max_far,
                "max_abs_P_huyg_over_face_err": max_huyg,
                "report_only": vid != "V0",
            }
            if vid == "V4" and v0_ref_800 is not None:
                entry["v4_vs_v0_800_hz"] = _v4_compare_v0(h5_path, cfg, v0_ref_800)
            results.append(entry)

            elapsed = time.perf_counter() - t0
            eta = (elapsed / (i + 1)) * (n_var - i - 1) if i + 1 < n_var else None
            tag = "report"
            if vid == "V0":
                tag = f"far={'PASS' if far_ok else 'FAIL'} huyg={'PASS' if huyg_ok else 'FAIL'}"
            print(
                f"[c11 {i + 1}/{n_var}] {vid} {tag}"
                + (f"  ETA={eta:.0f}s" if eta is not None else ""),
                flush=True,
            )

    v0 = next(e for e in results if e["id"] == "V0")
    v0_rows = v0["frequencies"]
    g_far_ok, g_huyg_ok, g_far_max, g_huyg_max = _gate_power_rows(v0_rows)
    overall = g_far_ok and g_huyg_ok

    print("\n=== C11-bandedge variant report (diagnostic sweeps V1–V8) ===", flush=True)
    for entry in results:
        if entry["id"] == "V0":
            continue
        print(
            f"  {entry['id']}: max |P_far/P_face−1|={entry['max_abs_P_far_over_face_err']*100:.3f}%  "
            f"|P_huyg/P_face−1|={entry['max_abs_P_huyg_over_face_err']*100:.3f}%",
            flush=True,
        )

    print("\n=== C11 gates (V0 defaults @ 600–800 Hz) ===", flush=True)
    print(
        f"  G1-far: {'PASS' if g_far_ok else 'FAIL'}  "
        f"max |P_far/P_face−1|={g_far_max*100:.4f}% (tol ≤{_G_FAR_TOL*100:.1f}%)",
        flush=True,
    )
    print(
        f"  G2-huyg: {'PASS' if g_huyg_ok else 'FAIL'}  "
        f"max |P_huyg/P_face−1|={g_huyg_max*100:.4f}% (tol ≤{_G_HUYG_TOL*100:.1f}%)",
        flush=True,
    )
    print(f"  overall: {'PASS' if overall else 'FAIL'}", flush=True)

    payload: dict[str, Any] = {
        "check": "c11_bandedge",
        "timestamp_utc": stamp,
        "config": str(args.config),
        "base_decimation_m": base_m,
        "f_report_hz": list(_F_HZ),
        "r_sphere_m": _R_SPHERE_M,
        "quadrature": {"n_theta": _N_THETA, "n_phi": _N_PHI},
        "variants": results,
        "gates": {
            "g1_far_tol": _G_FAR_TOL,
            "g2_huyg_tol": _G_HUYG_TOL,
            "max_abs_P_far_over_face_err_V0": g_far_max,
            "max_abs_P_huyg_over_face_err_V0": g_huyg_max,
            "g1_far_pass": g_far_ok,
            "g2_huyg_pass": g_huyg_ok,
            "pass": overall,
        },
        "pass": overall,
        "phases_s": timer.phases,
    }
    out_json = args.out / f"c11_bandedge_{stamp}.json"
    out_json.write_text(json.dumps(vc.to_json_safe(payload), indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}", flush=True)
    vc.print_runtime_summary(timer, tuple(vid for vid, _, _ in variant_cfgs))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())