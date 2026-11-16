#!/usr/bin/env python3
"""C10-power: radiator, Huygens-surface, and far-field radiated power balance."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fdtd.config import (  # noqa: E402
    Config,
    DEFAULT_RAM_LIMIT_BYTES,
    derive,
    estimate_memory_bytes,
)
from fdtd.grid import Grid  # noqa: E402
from fdtd.post.kirchhoff import dpdn_from_vn, kh_pressure  # noqa: E402
from fdtd.post.zrad import zrad_from_h5  # noqa: E402
from fdtd.run import estimate_output_bytes, run  # noqa: E402
from fdtd.source import BoxBoundary  # noqa: E402
import validation._common as vc  # noqa: E402

_H_OVERRIDE_M = 0.005
_F_LOG_MIN_HZ = 100.0
_N_LOG = 12
_KL_EXTRA = (2.0, 5.0, 10.0)
_R_SPHERE_MULT = 50.0
_N_THETA = 24
_N_PHI = 48
_N_THETA_FINE = 36
_N_PHI_FINE = 72
_G_WH_TOL = 0.02
_G_WFF_TOL = 0.02
_G1_WH_NOTE = (
    "direct Huygens flux ½Re Σ p v_n* carries an O((ωΔt)², (kh)²) "
    "discrete-product error near f_max (≈1% at ~23 ppw; ≤0.3% for kL ≤ 10); "
    "far-field power is the primary energy check."
)
_C1_UPDATES_PER_S = 375.0e6
_V_REF = 1.0  # m/s transfer normalization


def _gate_line(name: str, ok: bool, detail: str) -> None:
    print(f"  {name}: {'PASS' if ok else 'FAIL'}  {detail}", flush=True)


def _dedupe_frequencies_hz(freqs: np.ndarray, *, sep_hz: float = 1.0) -> np.ndarray:
    arr = np.sort(np.asarray(freqs, dtype=np.float64))
    if arr.size == 0:
        return arr
    out: list[float] = [float(arr[0])]
    for f in arr[1:]:
        if f - out[-1] > float(sep_hz):
            out.append(float(f))
    return np.asarray(out, dtype=np.float64)


def _frequencies(cfg: Config, L: float, c: float) -> np.ndarray:
    d = derive(cfg)
    f_cfg = cfg.band.f_max
    f_max = float(f_cfg if f_cfg is not None else d.f_max)
    f_log = np.geomspace(_F_LOG_MIN_HZ, f_max, _N_LOG)
    f_kl = np.array([kl * c / (2.0 * math.pi * L) for kl in _KL_EXTRA], dtype=np.float64)
    return _dedupe_frequencies_hz(np.concatenate([f_log, f_kl]), sep_hz=1.0)


def _interp_z(freqs: np.ndarray, z: np.ndarray, f_tgt: float) -> complex:
    re = np.interp(f_tgt, freqs, np.real(z))
    im = np.interp(f_tgt, freqs, np.imag(z))
    return complex(re + 1j * im)


def _sphere_quadrature(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions and solid-angle weights (sum 4π). Gauss–Legendre in cos θ, uniform φ."""
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


def _w_rad(*, rho: float, c: float, S: float, z: complex, v_ref: float = _V_REF) -> float:
    return 0.5 * float(rho) * float(c) * float(S) * (float(v_ref) ** 2) * float(np.real(z))


def _w_huygens(
    p_face: np.ndarray,
    vn_face: np.ndarray,
    areas: np.ndarray,
    V: complex,
) -> float:
    """½ Re Σ p v_n* dA at |V| = 1 m/s (transfer functions via unified spectra / V)."""
    scale = abs(V)
    if scale < 1e-30:
        return float("nan")
    p_tf = p_face / V
    vn_tf = vn_face / V
    return float(0.5 * np.real(np.sum(p_tf * np.conj(vn_tf) * areas)))


def _w_far_field_sphere(
    *,
    r_m: float,
    n_theta: int,
    n_phi: int,
    patches: object,
    p_face: np.ndarray,
    vn_face: np.ndarray,
    V: complex,
    f_hz: float,
    rho: float,
    c: float,
) -> tuple[float, complex]:
    """Σ |H|²/(2ρc) r² w_i; also on-axis H at +z."""
    dirs, w_omega = _sphere_quadrature(n_theta, n_phi)
    pts = float(r_m) * dirs
    omega = np.asarray([2.0 * math.pi * f_hz], dtype=np.float64)
    k_sel = np.asarray([2.0 * math.pi * f_hz / c], dtype=np.float64)
    dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
    p_row = p_face[np.newaxis, :]
    label = f"KH sphere f={f_hz:.1f} Hz nθ={n_theta} nφ={n_phi}"
    with vc.timed_section(label):
        p_obs = kh_pressure(
            pts,
            patches,
            p_row,
            dpdn,
            k_sel,
            image_plane_z=None,
            near_subdivide=False,
        )[0]
    H = p_obs / V
    dens = (np.abs(H) ** 2) / (2.0 * float(rho) * float(c))
    w_ff = float(np.sum(dens * (float(r_m) ** 2) * w_omega))

    axis_pt = np.array([[0.0, 0.0, float(r_m)]], dtype=np.float64)
    with vc.timed_section(f"KH on-axis f={f_hz:.1f} Hz"):
        p_ax = kh_pressure(
            axis_pt,
            patches,
            p_row,
            dpdn,
            k_sel,
            image_plane_z=None,
            near_subdivide=False,
        )[0, 0]
    H_axis = complex(p_ax / V)
    return w_ff, H_axis


def _directivity_index(
    *,
    r_m: float,
    H_axis: complex,
    w_ff: float,
    rho: float,
    c: float,
) -> float:
    numer = 4.0 * math.pi * (float(r_m) ** 2) * (abs(H_axis) ** 2)
    denom = 2.0 * float(rho) * float(c) * max(float(w_ff), 1e-30)
    return float(10.0 * math.log10(max(numer / denom, 1e-30)))


def _geometry_assert(cfg: Config, d: object, *, label: str) -> tuple[float, float, float]:
    if cfg.box is None:
        raise RuntimeError(f"{label}: box required")
    tol = 1e-9
    rw = float(d.realized_W)  # type: ignore[attr-defined]
    rh = float(d.realized_H)
    rd = float(d.realized_D)
    h = float(d.h)  # type: ignore[attr-defined]
    ic = int(d.radiator_ic)  # type: ignore[attr-defined]
    jc = int(d.radiator_jc)
    xc, yc = ic * h, jc * h
    print(
        f"  [{label}] realized W,H,D=({rw:.9f},{rh:.9f},{rd:.9f}) m  "
        f"center=({xc:.9g},{yc:.9g}) m  h={h * 1e3:.4g} mm",
        flush=True,
    )
    for name, nom, got in (
        ("W", cfg.box.W, rw),
        ("H", cfg.box.H, rh),
        ("D", cfg.box.D, rd),
        ("x_c", cfg.radiator.x_c, xc),
        ("y_c", cfg.radiator.y_c, yc),
    ):
        if abs(got - nom) > tol:
            raise RuntimeError(
                f"{label}: {name} realized={got:.12g} m != nominal={nom:.12g} m"
            )
    L = max(rw, rh, rd)
    return rw, rh, L


def _plot_ratios(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping PNG", flush=True)
        return
    kL = np.asarray([r["kL"] for r in rows], dtype=np.float64)
    wh = np.asarray([r["W_H_over_W_rad"] for r in rows], dtype=np.float64)
    wf = np.asarray([r["W_ff_over_W_rad"] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4), layout="tight")
    ax.semilogx(kL, wh, "o-", ms=4, lw=1, label=r"$W_H/W_{rad}$")
    ax.semilogx(kL, wf, "s-", ms=4, lw=1, label=r"$W_{ff}/W_{rad}$")
    ax.axhline(1.0 + _G_WH_TOL, color="C0", ls=":", lw=0.8)
    ax.axhline(1.0 - _G_WH_TOL, color="C0", ls=":", lw=0.8)
    ax.axhline(1.0 + _G_WFF_TOL, color="C1", ls=":", lw=0.8)
    ax.axhline(1.0 - _G_WFF_TOL, color="C1", ls=":", lw=0.8)
    ax.set_xlabel(r"$kL$")
    ax.set_ylabel("power ratio")
    ax.legend(fontsize=8)
    ax.set_title("C10 power balance vs kL")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C10-power: radiated power self-consistency")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "c09_box.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "results")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--ram-gb", type=float, default=DEFAULT_RAM_LIMIT_BYTES / 1e9)
    parser.add_argument(
        "--keep-h5",
        action="store_true",
        help="Keep FDTD HDF5 under proj/boxed_piston/outputs/ (default: temporary)",
    )
    args = parser.parse_args(argv)
    with vc.keep_awake("c10_power"):
        return _c10_body(args)


def _c10_body(args: argparse.Namespace) -> int:
    ram_limit = float(args.ram_gb) * 1e9
    timer = vc.RunTimer()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_json(args.config.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg.numerics.h_override = _H_OVERRIDE_M

    rho, c = float(cfg.air.rho), float(cfg.air.c)
    d_plan = derive(cfg)
    est = estimate_memory_bytes(d_plan.nx, d_plan.ny, d_plan.nz, d_plan.cpml_cells)
    print("\n=== C10-power RAM check ===", flush=True)
    if est > ram_limit:
        raise vc.RamLimitExceeded("c10_power", d_plan.nx, float(est), float(ram_limit))
    wall_est = d_plan.n_steps * d_plan.total_cells / _C1_UPDATES_PER_S
    print(
        f"  grid {d_plan.nx}×{d_plan.ny}×{d_plan.nz}  mem≈{vc.fmt_bytes(est)}  "
        f"steps≈{d_plan.n_steps:,}",
        flush=True,
    )
    print(f"  estimated FDTD wall time: {wall_est:.0f} s", flush=True)

    vc.require_h5_disk(
        estimate_output_bytes(cfg, d_plan),
        vc.workdir_disk_parent(keep=args.keep_h5),
        label="c10_power",
    )

    rows: list[dict[str, Any]] = []
    quad_conv: dict[str, Any] = {}
    g_wh_max = 0.0
    g_wff_max = 0.0

    with vc.run_workdir("c10_power", keep=args.keep_h5) as wd:
        h5_path = wd / "fdtd.h5"
        timer.banner("fdtd")
        timer.begin("fdtd")
        run(cfg, out_path=h5_path, progress=True, ram_limit_bytes=ram_limit)
        timer.end("fdtd")

        d = derive(cfg)
        _, _, L = _geometry_assert(cfg, d, label="c10")
        grid = Grid(cfg, d)
        bound = BoxBoundary.from_grid(grid, d)
        S_grid = float(bound.s_grid)
        r_far = _R_SPHERE_MULT * L

        freqs_z, z_all = zrad_from_h5(h5_path, df_hz=2.0)
        f_set = _frequencies(cfg, L, c)
        f_top = float(np.max(f_set))

        print(
            f"\n=== C10 frequencies ({len(f_set)} pts, r={r_far:.4g} m, S_grid={S_grid:.6g} m²) ===",
            flush=True,
        )

        n_f = len(f_set)
        t0 = time.perf_counter()
        for i, f_hz in enumerate(f_set):
            f_key = float(f_hz)
            kL = f_key * 2.0 * math.pi * L / c
            patches, V, p_face, vn_face = vc.load_huygens_spectra(h5_path, d, f_key)
            areas = np.asarray(patches.areas, dtype=np.float64)
            z_f = _interp_z(freqs_z, z_all, f_key)

            w_rad = _w_rad(rho=rho, c=c, S=S_grid, z=z_f)
            w_h = _w_huygens(p_face, vn_face, areas, V)
            w_ff, h_axis = _w_far_field_sphere(
                r_m=r_far,
                n_theta=_N_THETA,
                n_phi=_N_PHI,
                patches=patches,
                p_face=p_face,
                vn_face=vn_face,
                V=V,
                f_hz=f_key,
                rho=rho,
                c=c,
            )
            di_db = _directivity_index(r_m=r_far, H_axis=h_axis, w_ff=w_ff, rho=rho, c=c)

            rh = w_h / max(w_rad, 1e-30)
            rf = w_ff / max(w_rad, 1e-30)
            err_h = abs(rh - 1.0)
            err_f = abs(rf - 1.0)
            g_wh_max = max(g_wh_max, err_h)
            g_wff_max = max(g_wff_max, err_f)

            row = {
                "f_hz": f_key,
                "kL": kL,
                "W_rad": w_rad,
                "W_H": w_h,
                "W_ff": w_ff,
                "W_H_over_W_rad": rh,
                "W_ff_over_W_rad": rf,
                "DI_db": di_db,
                "Re_z": float(np.real(z_f)),
                "V_mag": float(abs(V)),
            }
            rows.append(row)

            elapsed = time.perf_counter() - t0
            eta = (elapsed / (i + 1)) * (n_f - i - 1) if i + 1 < n_f else None
            print(
                f"[c10 {i + 1}/{n_f}] f={f_key:8.2f} Hz  kL={kL:6.3f}  "
                f"W_rad={w_rad:.4e}  W_H/W_rad={rh:.4f}  W_ff/W_rad={rf:.4f}  "
                f"DI={di_db:+.2f} dB"
                + (f"  ETA={eta:.0f}s" if eta is not None else ""),
                flush=True,
            )

            if abs(f_key - f_top) < 0.5:
                w_ff_fine, _ = _w_far_field_sphere(
                    r_m=r_far,
                    n_theta=_N_THETA_FINE,
                    n_phi=_N_PHI_FINE,
                    patches=patches,
                    p_face=p_face,
                    vn_face=vn_face,
                    V=V,
                    f_hz=f_key,
                    rho=rho,
                    c=c,
                )
                quad_conv = {
                    "f_hz": f_key,
                    "W_ff_n24_n48": w_ff,
                    "W_ff_n36_n72": w_ff_fine,
                    "W_ff_over_W_rad_coarse": rf,
                    "W_ff_over_W_rad_fine": w_ff_fine / max(w_rad, 1e-30),
                    "rel_change_ff": abs(w_ff_fine - w_ff) / max(abs(w_ff), 1e-30),
                }

        print("\n=== C10 power table ===", flush=True)
        print(
            f"  {'f':>10} {'kL':>8} {'W_rad':>12} {'W_H/W':>9} {'W_ff/W':>9} {'DI dB':>8}",
            flush=True,
        )
        for row in rows:
            print(
                f"  {row['f_hz']:10.2f} {row['kL']:8.4f} {row['W_rad']:12.4e} "
                f"{row['W_H_over_W_rad']:9.4f} {row['W_ff_over_W_rad']:9.4f} "
                f"{row['DI_db']:+8.2f}",
                flush=True,
            )

        if quad_conv:
            print(
                f"\n=== Far-field quadrature @ top f={quad_conv['f_hz']:.2f} Hz ===",
                flush=True,
            )
            print(
                f"  nθ={_N_THETA} nφ={_N_PHI}: W_ff/W_rad={quad_conv['W_ff_over_W_rad_coarse']:.5f}",
                flush=True,
            )
            print(
                f"  nθ={_N_THETA_FINE} nφ={_N_PHI_FINE}: "
                f"W_ff/W_rad={quad_conv['W_ff_over_W_rad_fine']:.5f}  "
                f"rel ΔW_ff={quad_conv['rel_change_ff']:.4e}",
                flush=True,
            )

        if not args.no_plot:
            _plot_ratios(args.out / f"c10_power_{stamp}.png", rows)

    g_wh_pass = g_wh_max <= _G_WH_TOL
    g_wff_pass = g_wff_max <= _G_WFF_TOL
    overall = g_wh_pass and g_wff_pass

    print("\n=== C10-power gates ===", flush=True)
    _gate_line(
        "G1-WH",
        g_wh_pass,
        f"max |W_H/W_rad−1|={g_wh_max:.4f} (tol ≤{_G_WH_TOL})",
    )
    print(f"  note: {_G1_WH_NOTE}", flush=True)
    _gate_line(
        "G2-Wff",
        g_wff_pass,
        f"max |W_ff/W_rad−1|={g_wff_max:.4f} (tol ≤{_G_WFF_TOL})",
    )
    print(f"  overall: {'PASS' if overall else 'FAIL'}", flush=True)

    payload: dict[str, Any] = {
        "check": "c10_power",
        "timestamp_utc": stamp,
        "config": str(args.config),
        "h_override_m": _H_OVERRIDE_M,
        "S_grid_m2": S_grid,
        "L_m": L,
        "r_sphere_m": r_far,
        "frequencies_hz": [float(f) for f in f_set],
        "rows": rows,
        "quadrature_convergence_top_f": quad_conv or None,
        "gates": {
            "g1_wh_tol": _G_WH_TOL,
            "g2_wff_tol": _G_WFF_TOL,
            "max_abs_WH_ratio_err": g_wh_max,
            "max_abs_Wff_ratio_err": g_wff_max,
            "g1_wh_pass": g_wh_pass,
            "g2_wff_pass": g_wff_pass,
            "pass": overall,
        },
        "pass": overall,
        "phases_s": timer.phases,
    }
    out_json = args.out / f"c10_power_{stamp}.json"
    out_json.write_text(json.dumps(vc.to_json_safe(payload), indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}", flush=True)
    vc.print_runtime_summary(timer, ("fdtd",))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())