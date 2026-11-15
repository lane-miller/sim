#!/usr/bin/env python3
"""C6-baffle: baffled piston Z_rad and far-field vs analytic reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from fdtd.post.kirchhoff import (  # noqa: E402
    SurfacePatches,
    dpdn_from_vn,
    far_field_at_radius,
    kh_far_field,
    kh_pressure,
)
from fdtd.post.spectra import spectrum, spectrum_rows  # noqa: E402
from fdtd.post.reference import z_baffled_circle  # noqa: E402
from fdtd.post.zrad import causal_end_taper  # noqa: E402
from fdtd.source import velocity_pulse  # noqa: E402
from fdtd.post.zrad import zrad_from_h5  # noqa: E402
from fdtd.run import estimate_output_bytes, run  # noqa: E402
from fdtd.source import BoxBoundary  # noqa: E402
import validation._common as vc  # noqa: E402

_G1_KA = 0.3
_G1_TOL = 0.02
_G2_TOL = 0.03
_G2_TIGHT = 0.01
_G3_DB = 0.3
_G3_PHASE_DEG = 5.0
_G3_DIR_DB = 0.5
_C1_UPDATES_PER_S = 375.0e6


def _geometry(cfg: Config, d: object) -> tuple[float, float]:
    grid = Grid(cfg, d)  # type: ignore[arg-type]
    bound = BoxBoundary.from_grid(grid, d)  # type: ignore[arg-type]
    S = float(bound.s_grid)
    a = math.sqrt(S / math.pi)
    return S, a


def _fit_low_ka(freqs: np.ndarray, z: np.ndarray, ka: np.ndarray) -> dict[str, float]:
    m = ka <= _G1_KA
    k = 2.0 * np.pi * freqs[m] / 343.0
    re = np.real(z[m])
    X = np.column_stack([k**2, k**4])
    A, AB = np.linalg.lstsq(X, re, rcond=None)[0]
    B = AB / A if abs(A) > 1e-30 else 0.0
    return {"A": float(A), "B": float(B)}


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


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.column_stack(
        [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ]
    )


def _h_exact_on_axis(z: float, f: float, *, a: float, rho: float, c: float) -> complex:
    """Transfer H = P/V for baffled piston on axis: ρc[e^{−jkz} − e^{−jk√(z²+a²)}]."""
    k = 2.0 * math.pi * f / c
    return rho * c * (
        np.exp(-1j * k * z) - np.exp(-1j * k * math.sqrt(z * z + a * a))
    )


def _h_transfer_err(h_ref: complex, h: complex) -> tuple[float, float]:
    db = 20.0 * math.log10(abs(h) / max(abs(h_ref), 1e-30))
    ph = float(np.rad2deg(np.angle(h) - np.angle(h_ref)))
    return db, ph


def maybe_plot_z(
    ka: np.ndarray,
    z: np.ndarray,
    z_ref: np.ndarray,
    path: Path,
    *,
    a: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = np.isfinite(ka) & np.isfinite(z.real)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), layout="tight")
    ax1.plot(ka[m], np.real(z[m]), "b.", ms=3, label="Re z FDTD")
    ax1.plot(ka[m], np.real(z_ref[m]), "k--", lw=1, label="Re z ref")
    ax2.plot(ka[m], np.imag(z[m]), "r.", ms=3, label="Im z FDTD")
    ax2.plot(ka[m], np.imag(z_ref[m]), "k--", lw=1, label="Im z ref")
    ax2.set_xlabel(r"$ka$")
    ax1.set_ylabel("Re z")
    ax2.set_ylabel("Im z")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.suptitle(f"C6-baffle  a={a:.4g} m")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def maybe_plot_dir(
    ka_tgt: float,
    theta: np.ndarray,
    p_fdtd: np.ndarray,
    p_ref: np.ndarray,
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    mag_fd = 20.0 * np.log10(np.maximum(np.abs(p_fdtd), 1e-30))
    mag_rf = 20.0 * np.log10(np.maximum(np.abs(p_ref), 1e-30))
    fig, ax = plt.subplots(figsize=(7, 4), layout="tight")
    ax.plot(np.rad2deg(theta), mag_fd, "b-", label="KH FDTD")
    ax.plot(np.rad2deg(theta), mag_rf, "k--", label="ref")
    ax.set_xlabel("θ deg")
    ax.set_ylabel("|p| dB")
    ax.set_title(f"Directivity  ka={ka_tgt:.2g}")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C6-baffle validation")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "c06_baffle.json",
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
    with vc.keep_awake("c06_baffle"):
        return _c06_body(args)


def _c06_body(args: argparse.Namespace) -> int:
    cfg = Config.from_json(args.config.read_text(encoding="utf-8"))
    d = derive(cfg)
    ram_limit = float(args.ram_gb) * 1e9
    est = estimate_memory_bytes(d.nx, d.ny, d.nz, d.cpml_cells)
    if est > ram_limit:
        raise vc.RamLimitExceeded(
            "c06 plan", d.nx, float(est), float(ram_limit)
        )

    timer = vc.RunTimer()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)

    vc.require_h5_disk(
        estimate_output_bytes(cfg, d),
        vc.workdir_disk_parent(keep=args.keep_h5),
        label="c06_baffle",
    )

    with vc.run_workdir("c06_baffle", keep=args.keep_h5) as wd:
        h5 = wd / "c06_baffle.h5"
        timer.banner("c06", f"grid {d.nx}×{d.ny}×{d.nz}  steps≈{d.n_steps:,}")
        timer.begin("c06")
        run(cfg, out_path=h5, progress=True, ram_limit_bytes=ram_limit)
        timer.end("c06")

        freqs, z = zrad_from_h5(h5, df_hz=2.0)
        S, a = _geometry(cfg, d)
        c = cfg.air.c
        k = 2.0 * np.pi * freqs / c
        ka = k * a
        ka_max = float(np.max(ka[freqs <= d.f_max]))
        z_ref = np.asarray(z_baffled_circle(ka), dtype=np.complex128)
        err = np.abs(z - z_ref)
        band = (freqs >= cfg.band.f_min) & (freqs <= d.f_max)
        fit = _fit_low_ka(freqs, z, ka)
        g1_ref = S / (2.0 * math.pi)
        g1_signed = fit["A"] / g1_ref - 1.0
        g1_pass = abs(g1_signed) <= _G1_TOL
        g2_pass = bool(np.max(err[band]) <= _G2_TOL)
        tight_mask = band & (err <= _G2_TIGHT)
        ka_tight = float(np.max(ka[tight_mask])) if np.any(tight_mask) else 0.0
        tail_db = _p_avg_tail_db(h5)

        # G3 far field at ka = 1, 3, 5
        import h5py

        g3_probe: dict[str, Any] = {}
        g3: dict[str, Any] = {}
        rho = cfg.air.rho
        with h5py.File(h5, "r") as f:
            centers = np.asarray(f["huygens/centers"][()], dtype=np.float64)
            normals = np.asarray(f["huygens/normals"][()], dtype=np.float64)
            areas = np.asarray(f["huygens/areas"][()], dtype=np.float64)
            patches = SurfacePatches(centers, normals, areas, float(d.h))
            t_p = np.asarray(f["huygens/times_p"][()], dtype=np.float64)
            t_v = np.asarray(f["huygens/times_vn"][()], dtype=np.float64)
            p_ts = np.asarray(f["huygens/p_face"][()], dtype=np.float64)
            vn_ts = np.asarray(f["huygens/vn"][()], dtype=np.float64)
            t_rad = np.asarray(f["radiator/times_p"][()], dtype=np.float64)
            v_rad = np.array(
                [velocity_pulse(float(t), d.pulse_sigma, d.pulse_t0) for t in t_rad],
                dtype=np.float64,
            )
            probe_centers = None
            probe_p_ts = None
            t_probe = None
            if "probes" in f:
                probe_centers = np.asarray(f["probes/centers"][()], dtype=np.float64)
                probe_p_ts = np.asarray(f["probes/p"][()], dtype=np.float64)
                t_probe = np.asarray(f["probes/times_p"][()], dtype=np.float64)
        n_samp = p_ts.shape[0]
        win_h = causal_end_taper(n_samp)
        p_win = p_ts * win_h[:, np.newaxis]
        vn_win = vn_ts * win_h[:, np.newaxis]
        theta = np.linspace(0.0, 0.5 * math.pi, 91)
        dir_ax = np.column_stack(
            [np.sin(theta), np.zeros_like(theta), np.cos(theta)]
        )
        r_far = 100.0 * a
        from scipy.special import j1

        for ka_tgt in (1.0, 3.0, 5.0):
            f_hz = ka_tgt * c / (math.pi * cfg.radiator.dims[0])
            with vc.timed_section(f"huygens spectrum ka={ka_tgt:g} f={f_hz:.1f} Hz"):
                V = complex(spectrum(v_rad, t_rad, f_hz))
                p_face = np.asarray(spectrum_rows(p_win, t_p, f_hz), dtype=np.complex128)
                vn_face = np.asarray(spectrum_rows(vn_win, t_v, f_hz), dtype=np.complex128)
            if abs(V) < 1e-30:
                continue
            omega = 2.0 * math.pi * f_hz
            dpdn_row = dpdn_from_vn(
                vn_face[np.newaxis, :],
                np.asarray([omega], dtype=np.float64),
                rho,
            )
            p_row = p_face[np.newaxis, :]
            k_sel = np.asarray([2.0 * math.pi * f_hz / c], dtype=np.float64)

            p_int = kh_pressure(
                np.array([[0.0, 0.0, r_far]], dtype=np.float64),
                patches,
                p_row,
                dpdn_row,
                k_sel,
                image_plane_z=0.0,
                near_subdivide=False,
            )[0, 0]
            H_kh = p_int / V
            H_ex = _h_exact_on_axis(r_far, f_hz, a=a, rho=rho, c=c)
            db_gate, ph_gate = _h_transfer_err(H_ex, H_kh)

            ff = kh_far_field(
                np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
                patches,
                p_row,
                dpdn_row,
                k_sel,
                image_plane_z=0.0,
            )[0, 0]
            p_ff = far_field_at_radius(
                np.asarray([[ff]], dtype=np.complex128), k_sel, R_ref=r_far
            )[0, 0]
            H_ff = p_ff / V
            db_ff, ph_ff = _h_transfer_err(H_kh, H_ff)

            pts = r_far * dir_ax
            with vc.timed_section(f"KH directivity ka={ka_tgt:g} f={f_hz:.1f} Hz"):
                p_dir = kh_pressure(
                    pts,
                    patches,
                    p_row,
                    dpdn_row,
                    k_sel,
                    image_plane_z=0.0,
                    near_subdivide=False,
                )[0]
            H_dir = p_dir / V
            H0 = H_dir[0]
            pat = np.ones_like(theta)
            u = ka_tgt * np.sin(theta)
            m = u > 1e-8
            pat[m] = np.abs(2.0 * j1(u[m]) / u[m])
            rel_fd = np.abs(H_dir / H0)
            dir_db_err = 20.0 * np.log10(np.maximum(rel_fd, 1e-30) / np.maximum(pat, 1e-30))
            pat_db = 20.0 * np.log10(np.maximum(pat, 1e-30))
            dir_gate = pat_db >= -20.0
            if np.any(dir_gate):
                dir_max = float(np.max(np.abs(dir_db_err[dir_gate])))
            else:
                dir_max = float(np.max(np.abs(dir_db_err)))
            dir_pass = dir_max <= _G3_DIR_DB

            g3[f"ka{ka_tgt:.0f}"] = {
                "f_hz": float(f_hz),
                "db_err": float(db_gate),
                "phase_err_deg": float(abs(ph_gate)),
                "ff_vs_kh_db": float(db_ff),
                "ff_vs_kh_phase_deg": float(abs(ph_ff)),
                "dir_max_db_err": dir_max,
                "dir_pass": bool(dir_pass),
            }

            if probe_centers is not None and probe_p_ts is not None and t_probe is not None:
                win_pr = causal_end_taper(probe_p_ts.shape[0])
                probes_kh: list[dict[str, float]] = []
                for qi in range(probe_p_ts.shape[1]):
                    z_m = float(probe_centers[qi, 2])
                    P_fd = complex(
                        spectrum(probe_p_ts[:, qi] * win_pr, t_probe, f_hz)
                    )
                    H_fd = P_fd / V
                    p_kh = kh_pressure(
                        probe_centers[qi].reshape(1, 3),
                        patches,
                        p_row,
                        dpdn_row,
                        k_sel,
                        image_plane_z=0.0,
                        near_subdivide=False,
                    )[0, 0]
                    H_kh_z = p_kh / V
                    H_ex_z = _h_exact_on_axis(z_m, f_hz, a=a, rho=rho, c=c)
                    db_kh, ph_kh = _h_transfer_err(H_fd, H_kh_z)
                    db_ex, ph_ex = _h_transfer_err(H_fd, H_ex_z)
                    probes_kh.append(
                        {
                            "z_m": z_m,
                            "kh_db": float(db_kh),
                            "kh_phase_deg": float(ph_kh),
                            "exact_db": float(db_ex),
                            "exact_phase_deg": float(ph_ex),
                        }
                    )
                g3_probe[f"ka{ka_tgt:.0f}"] = probes_kh

        g3_pass = all(
            abs(v["db_err"]) <= _G3_DB
            and v["phase_err_deg"] <= _G3_PHASE_DEG
            and v.get("dir_pass", False)
            for v in g3.values()
        )
        gate = g1_pass and g2_pass

    def _gate_line(name: str, passed: bool, detail: str) -> None:
        mark = "PASS" if passed else "FAIL"
        print(f"  {name}: {mark}  {detail}")

    print("\n--- C6-baffle gates & reports ---")
    _gate_line(
        "G1",
        g1_pass,
        f"signed Re-fit offset {g1_signed:+.4f} (tol ±{_G1_TOL}; ref S/2π={g1_ref:.4g})",
    )
    _gate_line(
        "G2",
        g2_pass,
        f"max |z−z_ref|={float(np.max(err[band])):.4f} (tol ≤{_G2_TOL}); "
        f"largest ka with |z−z_ref|≤{_G2_TIGHT}: ka={ka_tight:.3g}",
    )
    for key in sorted(g3.keys()):
        v = g3[key]
        ok_xfer = abs(v["db_err"]) <= _G3_DB and v["phase_err_deg"] <= _G3_PHASE_DEG
        ok_dir = bool(v.get("dir_pass", False))
        ok = ok_xfer and ok_dir
        _gate_line(
            f"G3 {key}",
            ok,
            f"H_KH vs H_exact Δ={v['db_err']:+.2f} dB, ∠{v['phase_err_deg']:.1f}° "
            f"(tol ±{_G3_DB}, ±{_G3_PHASE_DEG}); "
            f"directivity max |ΔdB|={v.get('dir_max_db_err', float('nan')):.3f} "
            f"(ref≥−20 dB, tol {_G3_DIR_DB}); "
            f"report H_FF−H_KH {v.get('ff_vs_kh_db', float('nan')):+.2f} dB",
        )
    print("  G3-probe (report): transfer H = P/V vs V(radiator times_p spectrum)")
    for key in sorted(g3_probe.keys()):
        print(f"    {key} @ {g3.get(key, {}).get('f_hz', float('nan')):.1f} Hz")
        for row in g3_probe[key]:
            print(
                f"      z={row['z_m']:.4g} m: "
                f"H_KH−H_FDTD {row['kh_db']:+.2f} dB, ∠{row['kh_phase_deg']:+.1f}°; "
                f"H_exact−H_FDTD {row['exact_db']:+.2f} dB, ∠{row['exact_phase_deg']:+.1f}°"
            )
    print(f"  tail: report-only  last-5% peak = {tail_db:.1f} dB re run peak (target ≤ −80 dB)")
    print(f"  overall gate: {'PASS' if gate else 'FAIL'}  (G1 ∧ G2)")
    print(f"  G3 summary pass: {'yes' if g3_pass else 'no'} (informational)")

    png_z = None if args.no_plot else args.out / f"c06_baffle_{stamp}.png"
    if png_z is not None:
        maybe_plot_z(ka, z, z_ref, png_z, a=a)

    payload: dict[str, Any] = {
        "config": str(args.config),
        "grid": {"nx": d.nx, "ny": d.ny, "nz": d.nz},
        "n_steps": d.n_steps,
        "h": d.h,
        "cells_across": cfg.radiator.dims[0] / d.h,
        "ka_max": ka_max,
        "tail_db": tail_db,
        "gates": {
            "g1_signed": g1_signed,
            "g1_pass": g1_pass,
            "g2_max_err": float(np.max(err[band])),
            "g2_pass": g2_pass,
            "ka_tight": ka_tight,
            "g3": g3,
            "g3_probe": g3_probe,
        },
        "pass": gate,
        "g3_pass": g3_pass,
        "wall_s_est": d.n_steps * d.total_cells / _C1_UPDATES_PER_S,
    }
    out_json = args.out / f"c06_baffle_{stamp}.json"
    out_json.write_text(json.dumps(vc.to_json_safe(payload), indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())