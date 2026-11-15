#!/usr/bin/env python3
"""C9-bempp: box radiation vs exterior Neumann BEM (bempp-cl Burton–Miller)."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
from fdtd.post.bem_ref import (  # noqa: E402
    BoxMeshCache,
    dedupe_frequencies_hz,
    format_bem_progress_line,
    h_target_from_frequency,
    h_target_sphere,
    solve_neumann_bm,
    zrad_sphere_analytic,
)
from fdtd.post.kirchhoff import (  # noqa: E402
    SurfacePatches,
    dpdn_from_vn,
    kh_pressure,
)
from fdtd.post.spectra import spectrum, spectrum_rows  # noqa: E402
from fdtd.post.zrad import causal_end_taper, zrad_from_h5  # noqa: E402
from fdtd.run import estimate_output_bytes, run  # noqa: E402
from fdtd.source import BoxBoundary, velocity_pulse  # noqa: E402
import validation._common as vc  # noqa: E402

_G0_KA = (0.2, 1.0, math.pi, 4.0)
_G0_TOL = 0.01
_G1_TOL = 0.02
_G2_DB = 0.3
_G2_PHASE_DEG = 5.0
_G2_DIR_DB = 0.5
_G2_DIR_LEVEL_DB = -20.0
_G2_KL = (2.0, 5.0, 10.0)
_G3_TOL = 0.005
_G3_EPW = 14
_G3_N_TOP = 3
_GEOM_TOL = 1e-9
_BEM_F_MIN_HZ = 100.0
_N_LOG_BEM = 12
_BEM_EPW = 10
_BEM_H_MAX_M = 0.02
_C1_UPDATES_PER_S = 375.0e6


def _gate_line(name: str, ok: bool, detail: str) -> None:
    print(f"  {name}: {'PASS' if ok else 'FAIL'}  {detail}", flush=True)


def _box_origin(cfg: Config, d: object) -> tuple[float, float, float]:
    xc, yc = float(cfg.radiator.x_c), float(cfg.radiator.y_c)
    rw = float(d.realized_W)  # type: ignore[attr-defined]
    rh = float(d.realized_H)
    rd = float(d.realized_D)
    return (-xc - 0.5 * rw, -yc - 0.5 * rh, -rd)


def _radiator_rect(cfg: Config) -> tuple[float, float]:
    if cfg.radiator.shape != "rect":
        raise RuntimeError("C9 config requires rectangular radiator")
    return float(cfg.radiator.dims[0]), float(cfg.radiator.dims[1])


def _assert_geometry(cfg: Config, d: object, *, label: str) -> tuple[float, float, float]:
    if cfg.box is None:
        raise RuntimeError(f"{label}: box required")
    tol = _GEOM_TOL
    rw = float(d.realized_W)  # type: ignore[attr-defined]
    rh = float(d.realized_H)
    rd = float(d.realized_D)
    h = float(d.h)
    ic = int(d.radiator_ic)  # type: ignore[attr-defined]
    jc = int(d.radiator_jc)
    xc, yc = ic * h, jc * h
    print(
        f"  [{label}] realized W,H,D=({rw:.9f},{rh:.9f},{rd:.9f}) m  "
        f"center=({xc:.9g},{yc:.9g}) m",
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


def _bem_frequencies(cfg: Config, L: float, c: float) -> np.ndarray:
    f_max = float(cfg.band.f_max if cfg.band.f_max is not None else derive(cfg).f_max)
    f_log = np.geomspace(_BEM_F_MIN_HZ, f_max, _N_LOG_BEM)
    f_kl = np.array([kl * c / (2.0 * math.pi * L) for kl in _G2_KL], dtype=np.float64)
    return dedupe_frequencies_hz(np.concatenate([f_log, f_kl]), sep_hz=1.0)


def _interp_z(freqs: np.ndarray, z: np.ndarray, f_tgt: float) -> complex:
    re = np.interp(f_tgt, freqs, np.real(z))
    im = np.interp(f_tgt, freqs, np.imag(z))
    return complex(re + 1j * im)


def _h_err(h_ref: complex, h: complex) -> tuple[float, float]:
    db = 20.0 * math.log10(abs(h) / max(abs(h_ref), 1e-30))
    ph = float(np.rad2deg(np.angle(h) - np.angle(h_ref)))
    while ph > 180.0:
        ph -= 360.0
    while ph < -180.0:
        ph += 360.0
    return db, ph


def _load_huygens_spectrum(
    path: Path, cfg: Config, d: object, f_hz: float
) -> tuple[SurfacePatches, complex, np.ndarray, np.ndarray, float]:
    import h5py

    rho = cfg.air.rho
    with h5py.File(path, "r") as f:
        centers = np.asarray(f["huygens/centers"][()], dtype=np.float64)
        normals = np.asarray(f["huygens/normals"][()], dtype=np.float64)
        areas = np.asarray(f["huygens/areas"][()], dtype=np.float64)
        patches = SurfacePatches(centers, normals, areas, float(d.h))  # type: ignore[attr-defined]
        t_p = np.asarray(f["huygens/times_p"][()], dtype=np.float64)
        t_v = np.asarray(f["huygens/times_vn"][()], dtype=np.float64)
        p_ts = np.asarray(f["huygens/p_face"][()], dtype=np.float64)
        vn_ts = np.asarray(f["huygens/vn"][()], dtype=np.float64)
        t_rad = np.asarray(f["radiator/times_p"][()], dtype=np.float64)
        v_rad = np.array(
            [velocity_pulse(float(t), d.pulse_sigma, d.pulse_t0) for t in t_rad],  # type: ignore[attr-defined]
            dtype=np.float64,
        )
    n_samp = p_ts.shape[0]
    win = causal_end_taper(n_samp)
    p_win = p_ts * win[:, np.newaxis]
    vn_win = vn_ts * win[:, np.newaxis]
    with vc.timed_section(f"huygens spectrum f={f_hz:.1f} Hz"):
        V = complex(spectrum(v_rad, t_rad, f_hz))
        p_face = np.asarray(spectrum_rows(p_win, t_p, f_hz), dtype=np.complex128)
        vn_face = np.asarray(spectrum_rows(vn_win, t_v, f_hz), dtype=np.complex128)
    return patches, V, p_face, vn_face, rho


def _directivity_cut(
    plane: str,
    r_m: float,
    f_hz: float,
    *,
    patches: SurfacePatches,
    p_face: np.ndarray,
    vn_face: np.ndarray,
    V: complex,
    rho: float,
    c: float,
    n_theta: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_theta)
    if plane == "xz":
        pts = np.column_stack([r_m * np.sin(theta), np.zeros_like(theta), r_m * np.cos(theta)])
    elif plane == "yz":
        pts = np.column_stack([np.zeros_like(theta), r_m * np.sin(theta), r_m * np.cos(theta)])
    elif plane == "diag45":
        ct, st = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
        rc = r_m * np.cos(theta)
        pts = np.column_stack([rc * ct, rc * st, r_m * np.sin(theta)])
    else:
        raise ValueError(f"unknown plane {plane!r}")
    omega = np.asarray([2.0 * math.pi * f_hz], dtype=np.float64)
    k_sel = np.asarray([2.0 * math.pi * f_hz / c], dtype=np.float64)
    dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
    p_row = p_face[np.newaxis, :]
    with vc.timed_section(f"KH directivity {plane} f={f_hz:.1f} Hz"):
        p_obs = kh_pressure(pts, patches, p_row, dpdn, k_sel, image_plane_z=None, near_subdivide=True)[0]
    H = p_obs / V
    axis_i = int(np.argmin(np.abs(theta)))
    H0 = H[axis_i]
    rel = np.abs(H / H0)
    db = 20.0 * np.log10(np.maximum(rel, 1e-30))
    return theta, db, H


def _bem_directivity(
    plane: str,
    r_m: float,
    f_hz: float,
    *,
    sol,
    V_fdtd: complex,
    c: float,
    n_theta: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_theta)
    if plane == "xz":
        pts = np.column_stack([r_m * np.sin(theta), np.zeros_like(theta), r_m * np.cos(theta)])
    elif plane == "yz":
        pts = np.column_stack([np.zeros_like(theta), r_m * np.sin(theta), r_m * np.cos(theta)])
    elif plane == "diag45":
        ct, st = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
        rc = r_m * np.cos(theta)
        pts = np.column_stack([rc * ct, rc * st, r_m * np.sin(theta)])
    else:
        raise ValueError(f"unknown plane {plane!r}")
    p_obs = sol.field_at(pts)
    H = p_obs / V_fdtd
    axis_i = int(np.argmin(np.abs(theta)))
    H0 = H[axis_i]
    rel = np.abs(H / H0)
    db = 20.0 * np.log10(np.maximum(rel, 1e-30))
    return theta, db, H


def _surface_report_points(
    origin: tuple[float, float, float], W: float, H: float, D: float
) -> list[tuple[str, np.ndarray]]:
    ox, oy, oz = origin
    z_front = oz + D
    z_rear = oz
    return [
        ("radiator_center", np.array([0.0, 0.0, z_front])),
        ("front_corner", np.array([ox + W, oy + H, z_front])),
        ("side_face", np.array([ox + W, oy + 0.5 * H, oz + 0.5 * D])),
        ("rear_center", np.array([ox + 0.5 * W, oy + 0.5 * H, z_rear])),
        ("edge_mid", np.array([ox + 0.5 * W, oy + H, oz + 0.5 * D])),
    ]


def _load_box_p_spectrum(path: Path, f_hz: float) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        if "box_surface" not in f:
            raise RuntimeError("HDF5 missing box_surface group")
        centers = np.asarray(f["box_surface/centers"][()], dtype=np.float64)
        t = np.asarray(f["box_surface/times"][()], dtype=np.float64)
        p_ts = np.asarray(f["box_surface/p"][()], dtype=np.float64)
    win = causal_end_taper(p_ts.shape[0])
    p_win = p_ts * win[:, np.newaxis]
    p_spec = np.asarray(spectrum_rows(p_win, t, f_hz), dtype=np.complex128)
    return centers, p_spec.ravel()


def _g0_sphere_check(rho: float, c: float) -> bool:
    import bempp_cl.api as bempp

    print("\n=== G0 pulsating-sphere self-check ===", flush=True)
    ok_all = True
    n = len(_G0_KA)
    t_batch = time.perf_counter()
    for i, ka in enumerate(_G0_KA):
        a = 0.05
        k = ka / a
        f_hz = k * c / (2.0 * math.pi)
        h_mesh = h_target_sphere(ka, a, epw=_BEM_EPW)
        grid = bempp.shapes.sphere(r=a, h=h_mesh)
        sol = solve_neumann_bm(
            grid, k, vn_P0=1.0, rho=rho, c=c, pulsating_sphere=True, log_dof=False
        )
        z_bem = sol.zrad()
        z_ref = zrad_sphere_analytic(ka)
        err = abs(z_bem - z_ref)
        ok = err <= _G0_TOL
        ok_all = ok_all and ok
        stats = sol.stats
        elapsed = time.perf_counter() - t_batch
        eta = (elapsed / (i + 1)) * (n - i - 1) if i + 1 < n else 0.0
        if stats is not None:
            print(
                format_bem_progress_line(
                    index=i + 1,
                    total=n,
                    f_hz=f_hz,
                    h_m=h_mesh,
                    stats=stats,
                    elapsed_s=elapsed,
                    eta_s=eta if i + 1 < n else None,
                    prefix="bem G0",
                ),
                flush=True,
            )
        print(
            f"  ka={ka:g}: |z_bem−z_ref|={err:.4f} (tol {_G0_TOL})  "
            f"z_ref={z_ref.real:.4f}{z_ref.imag:+.4f}j",
            flush=True,
        )
    if not ok_all:
        print("  G0 FAIL — aborting C9", flush=True)
    return ok_all


def _plot_z(
    path: Path,
    freqs: np.ndarray,
    z_fdtd: np.ndarray,
    z_bem: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=(7, 5), layout="tight", sharex=True)
    ax_re.plot(freqs, np.real(z_fdtd), label="FDTD")
    ax_re.plot(freqs, np.real(z_bem), "--", label="BEM")
    ax_im.plot(freqs, np.imag(z_fdtd), label="FDTD")
    ax_im.plot(freqs, np.imag(z_bem), "--", label="BEM")
    ax_re.set_ylabel("Re z")
    ax_im.set_ylabel("Im z")
    ax_im.set_xlabel("f (Hz)")
    ax_re.legend(fontsize=8)
    ax_im.legend(fontsize=8)
    ax_re.set_title("C9 z_rad: FDTD vs BEM")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dir(path: Path, *, plane: str, kl: float, curves: dict[str, np.ndarray], theta: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4), layout="tight")
    for name, db in curves.items():
        ax.plot(np.rad2deg(theta), db, lw=1, label=name)
    ax.set_xlabel("θ deg")
    ax.set_ylabel("rel |H| dB re on-axis")
    ax.set_title(f"C9 G2 directivity {plane}  kL={kl:g}")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C9-bempp: FDTD vs BEM")
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
    with vc.keep_awake("c09_bempp"):
        return _c09_body(args)


def _c09_body(args: argparse.Namespace) -> int:
    ram_limit = float(args.ram_gb) * 1e9
    timer = vc.RunTimer()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_json(args.config.read_text(encoding="utf-8"))
    rho, c = cfg.air.rho, cfg.air.c
    if not _g0_sphere_check(rho, c):
        return 1

    d_plan = derive(cfg)
    est = estimate_memory_bytes(d_plan.nx, d_plan.ny, d_plan.nz, d_plan.cpml_cells)
    print("\n=== C9 FDTD RAM check ===", flush=True)
    if est > ram_limit:
        raise vc.RamLimitExceeded("c09 FDTD", d_plan.nx, float(est), float(ram_limit))
    print(
        f"  grid {d_plan.nx}×{d_plan.ny}×{d_plan.nz}  mem≈{vc.fmt_bytes(est)}  "
        f"steps≈{d_plan.n_steps:,}",
        flush=True,
    )
    wall_est = d_plan.n_steps * d_plan.total_cells / _C1_UPDATES_PER_S
    print(f"  estimated FDTD wall time: {wall_est:.0f} s", flush=True)

    vc.require_h5_disk(
        estimate_output_bytes(cfg, d_plan),
        vc.workdir_disk_parent(keep=args.keep_h5),
        label="c09_bempp",
    )

    gates: dict[str, Any] = {}
    g1_pass = g2_pass = g3_pass = True

    with vc.run_workdir("c09_bempp", keep=args.keep_h5) as wd:
        h5 = wd / "fdtd.h5"
        timer.banner("fdtd")
        timer.begin("fdtd")
        run(cfg, out_path=h5, progress=True, ram_limit_bytes=ram_limit)
        timer.end("fdtd")
        d = derive(cfg)
        rw, rh, L = _assert_geometry(cfg, d, label="c09")
        rad_rect = _radiator_rect(cfg)
        origin = _box_origin(cfg, d)

        freqs_design, z_fdtd_all = zrad_from_h5(h5, df_hz=2.0)
        f_bem = _bem_frequencies(cfg, L, c)

        z_fdtd = np.array([_interp_z(freqs_design, z_fdtd_all, f) for f in f_bem], dtype=np.complex128)

        print(
            f"\n=== BEM solves (per-f mesh epw={_BEM_EPW}, h_max={_BEM_H_MAX_M * 1e3:.0f} mm) ===",
            flush=True,
        )
        mesh_cache = BoxMeshCache()
        W = float(d.realized_W)
        H = float(d.realized_H)
        D = float(d.realized_D)
        z_bem10: dict[float, complex] = {}
        sol_cache: dict[float, object] = {}
        f_order = np.sort(f_bem)[::-1]
        n_bem = len(f_order)
        t_bem0 = time.perf_counter()
        dt_first = None
        for i, f_hz in enumerate(f_order):
            f_key = float(f_hz)
            h_tgt = h_target_from_frequency(f_key, c, epw=_BEM_EPW, h_max=_BEM_H_MAX_M)
            grid = mesh_cache.get(W, H, D, rad_rect, origin, h_tgt)
            k = 2.0 * math.pi * f_key / c
            t0 = time.perf_counter()
            sol = solve_neumann_bm(grid, k, vn_P0=1.0, rho=rho, c=c, log_dof=False)
            dt = time.perf_counter() - t0
            wall_bem_est = None
            if i == 0:
                dt_first = dt
                wall_bem_est = dt * n_bem
                print(
                    f"  estimated BEM wall time (×{n_bem} freqs, worst case): {wall_bem_est:.0f} s",
                    flush=True,
                )
            z_bem10[f_key] = sol.zrad()
            sol_cache[f_key] = sol
            elapsed = time.perf_counter() - t_bem0
            if i == 0 and wall_bem_est is not None:
                eta = wall_bem_est - dt
            elif i + 1 < n_bem:
                eta = (elapsed / (i + 1)) * (n_bem - i - 1)
            else:
                eta = None
            if sol.stats is not None:
                print(
                    format_bem_progress_line(
                        index=i + 1,
                        total=n_bem,
                        f_hz=f_key,
                        h_m=h_tgt,
                        stats=sol.stats,
                        elapsed_s=elapsed,
                        eta_s=eta,
                    ),
                    flush=True,
                )
        print(f"  BEM batch: {time.perf_counter() - t_bem0:.1f} s", flush=True)

        z_bem_arr = np.array([z_bem10[float(f)] for f in f_bem], dtype=np.complex128)
        err_g1 = np.abs(z_fdtd - z_bem_arr)
        g1_max = float(np.max(err_g1))
        g1_pass = g1_max <= _G1_TOL
        gates["g1"] = {"max_err": g1_max, "tol": _G1_TOL, "pass": g1_pass}

        f_g3 = np.sort(f_bem)[-_G3_N_TOP:]
        print(
            f"\n=== G3 BEM mesh convergence (epw {_BEM_EPW} vs {_G3_EPW}, "
            f"top {_G3_N_TOP} freqs) ===",
            flush=True,
        )
        cache_g3 = BoxMeshCache()
        dz: list[float] = []
        n_g3 = len(f_g3)
        t_g3 = time.perf_counter()
        for i, f_hz in enumerate(np.sort(f_g3)[::-1]):
            f_key = float(f_hz)
            h14 = h_target_from_frequency(f_key, c, epw=_G3_EPW, h_max=_BEM_H_MAX_M)
            grid14 = cache_g3.get(W, H, D, rad_rect, origin, h14)
            k = 2.0 * math.pi * f_key / c
            sol14 = solve_neumann_bm(grid14, k, vn_P0=1.0, rho=rho, c=c, log_dof=False)
            z14 = sol14.zrad()
            dz.append(abs(z_bem10[f_key] - z14))
            elapsed = time.perf_counter() - t_g3
            eta = (elapsed / (i + 1)) * (n_g3 - i - 1) if i + 1 < n_g3 else None
            if sol14.stats is not None:
                print(
                    format_bem_progress_line(
                        index=i + 1,
                        total=n_g3,
                        f_hz=f_key,
                        h_m=h14,
                        stats=sol14.stats,
                        elapsed_s=elapsed,
                        eta_s=eta,
                        prefix="bem G3",
                    ),
                    flush=True,
                )
        g3_max = float(np.max(dz))
        g3_pass = g3_max <= _G3_TOL
        gates["g3"] = {
            "max_err": g3_max,
            "tol": _G3_TOL,
            "epw_coarse": _BEM_EPW,
            "epw_fine": _G3_EPW,
            "freqs_hz": [float(f) for f in f_g3],
            "pass": g3_pass,
        }

        print("\n=== G2 far field (KH FDTD vs BEM) ===", flush=True)
        g2_detail: dict[str, Any] = {}
        g2_pass = True
        r_far = 10.0 * L
        planes = ("xz", "yz", "diag45")
        f_report = float(f_bem[len(f_bem) // 2])

        for kl in _G2_KL:
            f_hz = kl * c / (2.0 * math.pi * L)
            key = f"kl{kl:g}"
            patches, V, p_face, vn_face, _ = _load_huygens_spectrum(h5, cfg, d, f_hz)
            i_near = int(np.argmin(np.abs(f_bem - f_hz)))
            f_key = float(f_bem[i_near])
            sol = sol_cache.get(f_key)
            if sol is None:
                h_tgt = h_target_from_frequency(f_key, c, epw=_BEM_EPW, h_max=_BEM_H_MAX_M)
                grid = mesh_cache.get(W, H, D, rad_rect, origin, h_tgt)
                k = 2.0 * math.pi * f_hz / c
                sol = solve_neumann_bm(grid, k, vn_P0=1.0, rho=rho, c=c, log_dof=False)
                sol_cache[f_key] = sol
            g2_detail[key] = {}
            for plane in planes:
                theta, db_fdtd, H_fdtd = _directivity_cut(
                    plane, r_far, f_hz,
                    patches=patches, p_face=p_face, vn_face=vn_face, V=V, rho=rho, c=c,
                )
                _, db_bem, H_bem = _bem_directivity(
                    plane, r_far, f_hz, sol=sol, V_fdtd=V, c=c,
                )
                on_i = int(np.argmin(np.abs(theta)))
                db_ax, ph_ax = _h_err(H_bem[on_i], H_fdtd[on_i])
                level = db_fdtd >= _G2_DIR_LEVEL_DB
                ddir = db_fdtd[level] - db_bem[level]
                dir_max = float(np.max(np.abs(ddir))) if np.any(level) else 0.0
                ok_ax = abs(db_ax) <= _G2_DB and abs(ph_ax) <= _G2_PHASE_DEG
                ok_dir = dir_max <= _G2_DIR_DB
                ok = ok_ax and ok_dir
                g2_pass = g2_pass and ok
                g2_detail[key][plane] = {
                    "on_axis_db_err": db_ax,
                    "on_axis_phase_deg": ph_ax,
                    "dir_max_db_err": dir_max,
                    "pass": ok,
                }
                print(
                    f"  kL={kl:g} {plane}: on-axis Δ={db_ax:+.2f} dB ∠{ph_ax:.1f}°  "
                    f"dir max |ΔdB|={dir_max:.3f}",
                    flush=True,
                )
                if not args.no_plot:
                    _plot_dir(
                        args.out / f"c09_dir_{key}_{plane}_{stamp}.png",
                        plane=plane,
                        kl=kl,
                        curves={"FDTD KH": db_fdtd, "BEM": db_bem},
                        theta=theta,
                    )
        gates["g2"] = g2_detail

        print(f"\n=== Surface pressure report @ f≈{f_report:.1f} Hz ===", flush=True)
        h_rep = h_target_from_frequency(f_report, c, epw=_BEM_EPW, h_max=_BEM_H_MAX_M)
        grid_rep = mesh_cache.get(W, H, D, rad_rect, origin, h_rep)
        k_rep = 2.0 * math.pi * f_report / c
        sol_rep = solve_neumann_bm(grid_rep, k_rep, vn_P0=1.0, rho=rho, c=c, log_dof=False)
        centers_fd, p_fd = _load_box_p_spectrum(h5, f_report)
        surf_points = _surface_report_points(origin, d.realized_W, d.realized_H, d.realized_D)
        for name, pt in surf_points:
            p_bem = complex(sol_rep.field_at(pt[np.newaxis, :])[0])
            j = int(np.argmin(np.sum((centers_fd - pt) ** 2, axis=1)))
            p_f = complex(p_fd[j])
            db, ph = _h_err(p_bem, p_f)
            print(f"  {name}: Δ={db:+.2f} dB  ∠{ph:+.1f}°", flush=True)

        if not args.no_plot:
            _plot_z(args.out / f"c09_z_{stamp}.png", f_bem, z_fdtd, z_bem_arr)

    print("\n=== C9 gates ===", flush=True)
    _gate_line("G0", True, "sphere self-check (pre-abort)")
    _gate_line("G1", g1_pass, f"max |z_FDTD−z_BEM|={g1_max:.4f} (tol ≤{_G1_TOL})")
    _gate_line("G2", g2_pass, f"far field kL={_G2_KL} r=10L (tol {_G2_DB} dB, {_G2_PHASE_DEG}°, dir {_G2_DIR_DB} dB)")
    _gate_line(
        "G3",
        g3_pass,
        f"max |z_epw{_BEM_EPW}−z_epw{_G3_EPW}|={g3_max:.4f} (tol ≤{_G3_TOL})",
    )
    overall = g1_pass and g2_pass and g3_pass
    print(f"  overall: {'PASS' if overall else 'FAIL'}", flush=True)

    payload: dict[str, Any] = {
        "config": str(args.config),
        "gates": gates,
        "g0_ka": list(_G0_KA),
        "bem_freqs_hz": f_bem.tolist(),
        "dt_first_bem_s": dt_first,
        "pass": overall,
        "phases_s": timer.phases,
    }
    out_json = args.out / f"c09_bempp_{stamp}.json"
    out_json.write_text(json.dumps(vc.to_json_safe(payload), indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())