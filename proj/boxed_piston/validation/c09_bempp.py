#!/usr/bin/env python3
"""C9-bempp: box radiation vs exterior Neumann BEM (bempp-cl Burton–Miller)."""

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

ROOT = Path(__file__).resolve().parents[1]
# ``validation`` package import (install boxrad separately).
sys.path.insert(0, str(ROOT))

from boxrad.config import (  # noqa: E402
    Config,
    DEFAULT_RAM_LIMIT_BYTES,
    derive,
    estimate_memory_bytes,
)
from boxrad.grid import Grid  # noqa: E402
from boxrad.post.bem_ref import (  # noqa: E402
    BoxMeshCache,
    c09_bem_cache_path,
    c09_geometry_signature,
    dedupe_frequencies_hz,
    format_bem_progress_line,
    h_target_from_frequency,
    h_target_sphere,
    load_c09_bem_cache,
    save_c09_bem_cache,
    NeumannBmResult,
    pulsating_sphere_H_transfer,
    solve_neumann_bm,
    zrad_sphere_analytic,
)
from boxrad.post.kirchhoff import (  # noqa: E402
    dpdn_from_vn,
    kh_pressure,
    kh_pressure_loop,
)
from boxrad.post.spectra import spectrum  # noqa: E402
from boxrad.post.zrad import causal_end_taper, zrad_from_h5  # noqa: E402
from boxrad.runner import estimate_output_bytes, run  # noqa: E402
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
_G3_EPW = 17
_G3_N_TOP = 3
_G3_RICH_P = 2.0
_GEOM_TOL = 1e-9
_BEM_F_MIN_HZ = 100.0
_N_LOG_BEM = 12
_BEM_EPW = 12
_BEM_H_MAX_M = 0.02
_C1_UPDATES_PER_S = 375.0e6
_KH_NORM_DB = 0.2
_KH_NORM_PHASE_DEG = 2.0
_H_REFINE_M = 0.0025
_H_REFINE_3_M = 0.2 / 120.0
_F_IM_DIAG_MAX_HZ = 1200.0
_G3_LOW_H_MAX_M = 0.01
_G3_LOW_RAD_DIV = 16
_G3_LOW_FREQS_HZ = (252.84318436080457, 639.2967587771188)
_G1X_TOL = 0.005
_G1X_FREQS_HZ = _G3_LOW_FREQS_HZ
_G2_PLANES = ("xz", "yz", "diag45")
_BEM_CACHE_DIR = ROOT / "validation" / "results" / "bem_cache"


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


def _directivity_cut(
    plane: str,
    r_m: float,
    f_hz: float,
    *,
    patches: object,
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


def _g2_theta_and_points(
    plane: str, r_m: float, *, n_theta: int = 181
) -> tuple[np.ndarray, np.ndarray]:
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
    return theta, pts


def _bem_H_pa(
    h: complex | np.ndarray, *, rho: float, c: float
) -> complex | np.ndarray:
    """Pa/(m/s) transfer for FDTD compare: ρc times BEM dimensionless ``transfer_at``."""
    scale = float(rho) * float(c)
    if isinstance(h, np.ndarray):
        return np.asarray(h, dtype=np.complex128) * scale
    return complex(scale * h)


def _H_to_dir_db(H: np.ndarray, theta: np.ndarray) -> np.ndarray:
    axis_i = int(np.argmin(np.abs(theta)))
    H0 = H[axis_i]
    rel = np.abs(H / H0)
    return 20.0 * np.log10(np.maximum(rel, 1e-30))


def _bem_directivity_from_h(
    plane: str,
    r_m: float,
    *,
    H_obs: np.ndarray,
    rho: float,
    c: float,
    n_theta: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BEM directivity from cached dimensionless ``transfer_at``; returns Pa/(m/s) H."""
    theta = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_theta)
    H = np.asarray(_bem_H_pa(H_obs, rho=rho, c=c), dtype=np.complex128).ravel()
    db = _H_to_dir_db(H, theta)
    return theta, db, H


def _bem_directivity(
    plane: str,
    r_m: float,
    *,
    sol: NeumannBmResult,
    rho: float,
    c: float,
    n_theta: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, pts = _g2_theta_and_points(plane, r_m, n_theta=n_theta)
    h_dim = np.asarray(sol.transfer_at(pts, on_surface=False), dtype=np.complex128).ravel()
    H = np.asarray(_bem_H_pa(h_dim, rho=rho, c=c), dtype=np.complex128).ravel()
    theta = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_theta)
    db = _H_to_dir_db(H, theta)
    return theta, db, H


def _nearest_kl_key(f_hz: float, L: float, c: float) -> str | None:
    for kl in _G2_KL:
        f_kl = kl * c / (2.0 * math.pi * L)
        if abs(f_hz - f_kl) / max(f_kl, 1.0) < 0.02:
            return f"kl{kl:g}"
    return None


def _g2_h_farfield(sol: NeumannBmResult, r_m: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for plane in _G2_PLANES:
        _, pts = _g2_theta_and_points(plane, r_m)
        out[plane] = np.asarray(
            sol.transfer_at(pts, on_surface=False), dtype=np.complex128
        ).ravel()
    return out


def _ensure_bem_sol(
    f_key: float,
    *,
    sol_cache: dict[float, NeumannBmResult],
    mesh_cache: BoxMeshCache,
    geo: dict[str, Any],
    W: float,
    H: float,
    D: float,
    rad_rect: tuple[float, float],
    origin: tuple[float, float, float],
    rho: float,
    c: float,
    epw: float,
) -> NeumannBmResult:
    sol = sol_cache.get(f_key)
    if sol is not None:
        return sol
    h_max = float(geo["h_max"])
    rad_div = int(geo.get("rad_div", 8))
    h_rad = min(rad_rect[0], rad_rect[1]) / float(rad_div)
    h_tgt = h_target_from_frequency(f_key, c, epw=epw, h_max=h_max)
    grid = mesh_cache.get(W, H, D, rad_rect, origin, h_tgt, h_rad=h_rad)
    k = 2.0 * math.pi * f_key / c
    sol = solve_neumann_bm(grid, k, vn_P0=1.0, rho=rho, c=c, log_dof=False)
    assert isinstance(sol, NeumannBmResult)
    sol_cache[f_key] = sol
    return sol


def _solve_bem_frequency(
    *,
    f_hz: float,
    epw: float,
    geo: dict[str, Any],
    cache_dir: Path,
    mesh_cache: BoxMeshCache,
    W: float,
    H: float,
    D: float,
    rad_rect: tuple[float, float],
    origin: tuple[float, float, float],
    rho: float,
    c: float,
    reuse: bool,
    surf_points: list[tuple[str, np.ndarray]] | None,
    r_far: float | None,
    L: float,
) -> tuple[NeumannBmResult | None, dict[str, Any], Path]:
    f_key = float(f_hz)
    path = c09_bem_cache_path(cache_dir, geo, epw=epw, f_hz=f_key)
    if reuse and path.is_file():
        try:
            cached = load_c09_bem_cache(path)
            if cached.get("post_complete", True) and cached.get("z_rad") is not None:
                print(f"[bem] f={f_key:.2f} Hz cache hit ({path.name})", flush=True)
                return None, cached, path
        except ValueError:
            print(f"[bem] f={f_key:.2f} Hz cache miss (stale or incomplete)", flush=True)
    else:
        print(f"[bem] f={f_key:.2f} Hz cache miss", flush=True)

    h_max = float(geo["h_max"])
    rad_div = int(geo.get("rad_div", 8))
    h_rad = min(rad_rect[0], rad_rect[1]) / float(rad_div)
    h_tgt = h_target_from_frequency(f_key, c, epw=epw, h_max=h_max)
    grid = mesh_cache.get(W, H, D, rad_rect, origin, h_tgt, h_rad=h_rad)
    k = 2.0 * math.pi * f_key / c
    sol = solve_neumann_bm(grid, k, vn_P0=1.0, rho=rho, c=c, log_dof=False)
    assert isinstance(sol, NeumannBmResult)
    n_dof = sol.stats.n_dofs if sol.stats is not None else 0
    p1_coeffs = np.asarray(sol.p_gf.coefficients, dtype=np.complex128).ravel()
    save_c09_bem_cache(
        path,
        f_hz=f_key,
        epw=epw,
        h_m=h_tgt,
        n_dof=n_dof,
        p1_coefficients=p1_coeffs,
        post_complete=False,
    )
    z_rad = sol.zrad()
    surface_names: list[str] = []
    surface_p: list[complex] = []
    if surf_points is not None:
        surface_names = [n for n, _ in surf_points]
        pts = np.asarray([pt for _, pt in surf_points], dtype=np.float64)
        h_surf = sol.transfer_at(pts, on_surface=True)
        # Cache field ``surface_p``: dimensionless ``transfer_at`` (same as v5 far-field cache).
        surface_p = [complex(h_surf[i]) for i in range(len(surf_points))]

    g2_planes: list[str] | None = None
    g2_theta: np.ndarray | None = None
    g2_p: dict[str, np.ndarray] | None = None
    if r_far is not None and _nearest_kl_key(f_key, L, c) is not None:
        g2_planes = list(_G2_PLANES)
        g2_theta, _ = _g2_theta_and_points(g2_planes[0], r_far)
        g2_p = _g2_h_farfield(sol, r_far)

    save_c09_bem_cache(
        path,
        f_hz=f_key,
        epw=epw,
        h_m=h_tgt,
        n_dof=n_dof,
        p1_coefficients=p1_coeffs,
        z_rad=z_rad,
        surface_names=surface_names,
        surface_p=surface_p,
        g2_planes=g2_planes,
        g2_theta=g2_theta,
        g2_p_by_plane=g2_p,
        post_complete=True,
    )
    data = load_c09_bem_cache(path)
    return sol, data, path


def _print_g1_table(
    f_bem: np.ndarray,
    z_fdtd: np.ndarray,
    z_bem: np.ndarray,
    *,
    L: float,
    c: float,
    label: str,
    err_tol: float | None = None,
) -> tuple[float, int, list[dict[str, Any]]]:
    err = np.abs(z_fdtd - z_bem)
    g1_max = float(np.max(err))
    i_max = int(np.argmax(err))
    two_pi = 2.0 * math.pi
    rows: list[dict[str, Any]] = []
    print(f"\n=== G1-report: default-resolution accuracy ({label}) ===", flush=True)
    print(
        f"  {'f':>10} {'kL':>8} {'|Δz|':>8} {'Re rat':>8} {'Im rat':>8}  "
        f"z_FDTD              z_BEM",
        flush=True,
    )
    for i, f_hz in enumerate(f_bem):
        f_key = float(f_hz)
        kL = f_key * two_pi * L / c
        zf = complex(z_fdtd[i])
        zb = complex(z_bem[i])
        re_rat = float(np.real(zf) / np.real(zb)) if abs(np.real(zb)) > 1e-30 else float("nan")
        im_rat = float(np.imag(zf) / np.imag(zb)) if abs(np.imag(zb)) > 1e-30 else float("nan")
        dz = float(err[i])
        rows.append(
            {
                "f_hz": f_key,
                "kL": kL,
                "dz": dz,
                "re_ratio_fdtd_over_bem": re_rat,
                "im_ratio_fdtd_over_bem": im_rat,
                "z_fdtd": [float(zf.real), float(zf.imag)],
                "z_bem": [float(zb.real), float(zb.imag)],
            }
        )
        print(
            f"  {f_key:10.2f} {kL:8.4f} {dz:8.4f} {re_rat:8.4f} {im_rat:8.4f}  "
            f"{zf.real:+.4f}{zf.imag:+.4f}j  {zb.real:+.4f}{zb.imag:+.4f}j",
            flush=True,
        )
    if err_tol is not None:
        print(
            f"  max |Δz|={g1_max:.4f} at f={float(f_bem[i_max]):.2f} Hz "
            f"(report tol ≤{err_tol})",
            flush=True,
        )
    else:
        print(
            f"  max |Δz|={g1_max:.4f} at f={float(f_bem[i_max]):.2f} Hz",
            flush=True,
        )
    return g1_max, i_max, rows


def _im_excess_scalar(z_fdtd: complex, z_bem: complex) -> float:
    im_b = float(np.imag(z_bem))
    if abs(im_b) < 1e-30:
        return float("nan")
    return float(np.imag(z_fdtd) / im_b - 1.0)


def _order_from_excess_pair(
    h_coarse: float, h_fine: float, e_coarse: float, e_fine: float
) -> float:
    if not (
        math.isfinite(e_coarse)
        and math.isfinite(e_fine)
        and e_coarse > 0.0
        and e_fine > 0.0
    ):
        return float("nan")
    if h_fine <= 0.0 or h_coarse <= 0.0 or abs(h_coarse - h_fine) < 1e-30:
        return float("nan")
    return float(math.log(e_fine / e_coarse) / math.log(h_fine / h_coarse))


def _fdtd_im_excess_order(
    h_coarse: float, h_mid: float, h_fine: float, e_c: float, e_m: float, e_f: float
) -> float:
    p_cm = _order_from_excess_pair(h_coarse, h_mid, e_c, e_m)
    p_mf = _order_from_excess_pair(h_mid, h_fine, e_m, e_f)
    vals = [p for p in (p_cm, p_mf) if math.isfinite(p)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _mean_im_excess(
    f_bem: np.ndarray,
    z_fdtd: np.ndarray,
    z_bem: np.ndarray,
    *,
    f_max_hz: float,
) -> float:
    mask = f_bem <= float(f_max_hz) + 1e-9
    ratios: list[float] = []
    for zf, zb in zip(z_fdtd[mask], z_bem[mask], strict=True):
        im_b = float(np.imag(zb))
        if abs(im_b) < 1e-30:
            continue
        ratios.append(float(np.imag(zf) / im_b) - 1.0)
    if not ratios:
        return float("nan")
    return float(np.mean(ratios))


def _box_surface_radiator_mask(
    centers: np.ndarray,
    *,
    rw: float,
    rh: float,
    z_front: float,
    h: float,
) -> np.ndarray:
    tol_z = max(0.51 * float(h), 1e-12)
    return (
        (np.abs(centers[:, 2] - float(z_front)) <= tol_z)
        & (np.abs(centers[:, 0]) <= 0.5 * float(rw) + 1e-9)
        & (np.abs(centers[:, 1]) <= 0.5 * float(rh) + 1e-9)
    )


def _print_absolute_scale_diagnostics(
    *,
    f_hz: float,
    rho: float,
    c: float,
    rw: float,
    rh: float,
    z_front: float,
    sol: NeumannBmResult,
    z_bem: complex,
    h5: Path,
    d: object,
    L: float,
    r_far: float,
    z_fdtd: complex,
    plane: str,
    patches: object,
    p_face: np.ndarray,
    vn_face: np.ndarray,
    V_fdtd: complex,
    bdata: dict[str, Any],
) -> dict[str, Any]:
    """Print on-surface and on-axis absolute H (Pa/(m/s)) checks at one frequency."""
    scale = float(rho * c)
    mean_p = sol.radiator_mean_pressure()
    z_from_area = mean_p / scale
    rel_z = abs(z_from_area - z_bem) / max(abs(z_bem), 1e-30)
    if rel_z > 1e-3:
        raise RuntimeError(
            f"BEM area-mean ⟨p⟩_rad/(ρc) vs z_BEM: relative error {rel_z:.4e} > 1e-3"
        )
    ctr_s = np.array([[0.0, 0.0, float(z_front)]], dtype=np.float64)
    off = np.array([[0.0, 0.0, float(z_front) + 1e-3]], dtype=np.float64)
    H_surf = complex(
        _bem_H_pa(sol.transfer_at(ctr_s, on_surface=True)[0], rho=rho, c=c)
    )
    H_off = complex(
        _bem_H_pa(sol.transfer_at(off, on_surface=False)[0], rho=rho, c=c)
    )

    centers_fd, p_fd = vc.load_box_surface_spectrum(h5, f_hz)
    V_rep = vc.load_radiator_velocity_spectrum(h5, d, f_hz)
    rad_mask = _box_surface_radiator_mask(
        centers_fd, rw=rw, rh=rh, z_front=z_front, h=float(d.h)  # type: ignore[attr-defined]
    )
    H_rad_fd = p_fd[rad_mask] / V_rep
    mean_H_fd = complex(np.mean(H_rad_fd)) if np.any(rad_mask) else complex(np.nan)
    z_ref_fd = scale * z_fdtd

    k = 2.0 * math.pi * f_hz / c
    S = float(rw * rh)
    H_mono = scale * k * S / (4.0 * math.pi * r_far)
    _, _, H_fd_axis = _directivity_cut(
        plane,
        r_far,
        f_hz,
        patches=patches,
        p_face=p_face,
        vn_face=vn_face,
        V=V_fdtd,
        rho=rho,
        c=c,
    )
    _, pts_ax = _g2_theta_and_points(plane, r_far)
    h_bem_axis = np.asarray(
        sol.transfer_at(pts_ax, on_surface=False), dtype=np.complex128
    ).ravel()
    H_bem_axis = np.asarray(_bem_H_pa(h_bem_axis, rho=rho, c=c), dtype=np.complex128)
    i_ax = int(np.argmin(np.abs(_g2_theta_and_points(plane, r_far)[0])))
    H_fd0 = complex(H_fd_axis[i_ax])
    H_bem0 = complex(H_bem_axis[i_ax])
    mono_ratio = abs(H_bem0) / max(float(H_mono), 1e-30)
    if mono_ratio > 2.0 or mono_ratio < 0.5:
        raise RuntimeError(
            f"on-axis |H_BEM|/monopole ρckS/(4πr)={mono_ratio:.4f} outside factor 2 "
            f"at kL=2"
        )

    print(f"\n=== Absolute-scale diagnostics @ f={f_hz:.2f} Hz ===", flush=True)
    print(
        f"  BEM  mean⟨p⟩_rad/(ρc)={z_from_area.real:+.5f}{z_from_area.imag:+.5f}j  "
        f"z_BEM={z_bem.real:+.5f}{z_bem.imag:+.5f}j  "
        f"rel|Δz|={rel_z:.5f} (≤1e-3)",
        flush=True,
    )
    print(
        f"  BEM  H(center)  "
        f"{abs(H_surf):.5f}∠{np.rad2deg(np.angle(H_surf)):.1f}°  "
        f"H_off(1 mm)  {abs(H_off):.5f}∠{np.rad2deg(np.angle(H_off)):.1f}°",
        flush=True,
    )
    print(
        f"  FDTD mean H_rad patches   "
        f"{abs(mean_H_fd):.5f}∠{np.rad2deg(np.angle(mean_H_fd)):.1f}°  "
        f"ρc·z_FDTD={abs(z_ref_fd):.5f}  |Δz|={abs(mean_H_fd / scale - z_fdtd):.5f}",
        flush=True,
    )
    print(
        f"  kL={f_hz * 2.0 * math.pi * L / c:.4g} @r={r_far:.3g} m  "
        f"|H_FDTD|={abs(H_fd0):.5f}  |H_BEM|={abs(H_bem0):.5f}  "
        f"monopole ρckS/(4πr)={H_mono:.5f}  |H_BEM|/mono={mono_ratio:.4f}",
        flush=True,
    )
    return {
        "f_hz": float(f_hz),
        "bem_z_from_mean_surf": [float(z_from_area.real), float(z_from_area.imag)],
        "z_bem": [float(z_bem.real), float(z_bem.imag)],
        "fdtd_mean_H_rad": [float(mean_H_fd.real), float(mean_H_fd.imag)],
        "z_fdtd": [float(z_fdtd.real), float(z_fdtd.imag)],
        "H_surf_center": [float(H_surf.real), float(H_surf.imag)],
        "H_off_1mm": [float(H_off.real), float(H_off.imag)],
        "on_axis": {
            "H_fdtd": [float(H_fd0.real), float(H_fd0.imag)],
            "H_bem": [float(H_bem0.real), float(H_bem0.imag)],
            "H_mono_mag": float(H_mono),
        },
    }


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
        for r_mult in (2.0, 10.0):
            r_m = r_mult * a
            pt = np.array([[0.0, 0.0, r_m]], dtype=np.float64)
            H_bem = complex(sol.transfer_at(pt, on_surface=False)[0])
            H_ref = pulsating_sphere_H_transfer(r_m, a, ka)
            herr = abs(H_bem - H_ref) / max(abs(H_ref), 1e-30)
            ok_h = herr <= 0.02
            ok_all = ok_all and ok_h
            print(
                f"    field_at r={r_mult:g}a: rel |H−H_ref|={herr:.4f} "
                f"({'PASS' if ok_h else 'FAIL'})",
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
    parser.add_argument(
        "--reuse-bem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load per-frequency BEM .npz cache when present (default: on)",
    )
    parser.add_argument(
        "--h-refine",
        action="store_true",
        help="Extra FDTD at 2.5 mm and 5/3 mm; Im-excess order, Richardson z_∞",
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
    grid_plan = Grid(cfg, d_plan)
    cfg.output.probes = [
        list(x) for x in vc.box_mode_probe_targets(cfg, grid_plan, d_plan)
    ]
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

    cfgs_disk = [cfg]
    if args.h_refine:
        for h_ov in (_H_REFINE_M, _H_REFINE_3_M):
            cfg_h = copy.deepcopy(cfg)
            cfg_h.numerics.h_override = h_ov
            cfgs_disk.append(cfg_h)
    vc.require_h5_disk(
        sum(estimate_output_bytes(c, derive(c)) for c in cfgs_disk),
        vc.workdir_disk_parent(keep=args.keep_h5),
        label="c09_bempp",
    )

    gates: dict[str, Any] = {}
    g2_pass = g3_pass = kh_norm_pass = g1x_pass = True
    g1x_rows: list[dict[str, Any]] = []
    dt_first = None
    h_refine_diag: dict[str, Any] = {}

    with vc.run_workdir("c09_bempp", keep=args.keep_h5) as wd:
        h5_path = wd / "fdtd.h5"
        timer.banner("fdtd")
        timer.begin("fdtd")
        run(cfg, out_path=h5_path, progress=True, ram_limit_bytes=ram_limit)
        timer.end("fdtd")
        d = derive(cfg)
        rw, rh, L = _assert_geometry(cfg, d, label="c09")
        rad_rect = _radiator_rect(cfg)
        origin = _box_origin(cfg, d)

        freqs_design, z_fdtd_all = zrad_from_h5(h5_path, df_hz=2.0)
        f_bem = _bem_frequencies(cfg, L, c)

        z_fdtd = np.array([_interp_z(freqs_design, z_fdtd_all, f) for f in f_bem], dtype=np.complex128)
        z_fdtd_by_label: dict[str, np.ndarray] = {"h5mm": z_fdtd}

        if args.h_refine:
            for h_ov, label, h5_name, phase in (
                (_H_REFINE_M, "c09 h-refine 2.5 mm", "fdtd_h2p5mm.h5", "fdtd_refine_2p5"),
                (_H_REFINE_3_M, "c09 h-refine 5/3 mm", "fdtd_h5_3mm.h5", "fdtd_refine_5_3"),
            ):
                print(f"\n=== FDTD h-refine (h_override={h_ov * 1e3:g} mm) ===", flush=True)
                cfg_r = copy.deepcopy(cfg)
                cfg_r.numerics.h_override = h_ov
                cfg_r.output.probes = [
                    list(x)
                    for x in vc.box_mode_probe_targets(
                        cfg_r, Grid(cfg_r, derive(cfg_r)), derive(cfg_r)
                    )
                ]
                d_r = derive(cfg_r)
                est_r = estimate_memory_bytes(d_r.nx, d_r.ny, d_r.nz, d_r.cpml_cells)
                if est_r > ram_limit:
                    raise vc.RamLimitExceeded(
                        f"c09 FDTD refine {label}", d_r.nx, float(est_r), float(ram_limit)
                    )
                _assert_geometry(cfg_r, d_r, label=label)
                h5_r = wd / h5_name
                timer.begin(phase)
                run(cfg_r, out_path=h5_r, progress=True, ram_limit_bytes=ram_limit)
                timer.end(phase)
                freqs_r, z_ref_all = zrad_from_h5(h5_r, df_hz=2.0)
                z_fdtd_ref = np.array(
                    [_interp_z(freqs_r, z_ref_all, f) for f in f_bem], dtype=np.complex128
                )
                key = "h2p5mm" if h_ov == _H_REFINE_M else "h5_3mm"
                z_fdtd_by_label[key] = z_fdtd_ref

        W = float(d.realized_W)
        H = float(d.realized_H)
        D = float(d.realized_D)
        r_far = 10.0 * L
        surf_points = _surface_report_points(origin, W, H, D)
        geo = c09_geometry_signature(
            W,
            H,
            D,
            rad_rect,
            origin,
            h_max=_BEM_H_MAX_M,
            rho=rho,
            c=c,
            rad_div=8,
        )
        _BEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        print(
            f"\n=== BEM solves (per-f mesh epw={_BEM_EPW}, h_max={_BEM_H_MAX_M * 1e3:.0f} mm) "
            f"cache={_BEM_CACHE_DIR.name} reuse={args.reuse_bem} ===",
            flush=True,
        )
        mesh_cache = BoxMeshCache()
        bem_data: dict[float, dict[str, Any]] = {}
        sol_cache: dict[float, NeumannBmResult] = {}
        f_order = np.sort(f_bem)[::-1]
        n_bem = len(f_order)
        t_bem0 = time.perf_counter()
        for i, f_hz in enumerate(f_order):
            f_key = float(f_hz)
            t0 = time.perf_counter()
            sol, data, cpath = _solve_bem_frequency(
                f_hz=f_key,
                epw=_BEM_EPW,
                geo=geo,
                cache_dir=_BEM_CACHE_DIR,
                mesh_cache=mesh_cache,
                W=W,
                H=H,
                D=D,
                rad_rect=rad_rect,
                origin=origin,
                rho=rho,
                c=c,
                reuse=bool(args.reuse_bem),
                surf_points=surf_points,
                r_far=r_far,
                L=L,
            )
            dt = time.perf_counter() - t0
            if i == 0 and sol is not None:
                dt_first = dt
                print(
                    f"  estimated BEM wall time (×{n_bem} freqs, fresh solve): "
                    f"{dt * n_bem:.0f} s",
                    flush=True,
                )
            bem_data[f_key] = data
            if sol is not None:
                sol_cache[f_key] = sol
            elapsed = time.perf_counter() - t_bem0
            eta = (elapsed / (i + 1)) * (n_bem - i - 1) if i + 1 < n_bem else None
            tag = "cached" if sol is None else "solved"
            print(
                f"[bem {i + 1}/{n_bem}] f={f_key:.2f} Hz {tag} "
                f"DOF={data['n_dof']} h={data['h_m'] * 1e3:.2f} mm "
                f"file={cpath.name} elapsed={elapsed:.1f}s"
                + (f" ETA={eta:.1f}s" if eta is not None else ""),
                flush=True,
            )
        print(f"  BEM batch: {time.perf_counter() - t_bem0:.1f} s", flush=True)

        z_bem_arr = np.array([complex(bem_data[float(f)]["z_rad"]) for f in f_bem], dtype=np.complex128)
        h_fdtd_m = float(d.h)  # type: ignore[attr-defined]
        g1_max, i_g1_max, g1_rows = _print_g1_table(
            f_bem,
            z_fdtd,
            z_bem_arr,
            L=L,
            c=c,
            label=f"FDTD h={h_fdtd_m * 1e3:g} mm vs BEM epw={_BEM_EPW:g}",
            err_tol=_G1_TOL,
        )

        z_front = float(origin[2] + D)
        f_kl2 = 2.0 * c / (2.0 * math.pi * L)
        i_kl2 = int(np.argmin(np.abs(f_bem - f_kl2)))
        f_kl2_key = float(f_bem[i_kl2])
        sol_kl2 = _ensure_bem_sol(
            f_kl2_key,
            sol_cache=sol_cache,
            mesh_cache=mesh_cache,
            geo=geo,
            W=W,
            H=H,
            D=D,
            rad_rect=rad_rect,
            origin=origin,
            rho=rho,
            c=c,
            epw=_BEM_EPW,
        )
        patches_kl2, V_kl2, p_face_kl2, vn_face_kl2 = vc.load_huygens_spectra(
            h5_path, d, f_kl2_key
        )
        abs_diag = _print_absolute_scale_diagnostics(
            f_hz=f_kl2_key,
            rho=rho,
            c=c,
            rw=float(rad_rect[0]),
            rh=float(rad_rect[1]),
            z_front=z_front,
            sol=sol_kl2,
            z_bem=complex(bem_data[f_kl2_key]["z_rad"]),
            h5=h5_path,
            d=d,
            L=L,
            r_far=r_far,
            z_fdtd=complex(z_fdtd[i_kl2]),
            plane="xz",
            patches=patches_kl2,
            p_face=p_face_kl2,
            vn_face=vn_face_kl2,
            V_fdtd=V_kl2,
            bdata=bem_data[f_kl2_key],
        )

        if args.h_refine:
            z_ref = z_fdtd_by_label["h2p5mm"]
            z_ref3 = z_fdtd_by_label["h5_3mm"]
            _print_g1_table(
                f_bem,
                z_ref,
                z_bem_arr,
                L=L,
                c=c,
                label=f"FDTD h={_H_REFINE_M * 1e3:g} mm vs BEM epw={_BEM_EPW:g}",
                err_tol=_G1_TOL,
            )
            _print_g1_table(
                f_bem,
                z_ref3,
                z_bem_arr,
                L=L,
                c=c,
                label=f"FDTD h={_H_REFINE_3_M * 1e3:g} mm vs BEM epw={_BEM_EPW:g}",
                err_tol=_G1_TOL,
            )
            ex_5 = _mean_im_excess(f_bem, z_fdtd, z_bem_arr, f_max_hz=_F_IM_DIAG_MAX_HZ)
            ex_25 = _mean_im_excess(f_bem, z_ref, z_bem_arr, f_max_hz=_F_IM_DIAG_MAX_HZ)
            ex_53 = _mean_im_excess(f_bem, z_ref3, z_bem_arr, f_max_hz=_F_IM_DIAG_MAX_HZ)
            p_mean = _fdtd_im_excess_order(
                h_fdtd_m, _H_REFINE_M, _H_REFINE_3_M, ex_5, ex_25, ex_53
            )
            h_refine_diag = {
                "f_max_hz": _F_IM_DIAG_MAX_HZ,
                "mean_im_excess_h5mm": ex_5,
                "mean_im_excess_h2p5mm": ex_25,
                "mean_im_excess_h5_3mm": ex_53,
                "p_im_excess_mean": p_mean,
                "h_levels_m": [h_fdtd_m, _H_REFINE_M, _H_REFINE_3_M],
            }
            print(
                f"\n=== FDTD reactance excess (mean Im(FDTD/BEM)−1, f≤{_F_IM_DIAG_MAX_HZ:g} Hz) ===",
                flush=True,
            )
            print(f"  h=5.0 mm:   excess={ex_5:.5f}", flush=True)
            print(f"  h=2.5 mm:   excess={ex_25:.5f}", flush=True)
            print(f"  h=5/3 mm:   excess={ex_53:.5f}", flush=True)
            if math.isfinite(p_mean):
                print(
                    f"  mean excess order p≈{p_mean:.3f} from successive Δ (r=2, 1.5; not through zero)",
                    flush=True,
                )
            richardson_rows: list[dict[str, Any]] = []
            z_fdtd_inf: dict[float, complex] = {}
            r_fine = float(_H_REFINE_M / _H_REFINE_3_M)
            print(
                f"\n=== FDTD Richardson z_FDTD,∞ (per-f excess order, r={r_fine:.4g}, "
                f"f≤{_F_IM_DIAG_MAX_HZ:g} Hz) ===",
                flush=True,
            )
            print(f"  {'f':>10} {'p_ex':>6} {'|Δz|_rep':>10}  z_∞", flush=True)
            for i, f_hz in enumerate(f_bem):
                if float(f_hz) > _F_IM_DIAG_MAX_HZ + 1e-9:
                    continue
                f_key = float(f_hz)
                z_c = complex(z_fdtd[i])
                z_m = complex(z_ref[i])
                z_f = complex(z_ref3[i])
                zb = complex(z_bem_arr[i])
                e_c = _im_excess_scalar(z_c, zb)
                e_m = _im_excess_scalar(z_m, zb)
                e_f = _im_excess_scalar(z_f, zb)
                p_f = _fdtd_im_excess_order(h_fdtd_m, _H_REFINE_M, _H_REFINE_3_M, e_c, e_m, e_f)
                if math.isfinite(p_f):
                    z_inf = vc.richardson_extrap_complex(z_m, z_f, r=r_fine, p=p_f)
                else:
                    z_inf = z_f
                z_fdtd_inf[f_key] = z_inf
                dz = abs(z_inf - zb)
                richardson_rows.append(
                    {
                        "f_hz": f_key,
                        "p_im_excess": p_f,
                        "dz_report": float(dz),
                        "z_inf": [float(z_inf.real), float(z_inf.imag)],
                        "z_bem": [float(zb.real), float(zb.imag)],
                    }
                )
                p_tag = f"{p_f:.2f}" if math.isfinite(p_f) else "—"
                print(
                    f"  {f_key:10.2f} {p_tag:>6} {dz:10.4f}  "
                    f"{z_inf.real:+.4f}{z_inf.imag:+.4f}j",
                    flush=True,
                )
            h_refine_diag["richardson"] = richardson_rows
            h_refine_diag["z_fdtd_inf"] = {
                str(k): [float(v.real), float(v.imag)] for k, v in z_fdtd_inf.items()
            }

        gates["g1_report"] = {
            "max_err": g1_max,
            "max_f_hz": float(f_bem[i_g1_max]),
            "report_tol": _G1_TOL,
            "rows": g1_rows,
            "h_refine": h_refine_diag if args.h_refine else None,
            "absolute_scale_kl2": abs_diag,
        }
        two_pi = 2.0 * math.pi

        f_g3 = np.sort(f_bem)[-_G3_N_TOP:]
        print(
            f"\n=== G3 BEM mesh convergence (epw {_BEM_EPW} vs {_G3_EPW}, "
            f"top {_G3_N_TOP} freqs) ===",
            flush=True,
        )
        mesh_g3 = BoxMeshCache()
        dz: list[float] = []
        n_g3 = len(f_g3)
        t_g3 = time.perf_counter()
        epw_ratio = float(_G3_EPW) / float(_BEM_EPW)
        g1_extrap: list[dict[str, Any]] = []
        print(
            f"\n  G1 Richardson (p={_G3_RICH_P:g}, epw {_BEM_EPW}→{_G3_EPW}, "
            f"r={epw_ratio:.4g}) at top {_G3_N_TOP} freqs:",
            flush=True,
        )
        for i, f_hz in enumerate(np.sort(f_g3)[::-1]):
            f_key = float(f_hz)
            _, data17, _ = _solve_bem_frequency(
                f_hz=f_key,
                epw=_G3_EPW,
                geo=geo,
                cache_dir=_BEM_CACHE_DIR,
                mesh_cache=mesh_g3,
                W=W,
                H=H,
                D=D,
                rad_rect=rad_rect,
                origin=origin,
                rho=rho,
                c=c,
                reuse=bool(args.reuse_bem),
                surf_points=None,
                r_far=None,
                L=L,
            )
            z17 = complex(data17["z_rad"])
            z12 = complex(bem_data[f_key]["z_rad"])
            dz.append(abs(z12 - z17))
            i_fd = int(np.argmin(np.abs(f_bem - f_key)))
            zf = complex(z_fdtd[i_fd])
            z_ex = vc.richardson_extrap_complex(
                z12, z17, r=epw_ratio, p=_G3_RICH_P
            )
            err_ex = abs(zf - z_ex)
            kL = f_key * two_pi * L / c
            g1_extrap.append(
                {
                    "f_hz": f_key,
                    "kL": kL,
                    "dz_extrap": err_ex,
                    "z_extrap": [float(z_ex.real), float(z_ex.imag)],
                }
            )
            print(
                f"    f={f_key:.2f} Hz kL={kL:.4f}: "
                f"|z_FDTD−z_BEM,extrap|={err_ex:.4f}",
                flush=True,
            )
            elapsed = time.perf_counter() - t_g3
            eta_s = (elapsed / (i + 1)) * (n_g3 - i - 1) if i + 1 < n_g3 else None
            print(
                f"[bem G3 {i + 1}/{n_g3}] f={f_key:.2f} Hz DOF={data17['n_dof']} "
                f"h={data17['h_m'] * 1e3:.2f} mm elapsed={elapsed:.1f}s"
                + (f" ETA={eta_s:.1f}s" if eta_s is not None else ""),
                flush=True,
            )
        g3_max = float(np.max(dz))
        g3_pass = g3_max <= _G3_TOL
        gates["g1_report"]["richardson_top_freqs"] = g1_extrap
        gates["g3"] = {
            "max_err": g3_max,
            "tol": _G3_TOL,
            "epw_coarse": _BEM_EPW,
            "epw_fine": _G3_EPW,
            "freqs_hz": [float(f) for f in f_g3],
            "pass": g3_pass,
        }

        print(
            f"\n=== G3-low BEM mesh (h_max={_G3_LOW_H_MAX_M * 1e3:g} mm, "
            f"rad_span/{_G3_LOW_RAD_DIV}) ===",
            flush=True,
        )
        geo_low = c09_geometry_signature(
            W,
            H,
            D,
            rad_rect,
            origin,
            h_max=_G3_LOW_H_MAX_M,
            rho=rho,
            c=c,
            rad_div=_G3_LOW_RAD_DIV,
        )
        mesh_g3_low = BoxMeshCache()
        g3_low_rows: list[dict[str, Any]] = []
        g3_low_pass = True
        bem_low_by_f: dict[float, dict[str, Any]] = {}
        for f_nom in _G3_LOW_FREQS_HZ:
            i_ref = int(np.argmin(np.abs(f_bem - float(f_nom))))
            f_key = float(f_bem[i_ref])
            _, data_low, cpath = _solve_bem_frequency(
                f_hz=f_key,
                epw=_BEM_EPW,
                geo=geo_low,
                cache_dir=_BEM_CACHE_DIR,
                mesh_cache=mesh_g3_low,
                W=W,
                H=H,
                D=D,
                rad_rect=rad_rect,
                origin=origin,
                rho=rho,
                c=c,
                reuse=bool(args.reuse_bem),
                surf_points=None,
                r_far=None,
                L=L,
            )
            z_low = complex(data_low["z_rad"])
            z_std = complex(bem_data[f_key]["z_rad"])
            dz = abs(z_low - z_std)
            ok = dz <= _G3_TOL
            g3_low_pass = g3_low_pass and ok
            row = {
                "f_hz": f_key,
                "dz": float(dz),
                "z_low_im": float(z_low.imag),
                "z_std_im": float(z_std.imag),
                "pass": ok,
            }
            g3_low_rows.append(row)
            bem_low_by_f[f_key] = data_low
            print(
                f"  f={f_key:.2f} Hz  |Δz|={dz:.5f} (tol ≤{_G3_TOL})  "
                f"Im z_low={z_low.imag:.5f}  Im z_std={z_std.imag:.5f}  "
                f"cache={cpath.name}",
                flush=True,
            )
        print(
            "\n=== BEM Richardson z_BEM,∞ (standard vs G3-low mesh) ===",
            flush=True,
        )
        bem_extrap: list[dict[str, Any]] = []
        for f_nom in _G1X_FREQS_HZ:
            i_ref = int(np.argmin(np.abs(f_bem - float(f_nom))))
            f_key = float(f_bem[i_ref])
            z_std = complex(bem_data[f_key]["z_rad"])
            z_low = complex(bem_low_by_f[f_key]["z_rad"])
            h_std = float(bem_data[f_key]["h_m"])
            h_low = float(bem_low_by_f[f_key]["h_m"])
            r_mesh = h_std / h_low if h_low > 0 else 1.0
            z_b1 = vc.richardson_extrap_complex(z_std, z_low, r=r_mesh, p=1.0)
            z_b2 = vc.richardson_extrap_complex(z_std, z_low, r=r_mesh, p=2.0)
            bem_extrap.append(
                {
                    "f_hz": f_key,
                    "r_mesh": r_mesh,
                    "z_bem_inf_p1": [float(z_b1.real), float(z_b1.imag)],
                    "z_bem_inf_p2": [float(z_b2.real), float(z_b2.imag)],
                }
            )
            print(
                f"  f={f_key:.2f} Hz  r={r_mesh:.4g}  "
                f"z_BEM,∞(p=1)={z_b1.real:+.5f}{z_b1.imag:+.5f}j  "
                f"z_BEM,∞(p=2)={z_b2.real:+.5f}{z_b2.imag:+.5f}j",
                flush=True,
            )
        gates["bem_extrap"] = bem_extrap

        if args.h_refine:
            print("\n=== G1x |z_FDTD,∞ − z_BEM,∞| (p=1 gate) ===", flush=True)
            for row in bem_extrap:
                f_key = float(row["f_hz"])
                z_b1 = complex(*row["z_bem_inf_p1"])
                z_b2 = complex(*row["z_bem_inf_p2"])
                z_fi = z_fdtd_inf.get(f_key)
                if z_fi is None:
                    g1x_pass = False
                    print(
                        f"  f={f_key:.2f} Hz: no FDTD z_∞ (f > {_F_IM_DIAG_MAX_HZ:g} Hz?) — FAIL",
                        flush=True,
                    )
                    continue
                err1 = abs(z_fi - z_b1)
                err2 = abs(z_fi - z_b2)
                ok = err1 <= _G1X_TOL
                g1x_pass = g1x_pass and ok
                g1x_rows.append(
                    {
                        "f_hz": f_key,
                        "dz_p1": float(err1),
                        "dz_p2": float(err2),
                        "pass": ok,
                    }
                )
                print(
                    f"  f={f_key:.2f} Hz  |z_FDTD,∞−z_BEM,∞(p=1)|={err1:.5f} "
                    f"(tol ≤{_G1X_TOL})  p=2 variant={err2:.5f}  "
                    f"({'PASS' if ok else 'FAIL'})",
                    flush=True,
                )
        gates["g1x"] = {
            "tol": _G1X_TOL,
            "requires_h_refine": True,
            "pass": g1x_pass if args.h_refine else None,
            "rows": g1x_rows,
        }

        gates["g3_low"] = {
            "tol": _G3_TOL,
            "h_max_m": _G3_LOW_H_MAX_M,
            "rad_div": _G3_LOW_RAD_DIV,
            "pass": g3_low_pass,
            "rows": g3_low_rows,
        }

        print("\n=== G2 far field (KH FDTD vs BEM) ===", flush=True)
        g2_detail: dict[str, Any] = {}
        g2_pass = True
        planes = _G2_PLANES
        f_report = float(f_bem[len(f_bem) // 2])

        for kl in _G2_KL:
            f_hz = kl * c / (2.0 * math.pi * L)
            key = f"kl{kl:g}"
            patches, V, p_face, vn_face = vc.load_huygens_spectra(h5_path, d, f_hz)
            i_near = int(np.argmin(np.abs(f_bem - f_hz)))
            f_key = float(f_bem[i_near])
            bdata = bem_data[f_key]
            g2_detail[key] = {}
            for plane in planes:
                theta, db_fdtd, H_fdtd = _directivity_cut(
                    plane, r_far, f_hz,
                    patches=patches, p_face=p_face, vn_face=vn_face, V=V, rho=rho, c=c,
                )
                if "g2_p_by_plane" in bdata and plane in bdata["g2_p_by_plane"]:
                    theta_b = np.asarray(bdata["g2_theta"], dtype=np.float64)
                    theta = theta_b
                    _, db_bem, H_bem = _bem_directivity_from_h(
                        plane,
                        r_far,
                        H_obs=bdata["g2_p_by_plane"][plane],
                        rho=rho,
                        c=c,
                    )
                else:
                    sol = sol_cache.get(f_key)
                    if sol is None:
                        sol, bdata, _ = _solve_bem_frequency(
                            f_hz=f_key,
                            epw=_BEM_EPW,
                            geo=geo,
                            cache_dir=_BEM_CACHE_DIR,
                            mesh_cache=mesh_cache,
                            W=W,
                            H=H,
                            D=D,
                            rad_rect=rad_rect,
                            origin=origin,
                            rho=rho,
                            c=c,
                            reuse=False,
                            surf_points=surf_points,
                            r_far=r_far,
                            L=L,
                        )
                        if sol is not None:
                            sol_cache[f_key] = sol
                        bem_data[f_key] = bdata
                    sol = sol_cache[f_key]
                    _, db_bem, H_bem = _bem_directivity(
                        plane, r_far, sol=sol, rho=rho, c=c
                    )
                on_i = int(np.argmin(np.abs(theta)))
                db_ax, ph_ax = vc.h_transfer_err(H_bem[on_i], H_fdtd[on_i])
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

        print(
            f"\n=== Surface transfer H=P/V report @ f≈{f_report:.1f} Hz ===",
            flush=True,
        )
        V_rep = vc.load_radiator_velocity_spectrum(h5_path, d, f_report)
        i_rep = int(np.argmin(np.abs(f_bem - f_report)))
        f_rep_key = float(f_bem[i_rep])
        rep_data = bem_data[f_rep_key]
        if "surface_p" not in rep_data or len(rep_data["surface_p"]) != len(surf_points):
            _, rep_data, _ = _solve_bem_frequency(
                f_hz=f_rep_key,
                epw=_BEM_EPW,
                geo=geo,
                cache_dir=_BEM_CACHE_DIR,
                mesh_cache=mesh_cache,
                W=W,
                H=H,
                D=D,
                rad_rect=rad_rect,
                origin=origin,
                rho=rho,
                c=c,
                reuse=bool(args.reuse_bem),
                surf_points=surf_points,
                r_far=None,
                L=L,
            )
            bem_data[f_rep_key] = rep_data
        sol_rep = _ensure_bem_sol(
            f_rep_key,
            sol_cache=sol_cache,
            mesh_cache=mesh_cache,
            geo=geo,
            W=W,
            H=H,
            D=D,
            rad_rect=rad_rect,
            origin=origin,
            rho=rho,
            c=c,
            epw=_BEM_EPW,
        )
        centers_fd, p_fd = vc.load_box_surface_spectrum(h5_path, f_report)
        for j, (name, pt) in enumerate(surf_points):
            H_bem = complex(
                _bem_H_pa(
                    sol_rep.transfer_at(
                        np.asarray([pt], dtype=np.float64), on_surface=True
                    )[0],
                    rho=rho,
                    c=c,
                )
            )
            k_fd = int(np.argmin(np.sum((centers_fd - pt) ** 2, axis=1)))
            H_fd = complex(p_fd[k_fd]) / V_rep
            db, ph = vc.h_transfer_err(H_bem, H_fd)
            print(f"  {name}: H_BEM vs H_FDTD Δ={db:+.2f} dB  ∠{ph:+.1f}°", flush=True)

        import h5py

        print(
            f"\n=== KH probe normalization @ f≈{f_report:.1f} Hz "
            f"(H_FDTD vs KH, tol {_KH_NORM_DB} dB / {_KH_NORM_PHASE_DEG}°) ===",
            flush=True,
        )
        patches_rep, V_kh, p_face_rep, vn_face_rep = vc.load_huygens_spectra(
            h5_path, d, f_report
        )
        assert abs(V_kh - V_rep) / max(abs(V_rep), 1e-30) < 1e-9
        with h5py.File(h5_path, "r") as f:
            if "probes" not in f:
                raise RuntimeError("C9 requires FDTD probes for KH normalization check")
            probe_centers = np.asarray(f["probes/centers"][()], dtype=np.float64)
            probe_p_ts = np.asarray(f["probes/p"][()], dtype=np.float64)
            t_probe = np.asarray(f["probes/times_p"][()], dtype=np.float64)
        win_pr = causal_end_taper(probe_p_ts.shape[0])
        omega = np.asarray([2.0 * math.pi * f_report], dtype=np.float64)
        k_sel = np.asarray([2.0 * math.pi * f_report / c], dtype=np.float64)
        dpdn = dpdn_from_vn(vn_face_rep[np.newaxis, :], omega, rho)
        p_row = p_face_rep[np.newaxis, :]
        qi = 0
        P_fd = complex(spectrum(probe_p_ts[:, qi] * win_pr, t_probe, f_report))
        H_fd = P_fd / V_kh
        p_kh = kh_pressure_loop(
            probe_centers[qi].reshape(1, 3),
            patches_rep,
            p_row,
            dpdn,
            k_sel,
        )[0, 0]
        H_kh = p_kh / V_kh
        db_kh, ph_kh = vc.h_transfer_err(H_fd, H_kh)
        kh_norm_pass = abs(db_kh) <= _KH_NORM_DB and abs(ph_kh) <= _KH_NORM_PHASE_DEG
        gates["kh_norm"] = {
            "f_hz": f_report,
            "probe": "front+z",
            "db_fdtd_vs_kh": db_kh,
            "phase_deg": ph_kh,
            "pass": kh_norm_pass,
        }
        print(
            f"  probe front+z: Δ={db_kh:+.3f} dB  ∠{ph_kh:+.2f}°  "
            f"({'PASS' if kh_norm_pass else 'FAIL'})",
            flush=True,
        )

        if not args.no_plot:
            _plot_z(args.out / f"c09_z_{stamp}.png", f_bem, z_fdtd, z_bem_arr)

    print("\n=== C9 gates ===", flush=True)
    _gate_line("G0", True, "sphere self-check (pre-abort)")
    _gate_line(
        "G1-report",
        True,
        f"default-resolution max |z_FDTD−z_BEM|={g1_max:.4f} (reference tol ≤{_G1_TOL})",
    )
    if gates.get("g1x", {}).get("pass") is not None:
        g1x_max = max((float(r["dz_p1"]) for r in g1x_rows), default=0.0)
        _gate_line(
            "G1x",
            bool(gates["g1x"]["pass"]),
            f"max |z_FDTD,∞−z_BEM,∞(p=1)|={g1x_max:.5f} (tol ≤{_G1X_TOL})",
        )
    _gate_line("G2", g2_pass, f"far field kL={_G2_KL} r=10L (tol {_G2_DB} dB, {_G2_PHASE_DEG}°, dir {_G2_DIR_DB} dB)")
    _gate_line(
        "G3",
        g3_pass,
        f"max |z_epw{_BEM_EPW}−z_epw{_G3_EPW}|={g3_max:.4f} (tol ≤{_G3_TOL})",
    )
    g3_low_max = max(
        (float(r["dz"]) for r in gates.get("g3_low", {}).get("rows", [])), default=0.0
    )
    _gate_line(
        "G3-low",
        g3_low_pass,
        f"max |Δz| vs standard mesh={g3_low_max:.5f} (tol ≤{_G3_TOL})",
    )
    _gate_line(
        "KH-norm",
        kh_norm_pass,
        f"probe H_FDTD vs KH @ report f (tol {_KH_NORM_DB} dB, {_KH_NORM_PHASE_DEG}°)",
    )
    g3_low_pass = bool(gates.get("g3_low", {}).get("pass", True))
    overall = g2_pass and g3_pass and g3_low_pass and kh_norm_pass
    if args.h_refine:
        overall = overall and g1x_pass
    print(f"  overall: {'PASS' if overall else 'FAIL'}", flush=True)

    payload: dict[str, Any] = {
        "config": str(args.config),
        "gates": gates,
        "g0_ka": list(_G0_KA),
        "bem_freqs_hz": f_bem.tolist(),
        "dt_first_bem_s": dt_first,
        "reuse_bem": bool(args.reuse_bem),
        "h_refine": bool(args.h_refine),
        "pass": overall,
        "phases_s": timer.phases,
    }
    out_json = args.out / f"c09_bempp_{stamp}.json"
    out_json.write_text(json.dumps(vc.to_json_safe(payload), indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())