#!/usr/bin/env python3
"""Step 10a2 gate C18(ii): FDTD (s3p/S4) vs BEM reference on C9 box."""

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
_OUT = _PKG / "outputs" / "step10a2"
_STEP10A = _PKG / "outputs" / "step10a"
_BEM_CACHE = _OUT / "bem_cache"
_C09_JSON = _PKG / "configs" / "c09_box.json"

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import Config  # noqa: E402
from boxrad.post.bem_ref import (  # noqa: E402
    BoxMeshCache,
    NeumannBmResult,
    c09_bem_cache_path,
    c09_geometry_signature,
    format_bem_progress_line,
    h_target_from_frequency,
    load_c09_bem_cache,
    save_c09_bem_cache,
    solve_neumann_bm,
)
from boxrad.runner import keep_awake  # noqa: E402
from validation._common import to_json_safe  # noqa: E402

_F64 = np.float64
_C128 = np.complex128
_FREQS_HZ = (300.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0)
_BEM_SELF_CHECK_HZ = (1000.0, 3000.0)
_BEM_EPW = 12.0
_BEM_H_MAX = 0.02
_BEM_RAD_DIV = 8
_BEM_EPW_REFINE = 18.0
_BEM_RAD_DIV_REFINE = 12
_BEM_RAD_REFINE_FREQS_HZ = (300.0, 1000.0)
_BEM_RAD_REFINE_H_RAD_M = (0.06 / 6.0, 0.06 / 12.0, 0.06 / 24.0)
_BEM_RAD_REFINE_H_RAD_REF_M = _BEM_RAD_REFINE_H_RAD_M[-1]
_FAR_M = 20.0
_FAR_PT = np.array([[0.0, 0.0, _FAR_M]], dtype=_F64)
_GATE_TILT_MARGIN = 0.01
_GATE_FF_DB = 0.2
_EDGE_REACTANCE_NOTE = (
    "radiator-edge reactance excess ≈1.5–2.6% at 10–12 cells across the short "
    "side, decreasing with resolution (C9)"
)
_A11_DIM_CORR = -0.0044


def _fdtd_keys(s4_suffix: str) -> tuple[tuple[str, str], ...]:
    return (
        ("s3p_h1", "s3p_h1.h5"),
        ("s3p_h3", "s3p_h3.h5"),
        ("S4_h1", f"s4_h1{s4_suffix}.h5"),
        ("S4_h3", f"s4_h3{s4_suffix}.h5"),
    )


def _load_c09_geometry() -> dict[str, Any]:
    cfg = Config.from_json(_C09_JSON.read_text(encoding="utf-8"))
    if cfg.box is None:
        raise RuntimeError("c09_box.json requires box geometry")
    W = float(cfg.box.W)
    H = float(cfg.box.H)
    D = float(cfg.box.D)
    rw, rh = float(cfg.radiator.dims[0]), float(cfg.radiator.dims[1])
    xc, yc = float(cfg.radiator.x_c), float(cfg.radiator.y_c)
    assert abs(W - 0.20) < 1e-15
    assert abs(H - 0.15) < 1e-15
    assert abs(D - 0.25) < 1e-15
    assert abs(rw - 0.08) < 1e-15
    assert abs(rh - 0.06) < 1e-15
    assert abs(xc - 0.03) < 1e-15
    assert abs(yc - 0.02) < 1e-15
    origin = (-xc - 0.5 * W, -yc - 0.5 * H, -D)
    assert abs(origin[0] - (-0.13)) < 1e-15
    assert abs(origin[1] - (-0.095)) < 1e-15
    assert abs(origin[2] - (-0.25)) < 1e-15
    rho = float(cfg.air.rho)
    c = float(cfg.air.c)
    return {
        "W": W,
        "H": H,
        "D": D,
        "rad_rect": (rw, rh),
        "origin": origin,
        "rho": rho,
        "c": c,
    }


def _wrap_phase_deg(ph: float) -> float:
    while ph > 180.0:
        ph -= 360.0
    while ph < -180.0:
        ph += 360.0
    return ph


def _z_errors(z_fd: complex, z_bem: complex) -> dict[str, float]:
    zb = complex(z_bem)
    den = max(abs(zb), 1e-30)
    dz = complex(z_fd) - zb
    return {
        "ez": float(abs(dz) / den),
        "dRe_rel": float(np.real(dz) / den),
        "dIm_rel": float(np.imag(dz) / den),
    }


def _h_fd_bem(
    H_fd: complex, H_bem: complex
) -> tuple[float, float]:
    ratio = complex(H_fd) / complex(H_bem)
    db = 20.0 * math.log10(max(abs(ratio), 1e-30))
    ph = _wrap_phase_deg(float(np.rad2deg(np.angle(ratio))))
    return float(db), ph


def _solve_bem_f(
    f_hz: float,
    *,
    epw: float,
    rad_div: int,
    geo: dict[str, Any],
    geom: dict[str, Any],
    mesh_cache: BoxMeshCache,
    sol_cache: dict[tuple[float, float, int], NeumannBmResult],
    reuse_cache: bool,
    cache_only: bool = False,
) -> NeumannBmResult:
    f_key = float(f_hz)
    cache_key = (f_key, float(epw), int(rad_div))
    if cache_key in sol_cache:
        return sol_cache[cache_key]
    W, H, D = geom["W"], geom["H"], geom["D"]
    origin = geom["origin"]
    rad_rect = geom["rad_rect"]
    rho, c = geom["rho"], geom["c"]
    rw, rh = rad_rect
    h_rad = min(rw, rh) / float(rad_div)
    h_tgt = h_target_from_frequency(f_key, c, epw=epw, h_max=_BEM_H_MAX)
    path = c09_bem_cache_path(_BEM_CACHE, geo, epw=epw, f_hz=f_key)
    if cache_only and not path.is_file():
        raise FileNotFoundError(
            f"BEM cache missing {path}; run step10a2 once without --s4-suffix to populate"
        )
    if reuse_cache and path.is_file():
        try:
            cached = load_c09_bem_cache(path)
            if cached.get("post_complete", True) and cached.get("z_rad") is not None:
                grid = mesh_cache.get(
                    W, H, D, rad_rect, origin, h_tgt, h_rad=h_rad
                )
                k = 2.0 * math.pi * f_key / c
                sol = solve_neumann_bm(
                    grid, k, vn_P0=1.0, rho=rho, c=c, log_dof=False
                )
                sol_cache[cache_key] = sol
                return sol
        except ValueError:
            pass
    if cache_only:
        raise FileNotFoundError(
            f"BEM cache unreadable or incomplete {path}; "
            "run step10a2 once without --s4-suffix to populate"
        )
    grid = mesh_cache.get(W, H, D, rad_rect, origin, h_tgt, h_rad=h_rad)
    k = 2.0 * math.pi * f_key / c
    sol = solve_neumann_bm(grid, k, vn_P0=1.0, rho=rho, c=c, log_dof=False)
    p1 = np.asarray(sol.p_gf.coefficients, dtype=np.complex128).ravel()
    z_rad = sol.zrad()
    save_c09_bem_cache(
        path,
        f_hz=f_key,
        epw=epw,
        h_m=h_tgt,
        n_dof=sol.stats.n_dofs if sol.stats else 0,
        p1_coefficients=p1,
        z_rad=z_rad,
        post_complete=True,
    )
    sol_cache[cache_key] = sol
    return sol


def _z_bem_cached(
    f_hz: float,
    *,
    epw: float,
    geo: dict[str, Any],
    sol: NeumannBmResult,
) -> complex:
    path = c09_bem_cache_path(_BEM_CACHE, geo, epw=epw, f_hz=f_hz)
    if path.is_file():
        try:
            data = load_c09_bem_cache(path)
            if data.get("z_rad") is not None:
                return complex(data["z_rad"])
        except ValueError:
            pass
    return sol.zrad()


def _H_bem(sol: NeumannBmResult, rho: float, c: float) -> complex:
    p = complex(sol.field_at(_FAR_PT)[0])
    return p / (rho * c)


def _bem_radiator_refinement_report(
    geom: dict[str, Any],
    mesh_cache: BoxMeshCache,
) -> dict[str, Any]:
    """C9 BEM solves varying radiator cap spacing only (no disk cache)."""
    W, H, D = geom["W"], geom["H"], geom["D"]
    origin = geom["origin"]
    rad_rect = geom["rad_rect"]
    rho, c = geom["rho"], geom["c"]
    rw, rh = rad_rect
    h_rad_main = min(rw, rh) / float(_BEM_RAD_DIV)
    print(
        f"\n=== BEM radiator refinement (report only; main h_rad="
        f"{h_rad_main * 1e3:.4g} mm) ===",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for f_hz in _BEM_RAD_REFINE_FREQS_HZ:
        f_key = float(f_hz)
        h_tgt = h_target_from_frequency(
            f_key, c, epw=_BEM_EPW, h_max=_BEM_H_MAX
        )
        solved: list[tuple[float, complex, int]] = []
        for h_rad in _BEM_RAD_REFINE_H_RAD_M:
            h_rad_f = float(h_rad)
            grid = mesh_cache.get(
                W, H, D, rad_rect, origin, h_tgt, h_rad=h_rad_f
            )
            k = 2.0 * math.pi * f_key / c
            sol = solve_neumann_bm(
                grid, k, vn_P0=1.0, rho=rho, c=c, log_dof=False
            )
            z = sol.zrad()
            n_dof = int(sol.stats.n_dofs) if sol.stats is not None else 0
            solved.append((h_rad_f, z, n_dof))
        im_ref = float(
            next(z for h, z, _ in solved if h == _BEM_RAD_REFINE_H_RAD_REF_M).imag
        )
        for h_rad_f, z, n_dof in solved:
            d_im = 0.0 if h_rad_f == _BEM_RAD_REFINE_H_RAD_REF_M else float(z.imag) - im_ref
            rows.append(
                {
                    "f_hz": f_key,
                    "h_tgt_m": float(h_tgt),
                    "h_rad_m": h_rad_f,
                    "h_rad_mm": float(h_rad_f * 1e3),
                    "n_dof": n_dof,
                    "z_real": float(z.real),
                    "z_imag": float(z.imag),
                    "dIm_vs_h_rad_finest": d_im,
                }
            )
            print(
                f"  bem-refine f={f_key:.0f} Hz h_rad={h_rad_f * 1e3:.4g} mm "
                f"DOF={n_dof}  Re z={z.real:+.6g}  Im z={z.imag:+.6g}  "
                f"dIm={d_im:+.6g}",
                flush=True,
            )
    return {
        "epw": _BEM_EPW,
        "h_max_m": _BEM_H_MAX,
        "main_h_rad_m": float(h_rad_main),
        "main_h_rad_mm": float(h_rad_main * 1e3),
        "h_rad_levels_m": list(_BEM_RAD_REFINE_H_RAD_M),
        "h_rad_ref_m": float(_BEM_RAD_REFINE_H_RAD_REF_M),
        "frequencies_hz": list(_BEM_RAD_REFINE_FREQS_HZ),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bem-refine",
        action="store_true",
        help="Report-only: radiator h_rad convergence at 300 and 1000 Hz",
    )
    parser.add_argument(
        "--s4-suffix",
        default="",
        metavar="SUFFIX",
        help='Load s4_h1{SUFFIX}.h5 / s4_h3{SUFFIX}.h5; write c18_bem{SUFFIX}.json/.png',
    )
    args = parser.parse_args(argv)
    s4_suffix = str(args.s4_suffix)
    fdtd_keys = _fdtd_keys(s4_suffix)
    bem_cache_only = bool(s4_suffix)
    out_stem = f"c18_bem{s4_suffix}"
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    _BEM_CACHE.mkdir(parents=True, exist_ok=True)
    geom = _load_c09_geometry()
    rho, c = geom["rho"], geom["c"]
    geo_bem = c09_geometry_signature(
        geom["W"],
        geom["H"],
        geom["D"],
        geom["rad_rect"],
        geom["origin"],
        h_max=_BEM_H_MAX,
        rho=rho,
        c=c,
        rad_div=_BEM_RAD_DIV,
    )

    for key, name in fdtd_keys:
        h5 = _STEP10A / name
        if not h5.is_file():
            hint = (
                f" showcase/step10a_c18.py --cfl … --s4-only"
                if s4_suffix
                else " showcase/step10a_c18.py first"
            )
            raise FileNotFoundError(f"Missing FDTD HDF5 {h5}; run{hint}")

    freqs = np.asarray(_FREQS_HZ, dtype=_F64)
    n_f = freqs.size
    mesh_cache = BoxMeshCache()
    sol_cache: dict[tuple[float, float, int], NeumannBmResult] = {}

    bem_z: dict[float, complex] = {}
    bem_H: dict[float, complex] = {}
    bem_h: dict[float, float] = {}

    with keep_awake("step10a2 C18(ii) BEM reference"):
        t_bem0 = time.perf_counter()
        for i, f_hz in enumerate(freqs):
            t_step = time.perf_counter()
            sol = _solve_bem_f(
                float(f_hz),
                epw=_BEM_EPW,
                rad_div=_BEM_RAD_DIV,
                geo=geo_bem,
                geom=geom,
                mesh_cache=mesh_cache,
                sol_cache=sol_cache,
                reuse_cache=True,
                cache_only=bem_cache_only,
            )
            bem_z[float(f_hz)] = _z_bem_cached(
                float(f_hz), epw=_BEM_EPW, geo=geo_bem, sol=sol
            )
            bem_H[float(f_hz)] = _H_bem(sol, rho, c)
            bem_h[float(f_hz)] = h_target_from_frequency(
                float(f_hz), c, epw=_BEM_EPW, h_max=_BEM_H_MAX
            )
            elapsed = time.perf_counter() - t_bem0
            eta = (elapsed / (i + 1)) * (n_f - i - 1) if i + 1 < n_f else None
            stats = sol.stats
            if stats is not None:
                print(
                    format_bem_progress_line(
                        index=i + 1,
                        total=n_f,
                        f_hz=float(f_hz),
                        h_m=bem_h[float(f_hz)],
                        stats=stats,
                        elapsed_s=elapsed,
                        eta_s=eta,
                        prefix="bem",
                    ),
                    flush=True,
                )
            else:
                print(
                    f"[bem {i + 1}/{n_f}] f={f_hz:.2f} Hz "
                    f"step={time.perf_counter() - t_step:.1f}s",
                    flush=True,
                )

        bem_self_check: list[dict[str, float]] = []
        for f_hz in _BEM_SELF_CHECK_HZ:
            z12 = bem_z[float(f_hz)]
            geo18 = c09_geometry_signature(
                geom["W"],
                geom["H"],
                geom["D"],
                geom["rad_rect"],
                geom["origin"],
                h_max=_BEM_H_MAX,
                rho=rho,
                c=c,
                rad_div=_BEM_RAD_DIV_REFINE,
            )
            sol18 = _solve_bem_f(
                float(f_hz),
                epw=_BEM_EPW_REFINE,
                rad_div=_BEM_RAD_DIV_REFINE,
                geo=geo18,
                geom=geom,
                mesh_cache=mesh_cache,
                sol_cache=sol_cache,
                reuse_cache=True,
                cache_only=bem_cache_only,
            )
            z18 = _z_bem_cached(float(f_hz), epw=_BEM_EPW_REFINE, geo=geo18, sol=sol18)
            rel = abs(z12 - z18) / max(abs(z18), 1e-30)
            bem_self_check.append(
                {"f_hz": float(f_hz), "rel_z_err_epw12_vs_18": float(rel)}
            )
            print(
                f"BEM self-check f={f_hz:.0f} Hz  "
                f"|z12-z18|/|z18|={rel:.4g}",
                flush=True,
            )

    fdtd: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for f_hz in freqs:
        f_key = float(f_hz)
        z_b = bem_z[f_key]
        H_b = bem_H[f_key]
        row: dict[str, Any] = {
            "f_hz": f_key,
            "z_bem_real": float(z_b.real),
            "z_bem_imag": float(z_b.imag),
            "H_bem_mag": float(abs(H_b)),
            "per_fdtd": {},
        }
        for tag, hname in fdtd_keys:
            with Result(_STEP10A / hname, frame="local") as res:
                _, z_arr = res.z(f_key)
                H_arr = res.field(_FAR_PT, f_key, frame="local") / (rho * c)
                z_fd = complex(z_arr.flat[0])
                H_fd = complex(H_arr.flat[0])
            ze = _z_errors(z_fd, z_b)
            db, ph = _h_fd_bem(H_fd, H_b)
            row["per_fdtd"][tag] = {
                "z_fd_real": float(z_fd.real),
                "z_fd_imag": float(z_fd.imag),
                **ze,
                "on_axis_db": db,
                "on_axis_phase_deg": ph,
            }
            if tag not in fdtd:
                fdtd[tag] = {"path": str(_STEP10A / hname), "by_f": {}}
            fdtd[tag]["by_f"][str(f_key)] = row["per_fdtd"][tag]
        rows.append(row)

    print("\n=== C18(ii) FDTD vs BEM (per frequency) ===", flush=True)
    for row in rows:
        f_key = float(row["f_hz"])
        print(f"  f={f_key:.0f} Hz  z_bem={row['z_bem_real']:+.5f}{row['z_bem_imag']:+.5f}j", flush=True)
        for tag, _ in fdtd_keys:
            d = row["per_fdtd"][tag]
            line = (
                f"    {tag}: ez={d['ez']:.4g}  "
                f"dRe/|zb|={d['dRe_rel']:+.4g}  dIm/|zb|={d['dIm_rel']:+.4g}  "
            )
            if tag.startswith("S4"):
                d_im_corr = float(d["dIm_rel"]) + _A11_DIM_CORR
                line += f"dIm_corr={d_im_corr:+.4g}  "
            line += (
                f"on-axis={d['on_axis_db']:+.3f} dB  "
                f"phase={d['on_axis_phase_deg']:+.2f} deg"
            )
            print(line, flush=True)

    bem_radiator_refinement: dict[str, Any] | None = None
    if args.bem_refine:
        bem_radiator_refinement = _bem_radiator_refinement_report(geom, mesh_cache)

    gates: dict[str, Any] = {"c18ii": {}, "report_only": {}}
    tilt_pass = True
    tilt_rows: list[dict[str, Any]] = []
    for row in rows:
        f_key = float(row["f_hz"])
        for hlab in ("h1", "h3"):
            ez_s3p = float(row["per_fdtd"][f"s3p_{hlab}"]["ez"])
            ez_s4 = float(row["per_fdtd"][f"S4_{hlab}"]["ez"])
            ok = ez_s4 <= ez_s3p + _GATE_TILT_MARGIN
            tilt_pass = tilt_pass and ok
            tilt_rows.append(
                {
                    "f_hz": f_key,
                    "h": hlab,
                    "ez_s3p": ez_s3p,
                    "ez_S4": ez_s4,
                    "pass": ok,
                }
            )

    ff_pass = True
    ff_rows: list[dict[str, Any]] = []
    for row in rows:
        f_key = float(row["f_hz"])
        db = abs(float(row["per_fdtd"]["S4_h3"]["on_axis_db"]))
        ok = db <= _GATE_FF_DB
        ff_pass = ff_pass and ok
        ff_rows.append({"f_hz": f_key, "on_axis_db_abs": db, "pass": ok})

    report_h3: list[dict[str, Any]] = []
    for row in rows:
        f_key = float(row["f_hz"])
        report_h3.append(
            {
                "f_hz": f_key,
                "ez_s3p_h3": float(row["per_fdtd"]["s3p_h3"]["ez"]),
                "ez_S4_h3": float(row["per_fdtd"]["S4_h3"]["ez"]),
            }
        )

    gates["c18ii"] = {
        "tilt_not_worse_than_aligned": {
            "margin": _GATE_TILT_MARGIN,
            "pass": tilt_pass,
            "rows": tilt_rows,
        },
        "S4_on_axis_h3_db": {
            "limit_db": _GATE_FF_DB,
            "pass": ff_pass,
            "rows": ff_rows,
        },
        "pass": bool(tilt_pass and ff_pass),
    }
    gates["report_only"] = {
        "edge_reactance_note": _EDGE_REACTANCE_NOTE,
        "ez_h3": report_h3,
    }

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    f_plot = freqs

    ax = axes[0]
    zr = np.array([bem_z[float(f)].real for f in f_plot], dtype=_F64)
    zi = np.array([bem_z[float(f)].imag for f in f_plot], dtype=_F64)
    ax.plot(f_plot, zr, "ko", mfc="none", ms=7, label="BEM Re")
    ax.plot(f_plot, zi, "k^", mfc="none", ms=6, label="BEM Im")
    styles = {"s3p_h1": "-", "s3p_h3": "--", "S4_h1": "-.", "S4_h3": ":"}
    for tag, hname in fdtd_keys:
        with Result(_STEP10A / hname, frame="local") as res:
            _, z_tab = res.z(f_plot)
            z_sel = np.asarray(z_tab, dtype=_C128).ravel()
        ls = styles[tag]
        ax.plot(f_plot, z_sel.real, ls, lw=1.0, label=f"{tag} Re")
        ax.plot(f_plot, z_sel.imag, ls, lw=0.9, alpha=0.85, label=f"{tag} Im")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("z")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Re/Im z: BEM vs FDTD")

    ax = axes[1]
    for tag, _ in fdtd_keys:
        ez = [float(r["per_fdtd"][tag]["ez"]) for r in rows]
        ax.semilogy(f_plot, ez, "o-", label=tag)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(r"$e_z$")
    ax.legend(fontsize=8)
    ax.set_title(r"$e_z = |z_{\mathrm{fd}}-z_{\mathrm{bem}}|/|z_{\mathrm{bem}}|$")

    ax = axes[2]
    for tag, _ in fdtd_keys:
        db = [float(r["per_fdtd"][tag]["on_axis_db"]) for r in rows]
        ax.plot(f_plot, db, "o-", label=tag)
    ax.axhline(_GATE_FF_DB, color="gray", ls=":", lw=0.8)
    ax.axhline(-_GATE_FF_DB, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(r"20 log10 $|H_{\mathrm{fd}}/H_{\mathrm{bem}}|$ (dB)")
    ax.legend(fontsize=8)
    ax.set_title("On-axis transfer ratio (dB)")

    fig.tight_layout()
    png_path = _OUT / f"{out_stem}.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    payload: dict[str, Any] = {
        "geometry": {
            "W": geom["W"],
            "H": geom["H"],
            "D": geom["D"],
            "rad_rect": list(geom["rad_rect"]),
            "origin": list(geom["origin"]),
        },
        "frequencies_hz": list(_FREQS_HZ),
        "bem_epw": _BEM_EPW,
        "bem_self_check": bem_self_check,
        "bem_z_by_f": {
            str(f): {"real": float(bem_z[f].real), "imag": float(bem_z[f].imag)}
            for f in bem_z
        },
        "table": rows,
        "fdtd_sources": fdtd,
        "gates": gates,
        "elapsed_s": elapsed,
    }
    if bem_radiator_refinement is not None:
        payload["bem_radiator_refinement"] = bem_radiator_refinement
    json_path = _OUT / f"{out_stem}.json"
    merged = payload
    if not s4_suffix and json_path.is_file():
        try:
            prior = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(prior, dict):
                prior.update(payload)
                merged = prior
        except json.JSONDecodeError:
            pass
    json_path.write_text(
        json.dumps(to_json_safe(merged), indent=2), encoding="utf-8"
    )
    print(f"\nwrote {json_path} and {png_path}  ({elapsed:.1f} s)", flush=True)
    print(
        f"C18(ii) gate pass={gates['c18ii']['pass']}  "
        f"tilt={tilt_pass}  S4 on-axis h3={ff_pass}",
        flush=True,
    )
    return 0 if gates["c18ii"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())