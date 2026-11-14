#!/usr/bin/env python3
"""C8-convergence: grid convergence (PPW sweep), Richardson extrapolation, KH probes."""

from __future__ import annotations

import argparse
import copy
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
    _radiator_D,
    derive,
    estimate_memory_bytes,
)
from fdtd.grid import Grid  # noqa: E402
from fdtd.post.kirchhoff import (  # noqa: E402
    SurfacePatches,
    dpdn_from_vn,
    kh_pressure,
    kh_pressure_loop,
)
from fdtd.post.reference import z_baffled_circle  # noqa: E402
from fdtd.post.spectra import spectrum, spectrum_rows  # noqa: E402
from fdtd.post.zrad import causal_end_taper, zrad_from_h5  # noqa: E402
from fdtd.recorder import resolve_huygens_offset  # noqa: E402
from fdtd.run import run  # noqa: E402
from fdtd.source import BoxBoundary, velocity_pulse  # noqa: E402
import validation._common as vc  # noqa: E402

_G1_PPW = (10, 15, 20, 30)
_G1_GATE_PPW = 20
_G1_ERR_TOL = 0.015
_G2_N = (24, 36, 54)
_G2_GATE_N = 36
_G2_REFINE = 1.5
_G2_GEOM_TOL = 1e-9
_G2_Z_TOL = 0.01
_G2_DIR_DB = 0.5
_G2_DIR_LEVEL_DB = -20.0
_G2_KL = (2.0, 5.0, 10.0)
_G3_DB = 0.2
_G3_PHASE_DEG = 2.0
_P_ORDER_FALLBACK = 2.0
_P_ORDER_RANGE = (1.0, 3.0)
_C1_UPDATES_PER_S = 375.0e6


def _fail_if_duplicate_h(h_by_key: dict[str, float], sweep: str) -> None:
    items = list(h_by_key.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            hi, hj = items[i][1], items[j][1]
            if abs(hi - hj) <= 1e-15 * max(hi, hj, 1e-12):
                raise RuntimeError(
                    f"C8 {sweep}: runs '{items[i][0]}' and '{items[j][0]}' share "
                    f"h={hi:.6g} m — λ/ppw capped h; lower f_max or raise cells_across"
                )


def _geometry(cfg: Config, d: object) -> tuple[float, float, float]:
    grid = Grid(cfg, d)  # type: ignore[arg-type]
    bound = BoxBoundary.from_grid(grid, d)  # type: ignore[arg-type]
    S = float(bound.s_grid)
    a = math.sqrt(S / math.pi)
    L = max(float(d.realized_W), float(d.realized_H), float(d.realized_D))  # type: ignore[attr-defined]
    return S, a, L


def _check_ram(label: str, d: object, ram_limit: float) -> None:
    est = estimate_memory_bytes(d.nx, d.ny, d.nz, d.cpml_cells)  # type: ignore[attr-defined]
    if est > ram_limit:
        raise vc.RamLimitExceeded(label, d.nx, float(est), float(ram_limit))  # type: ignore[attr-defined]
    print(
        f"  [{label}] grid {d.nx}×{d.ny}×{d.nz}  "  # type: ignore[attr-defined]
        f"mem≈{vc.fmt_bytes(est)}  steps≈{d.n_steps:,}",  # type: ignore[attr-defined]
        flush=True,
    )


def _print_h_control(label: str, cfg: Config, d: object) -> None:
    num = cfg.numerics
    D = _radiator_D(cfg.radiator)
    h = float(d.h)  # type: ignore[attr-defined]
    cells = D / h
    ppw = float(d.ppw_realized)  # type: ignore[attr-defined]
    if num.h_override is not None:
        driver = "h_override"
    else:
        h_ppw = float(d.lambda_min) / float(num.ppw)  # type: ignore[attr-defined]
        h_rad = D / float(num.cells_across_radiator)
        controlled = abs(h - h_ppw) <= 1e-12 * max(h_ppw, 1e-18)
        driver = "λ/ppw" if controlled or h <= h_ppw + 1e-15 else "D/cells"
        if not controlled and h >= h_rad - 1e-15:
            driver = "D/cells (radiator caps h)"
    print(
        f"  [{label}] h={h:.6g} m  driver={driver}  "
        f"cells_across≈{cells:.3f}  ppw@f_max={ppw:.2f}",
        flush=True,
    )


def _run_fdtd(
    cfg: Config,
    out_path: Path,
    *,
    ram_limit: float,
    timer: vc.RunTimer,
    label: str,
) -> object:
    d = derive(cfg)
    _check_ram(label, d, ram_limit)
    _print_h_control(label, cfg, d)
    timer.banner(label, f"grid {d.nx}×{d.ny}×{d.nz}")
    timer.begin(label)
    run(cfg, out_path=out_path, progress=True, ram_limit_bytes=ram_limit)
    timer.end(label)
    return d


def _design_f_max(cfg: Config, d: object) -> float:
    fm = cfg.band.f_max
    if fm is not None:
        return min(float(d.f_max), float(fm))  # type: ignore[attr-defined]
    return float(d.f_max)  # type: ignore[attr-defined]


def _load_z_band(
    path: Path, cfg: Config, d: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = cfg.air.c
    f_min = cfg.band.f_min
    f_max = _design_f_max(cfg, d)
    freqs, z = zrad_from_h5(path, df_hz=2.0)
    _, a, _ = _geometry(cfg, d)
    k = 2.0 * np.pi * freqs / c
    ka = k * a
    band = vc.design_band_mask(freqs, f_min=f_min, f_max=f_max, z=z)
    return freqs, z, ka, band


def _print_legacy_band_audit() -> None:
    print("\n=== C5–C7 band gate audit (report only; scripts unchanged) ===", flush=True)
    print(
        "  C5-lowfreq: gates on f_min..min(cfg.f_max, derived.f_max) — within design band",
        flush=True,
    )
    print(
        "  C6-baffle: G2 gate on cfg.f_min..derived.f_max (matches design when cfg sets f_max)",
        flush=True,
    )
    print(
        "  C7-subcell: G1/G2 compare on design band mask; G3/racetrack max |z−z_ref| uses",
        flush=True,
    )
    print(
        "    common-ka on isfinite(z) only — may include f outside [f_min,f_max] where",
        flush=True,
    )
    print("    |V|>−40 dB (zrad source band, not design band)", flush=True)


def _ppw_for_err(
    err_at_ref: float,
    ppw_ref: float,
    err_target: float,
    order_p: float,
) -> float | None:
    if err_at_ref <= 0 or err_at_ref <= err_target or order_p <= 0:
        return None
    return float(ppw_ref * (err_at_ref / err_target) ** (1.0 / order_p))


def _richardson_order(
    coarse: np.ndarray,
    mid: np.ndarray,
    fine: np.ndarray,
    *,
    refine: float = _G2_REFINE,
) -> tuple[float, bool, float, float]:
    """``p = ln(‖c−m‖/‖m−f‖) / ln(refine)`` with max-norm on aligned samples."""
    e_cm = float(np.max(np.abs(coarse - mid)))
    e_mf = float(np.max(np.abs(mid - fine)))
    if e_cm <= 0.0 or e_mf <= 0.0 or e_cm <= e_mf:
        return _P_ORDER_FALLBACK, True, e_cm, e_mf
    p = math.log(e_cm / e_mf) / math.log(refine)
    if p < _P_ORDER_RANGE[0] or p > _P_ORDER_RANGE[1]:
        return _P_ORDER_FALLBACK, True, e_cm, e_mf
    return float(p), False, e_cm, e_mf


def _richardson_order_complex(
    H_c: np.ndarray,
    H_m: np.ndarray,
    H_f: np.ndarray,
    gate: np.ndarray,
    *,
    refine: float = _G2_REFINE,
) -> tuple[float, bool]:
    """Max-norm of complex ΔH on gated angles (for reporting)."""
    if not np.any(gate):
        return _P_ORDER_FALLBACK, True
    e_cm = float(np.max(np.abs(H_c[gate] - H_m[gate])))
    e_mf = float(np.max(np.abs(H_m[gate] - H_f[gate])))
    if e_cm <= 0.0 or e_mf <= 0.0 or e_cm <= e_mf:
        return _P_ORDER_FALLBACK, True
    p = math.log(e_cm / e_mf) / math.log(refine)
    if p < _P_ORDER_RANGE[0] or p > _P_ORDER_RANGE[1]:
        return _P_ORDER_FALLBACK, True
    return float(p), False


def _assert_g2_nominal_geometry(cfg: Config, d: object, *, label: str) -> None:
    if cfg.box is None:
        raise RuntimeError(f"{label}: box config required for G2 geometry assert")
    box = cfg.box
    tol = _G2_GEOM_TOL
    rw = float(d.realized_W)  # type: ignore[attr-defined]
    rh = float(d.realized_H)
    rd = float(d.realized_D)
    h = float(d.h)
    ic = int(d.radiator_ic)  # type: ignore[attr-defined]
    jc = int(d.radiator_jc)
    xc = ic * h
    yc = jc * h
    print(
        f"  [{label}] realized W,H,D=({rw:.9f},{rh:.9f},{rd:.9f}) m  "
        f"center=({xc:.9g},{yc:.9g}) m  ppw@f_max={float(d.ppw_realized):.3f}",  # type: ignore[attr-defined]
        flush=True,
    )
    for name, nom, got in (
        ("W", box.W, rw),
        ("H", box.H, rh),
        ("D", box.D, rd),
        ("x_c", cfg.radiator.x_c, xc),
        ("y_c", cfg.radiator.y_c, yc),
    ):
        if abs(got - nom) > tol:
            raise RuntimeError(
                f"{label}: {name} realized={got:.12g} m != nominal={nom:.12g} m "
                f"(tol {tol:g})"
            )


def _richardson_extrap(
    z_coarse: np.ndarray,
    z_fine: np.ndarray,
    *,
    r: float,
    p: float,
) -> np.ndarray:
    """Extrapolate to h→0 from coarse and fine levels (fine = finer grid)."""
    denom = r**p - 1.0
    if abs(denom) < 1e-30:
        return z_fine.copy()
    return z_fine + (z_fine - z_coarse) / denom


def _g2_h_from_n(box: object, n_cells: int) -> float:
    L = float(box.W)  # type: ignore[attr-defined]
    return L / float(n_cells)


def _h_transfer_err(h_ref: complex, h: complex) -> tuple[float, float]:
    db = 20.0 * math.log10(abs(h) / max(abs(h_ref), 1e-30))
    ph = float(np.rad2deg(np.angle(h) - np.angle(h_ref)))
    while ph > 180.0:
        ph -= 360.0
    while ph < -180.0:
        ph += 360.0
    return db, ph


def _probe_targets_g3(cfg: Config, grid: Grid, derived: object) -> list[tuple[float, float, float]]:
    offset = resolve_huygens_offset(cfg)
    gap = int(derived.gap_cells)  # type: ignore[attr-defined]
    g_cells = 0.5 * (offset + gap)
    g = g_cells * grid.h
    W, D = grid.realized_W, grid.realized_D
    nominal = [
        (0.0, 0.0, g),
        (0.5 * W + g, 0.0, -0.5 * D),
        (0.0, 0.0, -D - g),
    ]
    snapped: list[tuple[float, float, float]] = []
    for x, y, z in nominal:
        i, j, k = grid.index_of(x, y, z)
        snapped.append(grid.coords_of(i, j, k))
    return snapped


def _plan_resources(cfgs: list[Config], ram_limit: float) -> tuple[float, float]:
    peak = 0.0
    wall = 0.0
    for cfg in cfgs:
        d = derive(cfg)
        est = estimate_memory_bytes(d.nx, d.ny, d.nz, d.cpml_cells)
        peak = max(peak, float(est))
        if est > ram_limit:
            raise vc.RamLimitExceeded("c08 plan", d.nx, float(est), float(ram_limit))
        wall += d.n_steps * d.total_cells / _C1_UPDATES_PER_S
    return wall, peak


def _load_huygens_spectra(
    path: Path,
    cfg: Config,
    d: object,
    f_hz: float,
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
    V = complex(spectrum(v_rad, t_rad, f_hz))
    p_face = np.asarray(spectrum_rows(p_win, t_p, f_hz), dtype=np.complex128)
    vn_face = np.asarray(spectrum_rows(vn_win, t_v, f_hz), dtype=np.complex128)
    return patches, V, p_face, vn_face, rho


def _directivity_cut(
    *,
    plane: str,
    r_m: float,
    f_hz: float,
    patches: SurfacePatches,
    p_face: np.ndarray,
    vn_face: np.ndarray,
    V: complex,
    rho: float,
    c: float,
    n_theta: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed Huygens KH (no image): |H(θ)| in dB re on-axis."""
    theta = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_theta)
    if plane == "xz":
        pts = np.column_stack([r_m * np.sin(theta), np.zeros_like(theta), r_m * np.cos(theta)])
    else:
        pts = np.column_stack([np.zeros_like(theta), r_m * np.sin(theta), r_m * np.cos(theta)])
    omega = np.asarray([2.0 * math.pi * f_hz], dtype=np.float64)
    k_sel = np.asarray([2.0 * math.pi * f_hz / c], dtype=np.float64)
    dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
    p_row = p_face[np.newaxis, :]
    p_obs = kh_pressure(pts, patches, p_row, dpdn, k_sel, image_plane_z=None, near_subdivide=True)[0]
    H = p_obs / V
    axis_i = int(np.argmin(np.abs(theta)))
    H0 = H[axis_i]
    rel = np.abs(H / H0)
    db = 20.0 * np.log10(np.maximum(rel, 1e-30))
    return theta, db, H


def _plot_g1_err(path: Path, series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4), layout="tight")
    for label, (ka, err) in series.items():
        m = np.isfinite(err)
        ax.semilogy(ka[m], err[m], ".", ms=3, label=label)
    ax.axhline(_G1_ERR_TOL, color="k", ls="--", lw=0.8, label=f"tol {_G1_ERR_TOL}")
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel(r"$|z - z_{\mathrm{ref}}|$")
    ax.legend(fontsize=8)
    ax.set_title("C8 G1 baffle dispersion (PPW sweep)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_directivity_overlay(
    path: Path,
    *,
    plane: str,
    kl: float,
    theta: np.ndarray,
    curves: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4), layout="tight")
    for name, db in curves.items():
        ax.plot(np.rad2deg(theta), db, lw=1, label=name)
    ax.set_xlabel("θ deg")
    ax.set_ylabel("rel |H| dB re on-axis")
    ax.set_title(f"C8 G2 directivity {plane}  kL={kl:g}")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C8-convergence validation")
    parser.add_argument(
        "--g1-config",
        type=Path,
        default=ROOT / "configs" / "c08_g1_baffle.json",
    )
    parser.add_argument(
        "--g2-config",
        type=Path,
        default=ROOT / "configs" / "c08_g2_box.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "results")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--ram-gb", type=float, default=DEFAULT_RAM_LIMIT_BYTES / 1e9)
    args = parser.parse_args(argv)

    ram_limit = float(args.ram_gb) * 1e9
    timer = vc.RunTimer()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)

    cfg_g1_base = Config.from_json(args.g1_config.read_text(encoding="utf-8"))
    cfg_g2_base = Config.from_json(args.g2_config.read_text(encoding="utf-8"))

    plan_cfgs: list[Config] = []
    for ppw in _G1_PPW:
        c = copy.deepcopy(cfg_g1_base)
        c.numerics.ppw = float(ppw)
        plan_cfgs.append(c)
    if cfg_g2_base.box is None:
        raise RuntimeError("G2 config requires box dimensions")
    for n_cells in _G2_N:
        c = copy.deepcopy(cfg_g2_base)
        c.numerics.h_override = _g2_h_from_n(c.box, n_cells)
        if n_cells == _G2_GATE_N:
            d_plan = derive(c)
            targets = _probe_targets_g3(c, Grid(c, d_plan), d_plan)
            c.output.probes = [[x, y, z] for x, y, z in targets]
        plan_cfgs.append(c)

    wall_est, peak_ram = _plan_resources(plan_cfgs, ram_limit)
    print("\n=== C8-convergence plan ===", flush=True)
    print(f"  FDTD runs: {len(plan_cfgs)}", flush=True)
    print(f"  estimated wall time: {wall_est:.0f} s ({wall_est / 60:.1f} min)", flush=True)
    print(f"  estimated peak RAM (largest single run): {vc.fmt_bytes(peak_ram)}", flush=True)
    _print_legacy_band_audit()

    gates: dict[str, Any] = {}
    g1_pass = True
    g2_z_pass = True
    g2_dir_pass = True
    g3_pass = True
    overall = False

    with tempfile.TemporaryDirectory(prefix="c08_") as tmpdir:
        tmp = Path(tmpdir)

        # --- G1: baffle PPW sweep vs analytic ---
        print("\n=== G1 baffle dispersion ===", flush=True)
        g1_h: dict[str, float] = {}
        g1_runs: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        g1_paths: dict[float, Path] = {}
        for ppw in _G1_PPW:
            cfg = copy.deepcopy(cfg_g1_base)
            cfg.numerics.ppw = float(ppw)
            h5 = tmp / f"g1_ppw{ppw:g}.h5"
            label = f"g1_ppw{ppw:g}"
            d = _run_fdtd(cfg, h5, ram_limit=ram_limit, timer=timer, label=label)
            g1_paths[ppw] = h5
            freqs, z, ka, band = _load_z_band(h5, cfg, d)
            g1_runs[ppw] = (ka, z, band)
            g1_h[str(ppw)] = float(d.h)

        _fail_if_duplicate_h(g1_h, "G1 PPW sweep")

        ka_g1, z_g1, g1_ka_meta = vc.runs_on_common_ka(
            g1_runs,
            ref_key=max(_G1_PPW),
            label="C8 G1",
            restrict_to_band=True,
            key_label="ppw",
        )
        z_ref_g1 = np.asarray(z_baffled_circle(ka_g1), dtype=np.complex128)
        g1_max: dict[str, float] = {}
        g1_err_at_fmax: dict[str, float] = {}
        g1_err_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ppw in _G1_PPW:
            err = np.abs(z_g1[ppw] - z_ref_g1)
            g1_max[str(ppw)] = float(np.max(err))
            g1_err_series[f"ppw {ppw:g}"] = (ka_g1, err)
            print(
                f"  G1 report ppw={ppw:g}: max |z−z_ref|={g1_max[str(ppw)]:.4f} "
                f"(design band, common ka)",
                flush=True,
            )

        g1_f_max = _design_f_max(cfg_g1_base, derive(cfg_g1_base))
        _, a_g1, _ = _geometry(cfg_g1_base, derive(cfg_g1_base))
        ka_fmax = 2.0 * math.pi * g1_f_max * a_g1 / cfg_g1_base.air.c
        for ppw in _G1_PPW:
            err = np.abs(z_g1[ppw] - z_ref_g1)
            e_f = vc.interp_vs_ka_real(ka_fmax, ka_g1, err)
            g1_err_at_fmax[str(ppw)] = e_f
            print(
                f"  G1 ppw={ppw:g} @ f_max={g1_f_max:.1f} Hz (ka≈{ka_fmax:.4g}): "
                f"|z−z_ref|={e_f:.4f}",
                flush=True,
            )

        g1_gate_err = g1_max[str(_G1_GATE_PPW)]
        g1_pass = g1_gate_err <= _G1_ERR_TOL
        g1_orders: list[float] = []
        for i in range(len(_G1_PPW) - 1):
            p0, p1 = _G1_PPW[i], _G1_PPW[i + 1]
            e0, e1 = g1_err_at_fmax[str(p0)], g1_err_at_fmax[str(p1)]
            if e0 > e1 > 0:
                g1_orders.append(
                    math.log(e0 / e1) / math.log(g1_h[str(p0)] / g1_h[str(p1)])
                )
        g1_order = float(np.mean(g1_orders)) if g1_orders else None
        g1_order_for_fit = g1_order if g1_order is not None else _P_ORDER_FALLBACK
        if g1_order is not None:
            print(
                f"  G1 observed order @ f_max (consecutive PPW): p≈{g1_order:.2f}",
                flush=True,
            )
        e20_fmax = g1_err_at_fmax[str(_G1_GATE_PPW)]
        ppw_for_001 = _ppw_for_err(e20_fmax, _G1_GATE_PPW, _G1_ERR_TOL, g1_order_for_fit)
        if ppw_for_001 is not None:
            print(
                f"  G1 extrapolated ppw for |z−z_ref|={_G1_ERR_TOL} @ f_max "
                f"(p={g1_order_for_fit:.2f}): ≈{ppw_for_001:.1f}",
                flush=True,
            )
        else:
            print(
                f"  G1 extrapolated ppw @ f_max: not reachable from ppw={_G1_GATE_PPW} "
                f"err={e20_fmax:.4f}",
                flush=True,
            )

        gates["g1"] = {
            "max_err": g1_max,
            "err_at_fmax": g1_err_at_fmax,
            "f_max_hz": g1_f_max,
            "ka_at_fmax": ka_fmax,
            "ppw_for_err_0p01_at_fmax": ppw_for_001,
            "gate_ppw": _G1_GATE_PPW,
            "gate_tol": _G1_ERR_TOL,
            "gate_err": g1_gate_err,
            "pass": g1_pass,
            "h_m": g1_h,
            "ka_align": g1_ka_meta,
            "order_at_fmax": g1_order,
        }

        # --- G2: box self-convergence (fixed L/N, ratio 1.5) ---
        print("\n=== G2 box self-convergence ===", flush=True)
        if cfg_g2_base.box is None:
            raise RuntimeError("G2 requires box in config")
        g2_h: dict[int, float] = {}
        g2_runs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        g2_paths: dict[int, Path] = {}
        g2_derived: dict[int, object] = {}

        for n_cells in _G2_N:
            cfg = copy.deepcopy(cfg_g2_base)
            cfg.numerics.h_override = _g2_h_from_n(cfg.box, n_cells)
            if n_cells == _G2_GATE_N:
                d_probe = derive(cfg)
                grid_probe = Grid(cfg, d_probe)
                cfg.output.probes = [
                    list(x) for x in _probe_targets_g3(cfg, grid_probe, d_probe)
                ]
            h5 = tmp / f"g2_N{n_cells}.h5"
            label = f"g2_N{n_cells}"
            d = _run_fdtd(cfg, h5, ram_limit=ram_limit, timer=timer, label=label)
            _assert_g2_nominal_geometry(cfg, d, label=label)
            g2_paths[n_cells] = h5
            g2_derived[n_cells] = d
            g2_h[n_cells] = float(d.h)
            _freqs, z, ka, band = _load_z_band(h5, cfg, d)
            g2_runs[n_cells] = (ka, z, band)

        n_fine, n_mid, n_coarse = _G2_N[2], _G2_N[1], _G2_N[0]
        ka_g2, z_g2, g2_ka_meta = vc.runs_on_common_ka(
            g2_runs,
            ref_key=n_fine,
            label="C8 G2",
            restrict_to_band=True,
            key_label="N",
        )
        p_obs, p_fallback, e_cm, e_mf = _richardson_order(
            z_g2[n_coarse], z_g2[n_mid], z_g2[n_fine], refine=_G2_REFINE
        )
        if p_fallback:
            print(
                f"  G2 Richardson z: p={p_obs:.2f} (fallback; ‖c−m‖={e_cm:.4g} "
                f"‖m−f‖={e_mf:.4g}, ratio refine={_G2_REFINE})",
                flush=True,
            )
        else:
            print(
                f"  G2 Richardson z: p≈{p_obs:.2f}  ‖c−m‖={e_cm:.4g}  ‖m−f‖={e_mf:.4g}",
                flush=True,
            )

        z_inf = _richardson_extrap(z_g2[n_mid], z_g2[n_fine], r=_G2_REFINE, p=p_obs)
        g2_z_max = float(np.max(np.abs(z_g2[n_mid] - z_inf)))
        g2_z_pass = g2_z_max <= _G2_Z_TOL

        g2_f_min = cfg_g2_base.band.f_min
        g2_f_max = _design_f_max(cfg_g2_base, g2_derived[n_mid])

        L_box = float(cfg_g2_base.box.W)
        c = cfg_g2_base.air.c
        r_far = 10.0 * L_box
        dir_report: dict[str, Any] = {}
        dir_plots: list[tuple[str, float, np.ndarray, dict[str, np.ndarray]]] = []

        for kl in _G2_KL:
            f_hz = kl * c / (2.0 * math.pi * L_box)
            if not (g2_f_min <= f_hz <= g2_f_max):
                raise RuntimeError(
                    f"G2 kL={kl:g} → f={f_hz:.2f} Hz outside design band "
                    f"[{g2_f_min}, {g2_f_max}]"
                )
            dir_by_n: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
            H_by_n: dict[str, dict[int, np.ndarray]] = {}
            for n_cells in _G2_N:
                patches, V, p_face, vn_face, rho = _load_huygens_spectra(
                    g2_paths[n_cells], cfg_g2_base, g2_derived[n_cells], f_hz
                )
                for plane in ("xz", "yz"):
                    theta, db, H = _directivity_cut(
                        plane=plane,
                        r_m=r_far,
                        f_hz=f_hz,
                        patches=patches,
                        p_face=p_face,
                        vn_face=vn_face,
                        V=V,
                        rho=rho,
                        c=c,
                    )
                    dir_by_n.setdefault(plane, {})[n_cells] = (theta, db)
                    H_by_n.setdefault(plane, {})[n_cells] = H

            for plane in ("xz", "yz"):
                theta = dir_by_n[plane][n_fine][0]
                db_fine = dir_by_n[plane][n_fine][1]
                db_mid = dir_by_n[plane][n_mid][1]
                db_coarse = dir_by_n[plane][n_coarse][1]
                H_c = H_by_n[plane][n_coarse]
                H_m = H_by_n[plane][n_mid]
                H_f = H_by_n[plane][n_fine]
                H_inf = _richardson_extrap(H_m, H_f, r=_G2_REFINE, p=p_obs)
                axis = int(np.argmin(np.abs(theta)))
                rel_mid = np.abs(H_m / H_m[axis])
                rel_inf = np.abs(H_inf / H_inf[axis])
                db_mid_pat = 20.0 * np.log10(np.maximum(rel_mid, 1e-30))
                db_inf_pat = 20.0 * np.log10(np.maximum(rel_inf, 1e-30))
                delta = db_mid_pat - db_inf_pat
                gate_mask = db_inf_pat >= _G2_DIR_LEVEL_DB
                if np.any(gate_mask):
                    dmax = float(np.max(np.abs(delta[gate_mask])))
                else:
                    dmax = float(np.max(np.abs(delta)))
                p_dir, p_dir_fb = _richardson_order_complex(
                    H_c, H_m, H_f, gate_mask, refine=_G2_REFINE
                )
                tag = f"kl{kl:g}_{plane}"
                dir_report[tag] = {
                    "f_hz": f_hz,
                    "max_dB_err_N36_vs_extrap": dmax,
                    "richardson_p_dir": p_dir,
                    "p_dir_fallback": p_dir_fb,
                    "pass": dmax <= _G2_DIR_DB,
                }
                ok = dmax <= _G2_DIR_DB
                g2_dir_pass = g2_dir_pass and ok
                print(
                    f"  G2 directivity {plane} kL={kl:g} (f={f_hz:.1f} Hz): "
                    f"max |ΔdB| N={n_mid} vs extrap={dmax:.3f}  "
                    f"p_dir≈{p_dir:.2f}{' (fallback)' if p_dir_fb else ''} "
                    f"({'PASS' if ok else 'FAIL'}, tol {_G2_DIR_DB})",
                    flush=True,
                )
                dir_plots.append(
                    (
                        plane,
                        kl,
                        theta,
                        {
                            f"N{n_coarse}": db_coarse,
                            f"N{n_mid}": db_mid,
                            f"N{n_fine}": db_fine,
                            "extrap": db_inf_pat,
                        },
                    )
                )

        gates["g2"] = {
            "box_cells_N": list(_G2_N),
            "gate_N": _G2_GATE_N,
            "h_m": {str(k): v for k, v in g2_h.items()},
            "ka_align": g2_ka_meta,
            "design_band_hz": [g2_f_min, g2_f_max],
            "richardson_p": p_obs,
            "p_fallback": p_fallback,
            "max_abs_z_gate_minus_zinf": g2_z_max,
            "richardson_refine": _G2_REFINE,
            "z_pass": g2_z_pass,
            "dir_pass": g2_dir_pass,
            "directivity": dir_report,
            "pass": g2_z_pass and g2_dir_pass,
            "L_m": L_box,
            "r_far_m": r_far,
        }

        # --- G3: KH vs FDTD probes (N=36 / gate level) ---
        print(f"\n=== G3 box-mode KH probe consistency (N={_G2_GATE_N}) ===", flush=True)
        cfg_g3 = copy.deepcopy(cfg_g2_base)
        cfg_g3.numerics.h_override = _g2_h_from_n(cfg_g3.box, _G2_GATE_N)
        d_g3 = g2_derived[_G2_GATE_N]
        h5_g3 = g2_paths[_G2_GATE_N]
        grid_g3 = Grid(cfg_g3, d_g3)  # type: ignore[arg-type]
        probe_req = _probe_targets_g3(cfg_g3, grid_g3, d_g3)
        probe_names = ("front+z", "+x mid", "rear−z")
        g3_rows: list[dict[str, Any]] = []
        rho = cfg_g3.air.rho

        import h5py

        with h5py.File(h5_g3, "r") as f:
            probe_centers = np.asarray(f["probes/centers"][()], dtype=np.float64)
            probe_p_ts = np.asarray(f["probes/p"][()], dtype=np.float64)
            t_probe = np.asarray(f["probes/times_p"][()], dtype=np.float64)

        g3_f_min = g2_f_min
        g3_f_max = g2_f_max
        win_pr = causal_end_taper(probe_p_ts.shape[0])
        for kl in _G2_KL:
            f_hz = kl * c / (2.0 * math.pi * L_box)
            if not (g3_f_min <= f_hz <= g3_f_max):
                raise RuntimeError(
                    f"G3 kL={kl:g} → f={f_hz:.2f} Hz outside design band "
                    f"[{g3_f_min}, {g3_f_max}]"
                )
            patches, V, p_face, vn_face, rho_h5 = _load_huygens_spectra(
                h5_g3, cfg_g3, d_g3, f_hz
            )
            assert abs(rho_h5 - rho) < 1e-9
            omega = np.asarray([2.0 * math.pi * f_hz], dtype=np.float64)
            k_sel = np.asarray([2.0 * math.pi * f_hz / c], dtype=np.float64)
            dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
            p_row = p_face[np.newaxis, :]
            for qi, name in enumerate(probe_names):
                P_fd = complex(spectrum(probe_p_ts[:, qi] * win_pr, t_probe, f_hz))
                H_fd = P_fd / V
                p_kh = kh_pressure_loop(
                    probe_centers[qi].reshape(1, 3),
                    patches,
                    p_row,
                    dpdn,
                    k_sel,
                )[0, 0]
                H_kh = p_kh / V
                db, ph = _h_transfer_err(H_fd, H_kh)
                ok = abs(db) <= _G3_DB and abs(ph) <= _G3_PHASE_DEG
                g3_pass = g3_pass and ok
                row = {
                    "kL": kl,
                    "f_hz": f_hz,
                    "probe": name,
                    "requested_m": list(probe_req[qi]),
                    "center_m": probe_centers[qi].tolist(),
                    "db_fdtd_vs_kh": float(db),
                    "phase_deg": float(ph),
                    "pass": ok,
                }
                g3_rows.append(row)
                print(
                    f"  G3 kL={kl:g} {name}: Δ={db:+.3f} dB  ∠{ph:+.2f}°  "
                    f"({'PASS' if ok else 'FAIL'})",
                    flush=True,
                )

        gates["g3"] = {"rows": g3_rows, "pass": g3_pass}

        print("\n  G3 probe table (H = P/V, KH full G vs FDTD):", flush=True)
        print(f"  {'kL':>4} {'probe':>10} {'ΔdB':>8} {'∠deg':>8} {'pass':>6}", flush=True)
        for row in g3_rows:
            print(
                f"  {row['kL']:4g} {row['probe']:>10} "
                f"{row['db_fdtd_vs_kh']:+8.3f} {row['phase_deg']:+8.2f} "
                f"{'yes' if row['pass'] else 'no':>6}",
                flush=True,
            )

        # --- Recommended ppw ---
        ppw_at_gate = float(g2_derived[_G2_GATE_N].ppw_realized)  # type: ignore[attr-defined]
        g1_ok = {float(p): g1_max[str(p)] <= _G1_ERR_TOL for p in _G1_PPW}
        g2_ok = g2_z_pass and g2_dir_pass
        rec: int | float | None = None
        for ppw in sorted(_G1_PPW):
            if g1_ok.get(float(ppw), False) and g2_ok:
                rec = int(ppw) if ppw == int(ppw) else ppw
                break
        print("\n=== Recommended default PPW (report only; config unchanged) ===", flush=True)
        print(
            f"  G2 gate level N={_G2_GATE_N}: realized ppw@f_max={ppw_at_gate:.2f}",
            flush=True,
        )
        if rec is not None:
            print(
                f"  smallest G1 sweep ppw with err≤{_G1_ERR_TOL} and G2 pass: {rec}",
                flush=True,
            )
        else:
            print("  no G1 ppw in {10,15,20,30} meets G1 and G2 together", flush=True)

        if not args.no_plot:
            _plot_g1_err(args.out / f"c08_g1_{stamp}.png", g1_err_series)
            for plane, kl, theta, curves in dir_plots:
                _plot_directivity_overlay(
                    args.out / f"c08_g2_dir_{plane}_kl{kl:g}_{stamp}.png",
                    plane=plane,
                    kl=kl,
                    theta=theta,
                    curves=curves,
                )

        overall = g1_pass and g2_z_pass and g2_dir_pass and g3_pass

        def _gate_line(name: str, passed: bool, detail: str) -> None:
            mark = "PASS" if passed else "FAIL"
            print(f"  {name}: {mark}  {detail}", flush=True)

        print("\n=== C8-convergence gates ===", flush=True)
        _gate_line(
            "G1",
            g1_pass,
            f"ppw={_G1_GATE_PPW} max |z−z_ref|={g1_gate_err:.4f} (tol ≤{_G1_ERR_TOL})",
        )
        _gate_line(
            "G2-z",
            g2_z_pass,
            f"max |z_N{n_mid}−z_∞|={g2_z_max:.4f} (tol ≤{_G2_Z_TOL}); "
            f"Richardson p={p_obs:.2f} (refine {_G2_REFINE})",
        )
        _gate_line("G2-directivity", g2_dir_pass, f"max |ΔdB| ≤{_G2_DIR_DB} (level ≥{_G2_DIR_LEVEL_DB} dB)")
        _gate_line(
            "G3",
            g3_pass,
            f"probes: |Δ|≤{_G3_DB} dB, ≤{_G3_PHASE_DEG}° vs KH (full G)",
        )
        print(f"  Overall: {'PASS' if overall else 'FAIL'}", flush=True)

        payload: dict[str, Any] = {
            "check": "c08_convergence",
            "timestamp_utc": stamp,
            "plan": {
                "n_runs": len(plan_cfgs),
                "wall_s_est": wall_est,
                "peak_ram_bytes_est": peak_ram,
            },
            "gates": gates,
            "recommended_ppw": rec,
            "pass": overall,
            "phases_s": timer.phases,
        }
        json_path = args.out / f"c08_convergence_{stamp}.json"
        json_path.write_text(json.dumps(vc.to_json_safe(payload), indent=2), encoding="utf-8")
        timer.banner("output", json_path.name)

    vc.print_runtime_summary(timer, tuple(timer.phases.keys()))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())