#!/usr/bin/env python3
"""C7-subcell: radiator sub-cell offsets and mask resolution (baffle mode)."""

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
from fdtd.grid import Grid, radiator_area_exact  # noqa: E402
from fdtd.post.reference import z_baffled_circle, z_baffled_rect  # noqa: E402
from fdtd.post.zrad import zrad_from_h5  # noqa: E402
from fdtd.run import run  # noqa: E402
from fdtd.source import BoxBoundary  # noqa: E402
import validation._common as vc  # noqa: E402

_G1_TOL = 0.01
_G2_TOL = 0.03
_G2_TIGHT = 0.01
_G3_TOL = 0.02
_G3_KA_MAX = 1.5
_G3_CELLS = (6, 10, 16)
_G3_GATE_CELLS = 10
_RT_CELLS = (10, 16, 24)
_C1_UPDATES_PER_S = 375.0e6


def _a_equiv(cfg: Config) -> float:
    s = radiator_area_exact(cfg.radiator)
    return math.sqrt(s / math.pi)


def _f_max_ka(*, ka_max: float, a_m: float, c: float) -> float:
    """Band top so ``ka = k a ≤ ka_max`` at ``f_max`` (``a = √(S/π)``)."""
    return ka_max * c / (2.0 * math.pi * a_m)


def _print_derived(label: str, cfg: Config, d: object) -> float:
    h = float(d.h)  # type: ignore[attr-defined]
    d_rad = _radiator_D(cfg.radiator)
    cells = d_rad / h
    ppw = float(d.ppw_realized)  # type: ignore[attr-defined]
    print(
        f"  [{label}] h={h:.6g} m  cells_across≈{cells:.4f}  "
        f"(min dim D={d_rad:.4g} m)  ppw={ppw:.2f}",
        flush=True,
    )
    return h


def _fail_if_duplicate_h(h_by_key: dict[str, float], sweep: str) -> None:
    items = list(h_by_key.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            hi, hj = items[i][1], items[j][1]
            if abs(hi - hj) <= 1e-15 * max(hi, hj, 1e-12):
                raise RuntimeError(
                    f"C7 {sweep}: runs '{items[i][0]}' and '{items[j][0]}' share "
                    f"h={hi:.6g} m — λ/ppw capped h; lower f_max so D/cells_across sets h"
                )


def _geometry(cfg: Config, d: object) -> tuple[float, float]:
    grid = Grid(cfg, d)  # type: ignore[arg-type]
    bound = BoxBoundary.from_grid(grid, d)  # type: ignore[arg-type]
    S = float(bound.s_grid)
    a = math.sqrt(S / math.pi)
    return S, a


def _check_ram(label: str, d: object, ram_limit: float) -> None:
    est = estimate_memory_bytes(d.nx, d.ny, d.nz, d.cpml_cells)  # type: ignore[attr-defined]
    if est > ram_limit:
        raise vc.RamLimitExceeded(label, d.nx, float(est), float(ram_limit))  # type: ignore[attr-defined]
    print(
        f"  [{label}] grid {d.nx}×{d.ny}×{d.nz}  "  # type: ignore[attr-defined]
        f"mem≈{vc.fmt_bytes(est)}  steps≈{d.n_steps:,}",  # type: ignore[attr-defined]
        flush=True,
    )


def _interp_vs_ka(ka_tgt: np.ndarray, ka_src: np.ndarray, z_src: np.ndarray) -> np.ndarray:
    if ka_tgt.size == 0:
        return np.asarray([], dtype=np.complex128)
    order = np.argsort(ka_src)
    ks = ka_src[order]
    zs = z_src[order]
    re = np.interp(ka_tgt, ks, np.real(zs))
    im = np.interp(ka_tgt, ks, np.imag(zs))
    return re + 1j * im


def _runs_on_common_ka(
    runs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    ref_key: int,
    label: str,
    a_by_key: dict[int, float],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    """Interpolate runs onto the reference run's ``ka`` where all runs overlap.

    Per-run span uses only finite ``z`` (``band`` is diagnostic, not applied).
    Overlap: ``[max(ka_min), min(ka_max)]`` over those finite samples.
    """
    print(f"  [{label}] common-ka alignment:", flush=True)
    valid: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    spans: list[tuple[float, float]] = []
    meta_runs: dict[str, Any] = {}
    for key in sorted(runs):
        ka, z, band = runs[key]
        ka = np.asarray(ka, dtype=np.float64)
        z = np.asarray(z, dtype=np.complex128)
        m_fin = np.isfinite(z)
        n_fin = int(np.sum(m_fin))
        n_band = int(np.sum(band & m_fin))
        a_m = float(a_by_key.get(key, float("nan")))
        if n_fin == 0:
            raise RuntimeError(f"C7 {label}: run cells={key} has no finite z")
        ka_fin = ka[m_fin]
        ka_span = (float(np.min(ka_fin)), float(np.max(ka_fin)))
        meta_runs[str(key)] = {
            "n_finite": n_fin,
            "n_band_mask": n_band,
            "ka_min": ka_span[0],
            "ka_max": ka_span[1],
            "a_m": a_m,
        }
        print(
            f"    cells={key}: finite={n_fin}  "
            f"(band∩finite={n_band})  "
            f"ka∈[{ka_span[0]:.4g}, {ka_span[1]:.4g}]  a={a_m:.6g} m",
            flush=True,
        )
        valid[key] = (ka_fin, z[m_fin])
        spans.append(ka_span)

    ka_min = max(s[0] for s in spans)
    ka_max = min(s[1] for s in spans)
    print(
        f"    overlap ka ∈ [{ka_min:.4g}, {ka_max:.4g}]  (ref cells={ref_key})",
        flush=True,
    )
    if ka_min >= ka_max - 1e-12:
        raise RuntimeError(
            f"C7 {label}: empty ka overlap [{ka_min:.4g}, {ka_max:.4g}] across runs"
        )

    if ref_key not in valid:
        raise RuntimeError(f"C7 {label}: reference key {ref_key} missing")
    ka_u, z_u = valid[ref_key]
    m_grid = (ka_u >= ka_min) & (ka_u <= ka_max)
    if not np.any(m_grid):
        raise RuntimeError(
            f"C7 {label}: reference run has no finite ka inside overlap"
        )
    ka_grid = ka_u[m_grid]
    aligned: dict[int, np.ndarray] = {ref_key: z_u[m_grid]}
    for key, (ka_v, z_v) in valid.items():
        if key == ref_key:
            continue
        aligned[key] = _interp_vs_ka(ka_grid, ka_v, z_v)
    meta: dict[str, Any] = {
        "runs": meta_runs,
        "ka_overlap": [ka_min, ka_max],
        "n_ka_common": int(ka_grid.size),
        "ref_key": ref_key,
    }
    return ka_grid, aligned, meta


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
    timer.banner(label, f"grid {d.nx}×{d.ny}×{d.nz}")
    _print_derived(label, cfg, d)
    timer.begin(label)
    run(cfg, out_path=out_path, progress=True, ram_limit_bytes=ram_limit)
    timer.end(label)
    return d


def _load_z(
    path: Path, cfg: Config, d: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = cfg.air.c
    f_min = cfg.band.f_min
    f_max = float(d.f_max)  # type: ignore[attr-defined]
    freqs, z = zrad_from_h5(path, df_hz=2.0)
    _, a = _geometry(cfg, d)
    k = 2.0 * np.pi * freqs / c
    ka = k * a
    band = (freqs >= f_min) & (freqs <= f_max) & np.isfinite(z)
    return freqs, z, ka, band


def _largest_ka_under(err: np.ndarray, ka: np.ndarray, mask: np.ndarray, tol: float) -> float:
    ok = mask & (err <= tol)
    if not np.any(ok):
        return 0.0
    return float(np.max(ka[ok]))


def _plot_z_vs_ref(
    path: Path,
    ka: np.ndarray,
    z: np.ndarray,
    z_ref: np.ndarray,
    mask: np.ndarray,
    *,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = mask & np.isfinite(z.real) & np.isfinite(z_ref.real)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), layout="tight")
    ax1.plot(ka[m], np.real(z[m]), "b.", ms=3, label="FDTD")
    ax1.plot(ka[m], np.real(z_ref[m]), "k--", lw=1, label="ref")
    ax2.plot(ka[m], np.imag(z[m]), "r.", ms=3, label="FDTD")
    ax2.plot(ka[m], np.imag(z_ref[m]), "k--", lw=1, label="ref")
    ax2.set_xlabel(r"$ka$")
    ax1.set_ylabel("Re z")
    ax2.set_ylabel("Im z")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_g1_offsets(
    path: Path,
    ka: np.ndarray,
    z0: np.ndarray,
    shifts: dict[str, np.ndarray],
    mask: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = mask & np.isfinite(z0.real)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), layout="tight")
    ax1.plot(ka[m], np.real(z0[m]), "k-", lw=1, label="cell-aligned")
    ax2.plot(ka[m], np.imag(z0[m]), "k-", lw=1)
    for name, z in shifts.items():
        ax1.plot(ka[m], np.real(z[m]), ".", ms=3, label=name)
        ax2.plot(ka[m], np.imag(z[m]), ".", ms=3, label=name)
    ax2.set_xlabel(r"$ka$")
    ax1.set_ylabel("Re z")
    ax2.set_ylabel("Im z")
    ax1.legend(fontsize=8)
    fig.suptitle("C7 G1 offset invariance (baffle circle)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_g3_err(
    path: Path,
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4), layout="tight")
    for label, (ka, err, mask) in series.items():
        m = mask & np.isfinite(err)
        ax.semilogy(ka[m], err[m], ".", ms=3, label=label)
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel(r"$|z - z_{\mathrm{ref}}|$")
    ax.legend(fontsize=8)
    ax.set_title("C7 G3 radiator resolution")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _print_gates_g1_g3(
    *,
    g1_pass: bool,
    g1_detail: dict[str, float],
    g2_pass: bool,
    g2_max: float,
    ka_tight2: float,
    g3_max_err: dict[int, float],
    g3_pass: bool,
    order: float | None,
    gates: dict[str, Any],
) -> None:
    def _line(name: str, passed: bool, detail: str) -> None:
        mark = "PASS" if passed else "FAIL"
        print(f"  {name}: {mark}  {detail}", flush=True)

    print("\n=== C7-subcell gates (G1–G3) ===", flush=True)
    _line(
        "G1",
        g1_pass,
        "max |z_shift−z_0| "
        + ", ".join(f"{k}={v:.4f}" for k, v in g1_detail.items())
        + f" (tol ≤{_G1_TOL})",
    )
    _line(
        "G2",
        g2_pass,
        f"max |z−z_ref|={g2_max:.4f} (tol ≤{_G2_TOL}); "
        f"largest ka with err≤{_G2_TIGHT}: {ka_tight2:.3g}",
    )
    if gates.get("g3_h_m"):
        print("  G3 grid spacing (report):", flush=True)
        for nc, hm in gates["g3_h_m"].items():
            print(f"    cells_across={nc}: h={hm:.6g} m", flush=True)
    for n_cells in _G3_CELLS:
        e = g3_max_err[n_cells]
        if n_cells == _G3_GATE_CELLS:
            _line(
                f"G3 ({n_cells} cells)",
                g3_pass,
                f"max |z−z_ref|={e:.4f}  gate tol ≤{_G3_TOL}",
            )
        else:
            print(f"  G3 report {n_cells} cells: max |z−z_ref|={e:.4f}", flush=True)
    if order is not None:
        print(f"  G3 observed convergence order (coarse pairs): p≈{order:.2f}", flush=True)


def _convergence_order(
    cells: tuple[int, ...],
    max_err: dict[int, float],
    ref_cells: int,
) -> float | None:
    coarse = [c for c in cells if c < ref_cells and c in max_err]
    if len(coarse) < 2:
        return None
    c0, c1 = coarse[0], coarse[1]
    e0, e1 = max_err[c0], max_err[c1]
    if e0 <= 0 or e1 <= 0 or e0 <= e1:
        return None
    return float(math.log(e0 / e1) / math.log(float(c1) / float(c0)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C7-subcell validation")
    parser.add_argument(
        "--g1-config",
        type=Path,
        default=ROOT / "configs" / "c07_g1.json",
    )
    parser.add_argument(
        "--g2-config",
        type=Path,
        default=ROOT / "configs" / "c07_g2_rect.json",
    )
    parser.add_argument(
        "--g3-config",
        type=Path,
        default=ROOT / "configs" / "c07_g3.json",
    )
    parser.add_argument(
        "--racetrack-config",
        type=Path,
        default=ROOT / "configs" / "c07_racetrack.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "validation" / "results")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--ram-gb", type=float, default=DEFAULT_RAM_LIMIT_BYTES / 1e9)
    args = parser.parse_args(argv)

    ram_limit = float(args.ram_gb) * 1e9
    timer = vc.RunTimer()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)

    gates: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    rt_h: dict[str, float] = {}
    rt_report: dict[str, Any] = {}
    gate_pass = True

    with tempfile.TemporaryDirectory(prefix="c07_") as tmpdir:
        tmp = Path(tmpdir)

        # --- G1: offset invariance (circle, baffle) ---
        cfg_g1 = Config.from_json(args.g1_config.read_text(encoding="utf-8"))
        d_align = derive(cfg_g1)
        h = d_align.h
        shifts_phys = {
            "shift_a": (0.25 * h, 0.25 * h),
            "shift_b": (0.5 * h, 0.37 * h),
        }
        g1_runs: dict[str, Path] = {}
        g1_cfgs: dict[str, Config] = {"aligned": cfg_g1}
        for name, (xc, yc) in shifts_phys.items():
            cpy = copy.deepcopy(cfg_g1)
            cpy.radiator.x_c = xc
            cpy.radiator.y_c = yc
            g1_cfgs[name] = cpy

        g1_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, cfg in g1_cfgs.items():
            h5 = tmp / f"g1_{name}.h5"
            _run_fdtd(cfg, h5, ram_limit=ram_limit, timer=timer, label=f"g1_{name}")
            g1_runs[name] = h5
            d_run = derive(cfg)
            g1_data[name] = _load_z(h5, cfg, d_run)

        freqs0, z0, ka0, band0 = g1_data["aligned"]
        g1_detail: dict[str, float] = {}
        g1_pass = True
        for name in shifts_phys:
            _, z_s, _, band_s = g1_data[name]
            m = band0 & band_s
            delta = float(np.max(np.abs(z_s[m] - z0[m])))
            g1_detail[name] = delta
            if delta > _G1_TOL:
                g1_pass = False
        gates["g1_pass"] = g1_pass
        gates["g1_max_delta"] = g1_detail

        # --- G2: rectangle vs reference ---
        cfg_g2 = Config.from_json(args.g2_config.read_text(encoding="utf-8"))
        h5_g2 = tmp / "g2_rect.h5"
        d_g2 = _run_fdtd(cfg_g2, h5_g2, ram_limit=ram_limit, timer=timer, label="g2_rect")
        freqs2, z2, ka2, band2 = _load_z(h5_g2, cfg_g2, d_g2)
        W, H = cfg_g2.radiator.dims[0], cfg_g2.radiator.dims[1]
        c = cfg_g2.air.c
        k2 = 2.0 * np.pi * freqs2 / c
        z_ref2 = np.asarray(z_baffled_rect(k2, W, H), dtype=np.complex128)
        err2 = np.abs(z2 - z_ref2)
        m2 = band2 & np.isfinite(z2) & np.isfinite(z_ref2)
        g2_max = float(np.max(err2[m2]))
        g2_pass = g2_max <= _G2_TOL
        ka_tight2 = _largest_ka_under(err2, ka2, m2, _G2_TIGHT)
        gates["g2_pass"] = g2_pass
        gates["g2_max_err"] = g2_max
        gates["g2_ka_tight"] = ka_tight2

        # --- G3: circle resolution ---
        cfg_g3_base = Config.from_json(args.g3_config.read_text(encoding="utf-8"))
        a_g3 = _a_equiv(cfg_g3_base)
        cfg_g3_base.band.f_max = _f_max_ka(
            ka_max=_G3_KA_MAX, a_m=a_g3, c=cfg_g3_base.air.c
        )
        g3_max_err: dict[int, float] = {}
        g3_h: dict[str, float] = {}
        g3_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        g3_runs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        g3_a: dict[int, float] = {}
        for n_cells in _G3_CELLS:
            cfg = copy.deepcopy(cfg_g3_base)
            cfg.numerics.cells_across_radiator = n_cells
            h5 = tmp / f"g3_{n_cells}.h5"
            d_g3 = _run_fdtd(
                cfg, h5, ram_limit=ram_limit, timer=timer, label=f"g3_{n_cells}c"
            )
            freqs, z, ka, band = _load_z(h5, cfg, d_g3)
            _, a_run = _geometry(cfg, d_g3)
            g3_a[n_cells] = a_run
            g3_runs[n_cells] = (ka, z, band)
            g3_h[str(n_cells)] = float(d_g3.h)

        _fail_if_duplicate_h(g3_h, "G3 cells_across sweep")

        ka_g3, z_g3_aligned, g3_ka_meta = _runs_on_common_ka(
            g3_runs,
            ref_key=max(_G3_CELLS),
            label="G3",
            a_by_key=g3_a,
        )
        z_ref_g3 = np.asarray(z_baffled_circle(ka_g3), dtype=np.complex128)
        for n_cells in _G3_CELLS:
            err = np.abs(z_g3_aligned[n_cells] - z_ref_g3)
            g3_max_err[n_cells] = float(np.max(err))
            g3_series[f"{n_cells} cells"] = (ka_g3, err, np.ones(ka_g3.shape, dtype=bool))

        g3_pass = g3_max_err.get(_G3_GATE_CELLS, float("inf")) <= _G3_TOL
        order = _convergence_order(_G3_CELLS, g3_max_err, ref_cells=max(_G3_CELLS))
        gates["g3_pass"] = g3_pass
        gates["g3_max_err"] = g3_max_err
        gates["g3_convergence_order"] = order
        gates["g3_h_m"] = g3_h
        gates["g3_ka_align"] = g3_ka_meta

        gate_pass = g1_pass and g2_pass and g3_pass

        if not args.no_plot:
            _plot_g1_offsets(
                args.out / f"c07_g1_{stamp}.png",
                ka0,
                z0,
                {k: g1_data[k][1] for k in shifts_phys},
                band0,
            )
            _plot_z_vs_ref(
                args.out / f"c07_g2_{stamp}.png",
                ka2,
                z2,
                z_ref2,
                m2,
                title="C7 G2 rectangle 0.10×0.05 m (baffle)",
            )
            _plot_g3_err(args.out / f"c07_g3_{stamp}.png", g3_series)

        _print_gates_g1_g3(
            g1_pass=g1_pass,
            g1_detail=g1_detail,
            g2_pass=g2_pass,
            g2_max=g2_max,
            ka_tight2=ka_tight2,
            g3_max_err=g3_max_err,
            g3_pass=g3_pass,
            order=order,
            gates=gates,
        )
        print(f"  Overall gate (G1∧G2∧G3): {'PASS' if gate_pass else 'FAIL'}", flush=True)

        json_path = args.out / f"c07_subcell_{stamp}.json"

        def _write_payload(extra_reports: dict[str, Any] | None = None) -> None:
            rep = dict(reports)
            if extra_reports:
                rep.update(extra_reports)
            payload: dict[str, Any] = {
                "check": "c07_subcell",
                "timestamp_utc": stamp,
                "h_aligned_m": h,
                "shift_offsets_m": {k: list(v) for k, v in shifts_phys.items()},
                "gates": gates,
                "reports": rep,
                "pass": gate_pass,
                "wall_s_est": derive(Config.from_json(args.g1_config.read_text())).n_steps
                * derive(Config.from_json(args.g1_config.read_text())).total_cells
                / _C1_UPDATES_PER_S,
                "phases_s": timer.phases,
            }
            json_path.write_text(
                json.dumps(vc.to_json_safe(payload), indent=2), encoding="utf-8"
            )

        _write_payload()
        timer.banner("output", f"{json_path.name} (G1–G3)")

        # --- Racetrack resolution (report only; must not block gates) ---
        print("\n=== C7 racetrack report (non-gated) ===", flush=True)
        try:
            cfg_rt = Config.from_json(args.racetrack_config.read_text(encoding="utf-8"))
            a_rt = _a_equiv(cfg_rt)
            cfg_rt.band.f_max = _f_max_ka(
                ka_max=_G3_KA_MAX, a_m=a_rt, c=cfg_rt.air.c
            )
            rt_z: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            rt_h = {}
            rt_a: dict[int, float] = {}
            for n_cells in _RT_CELLS:
                cfg = copy.deepcopy(cfg_rt)
                cfg.numerics.cells_across_radiator = n_cells
                h5 = tmp / f"rt_{n_cells}.h5"
                d_rt = _run_fdtd(
                    cfg, h5, ram_limit=ram_limit, timer=timer, label=f"rt_{n_cells}c"
                )
                _freqs, z, ka, band = _load_z(h5, cfg, d_rt)
                _, a_run = _geometry(cfg, d_rt)
                rt_a[n_cells] = a_run
                rt_z[n_cells] = (ka, z, band)
                rt_h[str(n_cells)] = float(d_rt.h)

            _fail_if_duplicate_h(rt_h, "racetrack cells_across sweep")

            ka_rt, z_rt_aligned, rt_ka_meta = _runs_on_common_ka(
                rt_z,
                ref_key=24,
                label="racetrack",
                a_by_key=rt_a,
            )
            z_ref_rt = z_rt_aligned[24]
            rt_report = {
                "h_m": rt_h,
                "ka_align": rt_ka_meta,
                **{
                    f"max_abs_z{n_cells}_minus_z24": float(
                        np.max(np.abs(z_rt_aligned[n_cells] - z_ref_rt))
                    )
                    for n_cells in (10, 16)
                },
            }
            reports["racetrack"] = rt_report
            for nc, hm in rt_h.items():
                print(f"    cells_across={nc}: h={hm:.6g} m", flush=True)
            lo, hi = rt_ka_meta["ka_overlap"]
            print(f"    common ka overlap [{lo:.4g}, {hi:.4g}], n={ka_rt.size}", flush=True)
            for n_cells in (10, 16):
                key = f"max_abs_z{n_cells}_minus_z24"
                print(f"    {key}={rt_report[key]:.4f}", flush=True)
            _write_payload()
            timer.banner("output", f"{json_path.name} (+racetrack)")
        except Exception as exc:
            print(f"  Racetrack report FAILED: {exc}", flush=True)
            reports["racetrack_error"] = str(exc)
            _write_payload()

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())