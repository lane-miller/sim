#!/usr/bin/env python3
"""Step 10b2: C13a rigid-sphere scattering ratio gate and C16 convergence."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_OUT = _PKG / "outputs" / "step10b2"
_CFG = _PKG / "configs" / "c13_sphere_ps.json"

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import Config  # noqa: E402
from boxrad.geom.geometry import build_geometry  # noqa: E402
from boxrad.geomrun import dry_run_geom, prepare_geom_run  # noqa: E402
from boxrad.post.transfer import h_transfer_err, probe_fdtd_pressure_spectrum  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402
from boxrad.validation.analytic import monopole_free, sphere_point_source  # noqa: E402
from validation._common import to_json_safe  # noqa: E402

_F64 = np.float64
_NS = (6, 9, 12, 18)
_FREQS_C13A = (300.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0)
_FREQS_C16 = (500.0, 1000.0, 2000.0, 3000.0)
_FAR_R = 2.0
_FAR_DIRS = (
    ("+z", np.array([0.0, 0.0, 1.0])),
    ("-z", np.array([0.0, 0.0, -1.0])),
    ("+x", np.array([1.0, 0.0, 0.0])),
    ("diag_xz", np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)),
)
_SHIFT_DELTAS = (0.25, 0.37, 0.5)
_CENTER = np.array([0.0, 0.0, 0.0], dtype=_F64)
_SPHERE_A = 0.05
_SPHERE_VOL = (4.0 / 3.0) * math.pi * _SPHERE_A**3
_GATE_C13A_DB = 0.1
_GATE_C13A_DEG = 1.0
_GATE_C16_P = 1.5
_GATE_SHIFT = 0.003
_CELL_ABORT = 15_000_000
_CFL_SWEEP_VALUES = (0.7, 0.5)
_CFL_SWEEP_REPORT = (0.99, *_CFL_SWEEP_VALUES)


def _probes() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, -0.15],
            [0.15, 0.0, 0.0],
            [0.0, 0.0, 0.15],
            [0.0, 1.0 / 15.0, 0.0],
            [0.0, 0.0, -1.0 / 15.0],
        ],
        dtype=_F64,
    )


def _load_base() -> dict[str, Any]:
    return json.loads(_CFG.read_text(encoding="utf-8"))


def _config_cfl_default() -> float:
    return float(_load_base()["numerics"]["cfl_safety"])


def _cfl_suffix(cfl_safety: float) -> str:
    return f"_cfl{cfl_safety:g}"


def _out_path(tag: str, file_suffix: str) -> Path:
    return _OUT / f"{tag}{file_suffix}.h5"


def _content_bbox(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    geom = build_geometry(cfg)
    src = np.asarray(cfg.numerics.test_source.position, dtype=_F64)  # type: ignore[union-attr]
    if geom.solid.faces.shape[0] == 0:
        lo = hi = src.copy()
    else:
        xlo, ylo, zlo, xhi, yhi, zhi = geom.solid.bbox
        lo = np.minimum(np.array([xlo, ylo, zlo], dtype=_F64), src)
        hi = np.maximum(np.array([xhi, yhi, zhi], dtype=_F64), src)
    return lo, hi


def _gap_cells(lo: np.ndarray, hi: np.ndarray, pts: np.ndarray, h: float) -> int:
    margins: list[float] = []
    for a in range(3):
        margins.append(max(0.0, float(lo[a]) - float(np.min(pts[:, a]))))
        margins.append(max(0.0, float(np.max(pts[:, a])) - float(hi[a])))
    return int(math.ceil(max(margins) / h)) + 3


def _build_cfg(
    *,
    n: int,
    sphere: bool,
    center: np.ndarray | None = None,
    cfl_safety: float | None = None,
) -> Config:
    raw = _load_base()
    h = 0.05 / n
    raw["numerics"]["h_override"] = h
    if cfl_safety is not None:
        raw["numerics"]["cfl_safety"] = float(cfl_safety)
    if not sphere:
        raw["parts"] = []
    elif center is not None:
        raw["parts"][0]["position"] = center.tolist()
    cfg = Config.from_dict(raw)
    pts = np.asarray(raw["output"]["probes"], dtype=_F64)
    lo, hi = _content_bbox(cfg)
    gap = _gap_cells(lo, hi, pts, h)
    raw["numerics"]["gap_cells"] = gap
    cfg = Config.from_dict(raw)
    est = dry_run_geom(cfg)
    h1 = float(est["h"])
    if abs(h1 - h) > 1e-9:
        gap = _gap_cells(lo, hi, pts, h1)
        raw["numerics"]["gap_cells"] = gap
        cfg = Config.from_dict(raw)
    return cfg


def _assert_probe_centers(cfg: Config) -> None:
    run = prepare_geom_run(cfg)
    h = float(run.derived.h)
    x0 = run.gg.x0.astype(_F64)
    for p in _probes():
        rel = np.round((p - x0) / h - 0.5)
        cc = x0 + (rel + 0.5) * h
        off = float(np.linalg.norm(cc - p))
        assert off <= 1e-9, f"probe {p} off grid by {off} m"


def _staircase_available(_raw: dict[str, Any]) -> bool:
    return False


def _h_fd(path: Path, f_hz: float, probe_i: int | None, pt: np.ndarray | None) -> complex:
    with Result(path, frame="world") as res:
        if probe_i is not None:
            with h5py.File(path, "r") as f:
                t_pr = np.asarray(f["probes/times_p"][()], dtype=_F64)
                p_ts = np.asarray(f["probes/p"][()], dtype=_F64)
            p_fd = probe_fdtd_pressure_spectrum(p_ts, t_pr, f_hz, probe_i)
            v = res._radiator_v_at_f(f_hz)
            return p_fd / v
        assert pt is not None
        p_fd = res.field(pt[np.newaxis, :], f_hz, frame="world")[0, 0]
        v = res._radiator_v_at_f(f_hz)
        return p_fd / v


def _h_an(
    pt: np.ndarray,
    f_hz: float,
    src: np.ndarray,
    rho: float,
    c: float,
    center: np.ndarray,
    *,
    sphere: bool,
) -> complex:
    f_scalar = float(np.asarray(f_hz, dtype=_F64).reshape(()))
    points = np.atleast_2d(np.asarray(pt, dtype=_F64))
    assert points.shape == (1, 3), f"points must be (1, 3), got {points.shape}"
    center_v = np.asarray(center, dtype=_F64).reshape(3)
    src_v = np.asarray(src, dtype=_F64).reshape(3)
    assert center_v.shape == (3,)
    assert src_v.shape == (3,)
    if sphere:
        return complex(
            sphere_point_source(points, f_scalar, center_v, _SPHERE_A, src_v, rho, c)[0]
        )
    _, p = monopole_free(points, f_scalar, src_v, rho, c)
    assert p.shape == (1, 1)
    return complex(p[0, 0])


def _ratio_db_deg(R_fd: complex, R_an: complex) -> tuple[float, float]:
    return h_transfer_err(R_an, R_fd)


def _run_labeled(label: str, cfg: Config, out: Path) -> None:
    print(f"\n>>> run {label} -> {out.name}")
    t0 = time.perf_counter()
    run(cfg, out_path=out, progress=True)
    print(f"    wall {time.perf_counter() - t0:.1f} s")


def _planned_run_tags() -> list[tuple[str, int, bool, np.ndarray | None]]:
    specs: list[tuple[str, int, bool, np.ndarray | None]] = []
    for n in _NS:
        specs.append((f"sphere_n{n}", n, True, None))
        specs.append((f"free_n{n}", n, False, None))
    h9 = 0.05 / 9
    for delta in _SHIFT_DELTAS:
        shift = delta * h9 * np.array([1.0, 1.0, 1.0], dtype=_F64)
        specs.append((f"sphere_n9_d{delta:g}", 9, True, _CENTER + shift))
    return specs


def _prepare_runs(
    cfl_override: float | None,
) -> tuple[list[dict[str, Any]], list[tuple[str, Config, Path]]]:
    rows: list[dict[str, Any]] = []
    run_specs: list[tuple[str, Config, Path]] = []
    file_suffix = _cfl_suffix(cfl_override) if cfl_override is not None else ""
    for tag, n, sphere, cen in _planned_run_tags():
        cfg = _build_cfg(n=n, sphere=sphere, center=cen, cfl_safety=cfl_override)
        est = dry_run_geom(cfg)
        shape = tuple(int(x) for x in est["grid_shape"])
        ncell = int(np.prod(shape))
        h = float(est["h"])
        gap = int(cfg.numerics.gap_cells)  # type: ignore[arg-type]
        n_steps = int(est["n_steps"])
        out = _out_path(tag, file_suffix)
        print(
            f"preflight {tag}{file_suffix}  n={n}  h={h:.6g} m  gap_cells={gap}  "
            f"grid={shape}  cells={ncell}  n_steps={n_steps}"
            + (f"  cfl={cfl_override:g}" if cfl_override is not None else "")
        )
        if ncell > _CELL_ABORT:
            raise SystemExit(
                f"abort: planned run {tag!r} needs {ncell} cells (limit {_CELL_ABORT}); "
                "check gap_cells (near-field probes only) and resolution."
            )
        rows.append(
            {
                "tag": tag,
                "file_suffix": file_suffix,
                "n": n,
                "h_m": h,
                "gap_cells": gap,
                "grid_shape": shape,
                "cells": ncell,
                "n_steps": n_steps,
                "cfl_safety": cfl_override if cfl_override is not None else _config_cfl_default(),
            }
        )
        run_specs.append((tag, cfg, out))
    return rows, run_specs


def _prepare_cfl_sweep_runs() -> tuple[list[dict[str, Any]], list[tuple[str, Config, Path]]]:
    rows: list[dict[str, Any]] = []
    run_specs: list[tuple[str, Config, Path]] = []
    for cfl in _CFL_SWEEP_VALUES:
        suf = _cfl_suffix(cfl)
        for sphere in (True, False):
            tag = f"{'sphere' if sphere else 'free'}_n9{suf}"
            out = _OUT / f"{tag}.h5"
            if out.is_file():
                print(f"preflight {tag}: skip FDTD (existing {out.name})")
                continue
            cfg = _build_cfg(n=9, sphere=sphere, cfl_safety=cfl)
            est = dry_run_geom(cfg)
            shape = tuple(int(x) for x in est["grid_shape"])
            ncell = int(np.prod(shape))
            h = float(est["h"])
            gap = int(cfg.numerics.gap_cells)  # type: ignore[arg-type]
            n_steps = int(est["n_steps"])
            n_floored = int(est.get("n_floored", 0))
            deq_h = float(est.get("delta_eq_over_h", 0.0))
            print(
                f"preflight {tag}  cfl={cfl}  h={h:.6g} m  gap_cells={gap}  "
                f"grid={shape}  cells={ncell}  n_steps={n_steps}  "
                f"n_floored={n_floored}  delta_eq/h={deq_h:.4g}"
            )
            if ncell > _CELL_ABORT:
                raise SystemExit(
                    f"abort: planned run {tag!r} needs {ncell} cells (limit {_CELL_ABORT}); "
                    "check gap_cells (near-field probes only) and resolution."
                )
            rows.append(
                {
                    "tag": tag,
                    "cfl_safety": cfl,
                    "n": 9,
                    "h_m": h,
                    "gap_cells": gap,
                    "grid_shape": shape,
                    "cells": ncell,
                    "n_steps": n_steps,
                    "n_floored": n_floored,
                    "delta_eq_over_h": deq_h,
                }
            )
            run_specs.append((tag, cfg, out))
    return rows, run_specs


def _collect_ratio_block(
    sphere_h5: Path,
    free_h5: Path,
    freqs: tuple[float, ...],
    center: np.ndarray,
    rho: float,
    c: float,
    src: np.ndarray,
    *,
    include_abs_solo: bool,
) -> dict[str, Any]:
    probes = _probes()
    out: dict[str, Any] = {"freqs_hz": list(freqs), "probes": [], "far_field": []}
    if include_abs_solo:
        out["solo_sphere"] = []
    for f_hz in freqs:
        row_p: dict[str, Any] = {"f_hz": f_hz, "entries": []}
        row_ff: dict[str, Any] = {"f_hz": f_hz, "entries": []}
        row_solo: dict[str, Any] | None = {"f_hz": f_hz, "entries": []} if include_abs_solo else None
        for qi, pt in enumerate(probes):
            h_s = _h_fd(sphere_h5, f_hz, qi, None)
            h_f = _h_fd(free_h5, f_hz, qi, None)
            R_fd = h_s / h_f
            p_s = _h_an(pt, f_hz, src, rho, c, center, sphere=True)
            p_f = _h_an(pt, f_hz, src, rho, c, center, sphere=False)
            R_an = p_s / p_f
            db, deg = _ratio_db_deg(R_fd, R_an)
            ran_db = 20.0 * math.log10(max(abs(R_an), 1e-30))
            row_p["entries"].append(
                {"probe": qi, "db": db, "deg": deg, "ran_db": ran_db, "err_cplx": complex(R_fd / R_an - 1.0)}
            )
            if row_solo is not None:
                db_s, deg_s = h_transfer_err(p_s, h_s)
                row_solo["entries"].append({"probe": qi, "db": db_s, "deg": deg_s})
            print(f"  f={f_hz:.0f} P{qi} ratio dB {db:+.4f} deg {deg:+.2f}  |R_an|={ran_db:.2f} dB")
        for dname, u in _FAR_DIRS:
            pt = _FAR_R * u
            h_s = _h_fd(sphere_h5, f_hz, None, pt)
            h_f = _h_fd(free_h5, f_hz, None, pt)
            R_fd = h_s / h_f
            p_s = _h_an(pt, f_hz, src, rho, c, center, sphere=True)
            p_f = _h_an(pt, f_hz, src, rho, c, center, sphere=False)
            R_an = p_s / p_f
            db, deg = _ratio_db_deg(R_fd, R_an)
            row_ff["entries"].append({"dir": dname, "db": db, "deg": deg})
            print(f"  f={f_hz:.0f} far {dname} ratio dB {db:+.4f} deg {deg:+.2f}")
        out["probes"].append(row_p)
        out["far_field"].append(row_ff)
        if row_solo is not None:
            out["solo_sphere"].append(row_solo)
    return out


def _c16_metric(
    sphere_path: Path,
    free_path: Path,
    center: np.ndarray,
    rho: float,
    c: float,
    src: np.ndarray,
) -> float:
    e = 0.0
    for f_hz in _FREQS_C16:
        for qi, pt in enumerate(_probes()):
            h_s = _h_fd(sphere_path, f_hz, qi, None)
            h_f = _h_fd(free_path, f_hz, qi, None)
            R_fd = h_s / h_f
            p_s = _h_an(pt, f_hz, src, rho, c, center, sphere=True)
            p_f = _h_an(pt, f_hz, src, rho, c, center, sphere=False)
            R_an = p_s / p_f
            e = max(e, abs(R_fd / R_an - 1.0))
    return float(e)


def _fit_order(hs: list[float], es: list[float]) -> float:
    log_h = np.log(np.asarray(hs, dtype=_F64))
    log_e = np.log(np.asarray(es, dtype=_F64))
    p, _ = np.polyfit(log_h, log_e, 1)
    return float(p)


def _pairwise_orders(hs: list[float], es: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(len(hs) - 1):
        if es[i] <= 0 or es[i + 1] <= 0:
            out.append(float("nan"))
        else:
            out.append(math.log(es[i] / es[i + 1]) / math.log(hs[i] / hs[i + 1]))
    return out


def _json_complex_safe(obj: Any) -> Any:
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, dict):
        return {str(k): _json_complex_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_complex_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_complex_safe(v) for v in obj]
    return obj


def _sphere_run_tags() -> list[str]:
    tags = [f"sphere_n{n}" for n in _NS]
    for delta in _SHIFT_DELTAS:
        tags.append(f"sphere_n9_d{delta:g}")
    return tags


def _cfg_for_sphere_tag(tag: str, cfl_override: float | None) -> Config:
    if tag.startswith("sphere_n9_d"):
        delta = float(tag.split("_d", 1)[1])
        h9 = 0.05 / 9
        cen = _CENTER + delta * h9 * np.array([1.0, 1.0, 1.0], dtype=_F64)
        return _build_cfg(n=9, sphere=True, center=cen, cfl_safety=cfl_override)
    n = int(tag.removeprefix("sphere_n"))
    return _build_cfg(n=n, sphere=True, center=None, cfl_safety=cfl_override)


def _free_tag_for_sphere(tag: str) -> str:
    if tag == "sphere_n9" or tag.startswith("sphere_n9_d"):
        return "free_n9"
    n = int(tag.removeprefix("sphere_n"))
    return f"free_n{n}"


def _center_for_sphere_tag(tag: str) -> np.ndarray:
    if tag.startswith("sphere_n9_d"):
        delta = float(tag.split("_d", 1)[1])
        h9 = 0.05 / 9
        return _CENTER + delta * h9 * np.array([1.0, 1.0, 1.0], dtype=_F64)
    return _CENTER.copy()


def _load_fraction_floor_stats(
    h5_path: Path, cfg: Config
) -> tuple[dict[str, Any], dict[str, Any], float]:
    with h5py.File(h5_path, "r") as f:
        if "meta" in f:
            h = float(f["meta"].attrs["h"])
            raw = f["meta"].attrs.get("geom_stats_json")
            if raw is not None:
                text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
                blob = json.loads(text)
                return (
                    dict(blob["fractions_stats"]),
                    dict(blob["floor_stats"]),
                    h,
                )
    run = prepare_geom_run(cfg)
    return (
        dict(run.info["fractions_stats"]),
        dict(run.info["floor_stats"]),
        float(run.derived.h),
    )


def _max_probe_ratio_err(
    sphere_h5: Path,
    free_h5: Path,
    f_hz: float,
    center: np.ndarray,
    rho: float,
    c: float,
    src: np.ndarray,
) -> float:
    err = 0.0
    for qi, pt in enumerate(_probes()):
        h_s = _h_fd(sphere_h5, f_hz, qi, None)
        h_f = _h_fd(free_h5, f_hz, qi, None)
        R_fd = h_s / h_f
        p_s = _h_an(pt, f_hz, src, rho, c, center, sphere=True)
        p_f = _h_an(pt, f_hz, src, rho, c, center, sphere=False)
        R_an = p_s / p_f
        err = max(err, abs(R_fd / R_an - 1.0))
    return float(err)


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    xv = np.asarray(x, dtype=_F64)
    yv = np.asarray(y, dtype=_F64)
    if float(np.std(xv)) < 1e-30 or float(np.std(yv)) < 1e-30:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _floor_diagnostic(
    paths: dict[str, Path],
    rho: float,
    c: float,
    src: np.ndarray,
    cfl_override: float | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_tag: dict[str, dict[str, Any]] = {}
    print(
        "\nfloor diagnostic (sphere runs)\n"
        f"{'run':<22} {'h_mm':>9} {'n_cut':>7} {'n_floor':>8} "
        f"{'dV/V_sph':>10} {'deq/h':>8} {'|R-1|@1k':>10} {'|R-1|@3k':>10}"
    )
    for tag in _sphere_run_tags():
        sp = paths[tag]
        free_tag = _free_tag_for_sphere(tag)
        fp = paths[free_tag]
        cfg = _cfg_for_sphere_tag(tag, cfl_override)
        center = _center_for_sphere_tag(tag)
        fr_st, fl_st, h = _load_fraction_floor_stats(sp, cfg)
        n_cut = int(fr_st.get("n_cut_cells", 0))
        n_floored = int(fl_st.get("n_floored", 0))
        dv = float(fl_st.get("delta_V", 0.0))
        deq_h = float(fl_st.get("delta_eq_over_h", 0.0))
        dv_over_v = dv / _SPHERE_VOL
        e1k = _max_probe_ratio_err(sp, fp, 1000.0, center, rho, c, src)
        e3k = _max_probe_ratio_err(sp, fp, 3000.0, center, rho, c, src)
        row = {
            "run": tag,
            "h_m": h,
            "n_cut_cells": n_cut,
            "n_floored": n_floored,
            "min_raw_beta_cut": float(fl_st.get("min_raw_beta_cut", 0.0)),
            "delta_V": dv,
            "delta_V_over_sphere_volume": dv_over_v,
            "delta_eq_over_h": deq_h,
            "max_abs_R_ratio_err_1khz_P0_P4": e1k,
            "max_abs_R_ratio_err_3khz_P0_P4": e3k,
        }
        rows.append(row)
        by_tag[tag] = row
        print(
            f"{tag:<22} {h * 1e3:9.4g} {n_cut:7d} {n_floored:8d} "
            f"{dv_over_v:10.4g} {deq_h:8.4g} {e1k:10.4g} {e3k:10.4g}"
        )

    shift_tags = ["sphere_n9"] + [f"sphere_n9_d{d:g}" for d in _SHIFT_DELTAS]
    deq_vals = [by_tag[t]["delta_eq_over_h"] for t in shift_tags]
    nf_vals = [float(by_tag[t]["n_floored"]) for t in shift_tags]
    e3_vals = [by_tag[t]["max_abs_R_ratio_err_3khz_P0_P4"] for t in shift_tags]
    r_deq = _pearson(deq_vals, e3_vals)
    r_nf = _pearson(nf_vals, e3_vals)
    print(
        f"\nn=9 shift Pearson vs max |R_fd/R_an-1| @3 kHz (n={len(shift_tags)}): "
        f"delta_eq/h corr={r_deq:+.4f}  n_floored corr={r_nf:+.4f}"
    )
    return {
        "rows": rows,
        "pearson_n9_shift_vs_err_3khz": {
            "tags": shift_tags,
            "delta_eq_over_h_vs_err": r_deq,
            "n_floored_vs_err": r_nf,
        },
    }


def _air_and_source() -> tuple[float, float, np.ndarray]:
    raw = _load_base()
    return (
        float(raw["air"]["rho"]),
        float(raw["air"]["c"]),
        np.array(raw["numerics"]["test_source"]["position"], dtype=_F64),
    )


def _register_tag_paths(
    paths: dict[str, Path],
    file_suffix: str,
    *,
    cfl_sweep: bool,
) -> None:
    for tag, _, _, _ in _planned_run_tags():
        paths[tag] = _out_path(tag, file_suffix)
    if cfl_sweep:
        for cfl in _CFL_SWEEP_VALUES:
            suf = _cfl_suffix(cfl)
            for stem in ("sphere_n9", "free_n9"):
                sweep_tag = f"{stem}{suf}"
                paths[sweep_tag] = _OUT / f"{sweep_tag}.h5"


def _paths_from_existing(
    file_suffix: str,
    *,
    cfl_sweep: bool,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    missing: list[Path] = []
    _register_tag_paths(paths, file_suffix, cfl_sweep=cfl_sweep)
    for tag, _, _, _ in _planned_run_tags():
        p = paths[tag]
        if not p.is_file():
            missing.append(p)
    if missing:
        msg = "post-only: missing HDF5 output(s); run the showcase without --post-only first:\n"
        msg += "\n".join(f"  {m}" for m in missing)
        raise SystemExit(msg)
    return paths


def _n_steps_from_h5(h5_path: Path) -> int | None:
    with h5py.File(h5_path, "r") as f:
        if "meta" not in f:
            return None
        raw = f["meta"].attrs.get("n_steps")
        if raw is None:
            return None
        return int(raw)


def _cfl_sweep_tag_for(cfl: float) -> tuple[str, str]:
    if abs(cfl - _config_cfl_default()) < 1e-12:
        return "sphere_n9", "free_n9"
    suf = _cfl_suffix(cfl)
    return f"sphere_n9{suf}", f"free_n9{suf}"


def _cfl_sweep_paths(paths: dict[str, Path]) -> list[tuple[float, Path, Path]]:
    out: list[tuple[float, Path, Path]] = []
    for cfl in _CFL_SWEEP_REPORT:
        if abs(cfl - _config_cfl_default()) < 1e-12:
            sp = _out_path("sphere_n9", "")
            fp = _out_path("free_n9", "")
        else:
            sp_tag, fp_tag = _cfl_sweep_tag_for(cfl)
            sp = paths.get(sp_tag, _OUT / f"{sp_tag}.h5")
            fp = paths.get(fp_tag, _OUT / f"{fp_tag}.h5")
        if not sp.is_file() or not fp.is_file():
            continue
        out.append((cfl, sp, fp))
    return out


def _max_ratio_errors_c13a(
    sphere_h5: Path,
    free_h5: Path,
    freqs: tuple[float, ...],
    center: np.ndarray,
    rho: float,
    c: float,
    src: np.ndarray,
    *,
    include_far_field: bool,
) -> dict[str, float]:
    probes = _probes()
    max_db = 0.0
    max_deg = 0.0
    max_ff_db = 0.0
    max_ff_deg = 0.0
    for f_hz in freqs:
        for qi, pt in enumerate(probes):
            h_s = _h_fd(sphere_h5, f_hz, qi, None)
            h_f = _h_fd(free_h5, f_hz, qi, None)
            R_fd = h_s / h_f
            p_s = _h_an(pt, f_hz, src, rho, c, center, sphere=True)
            p_f = _h_an(pt, f_hz, src, rho, c, center, sphere=False)
            R_an = p_s / p_f
            db, deg = _ratio_db_deg(R_fd, R_an)
            max_db = max(max_db, abs(db))
            max_deg = max(max_deg, abs(deg))
        if include_far_field:
            for _dname, u in _FAR_DIRS:
                pt = _FAR_R * u
                h_s = _h_fd(sphere_h5, f_hz, None, pt)
                h_f = _h_fd(free_h5, f_hz, None, pt)
                R_fd = h_s / h_f
                p_s = _h_an(pt, f_hz, src, rho, c, center, sphere=True)
                p_f = _h_an(pt, f_hz, src, rho, c, center, sphere=False)
                R_an = p_s / p_f
                db, deg = _ratio_db_deg(R_fd, R_an)
                max_ff_db = max(max_ff_db, abs(db))
                max_ff_deg = max(max_ff_deg, abs(deg))
    out = {"max_probe_db": max_db, "max_probe_deg": max_deg}
    if include_far_field:
        out["max_far_db"] = max_ff_db
        out["max_far_deg"] = max_ff_deg
    return out


def _cfl_sweep_report(
    paths: dict[str, Path],
    rho: float,
    c: float,
    src: np.ndarray,
    preflight_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    print(
        "\ncfl_sweep (n=9)\n"
        f"{'cfl':>5} {'n_steps':>8} {'n_floor':>8} {'deq/h':>8} "
        f"{'|dB|1k P0-4':>11} {'|deg|1k':>8} {'|dB|1k far':>10} "
        f"{'|dB|3k P0-4':>11} {'|deg|3k':>8} {'|dB|3k far':>10}"
    )
    for cfl, sp, fp in _cfl_sweep_paths(paths):
        cfg_cfl = _build_cfg(
            n=9,
            sphere=True,
            cfl_safety=None if abs(cfl - _config_cfl_default()) < 1e-12 else cfl,
        )
        _, fl_st, _h = _load_fraction_floor_stats(sp, cfg_cfl)
        n_floored = int(fl_st.get("n_floored", 0))
        deq_h = float(fl_st.get("delta_eq_over_h", 0.0))
        n_steps = _n_steps_from_h5(sp)
        if n_steps is None:
            sp_tag = "sphere_n9" if abs(cfl - _config_cfl_default()) < 1e-12 else _cfl_sweep_tag_for(cfl)[0]
            for row in preflight_rows:
                if row.get("tag") == sp_tag:
                    n_steps = int(row["n_steps"])
                    break
        if n_steps is None:
            est = dry_run_geom(cfg_cfl)
            n_steps = int(est["n_steps"])
        m1k = _max_ratio_errors_c13a(
            sp, fp, (1000.0,), _CENTER, rho, c, src, include_far_field=True
        )
        m3k = _max_ratio_errors_c13a(
            sp, fp, (3000.0,), _CENTER, rho, c, src, include_far_field=True
        )
        ent = {
            "cfl_safety": cfl,
            "n_steps": n_steps,
            "n_floored": n_floored,
            "delta_eq_over_h": deq_h,
            "at_1khz": m1k,
            "at_3khz": m3k,
        }
        entries.append(ent)
        print(
            f"{cfl:5.2f} {n_steps:8d} {n_floored:8d} {deq_h:8.4g} "
            f"{m1k['max_probe_db']:11.4f} {m1k['max_probe_deg']:8.2f} "
            f"{m1k.get('max_far_db', float('nan')):10.4f} "
            f"{m3k['max_probe_db']:11.4f} {m3k['max_probe_deg']:8.2f} "
            f"{m3k.get('max_far_db', float('nan')):10.4f}"
        )
    return {"entries": entries}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-only",
        action="store_true",
        help="Skip preflight and FDTD; post-process existing outputs/step10b2/*.h5",
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=None,
        metavar="X",
        help="Set numerics.cfl_safety=X on every run; suffix outputs and c13a_c16_cflX.* (default: config)",
    )
    parser.add_argument(
        "--cfl-sweep",
        action="store_true",
        help="Run only missing sphere_n9/free_n9 at cfl 0.7 and 0.5; report 0.99/0.7/0.5",
    )
    args = parser.parse_args(argv)

    cfl_override = args.cfl
    file_suffix = _cfl_suffix(cfl_override) if cfl_override is not None else ""
    report_stem = "c13a_c16" + (file_suffix if cfl_override is not None else "")

    _OUT.mkdir(parents=True, exist_ok=True)
    json_path = _OUT / f"{report_stem}.json"
    png_path = _OUT / f"{report_stem}.png"

    rho, c, src = _air_and_source()
    preflight: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}

    if args.post_only:
        print("=== post-only (existing HDF5) ===")
        paths = _paths_from_existing(file_suffix, cfl_sweep=args.cfl_sweep)
    elif args.cfl_sweep:
        print("=== cfl_sweep: preflight missing n=9 variants only ===")
        preflight, run_specs = _prepare_cfl_sweep_runs()
        _register_tag_paths(paths, file_suffix, cfl_sweep=True)
        if not run_specs:
            print("cfl_sweep: all variant outputs already present; no FDTD")
        t_all = time.perf_counter()
        with keep_awake("step10b2 c13a_c16 cfl_sweep"):
            for name, cfg, out in run_specs:
                paths[name] = out
                _run_labeled(name, cfg, out)
        if run_specs:
            print(f"cfl_sweep FDTD wall {time.perf_counter() - t_all:.1f} s")
        for tag, _, _, _ in _planned_run_tags():
            paths.setdefault(tag, _out_path(tag, file_suffix))
    else:
        print("=== preflight (all planned runs, before FDTD) ===")
        preflight, run_specs = _prepare_runs(cfl_override)
        cfg_ref = next(cfg for tag, cfg, _ in run_specs if tag == "sphere_n9")
        _assert_probe_centers(cfg_ref)
        t_all = time.perf_counter()
        with keep_awake("step10b2 c13a_c16"):
            for name, cfg, out in run_specs:
                paths[name] = out
                if out.is_file():
                    print(f"\n>>> skip {name} (existing {out.name})")
                    continue
                _run_labeled(name, cfg, out)
        print(f"total wall {time.perf_counter() - t_all:.1f} s")

    if _staircase_available(_load_base()):
        print("staircase: running n=9 comparison (not implemented in numerics)")
    else:
        print("staircase mode not available, comparison skipped")

    c13a = _collect_ratio_block(
        paths["sphere_n9"],
        paths["free_n9"],
        _FREQS_C13A,
        _CENTER,
        rho,
        c,
        src,
        include_abs_solo=True,
    )
    max_db = max(abs(e["db"]) for block in c13a["probes"] for e in block["entries"])
    max_deg = max(abs(e["deg"]) for block in c13a["probes"] for e in block["entries"])
    max_ff_db = max(abs(e["db"]) for block in c13a["far_field"] for e in block["entries"])
    max_ff_deg = max(abs(e["deg"]) for block in c13a["far_field"] for e in block["entries"])
    ok_c13a = (
        max_db <= _GATE_C13A_DB
        and max_deg <= _GATE_C13A_DEG
        and max_ff_db <= _GATE_C13A_DB
        and max_ff_deg <= _GATE_C13A_DEG
    )
    print(
        f"\ngate C13a: max probe dB={max_db:.4f} deg={max_deg:.2f}  "
        f"far dB={max_ff_db:.4f} deg={max_ff_deg:.2f}  {'PASS' if ok_c13a else 'FAIL'}"
    )

    hs: list[float] = []
    es: list[float] = []
    for n in _NS:
        h = 0.05 / n
        e = _c16_metric(paths[f"sphere_n{n}"], paths[f"free_n{n}"], _CENTER, rho, c, src)
        hs.append(h)
        es.append(e)
        print(f"C16 n={n} h={h * 1e3:.4g} mm  e(h)={e:.4g}")
    p_obs = _fit_order(hs, es)
    pair = _pairwise_orders(hs, es)
    ok_c16 = p_obs >= _GATE_C16_P
    print(f"C16 observed order p={p_obs:.3f}  pairwise={pair}  {'PASS' if ok_c16 else 'FAIL'}")

    shift_spread: list[dict[str, Any]] = []
    for f_gate in (1000.0, 3000.0):
        per_probe: list[dict[str, Any]] = []
        for qi in range(len(_probes())):
            errs: list[float] = []
            centers: list[np.ndarray] = [_CENTER]
            paths_shift = [paths["sphere_n9"]]
            h9 = 0.05 / 9
            for delta in _SHIFT_DELTAS:
                centers.append(_CENTER + delta * h9 * np.array([1.0, 1.0, 1.0]))
                paths_shift.append(paths[f"sphere_n9_d{delta:g}"])
            for cen, spath in zip(centers, paths_shift):
                pt = _probes()[qi]
                h_s = _h_fd(spath, f_gate, qi, None)
                h_f = _h_fd(paths["free_n9"], f_gate, qi, None)
                R_fd = h_s / h_f
                p_s = _h_an(pt, f_gate, src, rho, c, cen, sphere=True)
                p_f = _h_an(pt, f_gate, src, rho, c, cen, sphere=False)
                R_an = p_s / p_f
                errs.append(abs(R_fd / R_an - 1.0))
            spread = max(errs) - min(errs)
            per_probe.append({"probe": qi, "spread": spread, "errs": errs})
            print(f"shift f={f_gate:.0f} P{qi} spread={spread:.4f}")
        shift_spread.append({"f_hz": f_gate, "probes": per_probe})
    max_shift = max(p["spread"] for block in shift_spread for p in block["probes"])
    ok_shift = max_shift <= _GATE_SHIFT
    print(f"gate shift spread max={max_shift:.4f} (≤ {_GATE_SHIFT})  {'PASS' if ok_shift else 'FAIL'}")

    floor_diagnostic = _floor_diagnostic(paths, rho, c, src, cfl_override)

    cfl_sweep: dict[str, Any] | None = None
    if args.cfl_sweep:
        cfl_sweep = _cfl_sweep_report(paths, rho, c, src, preflight)

    results: dict[str, Any] = {
        "cfl_safety": cfl_override if cfl_override is not None else _config_cfl_default(),
        "output_suffix": file_suffix,
        "preflight": preflight,
        "c13a": c13a,
        "c16": {"h_m": hs, "e": es, "p_observed": p_obs, "pairwise_p": pair},
        "shift_spread": shift_spread,
        "floor_diagnostic": floor_diagnostic,
        "gates": {
            "c13a_pass": ok_c13a,
            "c16_pass": ok_c16,
            "shift_pass": ok_shift,
            "pass": bool(ok_c13a and ok_c16 and ok_shift),
        },
        "staircase": "not available",
    }
    if cfl_sweep is not None:
        results["cfl_sweep"] = cfl_sweep
    json_path.write_text(
        json.dumps(to_json_safe(_json_complex_safe(results)), indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for qi in range(5):
        dbs = [
            next(e["db"] for e in block["entries"] if e["probe"] == qi)
            for block in c13a["probes"]
        ]
        axes[0].plot(_FREQS_C13A, dbs, "o-", label=f"P{qi}", ms=4)
    axes[0].set_xlabel("f [Hz]")
    axes[0].set_ylabel("ratio error [dB]")
    axes[0].set_title("C13a probes")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].axhline(_GATE_C13A_DB, color="k", ls="--", lw=0.8)
    axes[0].axhline(-_GATE_C13A_DB, color="k", ls="--", lw=0.8)

    axes[1].loglog(hs, es, "o-", label="e(h)")
    h_ref = np.array(hs, dtype=_F64)
    e_ref = es[0] * (h_ref / h_ref[0]) ** 2
    axes[1].loglog(h_ref, e_ref, "k--", label="slope 2")
    axes[1].set_xlabel("h [m]")
    axes[1].set_ylabel("e(h)")
    axes[1].set_title(f"C16 p={p_obs:.2f}")
    axes[1].legend(fontsize=8)

    sp_1k = [p["spread"] for p in shift_spread[0]["probes"]]
    sp_3k = [p["spread"] for p in shift_spread[1]["probes"]]
    x = np.arange(5)
    w = 0.35
    axes[2].bar(x - w / 2, sp_1k, w, label="1 kHz")
    axes[2].bar(x + w / 2, sp_3k, w, label="3 kHz")
    axes[2].axhline(_GATE_SHIFT, color="k", ls="--", lw=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"P{i}" for i in range(5)])
    axes[2].set_ylabel("|R_fd/R_an − 1| spread")
    axes[2].set_title("sub-cell shift")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"wrote {json_path}  {png_path}")


if __name__ == "__main__":
    main(sys.argv[1:])