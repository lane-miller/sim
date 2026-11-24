#!/usr/bin/env python3
"""Step 10b2: C13a rigid-sphere scattering ratio gate and C16 convergence."""

from __future__ import annotations

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
_GATE_C13A_DB = 0.1
_GATE_C13A_DEG = 1.0
_GATE_C16_P = 1.5
_GATE_SHIFT = 0.003


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


def _sample_points() -> np.ndarray:
    ff = _FAR_R * np.stack([u for _, u in _FAR_DIRS], axis=0)
    return np.vstack([_probes(), ff])


def _load_base() -> dict[str, Any]:
    return json.loads(_CFG.read_text(encoding="utf-8"))


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
) -> Config:
    raw = _load_base()
    h = 0.05 / n
    raw["numerics"]["h_override"] = h
    if not sphere:
        raw["parts"] = []
    elif center is not None:
        raw["parts"][0]["position"] = center.tolist()
    cfg = Config.from_dict(raw)
    pts = _sample_points()
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
    if sphere:
        return complex(sphere_point_source(pt[np.newaxis, :], f_hz, center, _SPHERE_A, src, rho, c)[0])
    _, p = monopole_free(pt[np.newaxis, :], f_hz, src, rho, c)
    return complex(p[0, 0])


def _ratio_db_deg(R_fd: complex, R_an: complex) -> tuple[float, float]:
    return h_transfer_err(R_an, R_fd)


def _run_labeled(label: str, cfg: Config, out: Path) -> None:
    print(f"\n>>> run {label} -> {out.name}")
    t0 = time.perf_counter()
    run(cfg, out_path=out, progress=True)
    print(f"    wall {time.perf_counter() - t0:.1f} s")


def _preflight() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, int, bool, np.ndarray | None]] = []
    for n in _NS:
        specs.append((f"sphere_n{n}", n, True, None))
        specs.append((f"free_n{n}", n, False, None))
    for delta in _SHIFT_DELTAS:
        h = 0.05 / 9
        shift = delta * h * np.array([1.0, 1.0, 1.0], dtype=_F64)
        specs.append((f"sphere_n9_d{delta:g}", 9, True, _CENTER + shift))
    for name, n, sphere, cen in specs:
        cfg = _build_cfg(n=n, sphere=sphere, center=cen)
        est = dry_run_geom(cfg)
        ncell = int(np.prod(est["grid_shape"]))
        rows.append(
            {
                "name": name,
                "cells": ncell,
                "h_mm": float(est["h"]) * 1e3,
                "n_steps": int(est["n_steps"]),
                "gap_cells": cfg.numerics.gap_cells,
            }
        )
        print(
            f"dry-run {name}: cells={ncell}  h={est['h'] * 1e3:.4g} mm  "
            f"n_steps={est['n_steps']}  gap_cells={cfg.numerics.gap_cells}"
        )
    return rows


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


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    json_path = _OUT / "c13a_c16.json"
    png_path = _OUT / "c13a_c16.png"

    base_cfg = _build_cfg(n=9, sphere=True)
    _assert_probe_centers(base_cfg)
    rho = float(base_cfg.air.rho)
    c = float(base_cfg.air.c)
    src = np.array(base_cfg.numerics.test_source.position, dtype=_F64)  # type: ignore[union-attr]

    print("=== preflight dry-run ===")
    preflight = _preflight()

    paths: dict[str, Path] = {}
    run_specs: list[tuple[str, Config]] = []
    for n in _NS:
        run_specs.append((f"sphere_n{n}", _build_cfg(n=n, sphere=True)))
        run_specs.append((f"free_n{n}", _build_cfg(n=n, sphere=False)))
    for delta in _SHIFT_DELTAS:
        h = 0.05 / 9
        shift = delta * h * np.array([1.0, 1.0, 1.0], dtype=_F64)
        run_specs.append((f"sphere_n9_d{delta:g}", _build_cfg(n=9, sphere=True, center=_CENTER + shift)))

    t_all = time.perf_counter()
    with keep_awake("step10b2 c13a_c16"):
        for name, cfg in run_specs:
            out = _OUT / f"{name}.h5"
            paths[name] = out
            _run_labeled(name, cfg, out)
    print(f"total wall {time.perf_counter() - t_all:.1f} s")

    if _staircase_available(_load_base()):
        print("staircase: running n=9 comparison (not implemented in numerics)")
    else:
        print("staircase mode not available, comparison skipped")

    c13a = _collect_ratio_block(
        paths["sphere_n9"],
        paths["free_n9"],
        _CENTER,
        rho,
        c,
        src,
        _FREQS_C13A,
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

    results: dict[str, Any] = {
        "preflight": preflight,
        "c13a": c13a,
        "c16": {"h_m": hs, "e": es, "p_observed": p_obs, "pairwise_p": pair},
        "shift_spread": shift_spread,
        "gates": {
            "c13a_pass": ok_c13a,
            "c16_pass": ok_c16,
            "shift_pass": ok_shift,
            "pass": bool(ok_c13a and ok_c16 and ok_shift),
        },
        "staircase": "not available",
    }
    json_path.write_text(json.dumps(to_json_safe(results), indent=2), encoding="utf-8")

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
    main()