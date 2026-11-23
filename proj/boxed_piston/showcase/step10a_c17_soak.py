#!/usr/bin/env python3
"""Step 10a gate C17: closed-domain stability soak and late-time tails (S4, S6)."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_OUT = _PKG / "outputs" / "step10a"
_SHOWCASE = _PKG / "configs" / "showcase"

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import Config  # noqa: E402
from boxrad.cutkernel import CutOperator, step_cut  # noqa: E402
from boxrad.geom.geometry import build_geometry  # noqa: E402
from boxrad.geomop import GeomOpCache, build_geom_op_cache  # noqa: E402
from boxrad.cpml import make_cpml_state  # noqa: E402
from boxrad.geomderive import validate_geom  # noqa: E402
from boxrad.geomop import build_run_operator  # noqa: E402
from boxrad.geomrun import GeomRun, dry_run_geom  # noqa: E402
from boxrad.grid import allocate_fields  # noqa: E402
from boxrad.kernels import make_update_coeffs  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402
from boxrad.validation.energy import discrete_energy  # noqa: E402
from validation._common import to_json_safe  # noqa: E402

_F32 = np.float32
_F64 = np.float64
_H = 0.005
_N_SOAK = 200_000
_N_RECORD = 100
_P_BLOW = 1.0e6


def _rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=_F64)


def _rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=_F64)


def _c17_config(*, boss_z: float) -> Config:
    R = _rot_z(37.0) @ _rot_x(20.0)
    box_n = R[:, 2].tolist()
    box_x = R[:, 0].tolist()
    box_c = np.array([0.04, 0.03, 0.02], dtype=_F64)
    sph_c = np.array([-0.04, -0.03, -0.02], dtype=_F64)
    boss_size = [0.02, 0.02, 0.004]
    boss_c = np.array([0.035, -0.045, boss_z], dtype=_F64)
    rad_position = (boss_c + np.array([0.0, 0.0, boss_size[2] * 0.5], dtype=_F64)).tolist()
    raw: dict[str, Any] = {
        "geometry": "parametric",
        "air": {"c": 343.0, "rho": 1.204},
        "box": None,
        "band": {"f_min": 50.0, "f_max": 3000.0},
        "numerics": {
            "h_override": _H,
            "cells_across_radiator": 10,
            "cfl_safety": 0.99,
            "gap_cells": None,
            "cpml_cells": 16,
            "cpml_alpha_factor": 1.0,
            "cpml_m": 3.0,
            "cpml_R": 1e-6,
            "cpml_kappa_max": 1.0,
            "cell_budget": 60_000_000,
            "pulse_db": -40.0,
            "pulse_f_factor": 1.2,
            "n_transits": 0.01,
            "min_edge_cells": 3,
        },
        "output": {"path": ""},
        "parts": [
            {
                "sphere": {"d": 0.06},
                "position": sph_c.tolist(),
                "normal": [0.0, 0.0, 1.0],
            },
            {
                "box": {"size": [0.05, 0.04, 0.03]},
                "position": box_c.tolist(),
                "normal": box_n,
                "x_axis": box_x,
            },
            {
                "box": {"size": boss_size},
                "position": boss_c.tolist(),
                "normal": [0.0, 0.0, 1.0],
                "x_axis": [1.0, 0.0, 0.0],
            },
        ],
        "radiators": [
            {
                "name": "stub",
                "rect": {"w": 0.006, "h": 0.006},
                "position": rad_position,
                "normal": [0.0, 0.0, 1.0],
                "x_axis": [1.0, 0.0, 0.0],
            }
        ],
    }
    return Config.from_dict(raw)


def _boss_tiny_beta_stats(cfg: Config) -> dict[str, float | int]:
    geom = build_geometry(cfg)
    assert geom is not None
    from boxrad.geomderive import derive_geom

    derived, gg = derive_geom(cfg, geom)
    fr = build_geom_op_cache(cfg, derived, gg, geom).fractions
    b = fr.beta
    pos = b > 0.0
    min_raw = float(np.min(b[pos])) if np.any(pos) else 0.0
    n_tiny = int(np.sum((b > 0.0) & (b < 1.0e-3)))
    return {"min_raw_beta": min_raw, "n_beta_lt_1e3": n_tiny, "h": float(derived.h)}


def _prepare_c17() -> tuple[Config, dict[str, Any]]:
    boss_z = -0.0612
    note = ""
    for _ in range(5):
        cfg = _c17_config(boss_z=boss_z)
        stats = _boss_tiny_beta_stats(cfg)
        if stats["n_beta_lt_1e3"] > 0:
            stats["boss_z_m"] = boss_z
            return cfg, stats
        boss_z += 0.001
        note = "boss z shifted +1 mm (no beta < 1e-3 at nominal z)"
    cfg = _c17_config(boss_z=boss_z)
    stats = _boss_tiny_beta_stats(cfg)
    stats["boss_z_m"] = boss_z
    stats["boss_z_note"] = note or "no raw beta < 1e-3 after 1 mm sweep"
    return cfg, stats


def _prepare_closed_geom_run(
    cfg: Config,
    *,
    cache: GeomOpCache | None = None,
    driven: int = 0,
) -> GeomRun:
    geom = build_geometry(cfg)
    assert geom is not None
    from boxrad.geomderive import derive_geom

    derived, gg = derive_geom(cfg, geom)
    errors, warnings = validate_geom(cfg, derived, geom)
    if errors:
        raise ValueError("; ".join(errors))
    derived = dataclasses.replace(derived, cpml_sides=(False,) * 6)
    fields = allocate_fields(gg)
    coeffs = make_update_coeffs(cfg.air.rho, cfg.air.c, derived.h, derived.dt)
    cpml = make_cpml_state(gg, cfg, derived, derived.dt)
    op, info = build_run_operator(cfg, derived, gg, geom, driven=driven, cache=cache)
    return GeomRun(
        cfg=cfg,
        derived=derived,
        gg=gg,
        geometry=geom,
        fields=fields,
        coeffs=coeffs,
        cpml=cpml,
        op=op,
        info=info,
        sigma=derived.pulse_sigma,
        t0=derived.pulse_t0,
        dt=derived.dt,
        warnings=warnings,
    )


def _v_now_zeros(op: CutOperator) -> list[_F32]:
    n = len(op.radiators_fast) + len(op.radiators_patch)
    return [_F32(0.0)] * n


def _step_record_energy(run) -> float:
    fields = run.fields
    p_n = fields.p.copy()
    v_prev = (fields.vx.copy(), fields.vy.copy(), fields.vz.copy())
    step_cut(fields, run.coeffs, run.op, run.cpml, _v_now_zeros(run.op))
    rho = float(run.cfg.air.rho)
    c = float(run.cfg.air.c)
    h = float(run.derived.h)
    fr = run.info["fractions"]
    bt = run.info["beta_tilde"]
    v_now = (fields.vx.copy(), fields.vy.copy(), fields.vz.copy())
    return discrete_energy(p_n, v_now, v_prev, bt, fr.alpha, rho, c, h)


def _closed_soak(cfg: Config) -> dict[str, Any]:
    est = dry_run_geom(cfg)
    print(
        f"C17(a) closed soak: grid={est.get('grid_shape')}  "
        f"n_floored={est.get('n_floored')}  wall_est={est.get('wall_time_s_est', float('nan')):.3g} s"
    )
    run = _prepare_closed_geom_run(cfg)
    fluid = run.op.inv_beta.reshape(-1) > _F32(0)
    rng = np.random.default_rng(0)
    p = run.fields.p.reshape(-1)
    p[fluid] = rng.uniform(-1.0, 1.0, size=int(np.sum(fluid))).astype(_F32)
    run.fields.vx.fill(0.0)
    run.fields.vy.fill(0.0)
    run.fields.vz.fill(0.0)
    steps: list[int] = []
    rel_e: list[float] = []
    e_trace: list[float] = []
    t_loop = time.perf_counter()
    with keep_awake("step10a C17 closed soak"):
        for n in range(1, _N_SOAK + 1):
            e = _step_record_energy(run)
            if n == 1:
                e0 = e
                e_trace.append(e)
                steps.append(n)
                rel_e.append(0.0)
            elif n % _N_RECORD == 0:
                e_trace.append(e)
                steps.append(n)
                rel_e.append((e - e0) / max(abs(e0), 1e-30))
            if n % 20_000 == 0:
                elapsed = time.perf_counter() - t_loop
                eta = elapsed / n * (_N_SOAK - n)
                print(f"  step {n}/{_N_SOAK}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
    rel = np.asarray(rel_e, dtype=_F64)
    max_drift = float(np.max(np.abs(rel)))
    n10 = max(1, len(e_trace) // 10)
    mean_first = float(np.mean(e_trace[:n10]))
    mean_last = float(np.mean(e_trace[-n10:]))
    mean_delta = abs(mean_last - mean_first) / max(abs(mean_first), 1e-30)
    pass_drift = max_drift <= 1.0e-4
    pass_mean = mean_delta <= 1.0e-4
    return {
        "n_steps": _N_SOAK,
        "max_rel_drift": max_drift,
        "mean_first_last_rel": mean_delta,
        "pass_drift": pass_drift,
        "pass_mean": pass_mean,
        "pass": bool(pass_drift and pass_mean),
        "steps": steps,
        "rel_energy": rel_e,
        "elapsed_s": time.perf_counter() - t_loop,
    }


def _unfloored_blow(cfg: Config) -> dict[str, Any]:
    print("\nC17(b) floor disabled (expect blow-up)")
    geom = build_geometry(cfg)
    assert geom is not None
    from boxrad.geomderive import derive_geom

    derived, gg = derive_geom(cfg, geom)
    base_cache = build_geom_op_cache(cfg, derived, gg, geom)
    raw_cache = GeomOpCache(
        fractions=base_cache.fractions,
        beta_tilde=base_cache.fractions.beta.copy(),
        floor_stats=base_cache.floor_stats,
        thin_stats=base_cache.thin_stats,
    )
    run = _prepare_closed_geom_run(cfg, cache=raw_cache)
    fluid = run.op.inv_beta.reshape(-1) > _F32(0)
    rng = np.random.default_rng(0)
    p = run.fields.p.reshape(-1)
    p[fluid] = rng.uniform(-1.0, 1.0, size=int(np.sum(fluid))).astype(_F32)
    run.fields.vx.fill(0.0)
    run.fields.vy.fill(0.0)
    run.fields.vz.fill(0.0)
    v_now = _v_now_zeros(run.op)
    blow_step: int | None = None
    p_max_trace: list[float] = []
    step_trace: list[int] = []
    for n in range(1, 50_001):
        _step_record_energy(run)
        pmax = float(np.max(np.abs(run.fields.p)))
        if n % 50 == 0 or n <= 5:
            p_max_trace.append(pmax)
            step_trace.append(n)
        if not np.all(np.isfinite(run.fields.p)) or pmax > _P_BLOW:
            blow_step = n
            p_max_trace.append(pmax)
            step_trace.append(n)
            break
    passed = blow_step is not None
    print(f"  blow-up at step {blow_step}" if passed else "  no blow-up (unexpected)")
    return {
        "blow_step": blow_step,
        "pass": passed,
        "p_max_steps": step_trace,
        "p_max": p_max_trace,
    }


def _load_json(path: Path) -> Config:
    return Config.from_json(path.read_text(encoding="utf-8"))


def _scaled_transits(cfg: Config, factor: float) -> Config:
    raw = json.loads(cfg.to_json())
    raw["numerics"]["n_transits"] = float(cfg.numerics.n_transits) * factor
    return Config.from_dict(raw)


def _tail_metrics(h5: Path) -> dict[str, Any]:
    with Result(h5) as res:
        import h5py

        with h5py.File(h5, "r") as f:
            p = np.asarray(f["radiator/p_avg"][()], dtype=_F64)
        tail_db_meta = float(res.meta.get("tail_db", float("nan")))
    if p.size == 0:
        raise RuntimeError(f"empty p_avg in {h5}")
    pk = float(np.max(np.abs(p)))
    n10 = max(1, int(math.floor(0.10 * p.size)))
    tail10 = float(np.max(np.abs(p[-n10:])))
    tail10_db = 20.0 * math.log10(tail10 / max(pk, 1e-30))
    half = p[p.size // 2 :]
    wins = np.array_split(half, 10)
    win_max = [float(np.max(np.abs(w))) for w in wins if w.size]
    spikes_db: list[float] = []
    mono = True
    for i in range(1, len(win_max)):
        if win_max[i] > 0 and win_max[i - 1] > 0:
            spikes_db.append(20.0 * math.log10(win_max[i] / win_max[i - 1]))
            if spikes_db[-1] > 3.0:
                mono = False
    gate_tail = tail10_db <= tail_db_meta + 1e-9
    return {
        "tail10_db_re_peak": tail10_db,
        "meta_tail_db": tail_db_meta,
        "window_spike_db": spikes_db,
        "non_increasing_windows": mono,
        "pass_tail": gate_tail,
        "pass_windows": mono,
        "pass": bool(gate_tail and mono),
        "p_avg_peak": pk,
    }


def _open_domain_runs() -> dict[str, Any]:
    out: dict[str, Any] = {}
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, fname in (("S4", "s4_tilted.json"), ("S6", "s6_downfire.json")):
        cfg_base = _load_json(_SHOWCASE / fname)
        cfg = _scaled_transits(cfg_base, 10.0)
        h5 = _OUT / f"c17_{name.lower()}.h5"
        if not h5.is_file():
            est = dry_run_geom(cfg)
            print(
                f"C17(c) {name}: grid={est.get('grid_shape')}  n_steps={est.get('n_steps')}  "
                f"wall_est={est.get('wall_time_s_est', float('nan')):.3g} s"
            )
            with keep_awake(f"step10a C17 {name}"):
                run(cfg, out_path=h5, progress=True)
        else:
            print(f"C17(c) reuse {h5}")
        met = _tail_metrics(h5)
        with Result(h5) as res:
            import h5py

            with h5py.File(h5, "r") as f:
                t = np.asarray(f["radiator/times_p"][()], dtype=_F64)
                p = np.asarray(f["radiator/p_avg"][()], dtype=_F64)
        traces[name] = (t, p)
        out[name] = met
    out["pass"] = bool(out["S4"]["pass"] and out["S6"]["pass"])
    out["traces"] = {k: {"t": v[0].tolist(), "p": v[1].tolist()} for k, v in traces.items()}
    return out


def main() -> int:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)

    cfg, boss_stats = _prepare_c17()
    print(
        f"C17 geometry: min raw beta={boss_stats['min_raw_beta']:.4g}  "
        f"n(beta<1e-3)={boss_stats['n_beta_lt_1e3']}  boss_z={boss_stats.get('boss_z_m', float('nan')):.6g} m"
    )
    if boss_stats.get("boss_z_note"):
        print(f"  note: {boss_stats['boss_z_note']}")

    closed = _closed_soak(cfg)
    unfloored = _unfloored_blow(cfg)
    open_dom = _open_domain_runs()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax = axes[0]
    ax.plot(closed["steps"], closed["rel_energy"], lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("(E−E0)/E0")
    ax.set_title("(a) closed soak energy")
    ax.axhline(1e-4, color="k", ls=":", lw=0.8)
    ax.axhline(-1e-4, color="k", ls=":", lw=0.8)

    ax = axes[1]
    ax.semilogy(unfloored["p_max_steps"], unfloored["p_max"])
    ax.set_xlabel("step")
    ax.set_ylabel("max |p|")
    ax.set_title("(b) unfloored blow-up")

    ax = axes[2]
    for name, color in (("S4", "C0"), ("S6", "C1")):
        t, p = open_dom["traces"][name]["t"], open_dom["traces"][name]["p"]
        t_arr = np.asarray(t, dtype=_F64)
        p_arr = np.asarray(p, dtype=_F64)
        pk = float(np.max(np.abs(p_arr)))
        db = 20.0 * np.log10(np.maximum(np.abs(p_arr) / max(pk, 1e-30), 1e-30))
        ax.plot(t_arr, db, color=color, label=name)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|p_avg| dB re peak")
    ax.legend()
    ax.set_title("(c) late-time S4 / S6")

    fig.tight_layout()
    fig.savefig(_OUT / "c17.png", dpi=140)

    payload = to_json_safe(
        {
            "boss_stats": boss_stats,
            "closed_soak": {k: v for k, v in closed.items() if k not in ("steps", "rel_energy")},
            "closed_rel_energy": {"steps": closed["steps"], "rel": closed["rel_energy"]},
            "unfloored": unfloored,
            "open_domain": {
                k: v
                for k, v in open_dom.items()
                if k not in ("traces",)
            },
            "gates": {
                "closed": closed["pass"],
                "unfloored_expected_blowup": unfloored["pass"],
                "open_domain": open_dom["pass"],
            },
            "elapsed_s": time.perf_counter() - t0,
        }
    )
    (_OUT / "c17.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {_OUT / 'c17.json'} and c17.png  ({payload['elapsed_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())