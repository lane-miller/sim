#!/usr/bin/env python3
"""Step 7a gates: environment planes (C19 B′)."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_OUT = _PKG / "outputs" / "step07a"
_CONFIGS = _PKG / "configs"
_SHOWCASE = _CONFIGS / "showcase"

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import Config  # noqa: E402
from boxrad.geomrun import dry_run_geom  # noqa: E402
from boxrad.post.transfer import load_huygens_spectra  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402


def _load_json(path: Path) -> Config:
    return Config.from_json(path.read_text(encoding="utf-8"))


def _w_huygens(p_face: np.ndarray, vn_face: np.ndarray, areas: np.ndarray, V: complex) -> float:
    scale = abs(V)
    if scale < 1e-30:
        return float("nan")
    p_tf = p_face / V
    vn_tf = vn_face / V
    return float(0.5 * np.real(np.sum(p_tf * np.conj(vn_tf) * areas)))


def _probes_cfg(base: Config) -> Config:
    c = copy.deepcopy(base)
    c.output.probes = [
        [0.03, 0.02, 0.02],
        [0.12, 0.05, -0.10],
        [0.06, 0.09, 0.05],
    ]
    return c


def _s2_free(cfg: Config) -> Config:
    raw = json.loads(cfg.to_json())
    raw.pop("environment", None)
    return Config.from_dict(raw)


def _run_parametric(name: str, cfg: Config, out: Path) -> Path:
    est = dry_run_geom(cfg)
    print(
        f"{name}: grid={est.get('grid_shape')}  n_steps={est.get('n_steps')}  "
        f"wall_est={est.get('wall_time_s_est', float('nan')):.3g} s"
    )
    with keep_awake(f"step07a {name}"):
        run(cfg, out_path=out, progress=True)
    return out


def _symmetry_gates(full: Path, half: Path, quarter: Path) -> dict[str, Any]:
    with Result(full) as rf, Result(half) as rh, Result(quarter) as rq:
        tf = np.asarray(rf._file()["radiator/p_avg"][()], dtype=np.float64)
        th = np.asarray(rh._file()["radiator/p_avg"][()], dtype=np.float64)
        tq = np.asarray(rq._file()["radiator/p_avg"][()], dtype=np.float64)
        p_ok_h = float(np.max(np.abs(tf - th))) / max(float(np.max(np.abs(tf))), 1e-30) <= 1e-5
        p_ok_q = float(np.max(np.abs(tf - tq))) / max(float(np.max(np.abs(tf))), 1e-30) <= 1e-5
        f_f, z_f = rf.z()
        _, z_h = rh.z()
        _, z_q = rq.z()
        z_ok_h = float(np.max(np.abs(z_f - z_h))) / max(float(np.max(np.abs(z_f))), 1e-30) <= 1e-5
        z_ok_q = float(np.max(np.abs(z_f - z_q))) / max(float(np.max(np.abs(z_f))), 1e-30) <= 1e-5
    return {
        "p_avg_half": p_ok_h,
        "p_avg_quarter": p_ok_q,
        "z_half": z_ok_h,
        "z_quarter": z_ok_q,
        "pass": bool(p_ok_h and p_ok_q and z_ok_h and z_ok_q),
    }


def _power_open(h5: Path) -> bool:
    with Result(h5) as res:
        rho = float(res.config.air.rho)
        c = float(res.config.air.c)
        f_min, f_max = res.band
        freqs = np.geomspace(max(f_min, 0.1 * f_max), f_max, 8)
        ok_all = True
        for f_hz in freqs:
            _, z = res.z(f_hz)
            w_c = 0.5 * rho * c * res.S_grid * float(np.real(z[0]))
            patches, v_ref, p_f, vn_f = load_huygens_spectra(h5, res.derived, float(f_hz))
            w_h = _w_huygens(p_f, vn_f, np.asarray(patches.areas), complex(v_ref))
            ratio = abs(w_h / max(w_c, 1e-30) - 1.0)
            ok_all = ok_all and ratio <= 0.02
        return ok_all


def _image_ratio_report(s2: Path, s2_free: Path) -> None:
    freqs = [30.0, 50.0, 80.0, 120.0, 200.0, 400.0]
    with Result(s2) as rp, Result(s2_free) as rf:
        pos = np.asarray(rp.meta.get("radiator0_position", [0.0, 0.125, 0.10]), dtype=np.float64)
        s = pos
        images = (
            np.array([s[0], s[1], -s[2]]),
            np.array([s[0], -s[1], s[2]]),
            np.array([s[0], -s[1], -s[2]]),
        )
        print("\nS2 Re z ratio vs image formula (CHECK)")
        print(f"{'f Hz':>8} {'Re zp/Re zf':>14} {'image approx':>14}")
        for f_hz in freqs:
            _, zp = rp.z(f_hz)
            _, zf = rf.z(f_hz)
            ratio = float(np.real(zp[0]) / np.real(zf[0]))
            k = 2.0 * math.pi * f_hz / 343.0
            approx = 1.0
            for sm in images:
                d_m = float(np.linalg.norm(s - sm))
                approx += math.sin(k * d_m) / max(k * d_m, 1e-30)
            print(f"{f_hz:8.1f} {ratio:14.4g} {approx:14.4g}")


def main() -> int:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    c19_full = _probes_cfg(_load_json(_SHOWCASE / "c19_sym_full.json"))
    half = _probes_cfg(_load_json(_SHOWCASE / "c19_half_x.json"))
    quarter = _probes_cfg(_load_json(_SHOWCASE / "c19_quarter_xy.json"))
    s2 = _load_json(_SHOWCASE / "s2_bookshelf.json")
    s2f = _s2_free(s2)

    paths = {
        "c19_full": _run_parametric("c19_full", c19_full, _OUT / "c19_sym_full.h5"),
        "half": _run_parametric("half", half, _OUT / "c19_half_x.h5"),
        "quarter": _run_parametric("quarter", quarter, _OUT / "c19_quarter_xy.h5"),
        "s2": _run_parametric("s2", s2, _OUT / "s2.h5"),
        "s2_free": _run_parametric("s2_free", s2f, _OUT / "s2_free.h5"),
    }

    gates: dict[str, Any] = {
        "c19": _symmetry_gates(paths["c19_full"], paths["half"], paths["quarter"])
    }
    gates["power_half"] = _power_open(paths["half"])
    gates["power_quarter"] = _power_open(paths["quarter"])
    gates["power_s2"] = _power_open(paths["s2"])

    _image_ratio_report(paths["s2"], paths["s2_free"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    with Result(paths["c19_full"]) as r0, Result(paths["half"]) as r1, Result(paths["quarter"]) as r2:
        t0f = np.asarray(r0._file()["radiator/times_p"][()], dtype=np.float64)
        p0 = np.asarray(r0._file()["radiator/p_avg"][()], dtype=np.float64)
        p1 = np.asarray(r1._file()["radiator/p_avg"][()], dtype=np.float64)
        p2 = np.asarray(r2._file()["radiator/p_avg"][()], dtype=np.float64)
        ax = axes[0]
        ax.plot(t0f, p0, label="c19 full")
        ax.plot(t0f, p1, "--", label="half")
        ax.plot(t0f, p2, ":", label="quarter")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("p_avg")
        ax.legend()
        ax2 = ax.twinx()
        ax2.semilogy(t0f, np.maximum(np.abs(p0 - p1), 1e-30), color="C1", alpha=0.5, label="|Δ| half")
        ax2.semilogy(t0f, np.maximum(np.abs(p0 - p2), 1e-30), color="C2", alpha=0.5, label="|Δ| quarter")
        ax.set_title("(a) p_avg symmetry")

    freqs = np.geomspace(30.0, 400.0, 40)
    with Result(paths["s2"]) as rs, Result(paths["s2_free"]) as rf:
        ratios = []
        approx_line = []
        s = np.asarray(rs.meta.get("radiator0_position", [0.0, 0.125, 0.10]), dtype=np.float64)
        for sm in (
            np.array([s[0], s[1], -s[2]]),
            np.array([s[0], -s[1], s[2]]),
            np.array([s[0], -s[1], -s[2]]),
        ):
            _ = sm
        for f_hz in freqs:
            _, zp = rs.z(f_hz)
            _, zf = rf.z(f_hz)
            ratios.append(float(np.real(zp[0]) / np.real(zf[0])))
            k = 2.0 * math.pi * f_hz / 343.0
            approx = 1.0
            for sm in (
                np.array([s[0], s[1], -s[2]]),
                np.array([s[0], -s[1], s[2]]),
                np.array([s[0], -s[1], -s[2]]),
            ):
                d_m = float(np.linalg.norm(s - sm))
                approx += math.sin(k * d_m) / max(k * d_m, 1e-30)
            approx_line.append(approx)
    ax = axes[1]
    ax.semilogx(freqs, ratios, label="S2 / S2-free")
    ax.semilogx(freqs, approx_line, "--", label="image sum")
    ax.axhline(4.0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("Re z ratio")
    ax.set_xlabel("f (Hz)")
    ax.legend()
    ax.set_title("(b) S2 image check")

    fig.tight_layout()
    fig.savefig(_OUT / "gates.png", dpi=140)

    gates["elapsed_s"] = time.perf_counter() - t0
    gates["h5"] = {k: str(v) for k, v in paths.items()}
    (_OUT / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(f"\nwrote {_OUT / 'gates.json'} and gates.png  ({gates['elapsed_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())