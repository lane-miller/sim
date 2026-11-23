#!/usr/bin/env python3
"""Step 6b2a gates: parametric recording vs legacy C9 and power consistency."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_OUT = _PKG / "outputs" / "step06b2a"
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


def _gate_table(title: str, rows: list[tuple[str, float, float, float, bool]]) -> None:
    print(f"\n{title}")
    print(f"{'label':<16} {'value':>12} {'ref':>12} {'ratio':>12}  PASS")
    for label, val, ref, ratio, ok in rows:
        print(f"{label:<16} {val:12.4g} {ref:12.4g} {ratio:12.4g}  {'PASS' if ok else 'FAIL'}")


def _run_all(paths: dict[str, Path]) -> dict[str, Path]:
    outs: dict[str, Path] = {}
    for name, cfg_path in paths.items():
        cfg = _load_json(cfg_path)
        if cfg.geometry == "parametric":
            est = dry_run_geom(cfg)
        else:
            est = run(cfg, dry_run=True, progress=False)
            assert isinstance(est, dict)
        print(
            f"{name}: grid={est.get('grid_shape')}  "
            f"n_steps={est.get('n_steps')}  "
            f"wall_est={est.get('wall_time_s_est', float('nan')):.3g} s"
        )
        out = _OUT / f"{name}.h5"
        with keep_awake(f"step06b2a {name}"):
            run(cfg, out_path=out, progress=True)
        outs[name] = out
    return outs


def _c12a(leg_h5: Path, s3p_h5: Path) -> tuple[bool, dict[str, Any]]:
    with Result(leg_h5) as r_leg, Result(s3p_h5) as r_s3p:
        f_leg, z_leg = r_leg.z()
        f_p, z_p = r_s3p.z()
        assert np.allclose(f_leg, f_p)
        num = float(np.max(np.abs(z_p - z_leg)))
        den = float(np.max(np.abs(z_leg)))
        z_ok = num / max(den, 1e-30) <= 1e-4
        rho = float(r_leg.config.air.rho)
        c = float(r_leg.config.air.c)
        w_leg = 0.5 * rho * c * r_leg.S_grid * np.real(z_leg)
        w_p = 0.5 * rho * c * r_s3p.S_grid * np.real(z_p)
        w_ratio = float(np.max(np.abs(w_p - w_leg) / np.maximum(w_leg, 1e-30)))
        w_ok = w_ratio <= 1e-4
        f_min, f_max = r_leg.band
        f_dirs = np.geomspace(f_min, f_max, 5)
        theta = np.deg2rad(np.arange(0.0, 181.0, 5.0))
        max_db = 0.0
        for phi in (0.0, 0.5 * math.pi):
            d_leg = r_leg.directivity(theta, phi, f_dirs)
            d_p = r_s3p.directivity(theta, phi, f_dirs)
            ref = np.max(np.abs(d_leg))
            mask = 20 * np.log10(np.maximum(np.abs(d_leg) / max(ref, 1e-30), 1e-30)) > -30.0
            if np.any(mask):
                ratio = np.where(mask, d_p / np.where(mask, d_leg, 1.0), 1.0)
                db = np.abs(20 * np.log10(np.abs(ratio)))
                max_db = max(max_db, float(np.max(db)))
        di_ok = max_db <= 0.001
    payload = {
        "z_ratio": num / max(den, 1e-30),
        "w_ratio_max": w_ratio,
        "directivity_db_max": max_db,
        "pass": bool(z_ok and w_ok and di_ok),
    }
    _gate_table(
        "C12a (s3p vs legacy C9)",
        [
            ("z", num, den, num / max(den, 1e-30), z_ok),
            ("power", w_ratio, 0.0, w_ratio, w_ok),
            ("directivity dB", max_db, 0.001, max_db / 0.001, di_ok),
        ],
    )
    return z_ok and w_ok and di_ok, payload


def _power_gate(name: str, h5: Path) -> tuple[bool, dict[str, Any]]:
    with Result(h5) as res:
        rho = float(res.config.air.rho)
        c = float(res.config.air.c)
        f_min, f_max = res.band
        f_lo = max(f_min, 0.1 * f_max)
        freqs = np.geomspace(f_lo, f_max, 8)
        rows: list[tuple[str, float, float, float, bool]] = []
        ok_all = True
        for f_hz in freqs:
            _, z = res.z(f_hz)
            w_c = 0.5 * rho * c * res.S_grid * float(np.real(z[0]))
            patches, v_ref, p_f, vn_f = load_huygens_spectra(h5, res.derived, float(f_hz))
            w_h = _w_huygens(
                p_f,
                vn_f,
                np.asarray(patches.areas, dtype=np.float64),
                complex(v_ref),
            )
            ratio = abs(w_h / max(w_c, 1e-30) - 1.0)
            ok = ratio <= 0.02
            ok_all = ok_all and ok
            rows.append((f"{f_hz:.4g} Hz", w_h, w_c, ratio, ok))
        _gate_table(f"Power consistency ({name})", rows)
    return ok_all, {"freqs_hz": freqs.tolist(), "pass": ok_all}


def _c18_preview(s3p_h5: Path, s4_h5: Path) -> None:
    with Result(s3p_h5) as r_s3, Result(s4_h5) as r_s4:
        f, z_s3 = r_s3.z()
        _, z_s4 = r_s4.z()
        re_ratio = float(np.max(np.abs(np.real(z_s4 - z_s3)))) / max(
            float(np.max(np.abs(np.real(z_s3)))), 1e-30
        )
        im_ratio = float(np.max(np.abs(np.imag(z_s4 - z_s3)))) / max(
            float(np.max(np.abs(np.imag(z_s3)))), 1e-30
        )
    print(
        f"\nC18(i) preview (S4 vs s3p): Re ratio={re_ratio:.4g}  Im ratio={im_ratio:.4g}  (report only)"
    )


def main() -> int:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    cfg_map = {
        "c09": _CONFIGS / "c09_box.json",
        "s3p": _SHOWCASE / "s3p_c09_parametric.json",
        "s1": _SHOWCASE / "s1_cylinder.json",
        "s4": _SHOWCASE / "s4_tilted.json",
        "s6": _SHOWCASE / "s6_downfire.json",
    }
    h5s = _run_all(cfg_map)
    gates: dict[str, Any] = {}
    ok_c12, gates["c12a"] = _c12a(h5s["c09"], h5s["s3p"])
    for key in ("s1", "s4", "s6"):
        ok, gates[f"power_{key}"] = _power_gate(key, h5s[key])
        gates.setdefault("power_pass", []).append(ok)
    _c18_preview(h5s["s3p"], h5s["s4"])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    with Result(h5s["c09"]) as r0, Result(h5s["s3p"]) as r1, Result(h5s["s4"]) as r2:
        f, z0 = r0.z()
        _, z1 = r1.z()
        _, z2 = r2.z()
        ax = axes[0]
        ax.semilogx(f, np.real(z0), label="C9 legacy")
        ax.semilogx(f, np.real(z1), "--", label="s3p")
        ax.semilogx(f, np.real(z2), ":", label="S4")
        ax.set_ylabel("Re z")
        ax.legend()
        ax2 = ax.twinx()
        ax2.semilogx(f, np.imag(z0), color="C0", alpha=0.35)
        ax2.semilogx(f, np.imag(z1), color="C1", alpha=0.35)
        ax2.semilogx(f, np.imag(z2), color="C2", alpha=0.35)
        ax.set_xlabel("f (Hz)")
        ax.set_title("(a) z on log grid")

    ax = axes[1]
    for key, color in zip(("s1", "s4", "s6"), ("C0", "C1", "C2")):
        with Result(h5s[key]) as res:
            rho = float(res.config.air.rho)
            c = float(res.config.air.c)
            f_min, f_max = res.band
            freqs = np.geomspace(max(f_min, 0.1 * f_max), f_max, 8)
            ratios = []
            for f_hz in freqs:
                _, z = res.z(f_hz)
                w_c = 0.5 * rho * c * res.S_grid * float(np.real(z[0]))
                patches, v_ref, p_f, vn_f = load_huygens_spectra(h5s[key], res.derived, float(f_hz))
                w_h = _w_huygens(p_f, vn_f, np.asarray(patches.areas), complex(v_ref))
                ratios.append(w_h / max(w_c, 1e-30))
            ax.semilogx(freqs, ratios, color=color, label=key)
    ax.axhline(1.02, color="k", ls="--", lw=0.8)
    ax.axhline(0.98, color="k", ls="--", lw=0.8)
    ax.set_ylabel("W_huy / W_circuit")
    ax.set_xlabel("f (Hz)")
    ax.legend()
    ax.set_title("(b) power ratio ±2%")

    fig.tight_layout()
    fig.savefig(_OUT / "gates.png", dpi=140)

    gates["elapsed_s"] = time.perf_counter() - t0
    gates["h5"] = {k: str(v) for k, v in h5s.items()}
    (_OUT / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(f"\nwrote {_OUT / 'gates.json'} and gates.png  ({gates['elapsed_s']:.1f} s wall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())