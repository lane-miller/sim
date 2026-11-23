#!/usr/bin/env python3
"""Step 8a gates: multi-radiator runs (C20(i), reciprocity report)."""

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
_OUT = _PKG / "outputs" / "step08a"
_REF_07A = _PKG / "outputs" / "step07a" / "s2_free.h5"
_SHOWCASE = _PKG / "configs" / "showcase"

sys.path.insert(0, str(_PKG))

from boxrad.config import Config  # noqa: E402
from boxrad.geomrun import dry_run_geom  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402
from validation._common import to_json_safe  # noqa: E402

_F64 = np.float64
_RHO = 1.204
_C = 343.0


def _load_s2_free() -> Config:
    cfg = Config.from_json((_SHOWCASE / "s2_bookshelf.json").read_text(encoding="utf-8"))
    raw = json.loads(cfg.to_json())
    raw.pop("environment", None)
    return Config.from_dict(raw)


def _radiator_positions(cfg: Config) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for rd in cfg.geometry_spec.radiators:
        assert rd.pose is not None
        out.append(np.asarray(rd.pose.position, dtype=_F64).reshape(3))
    return out


def _read_z_row(path: Path) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    import h5py

    with h5py.File(path, "r") as f:
        f_hz = np.asarray(f["results/zrad_row"].attrs["f"], dtype=_F64)
        z_row = np.asarray(f["results/zrad_row"][()], dtype=np.complex128)
        s_list = json.loads(f["meta"].attrs["radiator_S_json"])
        j_drv = int(f["meta"].attrs["driven_index"])
    return f_hz, z_row, [float(x) for x in s_list], j_drv


def _Z(z_ij: np.ndarray, s_j: float) -> np.ndarray:
    return _RHO * _C * z_ij / s_j


def main() -> int:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    cfg = _load_s2_free()
    out_path = _OUT / "s2_multirad.h5"
    est = dry_run_geom(cfg, out_path=out_path)
    print(
        f"s2-free multirad: n_runs={est.get('n_runs')}  "
        f"n_steps={est.get('n_steps')}  wall_est={est.get('wall_time_s_est_total', float('nan')):.3g} s"
    )
    with keep_awake("step08a multirad"):
        run(cfg, out_path=out_path, progress=True)

    sib = _OUT / "s2_multirad.r1_tweeter.h5"
    gates: dict[str, Any] = {"h5_primary": str(out_path), "h5_tweeter": str(sib)}

    import h5py

    if _REF_07A.is_file():
        with h5py.File(out_path, "r") as fn, h5py.File(_REF_07A, "r") as fr:
            p_new = np.asarray(fn["radiator/p_avg"][()], dtype=_F64)
            p_ref = np.asarray(fr["radiator/p_avg"][()], dtype=_F64)
            n = min(p_new.size, p_ref.size)
            d_p = float(np.max(np.abs(p_new[:n] - p_ref[:n])))
            h_new = np.asarray(fn["huygens/p_face"][()], dtype=np.float32)
            h_ref = np.asarray(fr["huygens/p_face"][()], dtype=np.float32)
            n_h = min(h_new.shape[0], h_ref.shape[0])
            d_h = float(np.max(np.abs(h_new[:n_h] - h_ref[:n_h])))
        gates["c20i_p_avg_max_abs"] = d_p
        gates["c20i_huygens_p_face_max_abs"] = d_h
        gates["c20i_pass"] = bool(d_p == 0.0 and d_h == 0.0)
    else:
        gates["c20i_note"] = f"missing reference {_REF_07A}; skipped"
        gates["c20i_pass"] = None

    f0, z0, s_list, j0 = _read_z_row(out_path)
    f1, z1, _, j1 = _read_z_row(sib)
    assert j0 == 0 and j1 == 1
    f_hz = f0
    s0, s1 = s_list[0], s_list[1]
    # z_ij = pickup i in file j (radiator j driven); Z_ij = rho*c*z_ij / S_j
    z00 = z0[:, 0]
    z10 = z0[:, 1]
    z01 = z1[:, 0]
    z11 = z1[:, 1]
    z00_imp = _Z(z00, s0)
    z11_imp = _Z(z11, s1)
    z01_imp = _Z(z01, s1)
    z10_imp = _Z(z10, s0)
    num = np.abs(z01_imp - z10_imp)
    den = max(float(np.max(np.abs(z01_imp))), 1e-30)
    gates["reciprocity_max_rel"] = float(np.max(num) / den)
    gates["reciprocity_max_rel_re"] = float(
        np.max(np.abs(np.real(z01_imp) - np.real(z10_imp))) / max(float(np.max(np.abs(np.real(z01_imp)))), 1e-30)
    )
    gates["reciprocity_max_rel_im"] = float(
        np.max(np.abs(np.imag(z01_imp) - np.imag(z10_imp))) / max(float(np.max(np.abs(np.imag(z01_imp)))), 1e-30)
    )

    positions = _radiator_positions(cfg)
    d_sep = float(np.linalg.norm(positions[0] - positions[1]))
    gates["radiator_separation_m"] = d_sep

    lf_hz = [50.0, 80.0, 120.0, 200.0, 300.0]
    lf: dict[str, Any] = {}
    for f in lf_hz:
        i0 = int(np.argmin(np.abs(f_hz - f)))
        re_z10_over_z00 = (
            float(np.real(z10[i0]) / np.real(z00[i0])) if np.real(z00[i0]) != 0 else float("nan")
        )
        re_z01_over_z11 = (
            float(np.real(z01[i0]) / np.real(z11[i0])) if np.real(z11[i0]) != 0 else float("nan")
        )
        k = 2.0 * math.pi * f / _C
        sinc = math.sin(k * d_sep) / (k * d_sep) if abs(k * d_sep) > 1e-12 else 1.0
        lf[str(int(f))] = {
            "Re_z10_over_Re_z00_drive0": re_z10_over_z00,
            "Re_z01_over_Re_z11_drive1": re_z01_over_z11,
            "sinc_kd": sinc,
        }
    gates["low_f_mutual"] = lf

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    ax0 = axes[0]
    for label, z in (("Z00", z00_imp), ("Z11", z11_imp)):
        ax0.semilogx(f_hz, np.real(z), label=f"Re {label}")
        ax0.semilogx(f_hz, np.imag(z), ls="--", alpha=0.7, label=f"Im {label}")
    ax0.set_xlabel("f (Hz)")
    ax0.set_ylabel("Z (Pa·s/m³)")
    ax0.legend(fontsize=8, ncol=2)
    ax0.set_title("(a) self impedances Z00, Z11")

    ax1 = axes[1]
    for label, z in (("Z01", z01_imp), ("Z10", z10_imp)):
        ax1.semilogx(f_hz, np.real(z), label=f"Re {label}")
        ax1.semilogx(f_hz, np.imag(z), ls="--", alpha=0.7, label=f"Im {label}")
    ax1.set_xlabel("f (Hz)")
    ax1.set_ylabel("Z (Pa·s/m³)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_title("(a) mutual impedances Z01, Z10")

    ax2 = axes[2]
    sinc_line = []
    ratio_line = []
    for f in lf_hz:
        k = 2.0 * math.pi * f / _C
        sinc_line.append(math.sin(k * d_sep) / (k * d_sep) if abs(k * d_sep) > 1e-12 else 1.0)
        i0 = int(np.argmin(np.abs(f_hz - f)))
        ratio_line.append(float(np.real(z10[i0]) / np.real(z00[i0])))
    ax2.plot(lf_hz, ratio_line, "o-", label="Re z10 / Re z00 (drive 0)")
    ax2.plot(lf_hz, sinc_line, "s--", label="sin(kd)/(kd)")
    ax2.set_xscale("log")
    ax2.set_xlabel("f (Hz)")
    ax2.set_ylabel("ratio")
    ax2.legend()
    ax2.set_title("(b) low-f mutual vs sinc(kd)")

    fig.tight_layout()
    fig.savefig(_OUT / "gates.png", dpi=140)

    gates["elapsed_s"] = time.perf_counter() - t0
    (_OUT / "gates.json").write_text(
        json.dumps(to_json_safe(gates), indent=2),
        encoding="utf-8",
    )
    print(f"wrote {_OUT / 'gates.json'} and gates.png  ({gates['elapsed_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())