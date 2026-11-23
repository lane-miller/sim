#!/usr/bin/env python3
"""Step 7b gates: image KH C19(A), C19(B′), and S2 showcase."""

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
_OUT_A = _PKG / "outputs" / "step07a"
_OUT = _PKG / "outputs" / "step07b"
_SHOWCASE = _PKG / "configs" / "showcase"

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import Config  # noqa: E402
from boxrad.geomrec import huygens_patches_planes  # noqa: E402
from boxrad.geomrun import dry_run_geom  # noqa: E402
from boxrad.post.images import expand_patches, image_sources, rigid_sides_from_planes  # noqa: E402
from boxrad.post.kirchhoff import dpdn_from_vn, kh_far_field, kh_pressure  # noqa: E402
from boxrad.post.transfer import h_transfer_err  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402
from validation._common import to_json_safe  # noqa: E402

_C = 343.0
_RHO = 1.204
_H = 0.005
_HI = np.array([0.20, 0.23, 0.25], dtype=np.float64)
_SOURCE = np.array([0.05, 0.08, 0.10], dtype=np.float64)


def _load_json(path: Path) -> Config:
    return Config.from_json(path.read_text(encoding="utf-8"))


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


def _ensure_h5(name: str, cfg: Config, fname: str) -> Path:
    path = _OUT_A / fname
    if path.is_file():
        return path
    est = dry_run_geom(cfg)
    print(
        f"{name}: missing {path.name}; running  grid={est.get('grid_shape')}  "
        f"n_steps={est.get('n_steps')}"
    )
    with keep_awake(f"step07b {name}"):
        run(cfg, out_path=path, progress=True)
    return path


def _lo_for_planes(planes: list[dict]) -> np.ndarray:
    lo = np.full(3, -0.15, dtype=np.float64)
    for pl in planes:
        if pl["side"] == "-":
            ax = {"x": 0, "y": 1, "z": 2}[pl["axis"]]
            lo[ax] = float(pl["offset"])
    return lo


def _open_patches(planes: list[dict]):
    lo = _lo_for_planes(planes)
    rigid = rigid_sides_from_planes(planes)
    z = np.zeros(3, dtype=np.float64)
    return huygens_patches_planes(lo, _HI.copy(), _H, 0, rigid, z, z + 1.0)


def _monopole_p(points: np.ndarray, k: float, sources: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(points.shape[0], dtype=np.complex128)
    for s in sources:
        r = np.linalg.norm(points - s, axis=-1)
        r = np.maximum(r, 1e-12)
        out += np.exp(-1j * k * r) / r
    return out


def _monopole_dpdn(centers, normals, k: float, sources: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(centers.shape[0], dtype=np.complex128)
    for s in sources:
        eps = centers - s
        r = np.linalg.norm(eps, axis=-1, keepdims=True)
        r = np.maximum(r, 1e-12)
        rhat = eps / r
        p = np.exp(-1j * k * r[..., 0]) / r[..., 0]
        grad = -p[..., np.newaxis] * rhat * (1.0 / r + 1j * k)
        out += np.sum(grad * normals, axis=-1)
    return out


def _c19a_planes_cases() -> list[tuple[str, list[dict]]]:
    return [
        ("z", [{"axis": "z", "side": "-", "offset": 0.0}]),
        (
            "zy",
            [
                {"axis": "z", "side": "-", "offset": 0.0},
                {"axis": "y", "side": "-", "offset": 0.0},
            ],
        ),
        (
            "zyx",
            [
                {"axis": "z", "side": "-", "offset": 0.0},
                {"axis": "y", "side": "-", "offset": 0.0},
                {"axis": "x", "side": "-", "offset": 0.0},
            ],
        ),
    ]


def gate_c19a() -> dict[str, Any]:
    freqs = [300.0, 1500.0, 3000.0]
    thetas = np.deg2rad(np.arange(0, 91, 10, dtype=np.float64))
    phi = 0.7
    st = np.sin(thetas)
    dirs = np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(thetas)])
    near_pts = np.array(
        [[0.5, 0.6, 0.4], [1.0, 0.3, 0.05], [0.3, 1.2, 0.8]],
        dtype=np.float64,
    )
    report: dict[str, Any] = {}
    for label, planes in _c19a_planes_cases():
        patches0 = _open_patches(planes)
        report[label] = {"freqs": {}}
        for f_hz in freqs:
            k = 2.0 * math.pi * f_hz / _C
            sources = image_sources(_SOURCE, planes)
            p0 = _monopole_p(patches0.centers, k, sources)
            dpdn0 = _monopole_dpdn(patches0.centers, patches0.normals, k, sources)
            vn0 = dpdn0 / (-1j * 2.0 * math.pi * f_hz * _RHO)
            exp_p, p_face, vn_face = expand_patches(patches0, planes, p0, vn0)
            omega = np.array([2.0 * math.pi * f_hz], dtype=np.float64)
            k_sel = np.array([k], dtype=np.float64)
            dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, _RHO)
            kh_ff = kh_far_field(dirs, exp_p, p_face[np.newaxis, :], dpdn, k_sel)[0]
            ref_ff = np.zeros(dirs.shape[0], dtype=np.complex128)
            for s in sources:
                ref_ff += np.exp(1j * k * (dirs @ s))
            far_db = far_ph = 0.0
            for a, b in zip(kh_ff, ref_ff, strict=True):
                db, ph = h_transfer_err(b, a)
                far_db = max(far_db, db)
                far_ph = max(far_ph, abs(ph))
            kh_nf = kh_pressure(
                near_pts, exp_p, p_face[np.newaxis, :], dpdn, k_sel, near_subdivide=False
            )[0]
            ref_nf = _monopole_p(near_pts, k, sources)
            near_db = near_ph = 0.0
            for a, b in zip(kh_nf, ref_nf, strict=True):
                db, ph = h_transfer_err(b, a)
                near_db = max(near_db, db)
                near_ph = max(near_ph, abs(ph))
            report[label]["freqs"][str(int(f_hz))] = {
                "far_db_max": float(far_db),
                "far_deg_max": float(far_ph),
                "near_db_max": float(near_db),
                "near_deg_max": float(near_ph),
                "pass_far": bool(far_db <= 0.1 and far_ph <= 1.0),
                "pass_near": bool(near_db <= 0.1 and near_ph <= 1.0),
            }
        vals = list(report[label]["freqs"].values())
        report[label]["pass"] = all(v["pass_far"] and v["pass_near"] for v in vals)
    report["pass"] = all(v["pass"] for k, v in report.items() if k != "pass")
    return report


def _rel_max(a: np.ndarray, b: np.ndarray) -> float:
    den = np.maximum(np.abs(a), 1e-30)
    return float(np.max(np.abs(a - b) / den))


def gate_c19b(full: Path, half: Path, quarter: Path) -> dict[str, Any]:
    freqs = np.linspace(120.0, 2600.0, 5)
    thetas = np.deg2rad(np.arange(0, 91, 5, dtype=np.float64))
    phis = np.deg2rad(np.arange(0, 91, 5, dtype=np.float64))
    th, ph = np.meshgrid(thetas, phis, indexing="ij")
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)
    open_mask = np.all(dirs > 0.05, axis=-1)
    th_o = th[open_mask]
    ph_o = ph[open_mask]
    probes = np.array(
        [
            [0.03, 0.02, 0.02],
            [0.12, 0.05, -0.10],
            [0.06, 0.09, 0.05],
        ],
        dtype=np.float64,
    )
    exterior = probes[probes[:, 0] > 0.02]
    out: dict[str, Any] = {"half": {}, "quarter": {}}
    with Result(full) as rf, Result(half) as rh, Result(quarter) as rq:
        for tag, rx in (("half", rh), ("quarter", rq)):
            d_max = h_max = di_max = 0.0
            n_planes = len(rx.planes)
            di_plane_db = 10.0 * math.log10(2.0**n_planes)
            for f_hz in freqs:
                d_f = rf.directivity(th_o.ravel(), ph_o.ravel(), float(f_hz), frame="world")
                d_x = rx.directivity(th_o.ravel(), ph_o.ravel(), float(f_hz), frame="world")
                d_max = max(d_max, _rel_max(d_f, d_x))
                h_f = rf.field(exterior, float(f_hz), frame="world")
                h_x = rx.field(exterior, float(f_hz), frame="world")
                h_max = max(h_max, _rel_max(h_f, h_x))
                di_f = float(rf.directivity_index(float(f_hz), frame="world")[0])
                di_x = float(rx.directivity_index(float(f_hz), frame="world")[0])
                di_max = max(di_max, abs((di_x - di_plane_db) - di_f))
            out[tag] = {
                "directivity_rel_max": float(d_max),
                "field_rel_max": float(h_max),
                "di_db_max": float(di_max),
                "pass": bool(d_max <= 1e-5 and h_max <= 1e-5 and di_max <= 1e-4),
            }
    out["pass"] = bool(out["half"]["pass"] and out["quarter"]["pass"])
    return out


def _s2_image_on_axis_db(f_hz: float, d_w: float = 0.55) -> float:
    k = 2.0 * math.pi * f_hz / _C
    val = 2.0 * (1.0 + np.exp(-1j * k * 2.0 * d_w))
    return float(20.0 * math.log10(abs(val)))


def gate_s2(s2: Path, s2_free: Path) -> dict[str, Any]:
    freqs = [50.0, 80.0, 120.0, 200.0, 400.0, 1000.0]
    d_w = 0.55
    rows: list[dict[str, float]] = []
    with Result(s2) as rp, Result(s2_free) as rf:
        for f_hz in freqs:
            d_s2 = complex(rp.directivity(0.0, 0.0, f_hz, frame="local")[0, 0])
            d_fr = complex(rf.directivity(0.0, 0.0, f_hz, frame="local")[0, 0])
            ratio_db = 20.0 * math.log10(abs(d_s2) / max(abs(d_fr), 1e-30))
            ref_db = _s2_image_on_axis_db(f_hz, d_w)
            di_s2 = float(rp.directivity_index(f_hz, frame="local")[0])
            di_fr = float(rf.directivity_index(f_hz, frame="local")[0])
            rows.append(
                {
                    "f_hz": float(f_hz),
                    "on_axis_ratio_db": float(ratio_db),
                    "image_ref_db": float(ref_db),
                    "on_axis_err_db": float(abs(ratio_db - ref_db)),
                    "di_s2_db": float(di_s2),
                    "di_free_db": float(di_fr),
                    "di_offset_db": float(di_s2 - di_fr),
                }
            )
    low_f_offset = rows[0]["di_offset_db"] if rows else float("nan")
    return {
        "d_w_m": d_w,
        "rows": rows,
        "di_low_f_offset_db": float(low_f_offset),
        "limit_on_axis_db": 12.04,
        "limit_di_offset_db": 6.02,
    }


def _plot_c19a(c19a: dict[str, Any], ax) -> None:
    labels = ["z", "zy", "zyx"]
    freqs = [300, 1500, 3000]
    for i, lab in enumerate(labels):
        db = [c19a[lab]["freqs"][str(f)]["far_db_max"] for f in freqs]
        ax.plot(freqs, db, "o-", label=f"{lab} far dB max")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("max error (dB)")
    ax.set_title("(a) C19(A) far-field error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_s2(s2: dict[str, Any], axes) -> None:
    rows = s2["rows"]
    freqs = [r["f_hz"] for r in rows]
    ax = axes[1]
    ax.semilogx(
        freqs,
        [r["on_axis_ratio_db"] for r in rows],
        "o-",
        label="|D_S2/D_free| (dB)",
    )
    ax.semilogx(
        freqs,
        [r["image_ref_db"] for r in rows],
        "--",
        label="image reference",
    )
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("on-axis (dB)")
    ax.set_title("(b) S2 on-axis ratio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = axes[2]
    ax2.semilogx(freqs, [r["di_s2_db"] for r in rows], "o-", label="DI S2")
    ax2.semilogx(freqs, [r["di_free_db"] for r in rows], "s-", label="DI free")
    ax2.axhline(6.02, color="k", ls=":", lw=0.8, label="6.02 dB offset guide")
    ax2.set_xlabel("f (Hz)")
    ax2.set_ylabel("DI (dB)")
    ax2.set_title("(c) DI S2 vs free")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)


def main() -> int:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    c19_full = _probes_cfg(_load_json(_SHOWCASE / "c19_sym_full.json"))
    half = _probes_cfg(_load_json(_SHOWCASE / "c19_half_x.json"))
    quarter = _probes_cfg(_load_json(_SHOWCASE / "c19_quarter_xy.json"))
    s2_cfg = _load_json(_SHOWCASE / "s2_bookshelf.json")
    s2f_cfg = _s2_free(s2_cfg)

    paths = {
        "c19_full": _ensure_h5("c19_full", c19_full, "c19_sym_full.h5"),
        "half": _ensure_h5("half", half, "c19_half_x.h5"),
        "quarter": _ensure_h5("quarter", quarter, "c19_quarter_xy.h5"),
        "s2": _ensure_h5("s2", s2_cfg, "s2.h5"),
        "s2_free": _ensure_h5("s2_free", s2f_cfg, "s2_free.h5"),
    }

    gates: dict[str, Any] = {
        "c19a": gate_c19a(),
        "c19b": gate_c19b(paths["c19_full"], paths["half"], paths["quarter"]),
        "s2": gate_s2(paths["s2"], paths["s2_free"]),
        "elapsed_s": float(time.perf_counter() - t0),
        "h5": {k: str(v) for k, v in paths.items()},
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    _plot_c19a(gates["c19a"], axes[0])
    _plot_s2(gates["s2"], axes)
    fig.tight_layout()
    fig.savefig(_OUT / "gates.png", dpi=140)

    (_OUT / "gates.json").write_text(
        json.dumps(to_json_safe(gates), indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {_OUT / 'gates.json'} and gates.png  ({gates['elapsed_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())