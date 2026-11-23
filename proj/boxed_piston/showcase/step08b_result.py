#!/usr/bin/env python3
"""Step 8b gates: C19(B) floor mirror-and-sum, C20(iii) multi-radiator power."""

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
_OUT = _PKG / "outputs" / "step08b"
_SHOWCASE = _PKG / "configs" / "showcase"
_REF_08A = _PKG / "outputs" / "step08a" / "s2_multirad.h5"

sys.path.insert(0, str(_PKG))

from boxrad import Result  # noqa: E402
from boxrad.config import Config  # noqa: E402
from boxrad.geomderive import derive_geom  # noqa: E402
from boxrad.geom.geometry import build_geometry  # noqa: E402
from boxrad.geomrun import dry_run_geom  # noqa: E402
from boxrad.post.transfer import load_huygens_spectra  # noqa: E402
from boxrad.post.transfer import h_transfer_err  # noqa: E402
from boxrad.runner import keep_awake, run  # noqa: E402
from validation._common import to_json_safe  # noqa: E402

_F64 = np.float64
_PROBE_EXTRA = [(0.05, 0.30, 0.12), (-0.08, 0.40, 0.25)]


def _load_json(path: Path) -> Config:
    return Config.from_json(path.read_text(encoding="utf-8"))


def _probes_cfg(cfg: Config, extra: list[tuple[float, float, float]]) -> Config:
    raw = json.loads(cfg.to_json())
    probes = list(raw.get("output", {}).get("probes") or [])
    probes.extend([list(p) for p in extra])
    raw.setdefault("output", {})["probes"] = probes
    return Config.from_dict(raw)


def _load_s2_free() -> Config:
    cfg = _load_json(_SHOWCASE / "s2_bookshelf.json")
    raw = json.loads(cfg.to_json())
    raw.pop("environment", None)
    return Config.from_dict(raw)


def _min_p_avg_trace(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    traces: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for p in paths:
        with h5py.File(p, "r") as f:
            traces.append(np.asarray(f["radiator/p_avg"][()], dtype=_F64))
            times.append(np.asarray(f["radiator/times_p"][()], dtype=_F64))
    n = min(t.shape[0] for t in traces)
    t_com = times[0][:n]
    return t_com, np.stack([t[:n] for t in traces], axis=0)


def _probe_traces(path: Path, n: int) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        if "probes" not in f:
            raise RuntimeError(f"{path} has no probes group")
        return np.asarray(f["probes/p"][()][:n], dtype=_F64)


def _assert_grid_coincidence(cfg_floor: Config, cfg_mirror: Config) -> dict[str, Any]:
    g0 = build_geometry(cfg_floor)
    g1 = build_geometry(cfg_mirror)
    assert g0 is not None and g1 is not None
    d0, _ = derive_geom(cfg_floor, g0)
    d1, _ = derive_geom(cfg_mirror, g1)
    h0, h1 = float(d0.h), float(d1.h)
    assert h0 == h1
    z_extent = 0.30
    assert abs(z_extent / h0 - round(z_extent / h0)) < 1e-12
    return {"h_m": h0, "z_extent_m": z_extent, "z_over_h": z_extent / h0}


def _far_points_r3m(n_pts: int = 6) -> np.ndarray:
    theta = np.linspace(0.25, 0.85, n_pts) * math.pi
    phi = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    r = 3.0
    pts = np.column_stack(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )
    pts = pts[pts[:, 2] > 0.0]
    if pts.shape[0] < n_pts:
        z = np.linspace(0.2, 2.8, n_pts)
        ang = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
        pts = np.column_stack([0.5 * np.cos(ang), 0.5 * np.sin(ang), z])
    return pts[:n_pts].astype(_F64)


def _w_huygens_multirad(
    primary: Path,
    freqs: np.ndarray,
    u_vec: np.ndarray,
) -> np.ndarray:
    with Result(primary, frame="world") as res:
        s_list = res.S_list
        n_f = freqs.size
        out = np.zeros(n_f, dtype=_F64)
        patches0, _, _, _ = load_huygens_spectra(primary, res.derived, float(freqs[0]))
        for fi, f_hz in enumerate(freqs):
            p_sum = None
            vn_sum = None
            areas = patches0.areas
            for j, uj in enumerate(u_vec):
                path_j = res.radiator(j).path
                patches, v_ref, p_f, vn_f = load_huygens_spectra(path_j, res.derived, float(f_hz))
                if fi == 0 and j > 0:
                    if not np.allclose(patches.centers, patches0.centers, rtol=0.0, atol=1e-12):
                        raise AssertionError("Huygens centers differ between multi-radiator files")
                scale = complex(uj) / complex(s_list[j]) / v_ref
                term_p = p_f * scale
                term_vn = vn_f * scale
                p_sum = term_p if p_sum is None else p_sum + term_p
                vn_sum = term_vn if vn_sum is None else vn_sum + term_vn
            assert p_sum is not None and vn_sum is not None
            out[fi] = float(0.5 * np.real(np.sum(p_sum * np.conj(vn_sum) * areas)))
        return out


def main() -> int:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    gates: dict[str, Any] = {}

    cfg_floor = _probes_cfg(_load_json(_SHOWCASE / "c19b_floor.json"), _PROBE_EXTRA)
    cfg_mirror = _probes_cfg(_load_json(_SHOWCASE / "c19b_floor_mirror.json"), _PROBE_EXTRA)
    gates["grid_coincidence"] = _assert_grid_coincidence(cfg_floor, cfg_mirror)

    floor_h5 = _OUT / "c19b_floor.h5"
    mirror_h5 = _OUT / "c19b_floor_mirror.h5"
    est_f = dry_run_geom(cfg_floor, out_path=floor_h5)
    est_m = dry_run_geom(cfg_mirror, out_path=mirror_h5)
    print(
        f"c19b_floor: n_steps={est_f.get('n_steps')}  "
        f"wall_est={est_f.get('wall_time_s_est', float('nan')):.3g} s"
    )
    print(
        f"c19b_mirror: n_runs={est_m.get('n_runs')}  "
        f"wall_est={est_m.get('wall_time_s_est_total', float('nan')):.3g} s"
    )
    with keep_awake("step08b c19b"):
        run(cfg_floor, out_path=floor_h5, progress=True)
        run(cfg_mirror, out_path=mirror_h5, progress=True)

    mirror_m = _OUT / "c19b_floor_mirror.r1_woofer_m.h5"
    import h5py

    t_min, stack = _min_p_avg_trace([floor_h5, mirror_h5])
    n = t_min.shape[0]
    p_floor = stack[0, :]
    with h5py.File(mirror_m, "r") as fm:
        p_m1 = np.asarray(fm["radiator/p_avg_all"][()], dtype=_F64)[:n, 0]
    p_mirror_sum = stack[1, :] + p_m1
    denom = max(float(np.max(np.abs(p_floor))), 1e-30)
    diff_p = float(np.max(np.abs(p_floor - p_mirror_sum))) / denom
    gates["c19b_p_avg_max_rel"] = diff_p
    gates["c19b_p_avg_pass"] = bool(diff_p <= 1e-5)

    n_pr = n
    pf = _probe_traces(floor_h5, n_pr)
    pm0 = _probe_traces(mirror_h5, n_pr)
    pm1 = _probe_traces(mirror_m, n_pr)
    pr_sum = pm0 + pm1
    d_pr = float(np.max(np.abs(pf - pr_sum))) / max(float(np.max(np.abs(pf))), 1e-30)
    gates["c19b_probes_max_rel"] = d_pr
    gates["c19b_probes_pass"] = bool(d_pr <= 1e-5)

    with Result(mirror_h5, frame="world") as r_m, Result(floor_h5, frame="world") as r_f:
        s0, s1 = float(r_m.S_list[0]), float(r_m.S_list[1])
        pts = _far_points_r3m(6)
        freqs_ff = np.geomspace(200.0, min(1500.0, r_f.band[1]), 4)
        max_db = 0.0
        max_ph = 0.0
        for f_hz in freqs_ff:
            h_sum = r_m.field_sum(pts, f_hz, U=(s0, s1), frame="world")
            h_floor = r_f.field(pts, f_hz, frame="world")
            for q in range(pts.shape[0]):
                db, ph = h_transfer_err(h_floor[0, q], h_sum[0, q])
                max_db = max(max_db, abs(db))
                max_ph = max(max_ph, abs(ph))
        gates["c19b_farfield_max_db"] = max_db
        gates["c19b_farfield_max_deg"] = max_ph

    s2_h5 = _REF_08A
    if not s2_h5.is_file():
        s2_cfg = _load_s2_free()
        s2_h5 = _OUT / "s2_multirad.h5"
        with keep_awake("step08b s2-free"):
            run(s2_cfg, out_path=s2_h5, progress=True)

    with Result(s2_h5, frame="world") as r_s2:
        f_min, f_max = r_s2.band
        freqs = np.geomspace(0.1 * f_max, f_max, 8)
        u_cases = {"pp": np.array([1.0, 1.0]), "pm": np.array([1.0, -1.0])}
        c20: dict[str, Any] = {}
        for label, u_vec in u_cases.items():
            w_h = _w_huygens_multirad(r_s2.path, freqs, u_vec)
            ratios: list[float] = []
            for fi, f_hz in enumerate(freqs):
                z_mat = r_s2.Z_matrix(f_hz)
                z_nn = z_mat[0]
                w_z = float(0.5 * np.real(u_vec.conj() @ z_nn @ u_vec))
                ratios.append(float(w_h[fi] / max(w_z, 1e-30) - 1.0))
            c20[label] = {
                "freqs_hz": freqs.tolist(),
                "w_huy_over_w_z_minus_1": ratios,
                "max_abs_ratio": float(np.max(np.abs(ratios))),
                "pass": bool(np.max(np.abs(ratios)) <= 0.02),
            }
        gates["c20iii"] = c20

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    ax0 = axes[0]
    ax0.semilogy(t_min, np.maximum(np.abs(p_floor), 1e-30), label="floor p_avg")
    ax0.semilogy(t_min, np.maximum(np.abs(p_mirror_sum), 1e-30), label="mirror sum")
    ax0.semilogy(
        t_min,
        np.maximum(np.abs(p_floor - p_mirror_sum), 1e-30),
        ls="--",
        label="|diff|",
    )
    ax0.set_xlabel("t (s)")
    ax0.set_ylabel("|p| (Pa)")
    ax0.legend(fontsize=8)
    ax0.set_title("C19(B) p_avg floor vs mirror superposition")

    ax1 = axes[1]
    for label, sty in (("pp", "o-"), ("pm", "s--")):
        rr = c20[label]["w_huy_over_w_z_minus_1"]
        ax1.semilogx(freqs, rr, sty, label=f"U={label}")
    ax1.axhline(0.02, color="gray", ls=":", lw=0.8)
    ax1.axhline(-0.02, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("f (Hz)")
    ax1.set_ylabel(r"$W_{\mathrm{huy}}/W_Z - 1$")
    ax1.legend()
    ax1.set_title("C20(iii) Huygens vs circuit power")

    fig.tight_layout()
    fig.savefig(_OUT / "gates.png", dpi=140)

    gates["elapsed_s"] = time.perf_counter() - t0
    (_OUT / "gates.json").write_text(
        json.dumps(to_json_safe(gates), indent=2),
        encoding="utf-8",
    )
    print(
        f"C19(B) p_avg rel diff {gates['c19b_p_avg_max_rel']:.3e} "
        f"{'PASS' if gates['c19b_p_avg_pass'] else 'FAIL'}"
    )
    print(f"wrote {_OUT / 'gates.json'} and gates.png  ({gates['elapsed_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())