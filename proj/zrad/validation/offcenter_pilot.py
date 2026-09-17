"""Pilot: fixed-size finite baffle with centered vs near-tangent piston positions.

2 positions × 3 ka = 6 solves. Uses hybrid extraction (top-only far-field Re,
near-field Im) matching the dense validation sweep.
"""

import json
import os
import sys
import time

import numpy as np

from analytical import zrad_norm_analytical
from mesh import (
    BAFFLE_TAG,
    PISTON_TAG,
    THICKNESS,
    fixed_baffle_mesh_path,
    generate_fixed_baffle_mesh,
)
from solve import RTOL, solve_one

# ── Pilot geometry (origin at piston center) ─────────────────────────────────
A = 0.0125
LX = 8.0 * A  # 0.1 m
LY = 16.0 * A  # 0.2 m
EDGE_MARGIN = A / 10.0  # gap between piston rim and nearest baffle edge
THICKNESS_PILOT = A / 30.0

KA_VALUES = [0.2, 1.0, 5.0]
POSITIONS = {
    "centered": {"piston_dx": 0.0, "piston_dy": 0.0},
    "near_tangent": {
        "piston_dx": 0.0,
        "piston_dy": LY / 2.0 - A - EDGE_MARGIN,
    },
}

MESH_DIR = os.path.join(os.path.dirname(__file__), "meshes", "offcenter_pilot")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "offcenter_pilot")
L_BAFFLE = max(LX, LY) / 2.0

# Validation reference: hybrid (ff_top Re + near-field Im) from dense/centered study.
VALIDATION_REF = {
    0.2: {"source": "interp ka=0.17556 & ka=0.266835 ff_top"},
    1.0: {"source": "result_ka1_ff_pistref (ff surfaces=all, Re only)"},
    5.0: {"source": "result_ka5_ff_top"},
}


def _load_validation_hybrid(ka):
    """Return (Re_ff, Im_nf) from validation study at the given ka."""
    import json as _json

    results_dir = os.path.join(os.path.dirname(__file__), "results")

    def _load(path):
        with open(path, encoding="utf-8") as f:
            return _json.load(f)

    if ka == 5.0:
        row = _load(os.path.join(results_dir, "result_ka5_ff_top.json"))
        return row["Z_rad_norm_real_ff"], row["Z_rad_norm_imag"]

    if ka == 1.0:
        row = _load(os.path.join(results_dir, "result_ka1_ff_pistref.json"))
        return row["Z_rad_norm_real_ff"], row["Z_rad_norm_imag"]

    if ka == 0.2:
        lo = _load(os.path.join(results_dir, "result_ka0p17556_ff_top.json"))
        hi = _load(os.path.join(results_dir, "result_ka0p266835_ff_top.json"))
        t = (ka - lo["ka"]) / (hi["ka"] - lo["ka"])
        re_ff = (1 - t) * lo["Z_rad_norm_real_ff"] + t * hi["Z_rad_norm_real_ff"]
        im_nf = (1 - t) * lo["Z_rad_norm_imag"] + t * hi["Z_rad_norm_imag"]
        return re_ff, im_nf

    z = zrad_norm_analytical(ka)
    return float(np.real(z)), float(np.imag(z))


def check_surface_tagging(mesh_file, a=A):
    """Sanity-check piston vs baffle-top/bottom/side classification."""
    import bempp_cl.api as bempp

    grid = bempp.import_grid(mesh_file)
    domain_indices = grid.domain_indices.astype(int)
    centers = np.asarray(grid.centroids)  # (n_elements, 3)
    normals = np.asarray(grid.normals)

    piston = domain_indices == PISTON_TAG
    baffle = domain_indices == BAFFLE_TAG

    top_baffle = baffle & (normals[:, 2] > 0.5) & (centers[:, 2] > 0.0)
    bottom = baffle & (normals[:, 2] < -0.5)
    sides = baffle & (np.abs(normals[:, 2]) <= 0.5)

    r_piston = np.hypot(centers[piston, 0], centers[piston, 1])
    y_top_baffle = centers[top_baffle, 1]
    y_max_top = float(np.max(y_top_baffle)) if np.any(top_baffle) else float("nan")
    y_min_top = float(np.min(y_top_baffle)) if np.any(top_baffle) else float("nan")

    return {
        "n_piston": int(np.sum(piston)),
        "n_baffle_top": int(np.sum(top_baffle)),
        "n_baffle_bottom": int(np.sum(bottom)),
        "n_baffle_sides": int(np.sum(sides)),
        "r_piston_max": float(np.max(r_piston)) if np.any(piston) else float("nan"),
        "y_top_baffle_range": [y_min_top, y_max_top],
    }


def run_pilot(solver="auto"):
    os.makedirs(MESH_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("Off-center pilot: fixed baffle geometry")
    print(f"  a={A} m  Lx={LX} m  Ly={LY} m  edge_margin={EDGE_MARGIN} m")
    print(f"  near-tangent dy={POSITIONS['near_tangent']['piston_dy']:.6f} m")
    print(f"  gap piston-top to baffle-top = {EDGE_MARGIN:.6f} m (= a/10)")
    print(f"  ka values: {KA_VALUES}")
    print("=" * 72)

    all_results = []
    t0 = time.perf_counter()

    for pos_label, pos in POSITIONS.items():
        for ka in KA_VALUES:
            print(f"\n{'─' * 72}")
            print(f"Mesh + solve: ka={ka:g}  position={pos_label}")
            print(f"{'─' * 72}")

            mesh_info = generate_fixed_baffle_mesh(
                ka,
                a=A,
                lx=LX,
                ly=LY,
                piston_dx=pos["piston_dx"],
                piston_dy=pos["piston_dy"],
                position_label=pos_label,
                thickness=THICKNESS_PILOT,
                mesh_dir=MESH_DIR,
            )
            mesh_file = mesh_info["mesh_path"]

            tagging = check_surface_tagging(mesh_file, a=A)
            print(
                f"  Surface tags: piston={tagging['n_piston']}  "
                f"top={tagging['n_baffle_top']}  bottom={tagging['n_baffle_bottom']}  "
                f"sides={tagging['n_baffle_sides']}"
            )
            print(
                f"  Piston r_max={tagging['r_piston_max']:.6f} m (a={A})  "
                f"top-baffle y=[{tagging['y_top_baffle_range'][0]:.5f}, "
                f"{tagging['y_top_baffle_range'][1]:.5f}]"
            )

            result = solve_one(
                ka,
                mesh_file=mesh_file,
                solver=solver,
                rtol=RTOL,
                far_field_surfaces="top",
                a=A,
                L_baffle_m=L_BAFFLE,
            )

            re_hybrid = result["Z_rad_norm_real_ff"]
            im_hybrid = result["Z_rad_norm_imag"]
            z_ana = zrad_norm_analytical(ka)

            row = {
                "position": pos_label,
                "piston_dx_m": pos["piston_dx"],
                "piston_dy_m": pos["piston_dy"],
                "edge_margin_m": EDGE_MARGIN,
                "mesh": mesh_info,
                "tagging": tagging,
                "solve": result,
                "Z_hybrid_real": re_hybrid,
                "Z_hybrid_imag": im_hybrid,
                "Z_analytical_real": float(np.real(z_ana)),
                "Z_analytical_imag": float(np.imag(z_ana)),
            }
            all_results.append(row)

            info = result["linear_solve_info"]
            status = "OK" if info == 0 else f"FAIL({info})"
            print(
                f"  Solve {status}  elements={result['n_elements']}  "
                f"DOFs={result['n_dofs']}  solver={result['solver']}"
            )
            print(
                f"  Piston area ratio: {result['piston_area_ratio']:.6f}  "
                f"(BEM={result['piston_area_m2']:.8e}  πa²={result['piston_area_analytic_m2']:.8e})"
            )
            print(
                f"  Z/(ρc) hybrid = {re_hybrid:.6f} + {im_hybrid:.6f}j  "
                f"(Re=ff_top, Im=near-field)"
            )

            flags = []
            if re_hybrid < 0:
                flags.append("Re<0")
            if im_hybrid < 0:
                flags.append("Im<0")
            if not np.isfinite(re_hybrid) or not np.isfinite(im_hybrid):
                flags.append("NaN/Inf")
            if flags:
                print(f"  *** WARNING: {', '.join(flags)} ***")

    # Summary table
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(
        f"{'pos':>14}  {'ka':>5}  {'n_el':>6}  {'h_min':>9}  {'sliver':>6}  "
        f"{'info':>4}  {'Re':>10}  {'Im':>10}  {'area_ratio':>10}"
    )
    print("-" * 90)

    issues = []
    for row in all_results:
        m = row["mesh"]
        s = row["solve"]
        info = s["linear_solve_info"]
        print(
            f"{row['position']:>14}  {s['ka']:5g}  {m['n_elements']:6d}  "
            f"{m['h_min_m']:9.6f}  {m['n_sliver_elements']:6d}  "
            f"{info:4d}  {row['Z_hybrid_real']:10.6f}  {row['Z_hybrid_imag']:10.6f}  "
            f"{s['piston_area_ratio']:10.6f}"
        )
        if info != 0:
            issues.append(f"ka={s['ka']} {row['position']}: solve info={info}")
        # Low-ka coarse annulus triangles (λ >> baffle) can have q<0.05 without
        # being gap degeneracy; near-tangent gap strip is checked separately below.
        if m.get("top_n_sliver_elements", 0) > 0 and s["ka"] >= 5.0:
            issues.append(
                f"ka={s['ka']} {row['position']}: "
                f"{m['top_n_sliver_elements']} top-surface sliver elements"
            )
        if s["piston_area_ratio"] < 0.99 or s["piston_area_ratio"] > 1.01:
            issues.append(
                f"ka={s['ka']} {row['position']}: area ratio={s['piston_area_ratio']:.4f}"
            )

    # Centered vs validation comparison
    print(f"\n{'─' * 72}")
    print("Centered vs validation (hybrid: ff_top Re + near-field Im)")
    print(f"{'pos':>14}  {'ka':>5}  {'pilot Re':>10}  {'val Re':>10}  {'ΔRe%':>8}  "
          f"{'pilot Im':>10}  {'val Im':>10}  {'ΔIm%':>8}")
    print("-" * 90)

    for ka in KA_VALUES:
        pilot = next(r for r in all_results if r["position"] == "centered" and r["solve"]["ka"] == ka)
        val_re, val_im = _load_validation_hybrid(ka)
        pre = pilot["Z_hybrid_real"]
        pim = pilot["Z_hybrid_imag"]
        d_re = 100.0 * abs(pre - val_re) / max(abs(val_re), 1e-12)
        d_im = 100.0 * abs(pim - val_im) / max(abs(val_im), 1e-12)
        print(
            f"{'centered':>14}  {ka:5g}  {pre:10.6f}  {val_re:10.6f}  {d_re:8.2f}  "
            f"{pim:10.6f}  {val_im:10.6f}  {d_im:8.2f}"
        )
        # Small fixed baffle at low ka is not infinite-baffle-like; only flag mid/high ka.
        if ka >= 1.0 and (d_re > 15 or d_im > 15):
            issues.append(
                f"ka={ka} centered: large deviation from validation "
                f"(ΔRe={d_re:.1f}%, ΔIm={d_im:.1f}%)"
            )

    out_json = os.path.join(RESULTS_DIR, "pilot_results.json")
    payload = {
        "geometry": {
            "a_m": A,
            "Lx_m": LX,
            "Ly_m": LY,
            "edge_margin_m": EDGE_MARGIN,
            "positions": POSITIONS,
        },
        "ka_values": KA_VALUES,
        "results": all_results,
        "elapsed_s": time.perf_counter() - t0,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")
    print(f"Total elapsed: {time.perf_counter() - t0:.1f} s")

    if issues:
        print(f"\n*** ISSUES FOUND ({len(issues)}) — do not proceed to full sweep ***")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("\nAll checks passed — ready to define full off-center parametric sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(run_pilot())
