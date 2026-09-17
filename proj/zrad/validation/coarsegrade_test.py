"""Test relaxed far-field baffle mesh grading at ka=5 (centered pilot geometry).

Keeps piston-local resolution unchanged; coarsens the outer baffle beyond a
transition radius with smooth grading. Does not modify existing pilot meshes.
"""

import json
import os
import sys
import time

import gmsh
import numpy as np

from analytical import zrad_norm_analytical
from mesh import (
    BAFFLE_TAG,
    C,
    PISTON_TAG,
    _classify_surfaces,
    _compute_h_piston,
    _mesh_element_stats,
    _piston_rim_curves,
    _set_piston_boundary_mesh,
)
from solve import solve_one

# Pilot geometry — matches offcenter_pilot.py / time_band4_solve.py
A = 0.0125
LX = 8.0 * A
LY = 16.0 * A
THICKNESS = A / 30.0
L_BAFFLE = max(LX, LY) / 2.0
KA = 5.0
POSITION_LABEL = "centered_coarsegrade"

MESH_DIR = os.path.join(os.path.dirname(__file__), "meshes", "offcenter_pilot")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "offcenter_pilot")

# Grading parameters (reported in output)
R_INNER_FACTOR = 2.5   # start far-field coarsening beyond this many radii
R_OUTER_FACTOR = 4.0   # fully coarse by this many radii
H_FAR_WAVELENGTH_DIV = 2.5  # h_far = λ / 2.5  (~2.4× coarser than baseline λ/6)


def _setup_mesh_fields_coarse_far(
    piston_surfs,
    baffle_surfs,
    h_piston,
    h_outer,
    h_far,
    piston_curves,
    a,
    r_inner,
    r_outer,
    piston_center=(0.0, 0.0),
    thickness=None,
):
    """Baffle-only piecewise radial grading; piston/rim held at h_piston via Min."""
    if thickness is None:
        thickness = a / 30.0
    px, py = piston_center

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # r = distance from piston center; near ramp 0→a, flat h_outer to r_inner,
    # then smooth ramp h_outer→h_far over [r_inner, r_outer].
    gmsh.model.mesh.field.add("MathEval", 1)
    gmsh.model.mesh.field.setString(
        1,
        "F",
        (
            f"r=Sqrt(({px}-x)^2+({py}-y)^2);"
            f"hNear={h_piston}+({h_outer}-{h_piston})*Min(r/{a},1);"
            f"tFar=Min(Max((r-{r_inner})/({r_outer}-{r_inner}),0),1);"
            f"hNear+({h_far}-{h_outer})*tFar"
        ),
    )

    gmsh.model.mesh.field.add("Restrict", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumbers(2, "FacesList", baffle_surfs)

    # Constant sizing on piston disk faces.
    gmsh.model.mesh.field.add("Constant", 4)
    gmsh.model.mesh.field.setNumber(4, "VIn", h_piston)
    gmsh.model.mesh.field.add("Restrict", 5)
    gmsh.model.mesh.field.setNumber(5, "InField", 4)
    gmsh.model.mesh.field.setNumbers(5, "FacesList", piston_surfs)

    # Constant sizing on piston rim 1D edges.
    gmsh.model.mesh.field.add("Constant", 7)
    gmsh.model.mesh.field.setNumber(7, "VIn", h_piston)
    gmsh.model.mesh.field.add("Restrict", 8)
    gmsh.model.mesh.field.setNumber(8, "InField", 7)
    gmsh.model.mesh.field.setNumbers(8, "EdgesList", piston_curves)

    gmsh.model.mesh.field.add("Min", 6)
    gmsh.model.mesh.field.setNumbers(6, "FieldsList", [2, 5, 8])
    gmsh.model.mesh.field.setAsBackgroundMesh(6)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_piston)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_far)


def generate_coarsegrade_mesh(out_path=None):
    """Build centered pilot mesh with relaxed far-baffle grading."""
    k = KA / A
    f = k * C / (2.0 * np.pi)
    wavelength = 2.0 * np.pi / k
    h_outer = wavelength / 6.0
    h_far = wavelength / H_FAR_WAVELENGTH_DIV
    h_piston = _compute_h_piston(a=A)
    r_inner = R_INNER_FACTOR * A
    r_outer = R_OUTER_FACTOR * A

    x0 = -LX / 2.0
    y0 = -LY / 2.0
    L_ref = L_BAFFLE

    gmsh.initialize()
    gmsh.model.add(f"coarsegrade_ka{KA:g}_{POSITION_LABEL}")

    occ = gmsh.model.occ
    t = THICKNESS

    box = occ.addBox(x0, y0, -t / 2, LX, LY, t)
    occ.synchronize()

    disk = occ.addDisk(0.0, 0.0, t / 2, A, A, zAxis=[0, 0, 1])
    occ.fragment([(3, box)], [(2, disk)])
    occ.synchronize()

    piston_surfs, baffle_surfs = _classify_surfaces(L_ref, t, a=A, origin=(0.0, 0.0))

    if not piston_surfs:
        gmsh.finalize()
        raise RuntimeError("No piston surfaces found")

    gmsh.model.addPhysicalGroup(2, piston_surfs, tag=PISTON_TAG, name="piston")
    gmsh.model.addPhysicalGroup(2, baffle_surfs, tag=BAFFLE_TAG, name="baffle")

    piston_curves, n_circ = _set_piston_boundary_mesh(piston_surfs, h_piston, a=A)
    _setup_mesh_fields_coarse_far(
        piston_surfs,
        baffle_surfs,
        h_piston,
        h_outer,
        h_far,
        piston_curves,
        a=A,
        r_inner=r_inner,
        r_outer=r_outer,
        thickness=t,
    )
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(2)

    n_piston_tri = 0
    for st in piston_surfs:
        _, tags, _ = gmsh.model.mesh.getElements(2, st)
        n_piston_tri += sum(len(tg) for tg in tags)

    stats = _mesh_element_stats()
    if out_path is None:
        out_path = os.path.join(
            MESH_DIR, "mesh_ka5_a0p0125_centered_coarsegrade.msh"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gmsh.write(out_path)

    grading = {
        "h_piston_m": h_piston,
        "h_outer_m": h_outer,
        "h_far_m": h_far,
        "h_outer_wavelength_div": 6.0,
        "h_far_wavelength_div": H_FAR_WAVELENGTH_DIV,
        "coarsening_factor_vs_h_outer": h_far / h_outer,
        "r_inner_m": r_inner,
        "r_outer_m": r_outer,
        "r_inner_factor_a": R_INNER_FACTOR,
        "r_outer_factor_a": R_OUTER_FACTOR,
        "n_circ_rim": n_circ,
    }

    print("=" * 72)
    print("COARSE-GRADE MESH GENERATION")
    print("=" * 72)
    print(f"  ka={KA:g}  f={f:.1f} Hz  Lx={LX} m  Ly={LY} m  centered")
    print(f"  h_piston={h_piston:.6f} m  (unchanged from baseline)")
    print(f"  h_outer  ={h_outer:.6f} m  (λ/6, mid-baffle target)")
    print(f"  h_far    ={h_far:.6f} m  (λ/{H_FAR_WAVELENGTH_DIV}, "
          f"{h_far / h_outer:.2f}× coarser than h_outer)")
    print(f"  transition: r={R_INNER_FACTOR}a → {R_OUTER_FACTOR}a  "
          f"({r_inner:.5f} → {r_outer:.5f} m)")
    print(f"  Elements : {stats['n_elements']}  (piston: {n_piston_tri})")
    print(f"  h_min={stats['h_min_m']:.6f}  h_max={stats['h_max_m']:.6f}")
    print(
        f"  Quality  : min={stats['quality_min']:.4f}  mean={stats['quality_mean']:.4f}"
        f"  sliver(all)={stats['n_sliver_elements']}"
        f"  top-sliver={stats['top_n_sliver_elements']}"
    )
    print(f"  Written  : {out_path}")

    gmsh.finalize()

    return {
        "mesh_path": out_path,
        "ka": KA,
        "a_m": A,
        "lx_m": LX,
        "ly_m": LY,
        "position_label": POSITION_LABEL,
        "n_piston_elements": n_piston_tri,
        "grading": grading,
        **stats,
    }


def main():
    mesh_path = os.path.join(MESH_DIR, "mesh_ka5_a0p0125_centered_coarsegrade.msh")
    result_path = os.path.join(RESULTS_DIR, "coarsegrade_ka5_centered.json")

    t0 = time.perf_counter()
    mesh_info = generate_coarsegrade_mesh(out_path=mesh_path)

    result = solve_one(
        KA,
        mesh_file=mesh_path,
        far_field_surfaces="top",
        a=A,
        L_baffle_m=L_BAFFLE,
    )
    dt_wall = time.perf_counter() - t0

    z_analytical = zrad_norm_analytical(KA)
    z_hybrid_re = result["Z_rad_norm_real_ff"]
    z_hybrid_im = result["Z_rad_norm_imag"]

    # Baseline from pilot_results.json (ka=5 centered)
    baseline = {
        "n_elements": 14452,
        "n_dofs": 7228,
        "n_piston_elements": 300,
        "assembly_time_s": 161.8,
        "linear_solve_time_s": 41.3,
        "Z_hybrid_real": 0.9825751585997229,
        "Z_hybrid_imag": 0.17659034712226307,
        "h_max_m": 0.0034457692520754755,
        "quality_min": 0.15839162576519544,
        "n_sliver_elements": 0,
        "top_n_sliver_elements": 0,
    }

    elem_reduction = 1.0 - result["n_elements"] / baseline["n_elements"]
    dof_reduction = 1.0 - result["n_dofs"] / baseline["n_dofs"]

    report = {
        "test": "coarsegrade_far_baffle",
        "geometry": {"a_m": A, "Lx_m": LX, "Ly_m": LY, "position": "centered"},
        "grading": mesh_info["grading"],
        "mesh": mesh_info,
        "solve": result,
        "Z_hybrid_real": z_hybrid_re,
        "Z_hybrid_imag": z_hybrid_im,
        "Z_analytical_real": float(np.real(z_analytical)),
        "Z_analytical_imag": float(np.imag(z_analytical)),
        "baseline": baseline,
        "wall_clock_s": dt_wall,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(report, f, indent=2)

    re_err_baseline = abs(baseline["Z_hybrid_real"] - baseline["Z_hybrid_real"])  # 0
    re_err_test_vs_analytical = abs(z_hybrid_re - float(np.real(z_analytical)))
    re_err_baseline_vs_analytical = abs(
        baseline["Z_hybrid_real"] - float(np.real(z_analytical))
    )
    im_err_test_vs_analytical = abs(z_hybrid_im - float(np.imag(z_analytical)))
    im_err_baseline_vs_analytical = abs(
        baseline["Z_hybrid_imag"] - float(np.imag(z_analytical))
    )
    re_delta_vs_baseline = z_hybrid_re - baseline["Z_hybrid_real"]
    im_delta_vs_baseline = z_hybrid_im - baseline["Z_hybrid_imag"]

    print("\n" + "=" * 72)
    print("COARSE-GRADE TEST RESULTS vs BASELINE")
    print("=" * 72)
    print(f"  Elements : {result['n_elements']:,}  (baseline {baseline['n_elements']:,}, "
          f"Δ {elem_reduction * 100:+.1f}%)")
    print(f"  DOFs     : {result['n_dofs']:,}  (baseline {baseline['n_dofs']:,}, "
          f"Δ {dof_reduction * 100:+.1f}%)")
    print(f"  Piston el: {result['n_piston_elements']}  (baseline {baseline['n_piston_elements']})")
    print(f"  Assembly : {result['assembly_time_s']:.1f} s  (baseline {baseline['assembly_time_s']:.1f} s)")
    print(f"  Solve    : {result['linear_solve_time_s']:.1f} s  (baseline {baseline['linear_solve_time_s']:.1f} s)")
    print(f"  Asm+solve: {result['solve_time_s']:.1f} s  "
          f"(baseline {baseline['assembly_time_s'] + baseline['linear_solve_time_s']:.1f} s)")
    print(f"  Wall     : {dt_wall:.1f} s")
    print()
    print(f"  Re(Z) ff top : {z_hybrid_re:.6f}  (baseline {baseline['Z_hybrid_real']:.6f}, "
          f"analytical {float(np.real(z_analytical)):.6f})")
    print(f"    vs baseline Δ = {re_delta_vs_baseline:+.6f}")
    print(f"    |error| test={re_err_test_vs_analytical:.6f}  "
          f"baseline={re_err_baseline_vs_analytical:.6f}")
    print(f"  Im(Z) near   : {z_hybrid_im:.6f}  (baseline {baseline['Z_hybrid_imag']:.6f}, "
          f"analytical {float(np.imag(z_analytical)):.6f})")
    print(f"    vs baseline Δ = {im_delta_vs_baseline:+.6f}")
    print(f"    |error| test={im_err_test_vs_analytical:.6f}  "
          f"baseline={im_err_baseline_vs_analytical:.6f}")
    print()
    print(f"  Mesh quality : min={mesh_info['quality_min']:.4f}  "
          f"(baseline {baseline['quality_min']:.4f})")
    print(f"  h_max        : {mesh_info['h_max_m']:.6f} m  "
          f"(baseline {baseline['h_max_m']:.6f} m)")
    print(f"  Sliver elems : all={mesh_info['n_sliver_elements']}  top={mesh_info['top_n_sliver_elements']}"
          f"  (baseline 0/0)")
    print(f"  Results JSON : {result_path}")
    print("=" * 72)

    # Viability assessment
    viable = (
        elem_reduction >= 0.30
        and re_err_test_vs_analytical <= re_err_baseline_vs_analytical * 1.5
        and im_err_test_vs_analytical <= im_err_baseline_vs_analytical * 1.5
    )
    if viable:
        print("VERDICT: Viable path — substantial element reduction with comparable accuracy.")
    elif elem_reduction < 0.15:
        print("VERDICT: Marginal savings — element count reduction too small to justify.")
    else:
        print("VERDICT: Accuracy degraded or savings insufficient — fall back to capped-ka sweep.")
    print("=" * 72)

    if result.get("linear_solve_info", 0) != 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
