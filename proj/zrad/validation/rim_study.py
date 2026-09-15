"""Piston-rim discontinuity diagnostics (tagging check + rim refinement + low-ka LU).

Does not modify mesh.py defaults or baffle geometry. Study meshes go to meshes/rim_study/.
"""

import argparse
import json
import os
import sys

import bempp_cl.api as bempp
import gmsh
import numpy as np

from analytical import zrad_norm_analytical
from mesh import (
    A,
    BAFFLE_TAG,
    PISTON_TAG,
    THICKNESS,
    _classify_surfaces,
    baffle_half_width,
    frequency_hz,
    mesh_path,
)
from solve import RTOL, solve_one

STUDY_DIR = os.path.join(os.path.dirname(__file__), "meshes", "rim_study")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "rim_study")


def _study_mesh_path(ka, label):
    tag = f"{ka:g}".replace(".", "p")
    return os.path.join(STUDY_DIR, f"mesh_ka{tag}_{label}.msh")


def check_tagging(mesh_file, ka=0.5):
    """Print piston element count and area vs pi*a^2."""
    grid = bempp.import_grid(mesh_file)
    domain_indices, areas = grid.domain_indices.astype(int), grid.volumes.astype(float)
    piston_mask = domain_indices == PISTON_TAG
    n_piston = int(np.sum(piston_mask))
    area_piston = float(np.sum(areas[piston_mask]))
    area_analytic = np.pi * A**2

    verts = grid.vertices
    elems = grid.elements
    centers = np.mean(verts[:, elems], axis=1)
    r_piston = np.hypot(centers[0, piston_mask], centers[1, piston_mask])
    r_max = float(np.max(r_piston)) if n_piston else float("nan")
    r_min = float(np.min(r_piston)) if n_piston else float("nan")

    baffle_mask = domain_indices != PISTON_TAG
    top_baffle = baffle_mask & (centers[2] > 0)
    r_baffle_top = np.hypot(centers[0, top_baffle], centers[1, top_baffle])
    r_baffle_min = float(np.min(r_baffle_top)) if np.any(top_baffle) else float("nan")

    print(f"\n=== Tagging check: {os.path.basename(mesh_file)} (ka={ka:g}) ===")
    print(f"  Piston elements     : {n_piston}")
    print(f"  Piston area (BEM)   : {area_piston:.8e} m²")
    print(f"  Piston area (πa²)   : {area_analytic:.8e} m²")
    print(f"  Area ratio BEM/πa²  : {area_piston / area_analytic:.6f}")
    print(f"  Piston r range      : [{r_min:.6f}, {r_max:.6f}] m  (a={A} m)")
    print(f"  Top-baffle r_min    : {r_baffle_min:.6f} m  (gap at rim: {r_baffle_min - A:.2e} m)")
    return {
        "n_piston_elements": n_piston,
        "area_piston_m2": area_piston,
        "area_analytic_m2": area_analytic,
        "area_ratio": area_piston / area_analytic,
        "r_piston_min_m": r_min,
        "r_piston_max_m": r_max,
        "r_baffle_top_min_m": r_baffle_min,
    }


def _setup_grading(piston_surfs, baffle_surfs, h_piston, h_outer, rim_mode="baseline"):
    """Configure Gmsh background mesh fields."""
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    t = THICKNESS
    all_surfs = piston_surfs + baffle_surfs

    if rim_mode == "baseline":
        gmsh.model.mesh.field.add("Ball", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", h_piston)
        gmsh.model.mesh.field.setNumber(1, "VOut", h_outer)
        gmsh.model.mesh.field.setNumber(1, "Radius", A)
        gmsh.model.mesh.field.setNumber(1, "XCenter", 0.0)
        gmsh.model.mesh.field.setNumber(1, "YCenter", 0.0)
        gmsh.model.mesh.field.setNumber(1, "ZCenter", t / 2)

        gmsh.model.mesh.field.add("Restrict", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumbers(2, "FacesList", all_surfs)

        gmsh.model.mesh.field.add("Threshold", 3)
        gmsh.model.mesh.field.setNumber(3, "InField", 2)
        gmsh.model.mesh.field.setNumber(3, "SizeMin", h_piston)
        gmsh.model.mesh.field.setNumber(3, "SizeMax", h_outer)
        gmsh.model.mesh.field.setNumber(3, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(3, "DistMax", A)
        gmsh.model.mesh.field.setAsBackgroundMesh(3)

    elif rim_mode == "half_hpiston":
        # Same topology as baseline but halved h_piston everywhere.
        _setup_grading(piston_surfs, baffle_surfs, h_piston, h_outer, rim_mode="baseline")

    elif rim_mode == "rim_band":
        # Fine band at r≈a: Ball at piston edge + baseline interior grading, take Min.
        h_rim = h_piston / 2.0
        band = A / 6.0

        gmsh.model.mesh.field.add("Ball", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", h_piston)
        gmsh.model.mesh.field.setNumber(1, "VOut", h_outer)
        gmsh.model.mesh.field.setNumber(1, "Radius", A)
        gmsh.model.mesh.field.setNumber(1, "XCenter", 0.0)
        gmsh.model.mesh.field.setNumber(1, "YCenter", 0.0)
        gmsh.model.mesh.field.setNumber(1, "ZCenter", t / 2)

        gmsh.model.mesh.field.add("Restrict", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumbers(2, "FacesList", all_surfs)

        gmsh.model.mesh.field.add("Threshold", 3)
        gmsh.model.mesh.field.setNumber(3, "InField", 2)
        gmsh.model.mesh.field.setNumber(3, "SizeMin", h_piston)
        gmsh.model.mesh.field.setNumber(3, "SizeMax", h_outer)
        gmsh.model.mesh.field.setNumber(3, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(3, "DistMax", A)

        gmsh.model.mesh.field.add("Ball", 4)
        gmsh.model.mesh.field.setNumber(4, "VIn", h_rim)
        gmsh.model.mesh.field.setNumber(4, "VOut", h_piston)
        gmsh.model.mesh.field.setNumber(4, "Radius", band)
        gmsh.model.mesh.field.setNumber(4, "XCenter", A)
        gmsh.model.mesh.field.setNumber(4, "YCenter", 0.0)
        gmsh.model.mesh.field.setNumber(4, "ZCenter", t / 2)

        gmsh.model.mesh.field.add("Restrict", 5)
        gmsh.model.mesh.field.setNumber(5, "InField", 4)
        gmsh.model.mesh.field.setNumbers(5, "FacesList", all_surfs)

        gmsh.model.mesh.field.add("Min", 6)
        gmsh.model.mesh.field.setNumbers(6, "FieldsList", [3, 5])
        gmsh.model.mesh.field.setAsBackgroundMesh(6)
    else:
        raise ValueError(f"Unknown rim_mode: {rim_mode!r}")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_piston / 2 if rim_mode != "baseline" else h_piston)
    if rim_mode == "rim_band":
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_rim)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_outer)


def generate_study_mesh(ka, rim_mode="baseline", out_path=None):
    """Generate a study mesh with fixed baffle geometry; vary rim grading only."""
    k = ka / A
    f = frequency_hz(ka)
    L = baffle_half_width(ka)
    wavelength = 2.0 * np.pi / k
    h_outer = wavelength / 6.0
    h_piston = min(h_outer, A / 12.0)
    if rim_mode == "half_hpiston":
        h_piston /= 2.0

    if out_path is None:
        out_path = _study_mesh_path(ka, rim_mode)

    gmsh.initialize()
    gmsh.model.add(f"rim_study_ka{ka:g}_{rim_mode}")
    occ = gmsh.model.occ
    t = THICKNESS

    box = occ.addBox(-L, -L, -t / 2, 2 * L, 2 * L, t)
    occ.synchronize()
    disk = occ.addDisk(0, 0, t / 2, A, A, zAxis=[0, 0, 1])
    occ.fragment([(3, box)], [(2, disk)])
    occ.synchronize()

    piston_surfs, baffle_surfs = _classify_surfaces(L, t)
    if not piston_surfs:
        gmsh.finalize()
        raise RuntimeError(f"No piston surfaces for ka={ka}")

    gmsh.model.addPhysicalGroup(2, piston_surfs, tag=PISTON_TAG, name="piston")
    gmsh.model.addPhysicalGroup(2, baffle_surfs, tag=BAFFLE_TAG, name="baffle")

    _setup_grading(piston_surfs, baffle_surfs, h_piston, h_outer, rim_mode=rim_mode)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gmsh.write(out_path)

    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    n_tri = sum(len(tags) for tags in elem_tags)
    print(
        f"  Generated {rim_mode}: ka={ka:g}  h_piston={h_piston:.6f}  "
        f"elements={n_tri}  -> {out_path}"
    )
    gmsh.finalize()
    return out_path, h_piston, n_tri


def _errors(result, ka):
    z_bem = result["Z_rad_norm_real"] + 1j * result["Z_rad_norm_imag"]
    z_ana = zrad_norm_analytical(ka)
    err_re = 100.0 * abs(np.real(z_bem) - np.real(z_ana)) / max(abs(np.real(z_ana)), 1e-12)
    err_im = 100.0 * abs(np.imag(z_bem) - np.imag(z_ana)) / max(abs(np.imag(z_ana)), 1e-12)
    return z_bem, z_ana, err_re, err_im


def run_rim_refinement(ka=0.5, solver="gmres"):
    """Tagging check + baseline vs refined solves at fixed ka."""
    baseline_mesh = mesh_path(ka)
    if not os.path.isfile(baseline_mesh):
        print(f"Baseline mesh missing: {baseline_mesh}. Run mesh.py --ka {ka} first.")
        sys.exit(1)

    tagging = check_tagging(baseline_mesh, ka=ka)

    meshes = [("baseline", baseline_mesh)]
    for mode in ("half_hpiston", "rim_band"):
        path, h_p, n_el = generate_study_mesh(ka, rim_mode=mode)
        meshes.append((mode, path))

    print(f"\n=== Rim refinement solves (ka={ka:g}, solver={solver}) ===")
    z_ana = zrad_norm_analytical(ka)
    print(f"  Analytical Z/(ρc) = {z_ana.real:.6f} + {z_ana.imag:.6f}j")

    rows = []
    for label, path in meshes:
        print(f"\n--- {label} ---")
        result = solve_one(ka, mesh_file=path, solver=solver, rtol=RTOL)
        z_bem, _, err_re, err_im = _errors(result, ka)
        row = {
            "label": label,
            "mesh_file": os.path.basename(path),
            "h_piston_note": label,
            "n_elements": result["n_elements"],
            "n_dofs": result["n_dofs"],
            "n_piston_elements": result["n_piston_elements"],
            "solver": solver,
            "linear_solve_info": result["linear_solve_info"],
            "Z_rad_norm_real": result["Z_rad_norm_real"],
            "Z_rad_norm_imag": result["Z_rad_norm_imag"],
            "err_re_pct": err_re,
            "err_im_pct": err_im,
        }
        rows.append(row)
        print(
            f"  Err Re={err_re:.2f}%  Err Im={err_im:.2f}%  "
            f"elements={result['n_elements']}  info={result['linear_solve_info']}"
        )

    base = rows[0]
    print("\n=== Error delta vs baseline ===")
    for row in rows[1:]:
        d_re = row["err_re_pct"] - base["err_re_pct"]
        ratio_re = row["err_re_pct"] / max(base["err_re_pct"], 1e-12)
        print(
            f"  {row['label']:12s}: Re err {base['err_re_pct']:.2f}% -> "
            f"{row['err_re_pct']:.2f}%  (delta {d_re:+.2f} pp, ratio {ratio_re:.3f}x)"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"{ka:g}".replace(".", "p")
    out_json = os.path.join(RESULTS_DIR, f"rim_study_ka{tag}.json")
    payload = {"ka": ka, "tagging": tagging, "analytical": {"re": float(z_ana.real), "im": float(z_ana.imag)}, "runs": rows}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_json}")
    return rows


def run_low_ka_lu(ka_list=(0.05, 0.1)):
    """Direct LU at low ka (separate from mid-band rim study)."""
    print("\n=== Low-ka direct LU (solver non-convergence track) ===")
    rows = []
    for ka in ka_list:
        mesh_file = mesh_path(ka)
        if not os.path.isfile(mesh_file):
            generate_study_mesh(ka, rim_mode="baseline", out_path=mesh_file)
        print(f"\n--- ka={ka:g} direct LU ---")
        result = solve_one(ka, mesh_file=mesh_file, solver="direct")
        z_bem, z_ana, err_re, err_im = _errors(result, ka)
        row = {
            "ka": ka,
            "solver": "direct",
            "linear_solve_info": result["linear_solve_info"],
            "Z_rad_norm_real": result["Z_rad_norm_real"],
            "Z_rad_norm_imag": result["Z_rad_norm_imag"],
            "err_re_pct": err_re,
            "err_im_pct": err_im,
        }
        rows.append(row)
        print(
            f"  Ana={z_ana.real:.6f}+{z_ana.imag:.6f}j  "
            f"BEM={z_bem.real:.6f}+{z_bem.imag:.6f}j  "
            f"Err Re={err_re:.2f}%  Im={err_im:.2f}%"
        )

    out_json = os.path.join(RESULTS_DIR, "low_ka_direct_lu.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out_json}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Piston-rim discontinuity study.")
    parser.add_argument("--tagging-only", action="store_true", help="Only run tagging check.")
    parser.add_argument("--rim-study", action="store_true", help="Run ka=0.5 rim refinement.")
    parser.add_argument("--low-ka-lu", action="store_true", help="Direct LU at ka=0.05, 0.1.")
    parser.add_argument("--ka", type=float, default=0.5, help="ka for rim study (default 0.5).")
    parser.add_argument(
        "--solver",
        choices=("gmres", "direct"),
        default="gmres",
        help="Solver for rim study (default gmres, matches prior converged mid-band runs).",
    )
    args = parser.parse_args()

    run_all = not (args.tagging_only or args.rim_study or args.low_ka_lu)

    if args.tagging_only or run_all:
        baseline = mesh_path(args.ka)
        if not os.path.isfile(baseline):
            print(f"Need baseline mesh: {baseline}")
            sys.exit(1)
        check_tagging(baseline, ka=args.ka)

    if args.rim_study or run_all:
        run_rim_refinement(ka=args.ka, solver=args.solver)

    if args.low_ka_lu or run_all:
        run_low_ka_lu()


if __name__ == "__main__":
    main()
