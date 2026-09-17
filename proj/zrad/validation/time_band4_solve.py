"""Time a single band-4 (ka=5) solve to estimate full-sweep runtime.

Uses the off-center pilot's centered geometry (a=0.0125 m, Lx=8a, Ly=16a,
dy=0) and hybrid extraction (far-field top-only Re, near-field Im).

Reuses meshes/offcenter_pilot/mesh_ka5_a0p0125_centered.msh when present;
otherwise generates via generate_fixed_baffle_mesh().
"""

import argparse
import os
import sys
import time

from mesh import fixed_baffle_mesh_path, generate_fixed_baffle_mesh
from solve import DEVICE_CHOICES, RTOL, SOLVER_CHOICES, solve_one

# Pilot geometry (origin at piston center) — matches offcenter_pilot.py "centered"
A = 0.0125
LX = 8.0 * A
LY = 16.0 * A
THICKNESS_PILOT = A / 30.0
L_BAFFLE = max(LX, LY) / 2.0

KA_BAND4 = 5.0
POSITION_LABEL = "centered"
MESH_DIR = os.path.join(os.path.dirname(__file__), "meshes", "offcenter_pilot")

DEFAULT_FULL_SWEEP_N = 250


def _ensure_mesh(force_regenerate=False):
    """Return mesh path; generate only if no saved pilot mesh exists."""
    mesh_path = fixed_baffle_mesh_path(
        KA_BAND4, POSITION_LABEL, a=A, mesh_dir=MESH_DIR
    )

    if not force_regenerate and os.path.isfile(mesh_path):
        print(f"Reusing existing mesh: {mesh_path}")
        return mesh_path, None

    if force_regenerate and os.path.isfile(mesh_path):
        print(f"Regenerating mesh (--regenerate-mesh): {mesh_path}")
    else:
        print(f"No saved mesh at {mesh_path}; generating...")

    mesh_info = generate_fixed_baffle_mesh(
        KA_BAND4,
        a=A,
        lx=LX,
        ly=LY,
        piston_dx=0.0,
        piston_dy=0.0,
        position_label=POSITION_LABEL,
        thickness=THICKNESS_PILOT,
        mesh_dir=MESH_DIR,
    )
    return mesh_info["mesh_path"], mesh_info


def _print_timing_summary(result, dt_wall_s, full_sweep_n):
    asm = result["assembly_time_s"]
    solve = result["linear_solve_time_s"]
    solve_core = result["solve_time_s"]
    ff = result.get("far_field_extraction_time_s", 0.0)
    n_dofs = result["n_dofs"]
    n_elements = result["n_elements"]
    solver = result["solver"]
    device = result["device_interface"]
    info = result["linear_solve_info"]
    status = "OK" if info == 0 else f"FAIL({info})"

    per_solve = dt_wall_s
    projected = per_solve * full_sweep_n

    print("\n" + "=" * 72)
    print("BAND-4 TIMING SUMMARY  (ka=5, hybrid ff_top Re + near-field Im)")
    print("=" * 72)
    print(f"  mesh           : {result['mesh_file']}")
    print(f"  elements       : {n_elements}")
    print(f"  DOFs           : {n_dofs}  (P,1 pressure / DP,0 Neumann)")
    print(f"  device/solver  : {device} / {solver}  linear_solve_info={info} ({status})")
    print(f"  assembly       : {asm:.1f} s")
    print(f"  linear solve   : {solve:.1f} s")
    print(f"  asm + solve    : {solve_core:.1f} s")
    print(f"  far-field ext  : {ff:.1f} s")
    print(f"  wall-clock     : {dt_wall_s:.1f} s  (mesh load + BEM + far-field)")
    print(f"  projected {full_sweep_n}-solve sweep : {projected / 3600:.2f} h  ({projected:.0f} s)")
    print("=" * 72)

    if info != 0:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Time one band-4 (ka=5) centered-pilot solve for full-sweep estimation."
        )
    )
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="auto",
        help="Linear solver (default: auto).",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Bempp assembly backend (default: auto).",
    )
    parser.add_argument("--rtol", type=float, default=RTOL, help="GMRES tolerance.")
    parser.add_argument(
        "--regenerate-mesh",
        action="store_true",
        help="Force mesh regeneration even if pilot mesh exists.",
    )
    parser.add_argument(
        "--full-sweep-n",
        type=int,
        default=DEFAULT_FULL_SWEEP_N,
        metavar="N",
        help=f"Project wall-clock to N solves (default: {DEFAULT_FULL_SWEEP_N}).",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Band-4 timing run (highest ka)")
    print(f"  a={A} m  Lx={LX} m  Ly={LY} m  dy=0 (centered)")
    print(f"  ka={KA_BAND4:g}  hybrid: far-field top Re, near-field Im")
    print("=" * 72)

    t0 = time.perf_counter()
    mesh_file, mesh_info = _ensure_mesh(force_regenerate=args.regenerate_mesh)
    if mesh_info is not None:
        print(
            f"  Generated mesh: {mesh_info['n_elements']} elements"
            f"  (pilot ka=5 centered target ~14,400)"
        )

    result = solve_one(
        KA_BAND4,
        mesh_file=mesh_file,
        rtol=args.rtol,
        solver=args.solver,
        device=args.device,
        far_field_surfaces="top",
        a=A,
        L_baffle_m=L_BAFFLE,
    )
    dt_wall = time.perf_counter() - t0

    return _print_timing_summary(result, dt_wall, args.full_sweep_n)


if __name__ == "__main__":
    sys.exit(main())
