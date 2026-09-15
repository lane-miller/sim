"""Solve the pulsating-sphere BEM check and compute normalized radiation impedance."""

import argparse
import json
import os
import platform
import resource
import time

import bempp_cl.api as bempp
import numpy as np
from bempp_cl.api.linalg import gmres as bempp_gmres

from mesh import A, C, KA_VALUES, RHO, U0, frequency_hz, mesh_path

RTOL = 1e-5
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

bempp.DEFAULT_DEVICE_INTERFACE = "opencl"


def _mem_gb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if platform.system() == "Darwin":
        return rss / 1e9
    return rss / 1e6


def _log(msg):
    print(msg, flush=True)


def solve_one(ka, mesh_file=None, rtol=RTOL):
    """Load mesh, solve exterior Neumann Burton-Miller BEM, return Z_rad_norm."""
    if mesh_file is None:
        mesh_file = mesh_path(ka)

    if not os.path.isfile(mesh_file):
        raise FileNotFoundError(f"Mesh not found: {mesh_file}. Run mesh.py first.")

    k = ka / A
    freq = frequency_hz(ka)
    wavelength = 2.0 * np.pi / k

    _log(f"ka={ka:g}  f={freq:.1f} Hz  k={k:.4g}  λ={wavelength:.3f} m")

    t_load = time.perf_counter()
    grid = bempp.import_grid(mesh_file)
    space_p = bempp.function_space(grid, "P", 1)
    space_q = bempp.function_space(grid, "DP", 0)
    id_pp = bempp.operators.boundary.sparse.identity(space_p, space_p, space_p)
    id_qp = bempp.operators.boundary.sparse.identity(space_q, space_p, space_p)
    _log(f"  loaded mesh ({time.perf_counter() - t_load:.1f} s)")

    n_elements = grid.number_of_elements
    n_dofs = space_p.global_dof_count
    _log(
        f"  elements={n_elements}  DOFs={n_dofs} (P,1 pressure / DP,0 Neumann)"
        f"  mem={_mem_gb():.2f} GB"
    )

    alpha = 1j / k
    omega = k * C

    t_setup = time.perf_counter()

    @bempp.complex_callable
    def neumann_callable(point, normal, domain_index, result):
        # ∂p/∂n = +i ω ρ u  (exp(+iωt); matches analytical reference and bempp convention)
        result[0] = 1j * omega * RHO * U0

    neumann_fun = bempp.GridFunction(space_q, fun=neumann_callable)

    slp = bempp.operators.boundary.helmholtz.single_layer(space_q, space_p, space_p, k)
    dlp = bempp.operators.boundary.helmholtz.double_layer(space_p, space_p, space_p, k)
    adlp = bempp.operators.boundary.helmholtz.adjoint_double_layer(space_q, space_p, space_p, k)
    hyp = bempp.operators.boundary.helmholtz.hypersingular(space_p, space_p, space_p, k)

    # Exterior Neumann Burton-Miller with bempp-cl operator signs (hyp term is +αW, not −αW).
    lhs = 0.5 * id_pp - dlp + alpha * hyp
    rhs_op = -slp - alpha * (adlp + 0.5 * id_qp)
    _log(f"  operators ready ({time.perf_counter() - t_setup:.1f} s)")

    t_solve = time.perf_counter()
    p_gf, info = bempp_gmres(lhs, rhs_op * neumann_fun, tol=rtol, use_strong_form=True)
    dt_solve = time.perf_counter() - t_solve

    status = "OK" if info == 0 else f"FAIL({info})"
    _log(f"  GMRES {status} in {dt_solve:.1f} s  |p|={np.linalg.norm(p_gf.coefficients):.3e}")

    if info != 0:
        _log(f"  WARNING: linear solve did not converge (info={info})")

    t_post = time.perf_counter()
    p_centers = p_gf.evaluate_on_element_centers().ravel()
    areas = grid.volumes.astype(float)

    p_avg = np.sum(p_centers * areas) / np.sum(areas)
    # Conjugate maps bempp's time-harmonic phase to the analytical reference convention.
    z_rad_norm = np.conj(p_avg / U0 / (RHO * C))

    dt_post = time.perf_counter() - t_post
    _log(f"  post-process ({dt_post:.1f} s)")

    result = {
        "ka": ka,
        "f_hz": freq,
        "k": k,
        "a_sphere_m": A,
        "mesh_file": os.path.basename(mesh_file),
        "n_elements": int(n_elements),
        "n_dofs": int(n_dofs),
        "assembly_time_s": 0.0,
        "linear_solve_time_s": dt_solve,
        "gmres_iters": 0,
        "solve_time_s": dt_solve,
        "linear_solve_info": int(info),
        "Z_rad_norm_real": float(np.real(z_rad_norm)),
        "Z_rad_norm_imag": float(np.imag(z_rad_norm)),
    }

    _log(
        f"  Z_rad/(rho c) = {result['Z_rad_norm_real']:.6f}"
        f" + {result['Z_rad_norm_imag']:.6f}j"
        f"  (solve={dt_solve:.1f}s)"
    )

    return result


def save_result(result, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    ka = result["ka"]
    tag = f"{ka:g}".replace(".", "p")
    out_path = os.path.join(results_dir, f"result_ka{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="BEM solve for pulsating-sphere validation.")
    parser.add_argument(
        "--ka",
        type=float,
        nargs="*",
        default=None,
        help="ka values to solve (default: all five validation points)",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Path to a single mesh file (requires exactly one --ka value)",
    )
    parser.add_argument("--rtol", type=float, default=RTOL, help="GMRES relative tolerance.")
    args = parser.parse_args()

    if args.mesh is not None:
        if not args.ka or len(args.ka) != 1:
            parser.error("--mesh requires exactly one --ka value")
        result = solve_one(args.ka[0], mesh_file=args.mesh, rtol=args.rtol)
        save_result(result)
        return

    ka_list = args.ka if args.ka is not None else KA_VALUES
    t_total = time.perf_counter()
    for i, ka in enumerate(ka_list):
        _log(f"\n[{i + 1}/{len(ka_list)}] solve ka={ka:g}")
        result = solve_one(ka, rtol=args.rtol)
        save_result(result)
    _log(f"\nAll done in {time.perf_counter() - t_total:.1f} s")


if __name__ == "__main__":
    main()
