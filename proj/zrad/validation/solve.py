"""Solve the baffled-piston BEM validation case and compute normalized radiation impedance."""

import argparse
import json
import os
import platform
import resource
import time

import bempp_cl.api as bempp
import numpy as np
from bempp_cl.api.linalg import gmres as bempp_gmres
from scipy.linalg import lu_factor, lu_solve

from mesh import (
    A,
    BAFFLE_TAG,
    C,
    KD_MARGIN_DEFAULT,
    KA_VALUES,
    PISTON_TAG,
    RHO,
    THICKNESS,
    U0,
    DEFAULT_RESULT_SUFFIX,
    baffle_half_width,
    frequency_hz,
    mesh_path,
)
from analytical import zrad_norm_analytical

A_PISTON = np.pi * A ** 2
FAR_FIELD_SPHERE_POINTS = 900  # ~30×30 angular density (Fibonacci sphere)
# Default R = max(10λ, L + 2λ, 2L²/λ): Fraunhofer + clears baffle; per-kd R is OK
# when each case is in its own asymptotic zone (power is R-invariant there).
FAR_FIELD_WAVELENGTH_MULTIPLE = 10.0
FAR_FIELD_BAFFLE_MARGIN_WAVELENGTHS = 2.0
FAR_FIELD_FRAUNHOFER_FACTOR = 2.0  # 2 L²/λ, circular-aperture scale (L = half-width)
DEFAULT_OUT_SUFFIX = DEFAULT_RESULT_SUFFIX

RTOL = 1e-5

# ka at which capped alpha matches inv_k: k_cap = 1/k(ka=ALPHA_K_CAP_KA).
ALPHA_K_CAP_KA = 0.5

ALPHA_MODES = ("inv_k", "fixed", "capped")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# OpenCL single-buffer limit on macOS / many GPUs (~2 GiB).
OPENCL_MAX_DENSE_GB = 2.0
# Dense LU needs ~3× the strong-form matrix in RAM.
DIRECT_LU_MAX_DENSE_GB = 3.0

DEVICE_CHOICES = ("auto", "opencl", "numba")
SOLVER_CHOICES = ("auto", "direct", "gmres")
FAR_FIELD_SURFACE_CHOICES = ("all", "top")


def _mem_gb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if platform.system() == "Darwin":
        return rss / 1e9
    return rss / 1e6


def _log(msg):
    print(msg, flush=True)


def _estimate_dense_gb(n_dofs):
    return (n_dofs ** 2 * 16) / 1e9


def _configure_bempp(n_dofs, device="auto", solver="auto"):
    """Pick backend/solver for this mesh size; set bempp.DEFAULT_DEVICE_INTERFACE."""
    dense_gb = _estimate_dense_gb(n_dofs)
    notes = []

    if device == "auto":
        if dense_gb > OPENCL_MAX_DENSE_GB:
            device = "numba"
            notes.append(
                f"device→numba (dense {dense_gb:.2f} GB > {OPENCL_MAX_DENSE_GB} GB OpenCL buffer limit)"
            )
        else:
            device = "opencl"

    if solver == "auto":
        if dense_gb > DIRECT_LU_MAX_DENSE_GB:
            solver = "gmres"
            notes.append(
                f"solver→gmres (dense {dense_gb:.2f} GB; direct LU needs ~{3 * dense_gb:.0f} GB RAM)"
            )
        else:
            solver = "direct"

    bempp.DEFAULT_DEVICE_INTERFACE = device
    return device, solver, notes


def coupling_alpha(k, mode="inv_k"):
    """Burton-Miller coupling parameter alpha for exterior Neumann Helmholtz BEM."""
    if mode == "inv_k":
        return 1j / k
    if mode == "fixed":
        return 1j
    if mode == "capped":
        k_cap = 1.0 / (ALPHA_K_CAP_KA / A)
        return 1j * min(1.0 / k, k_cap)
    raise ValueError(f"Unknown alpha mode: {mode!r} (expected {ALPHA_MODES})")


def _collect_element_data(grid):
    """Return per-element domain index and area arrays."""
    return grid.domain_indices.astype(int), grid.volumes.astype(float)


def _top_surface_mask(grid):
    """Mask for piston + top annulus (exclude bottom face and side walls)."""
    domain_indices = grid.domain_indices.astype(int)
    centers = np.asarray(grid.centroids)
    normals = np.asarray(grid.normals)
    piston = domain_indices == PISTON_TAG
    baffle = domain_indices == BAFFLE_TAG
    top_baffle = baffle & (normals[:, 2] > 0.5) & (centers[:, 2] > 0.0)
    return piston | top_baffle


def _piston_center():
    """Center of the piston disk (top face of the baffle plate)."""
    return np.array([0.0, 0.0, THICKNESS / 2.0])


def _far_field_radius(wavelength, L_baffle, far_field_R=None):
    """Extraction-sphere radius [m]: max(10λ, L+2λ, 2L²/λ).

    The Fraunhofer term prevents extracting at a fixed 10λ when the baffle
    half-width L is many wavelengths (kd=20 at ka=0.05 needs R≈25 m, not 13 m).
    """
    if far_field_R is not None:
        if far_field_R <= L_baffle:
            raise ValueError(
                f"far_field_R={far_field_R:.3f} m must exceed baffle half-width "
                f"L_baffle={L_baffle:.3f} m"
            )
        return float(far_field_R)
    r_wavelength = FAR_FIELD_WAVELENGTH_MULTIPLE * wavelength
    r_clear = L_baffle + FAR_FIELD_BAFFLE_MARGIN_WAVELENGTHS * wavelength
    r_fraunhofer = FAR_FIELD_FRAUNHOFER_FACTOR * L_baffle ** 2 / wavelength
    return max(r_wavelength, r_clear, r_fraunhofer)


def _fibonacci_sphere(n_points, center, radius):
    """Approximately uniform points on a sphere (3 × N array)."""
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n_points, dtype=float)
    theta = 2.0 * np.pi * i / golden_ratio
    cos_phi = 1.0 - 2.0 * (i + 0.5) / n_points
    sin_phi = np.sqrt(np.maximum(0.0, 1.0 - cos_phi ** 2))
    x = radius * sin_phi * np.cos(theta)
    y = radius * sin_phi * np.sin(theta)
    z = radius * cos_phi
    return np.vstack([x, y, z]) + center[:, np.newaxis]


def _far_field_radiated_power(
    space_p,
    space_q,
    p_gf,
    neumann_fun,
    k,
    wavelength,
    L_baffle,
    n_points=FAR_FIELD_SPHERE_POINTS,
    far_field_R=None,
    far_field_surfaces="all",
    grid=None,
    a=None,
):
    """Integrate |p|^2/(2 rho c) over a full extraction sphere → W and front-hem ratio."""
    center = _piston_center()
    radius = _far_field_radius(wavelength, L_baffle, far_field_R=far_field_R)
    eval_points = _fibonacci_sphere(n_points, center, radius)

    slp_pot = bempp.operators.potential.helmholtz.single_layer(space_q, eval_points, k)
    if far_field_surfaces == "all":
        dlp_pot = bempp.operators.potential.helmholtz.double_layer(space_p, eval_points, k)
        dlp_fun = p_gf
    elif far_field_surfaces == "top":
        if grid is None:
            raise ValueError("grid is required when far_field_surfaces='top'")
        top_mask = _top_surface_mask(grid)
        p_centers = p_gf.evaluate_on_element_centers().ravel()
        p_top_coeffs = np.where(top_mask, p_centers, 0.0)
        dlp_fun = bempp.GridFunction(space_q, coefficients=p_top_coeffs)
        dlp_pot = bempp.operators.potential.helmholtz.double_layer(space_q, eval_points, k)
    else:
        raise ValueError(
            f"Unknown far_field_surfaces: {far_field_surfaces!r} "
            f"(expected {FAR_FIELD_SURFACE_CHOICES})"
        )
    # Exterior Kirchhoff–Helmholtz: p = S(∂p/∂n) − D(p).
    p_far = (slp_pot @ neumann_fun - dlp_pot @ dlp_fun).ravel()

    intensity = np.abs(p_far) ** 2 / (2.0 * RHO * C)
    d_omega = 4.0 * np.pi / n_points
    power_density = intensity * radius ** 2
    power_total = float(np.sum(power_density * d_omega))

    front_mask = eval_points[2, :] >= center[2]
    power_front = float(np.sum(power_density[front_mask] * d_omega))
    front_ratio = power_front / power_total if power_total > 0.0 else float("nan")

    a_piston = A_PISTON if a is None else np.pi * a ** 2
    z_rad_norm_real_ff = 2.0 * power_total / (U0 ** 2 * a_piston * RHO * C)

    return {
        "far_field_R_m": float(radius),
        "far_field_n_points": int(n_points),
        "radiated_power_W": power_total,
        "front_hemisphere_power_W": power_front,
        "front_hemisphere_power_ratio": front_ratio,
        "Z_rad_norm_real_ff": float(z_rad_norm_real_ff),
    }


def _rhs_in_pressure_space(rhs_op, neumann_fun, space_p):
    """Apply Neumann data and return a GridFunction in the pressure (P,1) space."""
    rhs_raw = rhs_op * neumann_fun
    return bempp.GridFunction(space_p, coefficients=rhs_raw.coefficients)


def _solve_neumann_bm(lhs, rhs_gf, space_p, rtol, solver):
    """Solve the strong-form exterior Neumann Burton-Miller system for pressure."""
    if solver == "direct":
        _log("  assembling strong-form system for direct LU...")
        t_asm = time.perf_counter()
        A_discrete = lhs.strong_form()
        b_vec = rhs_gf.coefficients
        dt_asm = time.perf_counter() - t_asm
        _log(f"  assembly done ({dt_asm:.1f} s)  mem={_mem_gb():.2f} GB")

        _log("  direct LU (dense)...")
        t_solve = time.perf_counter()
        lu, piv = lu_factor(A_discrete.to_dense())
        p_coeffs = lu_solve((lu, piv), b_vec)
        dt_solve = time.perf_counter() - t_solve
        p_gf = bempp.GridFunction(space_p, coefficients=p_coeffs)
        return p_gf, 0, 0, dt_asm, dt_solve

    if solver == "gmres":
        t_solve = time.perf_counter()
        p_gf, info = bempp_gmres(
            lhs, rhs_gf, tol=rtol, use_strong_form=True, maxiter=20000
        )
        dt_solve = time.perf_counter() - t_solve
        return p_gf, info, 0, 0.0, dt_solve

    raise ValueError(f"Unknown solver: {solver!r} (expected 'direct' or 'gmres')")


def solve_one(
    ka,
    mesh_file=None,
    rtol=RTOL,
    solver="auto",
    alpha_mode="inv_k",
    kd=None,
    device="auto",
    far_field_R=None,
    far_field_surfaces="all",
    a=None,
    L_baffle_m=None,
):
    """Load mesh, solve exterior Neumann Burton-Miller BEM, return Z_rad_norm."""
    if mesh_file is None:
        mesh_file = mesh_path(ka, kd=kd)

    if not os.path.isfile(mesh_file):
        raise FileNotFoundError(f"Mesh not found: {mesh_file}. Run mesh.py first.")

    a_eff = A if a is None else a
    k = ka / a_eff
    freq = k * C / (2.0 * np.pi)
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
    dense_gb = _estimate_dense_gb(n_dofs)
    device, solver, auto_notes = _configure_bempp(n_dofs, device=device, solver=solver)
    _log(
        f"  elements={n_elements}  DOFs={n_dofs} (P,1 pressure / DP,0 Neumann)"
        f"  dense-est={dense_gb:.2f} GB  device={device}  solver={solver}"
        f"  mem={_mem_gb():.2f} GB"
    )
    for note in auto_notes:
        _log(f"  {note}")

    alpha = coupling_alpha(k, alpha_mode)
    omega = k * C
    _log(f"  alpha-mode={alpha_mode}  alpha={alpha:.6g}")

    t_setup = time.perf_counter()

    @bempp.complex_callable
    def neumann_callable(point, normal, domain_index, result):
        # Prescribed normal velocity u = U0 on piston, 0 on baffle.
        # ∂p/∂n = +i ω ρ u  (exp(+iωt); same convention as sphere_check/)
        if domain_index == PISTON_TAG:
            result[0] = 1j * omega * RHO * U0
        else:
            result[0] = 0.0

    neumann_fun = bempp.GridFunction(space_q, fun=neumann_callable)

    slp = bempp.operators.boundary.helmholtz.single_layer(space_q, space_p, space_p, k)
    dlp = bempp.operators.boundary.helmholtz.double_layer(space_p, space_p, space_p, k)
    adlp = bempp.operators.boundary.helmholtz.adjoint_double_layer(space_q, space_p, space_p, k)
    hyp = bempp.operators.boundary.helmholtz.hypersingular(space_p, space_p, space_p, k)

    lhs = 0.5 * id_pp - dlp + alpha * hyp
    rhs_op = -slp - alpha * (adlp + 0.5 * id_qp)
    rhs_gf = _rhs_in_pressure_space(rhs_op, neumann_fun, space_p)
    _log(f"  operators ready ({time.perf_counter() - t_setup:.1f} s)")

    if solver == "gmres":
        _log(
            f"  solving (strong-form GMRES via {device};"
            " first numba run may JIT-compile kernels)..."
        )
    else:
        _log(f"  solving (strong-form direct LU via {device})...")
    p_gf, info, gmres_iters, dt_asm, dt_solve = _solve_neumann_bm(
        lhs, rhs_gf, space_p, rtol, solver
    )
    solve_time = dt_asm + dt_solve

    status = "OK" if info == 0 else f"FAIL({info})"
    if solver == "gmres":
        _log(
            f"  GMRES {status} in {dt_solve:.1f} s"
            f"  |p|={np.linalg.norm(p_gf.coefficients):.3e}"
        )
    else:
        _log(
            f"  LU {status} in {dt_solve:.1f} s"
            f"  |p|={np.linalg.norm(p_gf.coefficients):.3e}"
        )

    if info != 0:
        _log(f"  WARNING: linear solve did not converge (info={info})")

    t_post = time.perf_counter()
    domain_indices, areas = _collect_element_data(grid)
    piston_mask = domain_indices == PISTON_TAG

    if not np.any(piston_mask):
        raise RuntimeError("No piston elements found in mesh domain indices.")

    p_centers = p_gf.evaluate_on_element_centers().ravel()
    area_piston = areas[piston_mask]

    p_avg = np.sum(p_centers[piston_mask] * area_piston) / np.sum(area_piston)
    z_rad = p_avg / U0
    z_rad_norm = np.conj(z_rad / (RHO * C))

    kd_label = KD_MARGIN_DEFAULT if kd is None else kd
    L_baffle = baffle_half_width(ka, kd=kd) if L_baffle_m is None else L_baffle_m
    ff_radius = _far_field_radius(wavelength, L_baffle, far_field_R=far_field_R)
    _log(
        f"  far-field power extraction (kd={kd_label:g},"
        f" surfaces={far_field_surfaces},"
        f" R={ff_radius:.3f} m,"
        f" n={FAR_FIELD_SPHERE_POINTS})..."
    )
    t_ff = time.perf_counter()
    ff = _far_field_radiated_power(
        space_p,
        space_q,
        p_gf,
        neumann_fun,
        k,
        wavelength,
        L_baffle,
        far_field_R=far_field_R,
        far_field_surfaces=far_field_surfaces,
        grid=grid,
        a=a_eff,
    )
    dt_ff = time.perf_counter() - t_ff
    _log(f"  far-field done ({dt_ff:.1f} s)")

    dt_post = time.perf_counter() - t_post
    _log(f"  post-process ({dt_post:.1f} s)")

    z_analytical = zrad_norm_analytical(ka)
    a_piston = np.pi * a_eff ** 2
    area_piston_bem = float(np.sum(area_piston))
    area_piston_analytic = float(a_piston)

    result = {
        "ka": ka,
        "kd": kd_label,
        "a_m": a_eff,
        "f_hz": freq,
        "k": k,
        "L_baffle_m": L_baffle,
        "mesh_file": os.path.basename(mesh_file),
        "n_elements": int(n_elements),
        "n_dofs": int(n_dofs),
        "n_piston_elements": int(np.sum(piston_mask)),
        "piston_area_m2": area_piston_bem,
        "piston_area_analytic_m2": area_piston_analytic,
        "piston_area_ratio": area_piston_bem / area_piston_analytic,
        "device_interface": device,
        "solver": solver,
        "alpha_mode": alpha_mode,
        "alpha_real": float(np.real(alpha)),
        "alpha_imag": float(np.imag(alpha)),
        "assembly_time_s": dt_asm,
        "linear_solve_time_s": dt_solve,
        "gmres_iters": int(gmres_iters),
        "solve_time_s": solve_time,
        "linear_solve_info": int(info),
        "Z_rad_norm_real": float(np.real(z_rad_norm)),
        "Z_rad_norm_imag": float(np.imag(z_rad_norm)),
        "Z_rad_norm_real_analytical": float(np.real(z_analytical)),
        "Z_rad_norm_imag_analytical": float(np.imag(z_analytical)),
        **ff,
        "far_field_surfaces": far_field_surfaces,
        "far_field_extraction_time_s": dt_ff,
    }

    _log(
        f"  Z_rad/(rho c) near-field = {result['Z_rad_norm_real']:.6f}"
        f" + {result['Z_rad_norm_imag']:.6f}j"
    )
    _log(
        f"  Z_rad/(rho c) far-field Re = {result['Z_rad_norm_real_ff']:.6f}"
        f"  analytical Re = {result['Z_rad_norm_real_analytical']:.6f}"
        f"  front/total power = {result['front_hemisphere_power_ratio']:.4f}"
        f"  (asm={dt_asm:.1f}s + solve={dt_solve:.1f}s + ff={dt_ff:.1f}s)"
    )

    return result


def save_result(result, results_dir=RESULTS_DIR, out_suffix=DEFAULT_OUT_SUFFIX):
    os.makedirs(results_dir, exist_ok=True)
    ka = result["ka"]
    tag = f"{ka:g}".replace(".", "p")
    alpha_mode = result.get("alpha_mode", "inv_k")
    if out_suffix is None:
        out_suffix = ""
    elif out_suffix == DEFAULT_OUT_SUFFIX and alpha_mode != "inv_k":
        out_suffix = f"_alpha_{alpha_mode}{DEFAULT_OUT_SUFFIX}"
    out_path = os.path.join(results_dir, f"result_ka{tag}{out_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="BEM solve for baffled-piston validation.")
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
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="auto",
        help=(
            "Linear solver: auto (default), dense direct LU, or strong-form GMRES. "
            f"Auto picks gmres when dense-est > {DIRECT_LU_MAX_DENSE_GB} GB."
        ),
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help=(
            "Bempp assembly backend: auto (default), opencl, or numba. "
            f"Auto picks numba when dense-est > {OPENCL_MAX_DENSE_GB} GB."
        ),
    )
    parser.add_argument(
        "--alpha-mode",
        choices=ALPHA_MODES,
        default="inv_k",
        help=(
            "Burton-Miller coupling: inv_k=1j/k (default), fixed=1j, "
            f"capped=1j*min(1/k, 1/k(ka={ALPHA_K_CAP_KA}))"
        ),
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default=DEFAULT_OUT_SUFFIX,
        help=(
            "Suffix before .json (default: _ff_pistref — avoids overwriting "
            "near-field-only results). Use '' to write result_ka*.json."
        ),
    )
    parser.add_argument(
        "--kd-override",
        type=float,
        default=None,
        metavar="KD",
        help=(
            f"Rim margin kd for L_baffle (default {KD_MARGIN_DEFAULT:g}). "
            "Must match the mesh when using --mesh."
        ),
    )
    parser.add_argument(
        "--far-field-R",
        type=float,
        default=None,
        metavar="R",
        help=(
            "Override far-field extraction-sphere radius [m]. "
            f"Default: max({FAR_FIELD_WAVELENGTH_MULTIPLE:g}λ, "
            f"L + {FAR_FIELD_BAFFLE_MARGIN_WAVELENGTHS:g}λ, "
            f"{FAR_FIELD_FRAUNHOFER_FACTOR:g}L²/λ)."
        ),
    )
    parser.add_argument(
        "--far-field-surfaces",
        choices=FAR_FIELD_SURFACE_CHOICES,
        default="all",
        help=(
            "Surfaces included in the far-field double-layer D(p) term: "
            "all (default, full closed surface) or top (piston + top annulus only; "
            "single-layer S(∂p/∂n) unchanged)."
        ),
    )
    args = parser.parse_args()

    if args.mesh is not None:
        if not args.ka or len(args.ka) != 1:
            parser.error("--mesh requires exactly one --ka value")
        result = solve_one(
            args.ka[0],
            mesh_file=args.mesh,
            rtol=args.rtol,
            solver=args.solver,
            alpha_mode=args.alpha_mode,
            kd=args.kd_override,
            device=args.device,
            far_field_R=args.far_field_R,
            far_field_surfaces=args.far_field_surfaces,
        )
        save_result(result, out_suffix=args.out_suffix)
        return

    ka_list = args.ka if args.ka is not None else KA_VALUES
    t_total = time.perf_counter()
    for i, ka in enumerate(ka_list):
        _log(f"\n[{i + 1}/{len(ka_list)}] solve ka={ka:g}")
        result = solve_one(
            ka,
            rtol=args.rtol,
            solver=args.solver,
            alpha_mode=args.alpha_mode,
            kd=args.kd_override,
            device=args.device,
            far_field_R=args.far_field_R,
            far_field_surfaces=args.far_field_surfaces,
        )
        save_result(result, out_suffix=args.out_suffix)
    _log(f"\nAll done in {time.perf_counter() - t_total:.1f} s")


if __name__ == "__main__":
    main()
