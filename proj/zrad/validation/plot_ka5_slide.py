"""Presentation-ready mesh and |p| heat-map figures for ka=5 centered off-center sweep case."""

import argparse
import json
import os
import time

import bempp_cl.api as bempp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib import cm

from mesh import BAFFLE_TAG, C, PISTON_TAG, RHO, U0
from solve import (
    RTOL,
    _configure_bempp,
    _rhs_in_pressure_space,
    _solve_neumann_bm,
    _top_surface_mask,
    coupling_alpha,
)

# Centered (dy=0) ka=5 case from offcenter_sweep — highest ka in the sweep.
KA = 5.0
A_M = 0.0125
L_BAFFLE_M = 0.1
MESH_FILE = os.path.join(
    os.path.dirname(__file__),
    "meshes",
    "offcenter_sweep",
    "mesh_ka5_a0p0125_dy0.msh",
)
RESULT_JSON = os.path.join(
    os.path.dirname(__file__),
    "results",
    "offcenter_sweep",
    "result_dy0_ka5.json",
)
PRESSURE_CACHE = os.path.join(
    os.path.dirname(__file__),
    "results",
    "offcenter_sweep",
    "pressure_coeffs_dy0_ka5.npz",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")

DPI = 220
EDGE_COLOR = "#666666"
FACE_COLOR = "#fafafa"
EDGE_LW = 0.18
CMAP = "magma"
P_REF = RHO * C * U0  # reference for normalized |p|


def _log(msg):
    print(msg, flush=True)


def _top_surface_triangles(grid):
    """Return (n_tri, 3, 2) xy vertex arrays and element indices for the radiating top."""
    verts = np.asarray(grid.vertices)
    elems = np.asarray(grid.elements)
    mask = _top_surface_mask(grid)
    elem_idx = np.where(mask)[0]
    triangles = verts[:2, elems[:, elem_idx]].transpose(2, 0, 1)
    return triangles, elem_idx


def _axis_limits(triangles, pad_frac=0.01):
    xy = triangles.reshape(-1, 2)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = dx * pad_frac
    pad_y = dy * pad_frac
    return (xmin - pad_x, xmax + pad_x), (ymin - pad_y, ymax + pad_y)


def _style_axes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def solve_pressure_field(mesh_file, ka, a_m, solver="auto", rtol=RTOL):
    """Run the same BEM solve as full_sweep.py; return grid and pressure GridFunction."""
    k = ka / a_m
    _log(f"Solving ka={ka:g}  k={k:.4g}  mesh={os.path.basename(mesh_file)}")

    grid = bempp.import_grid(mesh_file)
    space_p = bempp.function_space(grid, "P", 1)
    space_q = bempp.function_space(grid, "DP", 0)
    id_pp = bempp.operators.boundary.sparse.identity(space_p, space_p, space_p)
    id_qp = bempp.operators.boundary.sparse.identity(space_q, space_p, space_p)

    n_dofs = space_p.global_dof_count
    device, solver, notes = _configure_bempp(n_dofs, device="auto", solver=solver)
    for note in notes:
        _log(f"  {note}")
    _log(f"  DOFs={n_dofs}  device={device}  solver={solver}")

    bempp.DEFAULT_DEVICE_INTERFACE = device
    alpha = coupling_alpha(k, "inv_k")
    omega = k * C

    @bempp.complex_callable
    def neumann_callable(point, normal, domain_index, result):
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

    t0 = time.perf_counter()
    p_gf, info, _, dt_asm, dt_solve = _solve_neumann_bm(lhs, rhs_gf, space_p, rtol, solver)
    _log(
        f"  solve info={info}  wall={time.perf_counter() - t0:.1f} s"
        f"  (asm={dt_asm:.1f}s + lu/gmres={dt_solve:.1f}s)"
    )
    if info != 0:
        raise RuntimeError(f"Linear solve failed (info={info})")
    return grid, p_gf


def load_or_solve_pressure(mesh_file, cache_path, solver="auto", force_solve=False):
    if not force_solve and os.path.isfile(cache_path):
        _log(f"Loading cached pressure: {cache_path}")
        data = np.load(cache_path)
        grid = bempp.import_grid(mesh_file)
        p_gf = bempp.GridFunction(
            bempp.function_space(grid, "P", 1),
            coefficients=data["coefficients"],
        )
        return grid, p_gf

    grid, p_gf = solve_pressure_field(mesh_file, KA, A_M, solver=solver)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, coefficients=p_gf.coefficients)
    _log(f"Cached pressure coefficients: {cache_path}")
    return grid, p_gf


def plot_mesh_grayscale(triangles, xlim, ylim, out_path):
    fig, ax = plt.subplots(figsize=(4.0, 8.0), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pc = PolyCollection(
        triangles,
        facecolors=FACE_COLOR,
        edgecolors=EDGE_COLOR,
        linewidths=EDGE_LW,
        antialiaseds=True,
    )
    ax.add_collection(pc)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _style_axes(ax)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    _log(f"Saved: {out_path}")


def plot_pressure_heatmap(triangles, p_mag_norm, xlim, ylim, out_path):
    fig, ax = plt.subplots(figsize=(4.6, 8.0), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    norm = Normalize(vmin=0.0, vmax=float(np.max(p_mag_norm)))
    cmap = cm.get_cmap(CMAP)
    facecolors = cmap(norm(p_mag_norm))

    pc = PolyCollection(
        triangles,
        facecolors=facecolors,
        edgecolors="none",
        antialiaseds=True,
    )
    ax.add_collection(pc)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _style_axes(ax)

    # Compact colorbar to the right of the mesh.
    cax = fig.add_axes([0.88, 0.22, 0.035, 0.56])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=8, length=2, width=0.6, pad=1)
    cb.set_label(r"$|p|/(\rho c U_0)$", fontsize=9, labelpad=4)

    fig.subplots_adjust(left=0, right=0.84, bottom=0, top=1)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    _log(f"Saved: {out_path}")


def verify_against_result(p_gf, grid, result_path):
    """Spot-check near-field Im(Z) against the saved sweep JSON."""
    with open(result_path, encoding="utf-8") as f:
        ref = json.load(f)
    domain_indices = grid.domain_indices.astype(int)
    areas = grid.volumes.astype(float)
    piston_mask = domain_indices == PISTON_TAG
    p_centers = p_gf.evaluate_on_element_centers().ravel()
    p_avg = np.sum(p_centers[piston_mask] * areas[piston_mask]) / np.sum(areas[piston_mask])
    z_rad_norm = np.conj(p_avg / U0 / (RHO * C))
    re_err = abs(np.real(z_rad_norm) - ref["Z_rad_norm_real"])
    im_err = abs(np.imag(z_rad_norm) - ref["Z_rad_norm_imag"])
    _log(
        f"  Verify near-field Z/(ρc): {np.real(z_rad_norm):.6f}+{np.imag(z_rad_norm):.6f}j"
        f"  (ref {ref['Z_rad_norm_real']:.6f}+{ref['Z_rad_norm_imag']:.6f}j"
        f"  ΔRe={re_err:.2e} ΔIm={im_err:.2e})"
    )


def main():
    parser = argparse.ArgumentParser(description="Render ka=5 slide mesh and pressure figures.")
    parser.add_argument("--solver", default="auto", help="BEM linear solver (default: auto).")
    parser.add_argument("--force-solve", action="store_true", help="Re-run BEM even if cache exists.")
    args = parser.parse_args()

    if not os.path.isfile(MESH_FILE):
        raise FileNotFoundError(MESH_FILE)
    if not os.path.isfile(RESULT_JSON):
        raise FileNotFoundError(RESULT_JSON)

    os.makedirs(OUT_DIR, exist_ok=True)
    mesh_out = os.path.join(OUT_DIR, "mesh_grayscale_ka5.png")
    pressure_out = os.path.join(OUT_DIR, "pressure_heatmap_ka5.png")

    grid, p_gf = load_or_solve_pressure(
        MESH_FILE, PRESSURE_CACHE, solver=args.solver, force_solve=args.force_solve
    )
    verify_against_result(p_gf, grid, RESULT_JSON)

    triangles, elem_idx = _top_surface_triangles(grid)
    xlim, ylim = _axis_limits(triangles)

    p_centers = p_gf.evaluate_on_element_centers().ravel()
    p_mag_norm = np.abs(p_centers[elem_idx]) / P_REF

    plot_mesh_grayscale(triangles, xlim, ylim, mesh_out)
    plot_pressure_heatmap(triangles, p_mag_norm, xlim, ylim, pressure_out)


if __name__ == "__main__":
    main()
