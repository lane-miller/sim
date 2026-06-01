"""
solve.py — FEM Helmholtz solver for the expansion chamber POC.

PETSc in simenv is compiled with *real* scalars, so the complex Helmholtz
problem is split into coupled real / imaginary equations on a mixed space.

Correct Sommerfeld ABC: ∂p/∂n = jkp at every absorbing boundary (outward n).
Both inlet and outlet absorb their respective outgoing waves:
  - outlet (outward n = +z): absorbs forward-traveling transmitted wave
  - inlet  (outward n = −z): absorbs backward-traveling reflected wave,
                              sources incident wave via RHS

This gives the weak form:

    ∫∇p·∇v dx − k²∫pv dx − jk∫pv ds_in − jk∫pv ds_out = −2jk·p_inc·∫v ds_in

Expanding into the 2×2 real block system (v_r tests eq. 1, v_i tests eq. 2):

  Row 1:  ∫∇p_r·∇v_r dx − k²∫p_r·v_r dx + k∫p_i·v_r ds_in + k∫p_i·v_r ds_out = 0
  Row 2:  ∫∇p_i·∇v_i dx − k²∫p_i·v_i dx − k∫p_r·v_i ds_in − k∫p_r·v_i ds_out
            = −2k·p_inc·∫v_i ds_in

Key: both outlet terms carry the SAME sign as the corresponding inlet terms.
The previous sign error (+ at outlet vs − at inlet) reflected transmitted energy
back into the domain, injecting artificial power and causing negative TL.

Results saved to outputs/fem_results.npz for comparison by validate.py.
"""

import os
import sys

import numpy as np
from mpi4py import MPI

import basix.ufl
import dolfinx
from dolfinx import fem, geometry
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import inner, grad, split

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from mode_matching import PLOT_FREQS

MESH_FILE  = os.path.join(os.path.dirname(__file__), "outputs", "chamber.msh")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

INLET_TAG  = 1
OUTLET_TAG = 2


# ── Mesh loading ──────────────────────────────────────────────────────────────

def load_mesh():
    """Read Gmsh .msh, return (mesh, cell_tags, facet_tags)."""
    data = gmshio.read_from_msh(MESH_FILE, MPI.COMM_WORLD, 0, gdim=3)
    mesh = data.mesh
    cell_tags  = data.cell_tags
    facet_tags = data.facet_tags
    # Ensure facet→cell connectivity for ds subdomain integration
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    return mesh, cell_tags, facet_tags


# ── Build mixed function space ────────────────────────────────────────────────

def _build_space(mesh):
    """P2 Lagrange mixed space for (p_r, p_i)."""
    el  = basix.ufl.element("Lagrange", mesh.basix_cell(), 2)
    mel = basix.ufl.mixed_element([el, el])
    return fem.functionspace(mesh, mel)


# ── Single-frequency Helmholtz solve ─────────────────────────────────────────

def solve_helmholtz(mesh, facet_tags, f, p_inc=1.0):
    """
    Solve at frequency f [Hz].  Returns (p_r_fn, p_i_fn) — two real-valued
    dolfinx Functions representing the real and imaginary parts of p.
    """
    W = _build_space(mesh)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=mesh)

    u = ufl.TrialFunction(W)
    w = ufl.TestFunction(W)

    p_r, p_i = split(u)
    v_r, v_i = split(w)

    k = 2.0 * np.pi * f / cfg.C

    a = (
        inner(grad(p_r), grad(v_r)) * dx
        - k**2 * inner(p_r, v_r) * dx
        + k * inner(p_i, v_r) * ds(INLET_TAG)
        + k * inner(p_i, v_r) * ds(OUTLET_TAG)   # same sign as inlet: −jk → +k on p_i
        + inner(grad(p_i), grad(v_i)) * dx
        - k**2 * inner(p_i, v_i) * dx
        - k * inner(p_r, v_i) * ds(INLET_TAG)
        - k * inner(p_r, v_i) * ds(OUTLET_TAG)   # same sign as inlet: −jk → −k on p_r
    )

    L = -2.0 * k * p_inc * v_i * ds(INLET_TAG)

    problem = LinearProblem(
        a, L,
        petsc_options_prefix="helm_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    u_h = problem.solve()

    # Collapse sub-functions to standalone Function objects
    p_r_fn = u_h.sub(0).collapse()
    p_i_fn = u_h.sub(1).collapse()
    return p_r_fn, p_i_fn


# ── TL sweep ─────────────────────────────────────────────────────────────────

def compute_tl_fem(mesh, facet_tags, freqs, p_inc=1.0):
    """
    Solve at each frequency and return TL [dB] array.

    TL = 10·log10( |p_inc|²·S_quarter / ∫_outlet (p_r²+p_i²) ds )

    The 1/(2ρc) normalisation and quarter-to-full factor of 4 cancel in
    the ratio.
    """
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    S_quarter = (cfg.INLET_W / 2.0) * (cfg.INLET_H / 2.0)
    TL = np.zeros(len(freqs))

    for i, f in enumerate(freqs):
        p_r_fn, p_i_fn = solve_helmholtz(mesh, facet_tags, f, p_inc=p_inc)

        # ∫ |p|² ds_out  (real arithmetic on collapsed functions)
        p_sq_form = fem.form(
            (p_r_fn * p_r_fn + p_i_fn * p_i_fn) * ds(OUTLET_TAG)
        )
        p_outlet_sq = fem.assemble_scalar(p_sq_form)

        if p_outlet_sq <= 0.0:
            TL[i] = np.inf
        else:
            TL[i] = 10.0 * np.log10(p_inc**2 * S_quarter / p_outlet_sq)

        if (i + 1) % 5 == 0 or i == 0 or i == len(freqs) - 1:
            print(f"  f = {f:7.1f} Hz   TL = {TL[i]:.2f} dB")

    return TL


# ── On-axis pressure extraction ───────────────────────────────────────────────

def extract_onaxis(mesh, p_r_fn, p_i_fn, nz=400):
    """
    Evaluate p = p_r + j·p_i at (ε, ε, z) along the symmetry-axis edge.

    A small offset ε=1e-10 avoids ambiguity at the x=y=0 boundary without
    affecting the acoustic result.

    Returns (z_mm, p_complex) — 1-D arrays of length ≤ nz.
    """
    eps    = 1e-10
    z_arr  = np.linspace(0.0, cfg.TOTAL_L, nz)
    points = np.column_stack([
        np.full(nz, eps),
        np.full(nz, eps),
        z_arr,
    ])

    bb = geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb, points)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)

    cells_found  = []
    pts_found    = []
    z_found      = []

    for i, pt in enumerate(points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            cells_found.append(links[0])
            pts_found.append(pt)
            z_found.append(z_arr[i])

    if not pts_found:
        return np.array([]), np.array([], dtype=complex)

    pts_np  = np.array(pts_found)
    pr_vals = p_r_fn.eval(pts_np, cells_found).ravel()
    pi_vals = p_i_fn.eval(pts_np, cells_found).ravel()

    z_mm = np.array(z_found) * 1e3
    return z_mm, pr_vals + 1j * pi_vals


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load mesh
    print("Loading mesh...")
    mesh, cell_tags, facet_tags = load_mesh()
    n_nodes = mesh.topology.index_map(0).size_local
    print(f"Mesh: {mesh.topology.dim}D, {n_nodes} nodes")

    # 2. Single-frequency sanity check at 500 Hz with energy balance
    print("\nSingle-freq test at 500 Hz...")
    p_r_test, p_i_test = solve_helmholtz(mesh, facet_tags, 500.0)
    max_pr = np.max(np.abs(p_r_test.x.array))
    max_pi = np.max(np.abs(p_i_test.x.array))
    max_p  = np.sqrt(max_pr**2 + max_pi**2)
    print(f"  max|p_r| = {max_pr:.4f}  max|p_i| = {max_pi:.4f}  max|p| ≈ {max_p:.4f} Pa  (expect ~1–2)")

    ds_check = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    S_quarter = (cfg.INLET_W / 2.0) * (cfg.INLET_H / 2.0)
    p_sq_in  = fem.assemble_scalar(fem.form(
        (p_r_test * p_r_test + p_i_test * p_i_test) * ds_check(INLET_TAG)))
    p_sq_out = fem.assemble_scalar(fem.form(
        (p_r_test * p_r_test + p_i_test * p_i_test) * ds_check(OUTLET_TAG)))
    print(f"  Energy balance at 500 Hz:")
    print(f"    ∫|p|² ds_inlet  = {p_sq_in:.6f}  (incident ref = {1.0**2 * S_quarter:.6f})")
    print(f"    ∫|p|² ds_outlet = {p_sq_out:.6f}  (must be < inlet integral for TL > 0)")
    tl_500 = 10.0 * np.log10(1.0**2 * S_quarter / p_sq_out) if p_sq_out > 0 else np.inf
    print(f"    TL at 500 Hz    = {tl_500:.2f} dB  (must be > 0)")

    # 3. Full TL sweep
    print(f"\nTL sweep over {len(cfg.FREQS)} frequencies...")
    TL_fem = compute_tl_fem(mesh, facet_tags, cfg.FREQS)

    # Comparison with mode-matching reference
    from mode_matching import compute_tl as mm_compute_tl, compute_overlap
    C_ov = compute_overlap(cfg.INLET_W, cfg.INLET_H,
                           cfg.CHAMBER_W, cfg.CHAMBER_H, 8, 8)
    TL_mm, _ = mm_compute_tl(cfg.FREQS, C_ov)
    print("\n{:>10s}  {:>8s}  {:>8s}  {:>8s}".format("Freq [Hz]", "TL_FEM", "TL_MM", "Diff"))
    print("-" * 44)
    step = max(1, len(cfg.FREQS) // 10)
    any_negative = False
    for i in range(0, len(cfg.FREQS), step):
        f, tl, ref = cfg.FREQS[i], TL_fem[i], TL_mm[i]
        flag = " *** NEGATIVE ***" if tl < 0 else ""
        if tl < 0:
            any_negative = True
        print(f"  {f:8.1f}    {tl:6.2f}    {ref:6.2f}    {tl-ref:+6.2f}{flag}")
    if any_negative:
        print("\nWARNING: negative TL values remain — fix is incomplete!")
    else:
        print("\nAll TL values non-negative. Fix appears successful.")

    # 4. On-axis pressure at PLOT_FREQS
    print(f"\nOn-axis pressure at {PLOT_FREQS} Hz...")
    onaxis_fem = {}
    for fq in PLOT_FREQS:
        p_r_fn, p_i_fn = solve_helmholtz(mesh, facet_tags, fq)
        z_mm, p_vals   = extract_onaxis(mesh, p_r_fn, p_i_fn)
        onaxis_fem[fq] = (z_mm, p_vals)
        print(f"  f = {fq:.0f} Hz   npts = {len(z_mm)}   max|p| = {np.max(np.abs(p_vals)):.4f} Pa")

    # 5. Save all results for validate.py
    save_dict = dict(freqs=cfg.FREQS, TL_fem=TL_fem)
    for fq in PLOT_FREQS:
        z_mm, p_vals = onaxis_fem[fq]
        save_dict[f"onaxis_z_{int(fq)}"] = z_mm
        save_dict[f"onaxis_p_{int(fq)}"] = p_vals

    np.savez(os.path.join(OUTPUT_DIR, "fem_results.npz"), **save_dict)
    print("\nFEM results saved to outputs/fem_results.npz")
    print("Done — solve.py complete.")
