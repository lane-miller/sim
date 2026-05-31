"""
solve.py — Baffled Piston POC
Axisymmetric acoustic Helmholtz equation with radial coordinate-stretch PML.

Weak form:
  The standard 3D Helmholtz weak form is reduced to the (r,z) half-plane by
  the axisymmetric assumption (no θ dependence, m=0 mode). The 2πr Jacobian
  of the cylindrical coordinate transform is applied, giving:

    ∫_Ω (∇p·∇q - k²pq) r dΩ = -iωρ v₀ ∫_Γ_piston q r dΓ

  where all integrals are over the 2D (r,z) domain. The constant 2π cancels.

PML formulation:
  SPHERICAL (radial) complex coordinate stretching applied in subdomain tag 11.
  The PML shell is bounded by circular arcs centred at the origin, so the
  absorbing direction is the spherical radius ρ = √(r²+z²) — NOT the cylindrical
  radius r. Stretching only r (a cylindrical PML) fails to absorb axially
  propagating waves near the symmetry axis and produces trapped cavity
  resonances; the spherical stretch absorbs in every outward direction.

  Complex radial coordinate: ρ̃ = ρ + (1/iω) ∫_{R_fluid}^{ρ} σ(s) ds
  Quadratic grading: σ(ρ) = σ_max * ((ρ - R_fluid) / d_pml)^2
  σ_max chosen via reflection coefficient target R_ref = 1e-6:
    σ_max = -(m+1) * c * ln(R_ref) / (2 * d_pml),  m=2

  Radial stretch factor s_ρ = dρ̃/dρ = 1 + σ(ρ)/(iω). The complex Jacobian of
  the spherical stretch gives an ANISOTROPIC stiffness tensor with eigenvalues
  a_ρ = ρ̃²/(ρ² s_ρ) along e_ρ=(r,z)/ρ and a_t = s_ρ tangentially. With the
  stretched cylindrical (axisymmetric) weight r̃ = (ρ̃/ρ) r the weak-form
  coefficients are the full 2×2 tensor D and mass M:
    D = r̃ [ a_ρ·(e_ρ e_ρᵀ) + a_t·(e_t e_tᵀ) ],   M = k² r̃ s_ρ (ρ̃/ρ)
  D has a cross term D_rz (∂_r p ∂_z q + ∂_z p ∂_r q). In the fluid (σ=0):
  s_ρ=1, ρ̃=ρ ⇒ D=r·I, M=k²r (the plain axisymmetric form).

Boundary conditions:
  tag 1 (axis)      : homogeneous Neumann — natural BC, no action
  tag 2 (piston)    : inhomogeneous Neumann: ∂p/∂n = -iωρ v₀, v₀=1 m/s
  tag 3 (baffle)    : homogeneous Neumann — natural BC, no action
  tag 4 (ff_arc)    : no BC — far-field extraction surface for postprocess.py
  tag 5 (pml_outer) : Dirichlet p = 0

Real-valued splitting:
  PETSc is built with real scalars. The complex problem is split into
  real and imaginary parts: p = p_r + i*p_i, solved as a coupled 2×2 system.

Solver: PETSc LU (MUMPS). Fresh assembly at each frequency.
"""

import numpy as np
from pathlib import Path
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import (
    bb_tree, compute_collisions_points, compute_colliding_cells
)
import basix.ufl
import ufl

# ── Parameters (must match mesh.py) ───────────────────────────────────────────
# Element size is set by f_max; PML thickness by f_min (longest wavelength).
a         = 0.010    # piston radius [m]
R_fluid   = 0.080    # fluid domain radius [m]  (= 8a, standoff from near field)
c         = 343.0    # speed of sound [m/s]
rho       = 1.21     # air density [kg/m³]
v0        = 1.0      # piston velocity amplitude [m/s]
f_max     = 40e3     # highest frequency [Hz]
f_min     = 10000.0   # lowest frequency [Hz]
h_fluid   = (c / f_max) / 12
pml_frac  = 0.50                       # PML thickness in longest wavelengths
d_pml     = pml_frac * (c / f_min)     # PML thickness set by f_min (≈ 0.01715 m)
R_PML     = R_fluid + d_pml

freqs     = np.logspace(np.log10(f_min), np.log10(f_max), 40).tolist()   # [Hz]

# PML absorption coefficient
m_pml     = 3                            # grading order (gentler ramp near interface)
R_ref     = 1e-6                         # target reflection coefficient
sigma_max = -(m_pml + 1) * c * np.log(R_ref) / (2 * d_pml)

OUTPUT_DIR = Path(__file__).parent / "outputs"
MESH_FILE  = OUTPUT_DIR / "baffled_piston.msh"

# ── Load mesh ─────────────────────────────────────────────────────────────────
if not MESH_FILE.exists():
    raise FileNotFoundError(f"Mesh not found: {MESH_FILE}. Run mesh.py first.")

mesh_data = gmshio.read_from_msh(
    str(MESH_FILE), MPI.COMM_WORLD, rank=0, gdim=2
)
mesh       = mesh_data[0]
cell_tags  = mesh_data[1]
facet_tags = mesh_data[2]

# ── Mixed function space (p_r, p_i) ──────────────────────────────────────────
cell_name = mesh.topology.cell_name()
el = basix.ufl.element("Lagrange", cell_name, 1)
mel = basix.ufl.mixed_element([el, el])
W = fem.functionspace(mesh, mel)

# DG0 space for piecewise-constant PML coefficient fields
W_dg = fem.functionspace(mesh, ("DG", 0))

# ── Measures with tags ────────────────────────────────────────────────────────
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# ── Spatial coordinate r ──────────────────────────────────────────────────────
x = ufl.SpatialCoordinate(mesh)
r = x[0]

# ── Collapsed scalar sub-space + geometry helpers for the PML probe check ─────
# The collapsed scalar space of W.sub(0) holds the same DOF layout as the
# real/imaginary pressure arrays extracted after each solve, so scalar pressure
# functions can be reconstructed on it for point evaluation.
V_scalar, _ = W.sub(0).collapse()

tdim = mesh.topology.dim
_tree = bb_tree(mesh, tdim)

def _pad_3d(pts):
    """Ensure points have shape (N, 3) for DOLFINx geometry routines."""
    pts = np.asarray(pts, dtype=mesh.geometry.x.dtype)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] < 3:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 3 - pts.shape[1]),
                                        dtype=pts.dtype)])
    return pts

def _find_cells(points_3d):
    candidates = compute_collisions_points(_tree, points_3d)
    colliding  = compute_colliding_cells(mesh, candidates, points_3d)
    cells = np.full(points_3d.shape[0], -1, dtype=np.int32)
    for i in range(points_3d.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells[i] = links[0]
    return cells

def _eval_function(func, points_3d, cells):
    mask   = cells >= 0
    values = np.zeros(points_3d.shape[0], dtype=func.x.array.dtype)
    if mask.any():
        result       = func.eval(points_3d[mask], cells[mask])
        values[mask] = result.flatten()
    return values

# PML probe points (on the baffle, z=0): fluid/PML interface and near outer PML
probe_inner = [R_fluid, 0.0, 0.0]               # fluid/PML interface on baffle
probe_outer = [R_fluid + d_pml * 0.9, 0.0, 0.0]  # near outer PML on baffle
_probe_pts   = _pad_3d(np.array([probe_inner, probe_outer]))
_probe_cells = _find_cells(_probe_pts)

# ── Helper: compute PML ρ̃ analytically ───────────────────────────────────────
def pml_rho_tilde(rho_s, omega):
    """ρ̃ = ρ + (1/iω) * ∫_{R_fluid}^{ρ} σ(s) ds (returns complex)."""
    iomega = 1j * omega
    depth = max(rho_s - R_fluid, 0.0)
    integral = sigma_max / (m_pml + 1) * (depth ** (m_pml + 1)) / (d_pml ** m_pml)
    return rho_s + integral / iomega

# ── Remove stale pressure files before sweep ─────────────────────────────────
for _stale in OUTPUT_DIR.glob("pressure_*.npy"):
    _stale.unlink()

# ── Frequency sweep ───────────────────────────────────────────────────────────
summary = []

cells_fluid = cell_tags.find(10)
cells_pml   = cell_tags.find(11)

for f in freqs:
    omega = 2 * np.pi * f
    k     = omega / c
    ka    = k * a

    # Compute complex spherical-PML tensor coefficients per cell, split Re/Im.
    #   D = r̃ [ a_ρ e_ρe_ρᵀ + a_t e_te_tᵀ ],   M = k² r̃ s_ρ (ρ̃/ρ)
    #   a_ρ = ρ̃²/(ρ² s_ρ),  a_t = s_ρ,  r̃ = (ρ̃/ρ) r,  e_ρ = (r,z)/ρ
    n_cells = mesh.topology.index_map(2).size_local
    Drr_r = np.zeros(n_cells, dtype=np.float64); Drr_i = np.zeros(n_cells)
    Drz_r = np.zeros(n_cells);                   Drz_i = np.zeros(n_cells)
    Dzz_r = np.zeros(n_cells);                   Dzz_i = np.zeros(n_cells)
    M_r   = np.zeros(n_cells);                   M_i   = np.zeros(n_cells)

    conn = mesh.topology.connectivity(2, 0)

    for cell in cells_fluid:
        midpt = mesh.geometry.x[conn.links(cell)].mean(axis=0)
        r_mid = midpt[0]
        # In fluid: s_ρ=1, ρ̃=ρ → D = r·I, M = k²r (all real)
        Drr_r[cell] = r_mid
        Dzz_r[cell] = r_mid
        M_r[cell]   = k**2 * r_mid

    for cell in cells_pml:
        midpt = mesh.geometry.x[conn.links(cell)].mean(axis=0)
        r_mid = midpt[0]
        z_mid = midpt[1]
        rho_s = np.hypot(r_mid, z_mid)                 # spherical radius (not density)
        depth = max(rho_s - R_fluid, 0.0)
        sig   = sigma_max * (depth / d_pml) ** m_pml
        s_rho = 1.0 + sig / (1j * omega)
        rho_t = pml_rho_tilde(rho_s, omega)

        a_rho = rho_t**2 / (rho_s**2 * s_rho)          # radial eigenvalue
        a_t   = s_rho                                  # tangential eigenvalue
        er2   = (r_mid / rho_s) ** 2                    # e_ρe_ρᵀ components
        ez2   = (z_mid / rho_s) ** 2
        erz   = (r_mid * z_mid) / rho_s**2
        r_til = (rho_t / rho_s) * r_mid                # stretched axisym weight

        Drr_val = r_til * (a_rho * er2 + a_t * ez2)
        Dzz_val = r_til * (a_rho * ez2 + a_t * er2)
        Drz_val = r_til * (a_rho - a_t) * erz
        M_val   = k**2 * r_til * s_rho * (rho_t / rho_s)

        Drr_r[cell] = Drr_val.real; Drr_i[cell] = Drr_val.imag
        Drz_r[cell] = Drz_val.real; Drz_i[cell] = Drz_val.imag
        Dzz_r[cell] = Dzz_val.real; Dzz_i[cell] = Dzz_val.imag
        M_r[cell]   = M_val.real;   M_i[cell]   = M_val.imag

    # Create DG0 coefficient functions
    Drr_r_f = fem.Function(W_dg); Drr_i_f = fem.Function(W_dg)
    Drz_r_f = fem.Function(W_dg); Drz_i_f = fem.Function(W_dg)
    Dzz_r_f = fem.Function(W_dg); Dzz_i_f = fem.Function(W_dg)
    M_r_f   = fem.Function(W_dg); M_i_f   = fem.Function(W_dg)

    Drr_r_f.x.array[:] = Drr_r; Drr_i_f.x.array[:] = Drr_i
    Drz_r_f.x.array[:] = Drz_r; Drz_i_f.x.array[:] = Drz_i
    Dzz_r_f.x.array[:] = Dzz_r; Dzz_i_f.x.array[:] = Dzz_i
    M_r_f.x.array[:]   = M_r;   M_i_f.x.array[:]   = M_i

    for fn in (Drr_r_f, Drr_i_f, Drz_r_f, Drz_i_f,
               Dzz_r_f, Dzz_i_f, M_r_f, M_i_f):
        fn.x.scatter_forward()

    # ── Variational form (real-valued 2×2 block) ─────────────────────────────
    (p_r, p_i) = ufl.TrialFunctions(W)
    (q_r, q_i) = ufl.TestFunctions(W)

    def A_real(p, q):
        """Bilinear form with the real-part PML coefficients."""
        return (
            Drr_r_f * p.dx(0) * q.dx(0)
            + Drz_r_f * (p.dx(0) * q.dx(1) + p.dx(1) * q.dx(0))
            + Dzz_r_f * p.dx(1) * q.dx(1)
            - M_r_f * p * q
        ) * ufl.dx

    def A_imag(p, q):
        """Bilinear form with the imaginary-part PML coefficients."""
        return (
            Drr_i_f * p.dx(0) * q.dx(0)
            + Drz_i_f * (p.dx(0) * q.dx(1) + p.dx(1) * q.dx(0))
            + Dzz_i_f * p.dx(1) * q.dx(1)
            - M_i_f * p * q
        ) * ufl.dx

    # Real eq: A_r p_r - A_i p_i ;  Imag eq: A_i p_r + A_r p_i
    a_form = (
        A_real(p_r, q_r) - A_imag(p_i, q_r)
        + A_imag(p_r, q_i) + A_real(p_i, q_i)
    )

    # RHS: L = iωρv₀ ∫ q r ds(2) → Re part = 0, Im part = ωρv₀ ∫ q_i r ds(2)
    L_form = omega * rho * v0 * q_i * r * ds(2)

    # ── Dirichlet BC: p_r=0, p_i=0 on pml_outer (tag 5) ─────────────────────
    pml_facets = facet_tags.find(5)
    dofs_re = fem.locate_dofs_topological(W.sub(0), 1, pml_facets)
    dofs_im = fem.locate_dofs_topological(W.sub(1), 1, pml_facets)
    bc_re = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_re, W.sub(0))
    bc_im = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_im, W.sub(1))

    # ── Solve ─────────────────────────────────────────────────────────────────
    problem = LinearProblem(
        a_form, L_form, bcs=[bc_re, bc_im],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="baffled_piston_"
    )
    wh = problem.solve()

    # ── Extract complex pressure from mixed solution ──────────────────────────
    p_r_arr = wh.sub(0).collapse().x.array.copy()
    p_i_arr = wh.sub(1).collapse().x.array.copy()
    p_complex = p_r_arr + 1j * p_i_arr

    # ── Save solution ─────────────────────────────────────────────────────────
    out_file = OUTPUT_DIR / f"pressure_f{int(round(f))}.npy"
    np.save(out_file, p_complex)

    max_p = np.max(np.abs(p_complex))
    summary.append((f, k, ka, max_p))

    # ── PML verification check ──────────────────────────────────────────────
    # p=0 is enforced by the Dirichlet BC on the outer boundary (tag 5), so the
    # outer-boundary field is trivially zero. Instead probe the field inside the
    # PML: |p| at the fluid/PML interface (probe_inner) vs near the outer PML
    # (probe_outer). A working PML attenuates the wave strongly between them.
    p_r_probe = fem.Function(V_scalar)
    p_i_probe = fem.Function(V_scalar)
    p_r_probe.x.array[:] = p_complex.real
    p_i_probe.x.array[:] = p_complex.imag
    p_r_probe.x.scatter_forward()
    p_i_probe.x.scatter_forward()

    p_probe = (_eval_function(p_r_probe, _probe_pts, _probe_cells)
               + 1j * _eval_function(p_i_probe, _probe_pts, _probe_cells))
    val_inner = np.abs(p_probe[0])
    val_outer = np.abs(p_probe[1])
    ratio     = val_outer / val_inner if val_inner > 0 else np.inf
    status    = "WARN" if ratio > 0.05 else "OK"
    print(f"PML CHECK f={int(f):6d} Hz: |p|_inner={val_inner:.3e} Pa, "
          f"|p|_outer={val_outer:.3e} Pa, attenuation={ratio:.2e} [{status}]")

# ── Save mesh metadata for postprocess.py ────────────────────────────────────
np.savez(
    OUTPUT_DIR / "mesh_tags.npz",
    R_fluid=R_fluid, R_PML=R_PML, a=a, c=c, rho=rho, d_pml=d_pml,
    freqs=np.array(freqs)
)

# ── Summary table (truncated: header + every 10th frequency) ──────────────────
print(f"\n{'freq [Hz]':>10} {'k [rad/m]':>10} {'ka':>6} {'max|p| [Pa]':>12}")
print("-" * 44)
for row in summary[::10]:
    print(f"{int(round(row[0])):>10d} {row[1]:>10.3f} {row[2]:>6.3f} {row[3]:>12.4f}")
