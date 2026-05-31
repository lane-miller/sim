"""
validate.py — Baffled Piston POC
Postprocessing and validation. Produces frequency × angle heatmaps comparing the
FEM solution to the analytical baffled-piston field, plus a PML-attenuation curve
and an absolute on-axis pressure check. The near/far-field heatmaps are displayed
in dB NORMALIZED per-dataset to its own max and clipped to a 60 dB range (0 → -60
dB): this compares directivity shape and keeps directivity nulls (|p|→0, dB→-∞)
from blowing up the error map. The underlying FEM field is still absolute Pa (the
on-axis check below confirms calibration).

Three validations (all sampled over the swept frequency band on the x-axis and
the polar observation angle θ on the y-axis):

  1. PML attenuation vs frequency — an amplitude-invariant, null-robust check of
     the absorbing layer. We compare the acoustic energy on a spherical arc near
     the outer PML (ρ=R_fluid+0.9·d_pml) to that on the fluid/PML interface arc,
     using a sinθ-weighted energy norm:
       A(f) = 10·log10( ∫|p_out|²·sinθ dθ / ∫|p_in|²·sinθ dθ ).
     A PML is judged by REFLECTIONLESSNESS, not by the absolute field magnitude
     inside it. That magnitude merely tracks the (huge, v₀=1 m/s) source drive —
     the in-layer field is the attenuated analytic continuation of the outgoing
     wave, so ~100 dB SPL near the boundary is expected and says nothing about
     absorber quality. A 20 µPa "noise-floor" target is a human-hearing reference,
     irrelevant to a numerical absorber and not even amplitude-invariant. The
     energy ratio fixes both problems: it is independent of source amplitude and,
     by integrating over θ, immune to the directivity-null degeneracy (0/0) that
     plagued the old per-ray ratio. The measured A(f) should sit at or below the
     analytic one-way design decay (dashed target line); a curve well above it
     flags discretization reflection (e.g. too few PML element layers).
     The PRIMARY PML verdict, however, is checks 2–3: a reflectionless PML ⇔ the
     interior matches the analytic free field, so small near/far-field error IS
     the definitive evidence the PML is not contaminating the physical solution.

  2. Near-field pressure (inside the fluid domain, just inside the boundary) —
     FEM, Analytical, and Error heatmaps of |p| on a spherical arc at
     R_near = 0.95·R_fluid. The analytical reference is the EXACT half-space
     Rayleigh (King) integral — valid at ANY distance — rather than the far-field
     amplitude law, so the comparison stays correct even where R_near is within a
     few wavelengths of the piston at the low end of the band.

  3. Far-field pressure (at R_obs = 1 m, OUTSIDE the PML) — FEM, Analytical, and
     Error heatmaps. The FEM field is extrapolated to R_obs with the axisymmetric
     Kirchhoff-Helmholtz integral (radius-averaged over interior arcs) and scaled
     to absolute Pascals (see KH_PREFACTOR below).

Far-field extraction method: axisymmetric Kirchhoff-Helmholtz integral with
the rigid-baffle image (half-space Green's function)
  The axisymmetric FEM solves in the (r,z) half-plane but recovers the full
  3D field via the 2πr Jacobian. The correct far-field postprocessing must
  therefore use the AZIMUTHALLY-INTEGRATED 3D Green's function, NOT the 2D
  Hankel function (which would correspond to a 2D line-source problem).

  CLOSURE / BAFFLE IMAGE
  A quarter-circle arc is not a closed surface; the rest of the closure is the
  rigid baffle at z=0. We enforce it by image theory: the rigid baffle
  (∂p/∂n=0 on z=0) is equivalent to a full-space problem whose field is mirror-
  symmetric about z=0. The K-H integral over the closed image-sphere reduces to
  a single integral over the hemisphere using the half-space Green's function
  G_N = direct(z_s) + image(-z_s). Omitting the image is a dominant error (the
  open surface gives a wrong, often >1 and oscillatory pattern).

  AZIMUTHAL INTEGRATION  →  J₀ AND J₁
  For a source point (r_s, z_s) and observation direction θ (from the z-axis),
  the far-field free-space kernel is exp(-ik θ̂·s); azimuthal integration over
  φ ∈ [0, 2π] of the pressure and the OUTWARD-NORMAL derivative ∂G/∂n_s =
  -ik(θ̂·n̂)G_s gives, with β = k r_s sinθ and ψ = k z_s cosθ (direct + image):

    ∂p/∂n term : -dpdn * J₀(β) * cos(ψ)
    pressure   : -k * p * [ cosθ·n_z·J₀(β)·sin(ψ) + sinθ·n_r·J₁(β)·cos(ψ) ]

  The radial component of the surface normal couples to cos(φ) under the
  azimuthal integral and therefore produces a J₁ (dipole) term, NOT J₀ — the
  axial component produces the J₀ term. The far-field K-H integral is then:

    I(θ) = ∫_Γ [ p · ∂G_N/∂n - G_N · ∂p/∂n ] r_s ds

  where n = (n_r, n_z) is the outward (away-from-origin) arc normal.

  ABSOLUTE SCALING (KH_PREFACTOR)
  The kernels above drop the constants exp(ikR_obs)/(4πR_obs) (Green's-function
  prefactor), the 2π from the azimuthal integration, and the factor 2 from the
  baffle image. Restoring magnitudes: |p(θ)| = |I(θ)| · 2π · 1/(4πR_obs) · 2
  = |I(θ)| / R_obs. So KH_PREFACTOR = 1/R_obs converts the kernel integral to
  absolute Pascals. (If the far-field error heatmap shows a uniform dB offset,
  this single constant is the place to correct it.)

  EXTRACTION SURFACE AND RADIUS AVERAGING
  ∂p/∂n is taken from the exact (piecewise-constant) gradient of the P1 field.
  Rather than the ff_arc (tag 4), which sits on the fluid/PML interface where
  the field is most contaminated by residual reflection, we integrate over a
  band of interior arcs (radii R_e < R_fluid) and average the complex far field:
  the genuine outgoing wave is surface-invariant and adds coherently, while any
  radius-dependent (standing-wave) reflection partly averages out.

Analytical reference:
  Near-field p  exact half-space Rayleigh integral p = (iωρv₀/2π)∫_disk e^{ikd}/d dS
                (valid at any distance; this is what the near-field heatmap uses).
  Directivity   D(θ) = |2 J₁(ka sinθ)/(ka sinθ)|,  D(0)=1 by L'Hôpital.
  Far-field p   |p(r,θ)| = ρ c v₀ k a²/(2 r) · D(θ)   (baffled piston, r ≫ a²/λ;
                used for the 1 m far-field heatmap where kr ≫ 1).
  On-axis p     |p(z)|  = 2 ρ c v₀ |sin( (k/2)(√(z²+a²) − z) )|   (exact).
  Valid because the axisymmetric FEM with the 2πr Jacobian recovers the full 3D
  problem.
"""

import numpy as np
from pathlib import Path
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import basix.ufl
import ufl
from scipy.special import j0, j1
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── Parameters ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "outputs"
MESH_FILE  = OUTPUT_DIR / "baffled_piston.msh"
N_theta    = 180
P_REF      = 1.0      # dB reference pressure [Pa]  (dB re 1 Pa, near/far-field plots)
R_obs      = 1.0      # far-field observation radius (outside PML) [m]

# PML design constants (must match solve.py) — used only to draw the analytic
# one-way attenuation target on the PML-attenuation curve.
m_pml      = 3        # PML grading order (must match solve.py)
R_ref      = 1e-6     # target round-trip reflection coefficient
PML_PROBE_FRAC = 0.9  # probe arc depth as a fraction of d_pml (matches R_pml_out)

# ── Load mesh metadata ────────────────────────────────────────────────────────
meta     = np.load(OUTPUT_DIR / "mesh_tags.npz")
R_fluid  = float(meta["R_fluid"])
R_PML    = float(meta["R_PML"])
a        = float(meta["a"])
c        = float(meta["c"])
rho      = float(meta["rho"])
d_pml    = float(meta["d_pml"])
freqs    = meta["freqs"].tolist()
v0       = 1.0      # piston velocity amplitude [m/s] (matches solve.py)

KH_PREFACTOR = 1.0 / R_obs   # kernel integral → absolute Pa (see module docstring)

# Sampling radii (all spherical, matching the spherical PML / quarter-circle arcs)
R_near    = 0.95 * R_fluid              # near-field probe, just inside the boundary
R_pml_in  = R_fluid                     # fluid/PML interface
R_pml_out = R_fluid + 0.9 * d_pml       # near the outer PML (inside the domain)

# ── Representative frequencies for the on-axis 2D check (4 closest to targets) ─
_target_freqs = [10000.0, 18000.0, 30000.0, 40000.0]
_freqs_arr    = np.array(freqs)
rep_idx       = [int(np.argmin(np.abs(_freqs_arr - tf))) for tf in _target_freqs]

# ── Load mesh ─────────────────────────────────────────────────────────────────
if not MESH_FILE.exists():
    raise FileNotFoundError(f"Mesh not found: {MESH_FILE}. Run mesh.py first.")

mesh_data  = gmshio.read_from_msh(str(MESH_FILE), MPI.COMM_WORLD, rank=0, gdim=2)
msh        = mesh_data[0]
cell_tags  = mesh_data[1]
facet_tags = mesh_data[2]

# ── Scalar function space (matches each sub-space of mixed W in solve.py) ─────
cell_name = msh.topology.cell_name()
el = basix.ufl.element("Lagrange", cell_name, 1)
V  = fem.functionspace(msh, el)

# ── Geometry helpers ──────────────────────────────────────────────────────────
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
geom = msh.geometry.x

def _pad_3d(pts):
    """Ensure points have shape (N, 3) for DOLFINx geometry routines."""
    pts = np.asarray(pts, dtype=geom.dtype)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] < 3:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 3 - pts.shape[1]),
                                        dtype=pts.dtype)])
    return pts

_tree = bb_tree(msh, tdim)

def _find_cells(points_3d):
    candidates = compute_collisions_points(_tree, points_3d)
    colliding  = compute_colliding_cells(msh, candidates, points_3d)
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

def _eval_vector(func, points_3d, cells):
    """Evaluate a vector-valued (tdim-component) function at points."""
    mask   = cells >= 0
    values = np.zeros((points_3d.shape[0], tdim), dtype=func.x.array.dtype)
    if mask.any():
        values[mask] = func.eval(points_3d[mask], cells[mask])
    return values

# ── Vector DG0 space for the exact P1 gradient (∂p/∂n is piecewise constant) ──
Vg       = fem.functionspace(msh, ("DG", 0, (tdim,)))
_grad_ip = Vg.element.interpolation_points

# ── Far-field extraction arcs (interior quarter-circles, radius-averaged) ─────
N_arc         = 121                       # angular samples per extraction arc
N_radii       = 40                        # arcs in the averaging band
frac_lo       = 0.45                      # innermost arc radius / R_fluid
frac_hi       = 0.92                      # outermost arc radius / R_fluid
extract_radii = np.linspace(frac_lo * R_fluid, frac_hi * R_fluid, N_radii)

def _arc_geometry(R_e):
    """Quarter-circle arc midpoints (r,z), outward normals, arc-length weights."""
    ang  = np.linspace(0.0, np.pi / 2, N_arc + 1)
    amid = 0.5 * (ang[:-1] + ang[1:])
    seg  = R_e * (ang[1:] - ang[:-1])             # arc-length quadrature weights
    r_s  = R_e * np.cos(amid)                     # r=0 at top, z=0 at base
    z_s  = R_e * np.sin(amid)
    mids = np.column_stack([r_s, z_s])
    return mids, mids / R_e, seg                  # normals = (r,z)/R_e (outward)

# ── Observation angles ────────────────────────────────────────────────────────
theta     = np.linspace(0, np.pi / 2, N_theta, endpoint=True)
theta_deg = np.degrees(theta)
_sin_w    = np.sin(theta)   # axisymmetric surface weight (dS ∝ sinθ dθ)

# np.trapz was renamed np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

def energy_atten_db(p_in_arc, p_out_arc):
    """sinθ-weighted energy attenuation [dB] from the interface arc to the outer
    arc: A = 10·log10( ∫|p_out|²sinθ dθ / ∫|p_in|²sinθ dθ ). Amplitude-invariant
    and immune to directivity nulls (they contribute ≈0 to both integrals)."""
    e_in  = _trapz(np.abs(p_in_arc) ** 2  * _sin_w, theta)
    e_out = _trapz(np.abs(p_out_arc) ** 2 * _sin_w, theta)
    return 10.0 * np.log10(max(e_out, 1e-30) / max(e_in, 1e-30))

# Analytic one-way amplitude decay through the PML to the probe depth, in dB.
# The complex coordinate stretch attenuates the outgoing wave by
# exp(-(1/c)∫σ ds); with the quadratic σ tuned to R_ref this reduces to a
# frequency-INDEPENDENT constant: (1/c)∫₀^{depth}σ = -(ln R_ref / 2)·frac^{m+1}.
PML_DESIGN_ATTEN_DB = (20.0 / np.log(10.0)) * (np.log(R_ref) / 2.0) \
    * PML_PROBE_FRAC ** (m_pml + 1)

# ── Axisymmetric far-field K-H integral over one arc ──────────────────────────
def compute_farfield_axi(p_arc, dpdn_arc, mids, normals, seg, k):
    """
    Axisymmetric far-field K-H integral over one quarter-circle arc, including
    the rigid-baffle image. Returns the complex kernel amplitude I(θ) at each
    angle in theta (absolute Pa = |I| · KH_PREFACTOR). With β = k r_s sinθ and
    ψ = k z_s cosθ (direct + image):

      ∂p/∂n term : -dpdn * J0(β) * cos(ψ)
      pressure   : -k * p * [ cosθ·n_z·J0(β)·sin(ψ) + sinθ·n_r·J1(β)·cos(ψ) ]

    The radial normal component produces the J1 (dipole) term, the axial
    component the J0 term. The exp(ikR)/R prefactor, the azimuthal 2π and the
    image factor 2 are restored via KH_PREFACTOR at the call site.
    """
    r_s = mids[:, 0]
    z_s = mids[:, 1]
    n_r = normals[:, 0]
    n_z = normals[:, 1]

    p_ff = np.zeros(N_theta, dtype=np.complex128)

    for i, th in enumerate(theta):
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        beta   = k * r_s * sin_th
        psi    = k * z_s * cos_th
        J0b    = j0(beta)
        J1b    = j1(beta)

        p_kernel  = -k * (cos_th * n_z * J0b * np.sin(psi)
                          + sin_th * n_r * J1b * np.cos(psi))
        dn_kernel = J0b * np.cos(psi)

        # K-H integrand with arc-length quadrature weights and r_s azimuthal weight
        integrand = r_s * (p_arc * p_kernel - dpdn_arc * dn_kernel) * seg
        p_ff[i]   = np.sum(integrand)

    return p_ff

def _eval_on_arc(p_r_func, p_i_func, grad_r, grad_i, R_e):
    """Complex p and its outward normal derivative ∂p/∂n on the arc of radius R_e."""
    mids, normals, seg = _arc_geometry(R_e)
    pts_3d = _pad_3d(mids)
    cells  = _find_cells(pts_3d)
    n_miss = np.sum(cells < 0)
    if n_miss > 0:
        print(f"  WARNING: {n_miss}/{len(cells)} arc points (R_e={R_e:.4f}) not found")

    p_arc = (_eval_function(p_r_func, pts_3d, cells)
             + 1j * _eval_function(p_i_func, pts_3d, cells))
    grad  = (_eval_vector(grad_r, pts_3d, cells)
             + 1j * _eval_vector(grad_i, pts_3d, cells))
    dpdn  = grad[:, 0] * normals[:, 0] + grad[:, 1] * normals[:, 1]
    return p_arc, dpdn, mids, normals, seg

# ── Analytical references ─────────────────────────────────────────────────────
def analytical_directivity(ka, theta_arr):
    """D(θ) = |2 J₁(ka sinθ)/(ka sinθ)|, D(0)=1 by L'Hôpital."""
    D      = np.ones_like(theta_arr)
    sin_th = np.sin(theta_arr)
    mask   = np.abs(sin_th) > 1e-12
    arg    = ka * sin_th[mask]
    D[mask] = np.abs(2.0 * j1(arg) / arg)
    return D

def analytical_pressure(r_obs, k, theta_arr):
    """Far-field baffled-piston pressure magnitude [Pa]:
       |p(r,θ)| = ρ c v₀ k a²/(2 r) · D(θ).  Valid for r ≫ a²/λ."""
    D = analytical_directivity(k * a, theta_arr)
    return rho * c * v0 * k * a**2 / (2.0 * r_obs) * D

def analytical_onaxis(z_arr, k):
    """Exact on-axis pressure magnitude for a baffled circular piston:
       |p(z)| = 2 ρ c v₀ |sin( (k/2)(√(z²+a²) − z) )|."""
    return 2.0 * rho * c * v0 * np.abs(
        np.sin(0.5 * k * (np.sqrt(z_arr**2 + a**2) - z_arr))
    )

# ── Exact Rayleigh (King) integral for the near-field reference ───────────────
# The near-field arc (R_near ≈ R_fluid) is only a fraction of a wavelength from
# the piston at the low end of the band (kR_near ≈ 1.7 at 2 kHz), so the
# far-field amplitude law ρcv₀ka²/(2r)·D(θ) is NOT valid there. The exact field
# of a baffled piston with uniform velocity v₀ is the half-space Rayleigh
# integral, valid at ANY distance:
#     p(x) = (iωρv₀ / 2π) ∫_disk e^{ikd}/d dS,   d = |x − x_s|
# (the 2π — rather than 4π — is the baffle/image factor). On axis this reduces
# analytically to the 2ρcv₀|sin(...)| form above, confirming the constant.
_Ns_quad, _Nphi_quad = 64, 128
_s_nodes, _s_wts = np.polynomial.legendre.leggauss(_Ns_quad)
_s_nodes = 0.5 * a * (_s_nodes + 1.0)          # Gauss nodes mapped [-1,1] → [0,a]
_s_wts   = 0.5 * a * _s_wts                     # matching ds weights
_phi     = np.linspace(0.0, 2 * np.pi, _Nphi_quad, endpoint=False)
_dphi    = 2 * np.pi / _Nphi_quad
_S, _PHI = np.meshgrid(_s_nodes, _phi, indexing="ij")          # (Ns, Nphi)
_SW       = (_s_wts[:, None] * _dphi) * _S                      # area weight s·ds·dφ
_s_cosphi = _S * np.cos(_PHI)
_s_sq     = _S ** 2

def rayleigh_pressure(r_o, z_o, k):
    """Exact baffled-piston |p| [Pa] at field points (r_o, z_o), vectorized.
    Valid in both the near and far field (this is the reference the near-field
    FEM heatmap is compared against)."""
    omega = k * c
    r_o   = np.atleast_1d(r_o)[:, None, None]                  # (Np,1,1)
    z_o   = np.atleast_1d(z_o)[:, None, None]
    d     = np.sqrt(r_o**2 + _s_sq[None] - 2.0 * r_o * _s_cosphi[None] + z_o**2)
    integ = (np.exp(1j * k * d) / d) * _SW[None]               # (Np,Ns,Nphi)
    p     = (1j * omega * rho * v0 / (2.0 * np.pi)) * integ.sum(axis=(1, 2))
    return np.abs(p)

# ── Sample |p| along a spherical arc (constant ρ=R_e) at the observation angles ─
def abs_p_on_sphere(p_r_func, p_i_func, R_e, theta_arr):
    """|p| at the points (r=R_e sinθ, z=R_e cosθ) for θ in theta_arr [Pa].
    A small radial floor avoids the r=0 singularity of the point lookup on axis."""
    rr  = np.maximum(R_e * np.sin(theta_arr), 1e-4)
    zz  = R_e * np.cos(theta_arr)
    pts = _pad_3d(np.column_stack([rr, zz]))
    cells = _find_cells(pts)
    p = (_eval_function(p_r_func, pts, cells)
         + 1j * _eval_function(p_i_func, pts, cells))
    return np.abs(p)

def to_db(p_abs):
    """SPL in dB re P_REF (floored to avoid log of zero)."""
    return 20.0 * np.log10(np.maximum(p_abs, 1e-12) / P_REF)

def _load_pressure_funcs(f):
    """Load the saved complex pressure for frequency f and reconstruct the
    scalar real/imaginary FEM functions."""
    p_file = OUTPUT_DIR / f"pressure_f{int(round(f))}.npy"
    if not p_file.exists():
        raise FileNotFoundError(f"{p_file} not found. Run solve.py first.")
    p_complex = np.load(p_file)
    p_r_func = fem.Function(V)
    p_i_func = fem.Function(V)
    p_r_func.x.array[:] = p_complex.real
    p_i_func.x.array[:] = p_complex.imag
    p_r_func.x.scatter_forward()
    p_i_func.x.scatter_forward()
    return p_r_func, p_i_func

# ── Sweep: assemble frequency × angle fields ──────────────────────────────────
nf, nt = len(freqs), N_theta
pml_in_abs  = np.zeros((nf, nt))   # |p| at fluid/PML interface
pml_out_abs = np.zeros((nf, nt))   # |p| near outer PML
near_fem    = np.zeros((nf, nt))   # FEM |p| at R_near                 [Pa]
near_ana    = np.zeros((nf, nt))   # analytical |p| at R_near          [Pa]
far_fem     = np.zeros((nf, nt))   # FEM |p| extrapolated to R_obs     [Pa]
far_ana     = np.zeros((nf, nt))   # analytical |p| at R_obs           [Pa]

print(f"Sweeping {nf} frequencies ({freqs[0]:.0f}–{freqs[-1]:.0f} Hz)…")
print(f"{'freq [Hz]':>10} {'ka':>7} {'PMLatten[dB]':>13} "
      f"{'near rel':>9} {'far rel':>9}")
print("-" * 52)

for i, f in enumerate(freqs):
    omega = 2 * np.pi * f
    k     = omega / c
    ka    = k * a

    p_r_func, p_i_func = _load_pressure_funcs(f)

    # PML absorption + near-field (direct, absolute FEM pressure)
    pml_in_abs[i]  = abs_p_on_sphere(p_r_func, p_i_func, R_pml_in,  theta)
    pml_out_abs[i] = abs_p_on_sphere(p_r_func, p_i_func, R_pml_out, theta)
    near_fem[i]    = abs_p_on_sphere(p_r_func, p_i_func, R_near,    theta)
    # Exact (near-field-valid) Rayleigh integral on the same arc
    near_ana[i]    = rayleigh_pressure(R_near * np.sin(theta),
                                       R_near * np.cos(theta), k)

    # Far-field: K-H extrapolation to R_obs (radius-averaged), scaled to abs Pa
    grad_r = fem.Function(Vg)
    grad_i = fem.Function(Vg)
    grad_r.interpolate(fem.Expression(ufl.grad(p_r_func), _grad_ip))
    grad_i.interpolate(fem.Expression(ufl.grad(p_i_func), _grad_ip))

    p_ff = np.zeros(N_theta, dtype=np.complex128)
    for R_e in extract_radii:
        p_arc, dpdn_arc, mids, normals, seg = _eval_on_arc(
            p_r_func, p_i_func, grad_r, grad_i, R_e
        )
        p_ff += compute_farfield_axi(p_arc, dpdn_arc, mids, normals, seg, k)
    p_ff /= len(extract_radii)

    far_fem[i] = np.abs(p_ff) * KH_PREFACTOR
    far_ana[i] = analytical_pressure(R_obs, k, theta)

    if i % 10 == 0 or i == nf - 1:
        atten   = energy_atten_db(pml_in_abs[i], pml_out_abs[i])
        near_rel = (np.max(np.abs(near_fem[i] - near_ana[i]))
                    / max(np.max(near_ana[i]), 1e-30))
        far_rel  = (np.max(np.abs(far_fem[i] - far_ana[i]))
                    / max(np.max(far_ana[i]), 1e-30))
        print(f"{int(round(f)):>10d} {ka:>7.2f} {atten:>13.1f} "
              f"{near_rel:>9.3f} {far_rel:>9.3f}")

# ── Derived fields ────────────────────────────────────────────────────────────
# Per-angle ratio (kept for reference/saved data; degenerate at directivity nulls)
pml_atten_db = 20.0 * np.log10(np.maximum(pml_out_abs, 1e-12)
                               / np.maximum(pml_in_abs, 1e-12))
# Amplitude-invariant, null-robust energy attenuation per frequency [dB]
pml_atten_energy_db = np.array([
    energy_atten_db(pml_in_abs[i], pml_out_abs[i]) for i in range(nf)
])
# Each dataset is normalized to ITS OWN overall max and expressed in dB, then
# clipped to a fixed DYN_RANGE_DB window (0 → -60 dB). Normalizing per-dataset
# compares directivity shape rather than absolute calibration, and the -60 dB
# floor stops the error map from blowing up in directivity nulls (where |p|→0
# drives the raw dB to -∞).
DYN_RANGE_DB = 60.0

def norm_db(field):
    """|p| normalized to its overall max, in dB, clipped to [-DYN_RANGE_DB, 0]."""
    ref = max(float(np.max(field)), 1e-30)
    db  = 20.0 * np.log10(np.maximum(field, 1e-30) / ref)
    return np.clip(db, -DYN_RANGE_DB, 0.0)

near_fem_db, near_ana_db = norm_db(near_fem), norm_db(near_ana)
far_fem_db,  far_ana_db  = norm_db(far_fem),  norm_db(far_ana)
near_err_db = near_fem_db - near_ana_db
far_err_db  = far_fem_db  - far_ana_db

# ── Plotting helpers (freq on x, angle on y) ──────────────────────────────────
F_edges = _freqs_arr            # cell centers; pcolormesh handles with shading='gouraud'

def _heatmap(ax, Z, title, cmap, vmin=None, vmax=None, norm=None, cbar_label="dB"):
    mesh = ax.pcolormesh(_freqs_arr, theta_deg, Z.T, cmap=cmap,
                         shading="gouraud", vmin=vmin, vmax=vmax, norm=norm)
    ax.set_xscale("log")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$\theta$ [deg]")
    ax.set_title(title)
    cb = ax.figure.colorbar(mesh, ax=ax)
    cb.set_label(cbar_label)
    return mesh

# ── Figure 1: PML attenuation vs frequency (amplitude-invariant, null-robust) ─
# Energy ratio between the interface arc and the 0.9·d_pml arc, with the analytic
# one-way design decay overlaid. This judges the absorber by reflectionlessness,
# not by absolute field magnitude (which only tracks the v₀=1 m/s source drive).
# The DEFINITIVE PML verdict is the near/far-field error heatmaps (Figs 2–3): a
# reflectionless PML ⇔ the interior matches the analytic free field.
fig_pml, ax_pml = plt.subplots(figsize=(8, 5))
ax_pml.semilogx(_freqs_arr, pml_atten_energy_db, color="C0", lw=1.8,
                label="measured energy attenuation")
ax_pml.axhline(PML_DESIGN_ATTEN_DB, color="C3", ls="--", lw=1.5,
               label=f"analytic design decay ({PML_DESIGN_ATTEN_DB:.1f} dB)")
ax_pml.set_xlabel("frequency [Hz]")
ax_pml.set_ylabel("PML attenuation [dB]")
ax_pml.set_title(
    "PML attenuation: interface arc → outer arc "
    f"($\\rho$={R_fluid + PML_PROBE_FRAC * d_pml:.4f} m, sin$\\theta$-weighted energy)"
)
ax_pml.grid(True, which="both", ls=":", alpha=0.5)
ax_pml.legend(loc="best", fontsize=9)
fig_pml.tight_layout()
fig_pml.savefig(OUTPUT_DIR / "pml_check.png", dpi=150)

# ── Figure 2: Near-field FEM / Analytical / Error (normalized, 60 dB range) ───
near_emax = float(min(12.0, np.max(np.abs(near_err_db))))

fig_near, ax_near = plt.subplots(1, 3, figsize=(16, 4.5))
_heatmap(ax_near[0], near_fem_db, f"Near-field FEM (R={R_near*1e3:.1f} mm, normalized)",
         cmap="inferno", vmin=-DYN_RANGE_DB, vmax=0.0, cbar_label="dB (norm. to max)")
_heatmap(ax_near[1], near_ana_db, "Near-field Analytical (normalized)",
         cmap="inferno", vmin=-DYN_RANGE_DB, vmax=0.0, cbar_label="dB (norm. to max)")
_heatmap(ax_near[2], near_err_db, "Near-field Error (FEM − Analytical, normalized)",
         cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -near_emax, near_emax),
         cbar_label="Δ dB")
fig_near.tight_layout()
fig_near.savefig(OUTPUT_DIR / "nearfield_validation.png", dpi=150)

# ── Figure 3: Far-field FEM / Analytical / Error (normalized, 60 dB range) ────
far_emax = float(min(12.0, np.max(np.abs(far_err_db))))

fig_far, ax_far = plt.subplots(1, 3, figsize=(16, 4.5))
_heatmap(ax_far[0], far_fem_db, f"Far-field FEM (K-H, R={R_obs:.1f} m, normalized)",
         cmap="inferno", vmin=-DYN_RANGE_DB, vmax=0.0, cbar_label="dB (norm. to max)")
_heatmap(ax_far[1], far_ana_db, "Far-field Analytical (normalized)",
         cmap="inferno", vmin=-DYN_RANGE_DB, vmax=0.0, cbar_label="dB (norm. to max)")
_heatmap(ax_far[2], far_err_db, "Far-field Error (FEM − Analytical, normalized)",
         cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -far_emax, far_emax),
         cbar_label="Δ dB")
fig_far.tight_layout()
fig_far.savefig(OUTPUT_DIR / "farfield_validation.png", dpi=150)

# ── Save heatmap data ─────────────────────────────────────────────────────────
np.savez(
    OUTPUT_DIR / "heatmap_data.npz",
    freqs=_freqs_arr, theta_deg=theta_deg,
    pml_atten_db=pml_atten_db,
    pml_atten_energy_db=pml_atten_energy_db,
    pml_design_atten_db=PML_DESIGN_ATTEN_DB,
    near_fem_db=near_fem_db, near_ana_db=near_ana_db, near_err_db=near_err_db,
    far_fem_db=far_fem_db,   far_ana_db=far_ana_db,   far_err_db=far_err_db,
)
print(f"\nFigure saved: {OUTPUT_DIR / 'pml_check.png'}")
print(f"Figure saved: {OUTPUT_DIR / 'nearfield_validation.png'}")
print(f"Figure saved: {OUTPUT_DIR / 'farfield_validation.png'}")
print(f"Data saved:   {OUTPUT_DIR / 'heatmap_data.npz'}")

# ── On-axis ABSOLUTE pressure sanity check (numeric only — no plot) ────────────
# Confirms the FEM output is in absolute Pascals (the solve is driven by the
# v₀=1 m/s Neumann BC) by comparing the on-axis |p(z)| against the exact
# closed-form baffled-piston result. This is a console check only; all visual
# diagnostics are the frequency × angle heatmaps above.
N_axis     = 200
epsilon    = 1e-4                                  # slightly off-axis (avoid r=0)
z_axis     = np.linspace(0.001, R_fluid * 0.95, N_axis)
axis_pts   = _pad_3d(np.column_stack([np.full(N_axis, epsilon), z_axis]))
axis_cells = _find_cells(axis_pts)

print(f"\nOn-axis absolute pressure check (FEM Pa vs analytical Pa)")
print(f"{'freq [Hz]':>10} {'ka':>8} {'max|err| [Pa]':>14} "
      f"{'max|p|_ana [Pa]':>16} {'rel_error':>10}")
print("-" * 62)

for idx in rep_idx:
    f     = freqs[idx]
    k     = (2 * np.pi * f) / c
    ka    = k * a

    p_r_func, p_i_func = _load_pressure_funcs(f)
    p_axis = (_eval_function(p_r_func, axis_pts, axis_cells)
              + 1j * _eval_function(p_i_func, axis_pts, axis_cells))
    p_fem_abs = np.abs(p_axis)
    p_ana_abs = analytical_onaxis(z_axis, k)

    max_err   = np.max(np.abs(p_fem_abs - p_ana_abs))
    max_ana   = np.max(p_ana_abs)
    rel_error = max_err / max_ana if max_ana > 0 else np.inf
    print(f"{int(round(f)):>10d} {ka:>8.3f} {max_err:>14.4e} "
          f"{max_ana:>16.4e} {rel_error:>10.4f}")

plt.show(block=True)   # hold the three heatmap figures open until the user closes them
