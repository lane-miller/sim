"""
extra_plots.py — Baffled Piston POC
2-D polar far-field pressure directivity at r = 1 m, p_ref = 1 Pa.
FEA (axisymmetric Kirchhoff–Helmholtz extrapolation) vs Analytical.

All requested frequencies are overlaid on one polar plot.  Each frequency
gets a unique colour; line style distinguishes FEA (dashed) from Analytical
(solid).  The radial axis is SPL in dB re 1 Pa over a fixed dynamic range.
"""

# ── User-configurable frequency list ──────────────────────────────────────────
# Set to any subset of the solved frequencies (Hz).  Leave empty to use four
# representative frequencies spread across the solved band automatically.
FREQS = [10000.0, 16000.0, 24000.0, 32000.0]

# ── Dynamic range for the polar plot (dB below the on-axis peak) ───────────────
DYN_RANGE_DB = 40.0

# ── Frequency for the 2D pressure field heatmap ────────────────────────────────
FHEATMAP = 20e3   # [Hz]

# ── Colormap for the pressure heatmap ─────────────────────────────────────────
# Perceptually-uniform sequential options (all work well for SPL fields):
#   "inferno"  — black → purple → orange → yellow  (high contrast, dark bg)
#   "magma"    — black → purple → salmon → white   (softer highlight)
#   "plasma"   — blue  → magenta → yellow          (vibrant)
#   "viridis"  — navy  → teal   → yellow           (colourblind-safe default)
#   "turbo"    — blue  → green  → red              (high detail, rainbow-like)
#   "hot"      — black → red    → yellow → white
HEATMAP_CMAP = "inferno"

# ── Imports ───────────────────────────────────────────────────────────────────
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
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "outputs"
MESH_FILE  = OUTPUT_DIR / "baffled_piston.msh"

# ── Physical constants ────────────────────────────────────────────────────────
R_obs  = 1.0   # far-field observation radius [m]
P_REF  = 1.0   # dB reference pressure [Pa]
v0     = 1.0   # piston velocity amplitude [m/s]  (matches solve.py)

KH_PREFACTOR = 1.0 / R_obs   # kernel integral → absolute Pa

# ── Load mesh metadata ────────────────────────────────────────────────────────
meta      = np.load(OUTPUT_DIR / "mesh_tags.npz")
R_fluid   = float(meta["R_fluid"])
a         = float(meta["a"])
c         = float(meta["c"])
rho       = float(meta["rho"])
all_freqs = meta["freqs"].tolist()
_freqs_arr = np.array(all_freqs)

# ── Resolve requested frequencies against the solved set ─────────────────────
if not FREQS:
    _targets = [10000.0, 18000.0, 30000.0, 40000.0]
    FREQS = [all_freqs[int(np.argmin(np.abs(_freqs_arr - t)))] for t in _targets]

plot_freqs = []
for f in FREQS:
    solved = all_freqs[int(np.argmin(np.abs(_freqs_arr - f)))]
    if solved not in plot_freqs:
        plot_freqs.append(solved)

# ── Observation angles (0 = on-axis, π/2 = in the baffle plane) ──────────────
N_theta = 361
theta   = np.linspace(0.0, np.pi / 2.0, N_theta)

# ── Load mesh ─────────────────────────────────────────────────────────────────
if not MESH_FILE.exists():
    raise FileNotFoundError(f"Mesh not found: {MESH_FILE}. Run mesh.py first.")

mesh_data  = gmshio.read_from_msh(str(MESH_FILE), MPI.COMM_WORLD, rank=0, gdim=2)
msh        = mesh_data[0]
cell_tags  = mesh_data[1]

cell_name  = msh.topology.cell_name()
el = basix.ufl.element("Lagrange", cell_name, 1)
V  = fem.functionspace(msh, el)

tdim = msh.topology.dim
msh.topology.create_connectivity(tdim - 1, tdim)
geom = msh.geometry.x

def _pad_3d(pts):
    """Ensure shape (N, 3) as required by DOLFINx geometry routines."""
    pts = np.asarray(pts, dtype=geom.dtype)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] < 3:
        pts = np.hstack([pts, np.zeros((pts.shape[0], 3 - pts.shape[1]),
                                        dtype=pts.dtype)])
    return pts

_tree = bb_tree(msh, tdim)

def _find_cells(pts3d):
    cands     = compute_collisions_points(_tree, pts3d)
    colliding = compute_colliding_cells(msh, cands, pts3d)
    cells     = np.full(pts3d.shape[0], -1, dtype=np.int32)
    for i in range(pts3d.shape[0]):
        links = colliding.links(i)
        if len(links):
            cells[i] = links[0]
    return cells

def _eval_scalar(func, pts3d, cells):
    mask   = cells >= 0
    values = np.zeros(pts3d.shape[0], dtype=func.x.array.dtype)
    if mask.any():
        values[mask] = func.eval(pts3d[mask], cells[mask]).flatten()
    return values

def _eval_vector(func, pts3d, cells):
    mask   = cells >= 0
    values = np.zeros((pts3d.shape[0], tdim), dtype=func.x.array.dtype)
    if mask.any():
        values[mask] = func.eval(pts3d[mask], cells[mask])
    return values

# Vector DG0 for exact P1 gradient
Vg       = fem.functionspace(msh, ("DG", 0, (tdim,)))
_grad_ip = Vg.element.interpolation_points

# ── Far-field extraction: radius-averaged K-H integral (matches validate.py) ──
N_arc         = 121
N_radii       = 40
extract_radii = np.linspace(0.45 * R_fluid, 0.92 * R_fluid, N_radii)

def _arc_geometry(R_e):
    """Quarter-circle midpoints, outward unit normals, arc-length weights."""
    ang  = np.linspace(0.0, np.pi / 2.0, N_arc + 1)
    amid = 0.5 * (ang[:-1] + ang[1:])
    seg  = R_e * (ang[1:] - ang[:-1])
    r_s  = R_e * np.cos(amid)
    z_s  = R_e * np.sin(amid)
    mids = np.column_stack([r_s, z_s])
    return mids, mids / R_e, seg   # normals = (r, z) / R_e

def _kh_farfield(p_arc, dpdn_arc, mids, normals, seg, k):
    """
    Axisymmetric K-H integral with rigid-baffle image.
    Returns complex kernel amplitude I(θ) for each angle in `theta`.
    Absolute Pa = |I(θ)| * KH_PREFACTOR.
    """
    r_s, z_s = mids[:, 0], mids[:, 1]
    n_r, n_z = normals[:, 0], normals[:, 1]
    p_ff = np.zeros(N_theta, dtype=np.complex128)
    for i, th in enumerate(theta):
        sth = np.sin(th);  cth = np.cos(th)
        beta = k * r_s * sth;  psi = k * z_s * cth
        J0b  = j0(beta);       J1b = j1(beta)
        p_kernel  = -k * (cth * n_z * J0b * np.sin(psi)
                          + sth * n_r * J1b * np.cos(psi))
        dn_kernel = J0b * np.cos(psi)
        p_ff[i]   = np.sum(r_s * (p_arc * p_kernel - dpdn_arc * dn_kernel) * seg)
    return p_ff

def _eval_on_arc(p_r, p_i, gr, gi, R_e):
    mids, normals, seg = _arc_geometry(R_e)
    pts3d  = _pad_3d(mids)
    cells  = _find_cells(pts3d)
    p_arc  = _eval_scalar(p_r, pts3d, cells) + 1j * _eval_scalar(p_i, pts3d, cells)
    grad   = _eval_vector(gr, pts3d, cells)  + 1j * _eval_vector(gi, pts3d, cells)
    dpdn   = grad[:, 0] * normals[:, 0] + grad[:, 1] * normals[:, 1]
    return p_arc, dpdn, mids, normals, seg

# ── Analytical far-field baffled-piston pressure at r = R_obs ─────────────────
def analytical_farfield(k, theta_arr):
    """
    |p(r, θ)| = ρ c v₀ k a² / (2 r) · D(θ),
    D(θ) = |2 J₁(ka sinθ) / (ka sinθ)|,   D(0) = 1 (L'Hôpital).
    """
    sin_th = np.sin(theta_arr)
    D      = np.ones_like(theta_arr)
    mask   = np.abs(sin_th) > 1e-12
    arg    = k * a * sin_th[mask]
    D[mask] = np.abs(2.0 * j1(arg) / arg)
    return rho * c * v0 * k * a**2 / (2.0 * R_obs) * D

# ── Compute FEA and analytical patterns for each frequency ────────────────────
results = {}   # freq → (p_fem [Pa], p_ana [Pa])

print(f"Computing far-field patterns for {len(plot_freqs)} frequencies …")
for f in plot_freqs:
    k = 2.0 * np.pi * f / c
    print(f"  f = {f:>8.1f} Hz  (ka = {k * a:.2f})")

    p_file = OUTPUT_DIR / f"pressure_f{int(round(f))}.npy"
    if not p_file.exists():
        raise FileNotFoundError(f"{p_file} not found — run solve.py first.")
    p_complex = np.load(p_file)

    p_r = fem.Function(V);  p_r.x.array[:] = p_complex.real;  p_r.x.scatter_forward()
    p_i = fem.Function(V);  p_i.x.array[:] = p_complex.imag;  p_i.x.scatter_forward()

    gr = fem.Function(Vg)
    gi = fem.Function(Vg)
    gr.interpolate(fem.Expression(ufl.grad(p_r), _grad_ip))
    gi.interpolate(fem.Expression(ufl.grad(p_i), _grad_ip))

    p_ff = np.zeros(N_theta, dtype=np.complex128)
    for R_e in extract_radii:
        p_arc, dpdn_arc, mids, normals, seg = _eval_on_arc(p_r, p_i, gr, gi, R_e)
        p_ff += _kh_farfield(p_arc, dpdn_arc, mids, normals, seg, k)
    p_ff /= len(extract_radii)

    results[f] = (np.abs(p_ff) * KH_PREFACTOR, analytical_farfield(k, theta))

print("Done.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_db(p_abs):
    return 20.0 * np.log10(np.maximum(p_abs, 1e-12) / P_REF)

# ── Polar plot ────────────────────────────────────────────────────────────────
# Layout:  on-axis (θ = 0) points upward; θ increases clockwise toward the baffle
# plane (θ = 90°).  The pattern is mirrored to show both halves of the baffled
# half-space (left/right).  The radial axis is clipped to DYN_RANGE_DB below the
# global SPL peak.

# Determine global SPL ceiling for the radial axis
all_spl = np.concatenate([
    [to_db(p_fem), to_db(p_ana)]
    for p_fem, p_ana in results.values()
])
spl_ceil = float(np.max(all_spl))
spl_floor = spl_ceil - DYN_RANGE_DB

def _r(spl):
    """Map SPL [dB] to polar radius, clamped to [0, DYN_RANGE_DB]."""
    return np.clip(spl - spl_floor, 0.0, DYN_RANGE_DB)

n_f    = len(plot_freqs)
colors = [cm.tab10(i / max(n_f - 1, 1)) for i in range(n_f)]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")   # on-axis (θ = 0) at top
ax.set_theta_direction(-1)        # clockwise (θ increases toward baffle plane)
ax.set_thetamin(-90)
ax.set_thetamax(90)

# The solved pattern covers θ ∈ [0, π/2].  Mirror to the left side for the plot
# (negative angles in matplotlib polar = left of vertical).
th_right = theta          # θ : 0 → +π/2  (right half)
th_left  = -theta[1:]     # θ : 0 → -π/2  (left half, skip duplicate at 0)

for i, f in enumerate(plot_freqs):
    p_fem, p_ana = results[f]
    color = colors[i]

    r_ana = _r(to_db(p_ana))
    r_fem = _r(to_db(p_fem))

    # Analytical — solid
    ax.plot(th_right, r_ana,     color=color, ls="-",  lw=1.8)
    ax.plot(th_left,  r_ana[1:], color=color, ls="-",  lw=1.8)
    # FEA — dashed
    ax.plot(th_right, r_fem,     color=color, ls="--", lw=1.8)
    ax.plot(th_left,  r_fem[1:], color=color, ls="--", lw=1.8)

# ── Radial axis ticks (every 10 dB) ──────────────────────────────────────────
r_ticks = np.arange(0, DYN_RANGE_DB + 1, 10)
ax.set_rticks(r_ticks)
ax.set_yticklabels(
    [f"{spl_floor + r:.0f} dBSPL" for r in r_ticks],
    fontsize=7, color="grey",
)
ax.set_rlim(0, DYN_RANGE_DB)
ax.tick_params(axis="y", labelrotation=45)

# ── Angular axis labels ────────────────────────────────────────────────────────
ax.set_xticks(np.deg2rad([-90, -60, -30, 0, 30, 60, 90]))
ax.set_xticklabels(["90°", "60°", "30°", "0°", "30°", "60°", "90°"])

# ── Legend ─────────────────────────────────────────────────────────────────────
# Section 1: one coloured patch per frequency
freq_handles = [
    Line2D([0], [0], color=colors[i], lw=2.5, label=f"{f:.0f} Hz")
    for i, f in enumerate(plot_freqs)
]
# Section 2: line style legend
style_handles = [
    Line2D([0], [0], color="k", ls="-",  lw=2, label="Analytical"),
    Line2D([0], [0], color="k", ls="--", lw=2, label="FEA (K-H)"),
]
leg = ax.legend(
    handles=freq_handles + style_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=min(n_f + 2, 6),
    fontsize=9,
    framealpha=0.85,
)

ax.set_title(
    f"Far-field SPL  (r = {R_obs:.0f} m,  $p_{{\\rm ref}}$ = {P_REF:.0f} Pa)\n"
    r"$\bf{—}$ Analytical  $\bf{-\,-}$ FEA (K–H)",
    pad=18, fontsize=11,
)
ax.grid(True, linestyle=":", alpha=0.5)

fig.tight_layout()
out_path = OUTPUT_DIR / "farfield_polar.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved: {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 2D Pressure Field Heatmap — FEA vs Analytical at FHEATMAP
# Domain: fluid quarter-circle (0 ≤ r, z; r²+z² ≤ R_fluid²)
# ══════════════════════════════════════════════════════════════════════════════

# ── Rayleigh (King) integral — exact near/far-field baffled-piston reference ──
# p(x) = (iωρv₀ / 2π) ∫_disk e^{ikd}/d dS,  d = |x − x_s|
# Uses Gauss–Legendre quadrature in s (radial) and uniform nodes in φ (azimuthal).
_Ns_hm,  _Nphi_hm = 64, 128
_s_nd_hm, _s_wt_hm = np.polynomial.legendre.leggauss(_Ns_hm)
_s_nd_hm  = 0.5 * a * (_s_nd_hm + 1.0)       # map [-1,1] → [0, a]
_s_wt_hm  = 0.5 * a * _s_wt_hm
_phi_hm   = np.linspace(0.0, 2.0 * np.pi, _Nphi_hm, endpoint=False)
_dphi_hm  = 2.0 * np.pi / _Nphi_hm
_SHM, _PHIHM   = np.meshgrid(_s_nd_hm, _phi_hm, indexing="ij")  # (Ns, Nphi)
_SW_HM         = (_s_wt_hm[:, None] * _dphi_hm) * _SHM           # area weight s·ds·dφ
_s_cosphi_hm   = _SHM * np.cos(_PHIHM)
_s_sq_hm       = _SHM ** 2

def _rayleigh_chunk(r_pts, z_pts, k):
    """Exact baffled-piston |p| [Pa] for a batch of (r, z) field points."""
    omega = k * c
    r_o   = np.asarray(r_pts)[:, None, None]   # (Np, 1, 1)
    z_o   = np.asarray(z_pts)[:, None, None]
    d     = np.sqrt(r_o**2 + _s_sq_hm[None]
                    - 2.0 * r_o * _s_cosphi_hm[None] + z_o**2)
    p     = ((1j * omega * rho * v0 / (2.0 * np.pi))
             * (np.exp(1j * k * d) / d * _SW_HM[None]).sum(axis=(1, 2)))
    return np.abs(p)

# ── Resolve FHEATMAP to nearest solved frequency ──────────────────────────────
_f_hm   = all_freqs[int(np.argmin(np.abs(_freqs_arr - FHEATMAP)))]
k_hm    = 2.0 * np.pi * _f_hm / c
print(f"\nComputing pressure heatmap at f = {_f_hm/1e3:.1f} kHz  (ka = {k_hm*a:.2f}) …")

# ── Load FEA pressure for FHEATMAP ────────────────────────────────────────────
_p_hm_file = OUTPUT_DIR / f"pressure_f{int(round(_f_hm))}.npy"
if not _p_hm_file.exists():
    raise FileNotFoundError(f"{_p_hm_file} not found — run solve.py first.")
_p_hm_cplx = np.load(_p_hm_file)

# ── Polar grid shared by both tiles ───────────────────────────────────────────
# Every grid point satisfies ρ ≤ R_fluid by construction, so the outermost
# column lands exactly on the arc — clean edge, no clipping required.
#
#   ρ : radial distance from origin  [0, R_fluid]
#   φ : polar angle in (r,z) plane   [0, π/2]
#   r = ρ cos φ  (cylindrical radius)
#   z = ρ sin φ  (axial coordinate)
#
# FEA: fine polar grid, scipy LinearNDInterpolator from scattered DOF values.
# Analytical: coarser polar grid keeps Rayleigh integral fast (all pts inside).
NR_FEA, NPHI_FEA     = 300, 300   # FEA polar grid  (scipy interp — essentially free)
NR_ANA, NPHI_ANA     = 80,  80    # Analytical polar grid  (~6 400 Rayleigh evals)

_phi_fea = np.linspace(0.0, np.pi / 2.0, NPHI_FEA)
_rho_fea = np.linspace(0.0, R_fluid,     NR_FEA)
_RHO_fea, _PHI_fea = np.meshgrid(_rho_fea, _phi_fea, indexing="ij")  # (NR, NPHI)
_R_fea = _RHO_fea * np.cos(_PHI_fea)
_Z_fea = _RHO_fea * np.sin(_PHI_fea)

_phi_ana = np.linspace(0.0, np.pi / 2.0, NPHI_ANA)
_rho_ana = np.linspace(0.0, R_fluid,     NR_ANA)
_RHO_ana, _PHI_ana = np.meshgrid(_rho_ana, _phi_ana, indexing="ij")  # (NR, NPHI)
_R_ana = _RHO_ana * np.cos(_PHI_ana)
_Z_ana = _RHO_ana * np.sin(_PHI_ana)

# ── FEA: scatter DOF values → polar grid via scipy barycentric interpolation ──
_dof_coords = V.tabulate_dof_coordinates()          # (N_dofs, 3)
_r_dof = _dof_coords[:, 0]
_z_dof = _dof_coords[:, 1]
_fluid_dof = (_r_dof**2 + _z_dof**2) <= R_fluid**2

_interp_fea = LinearNDInterpolator(
    np.column_stack([_r_dof[_fluid_dof], _z_dof[_fluid_dof]]),
    np.abs(_p_hm_cplx[_fluid_dof]),
    fill_value=0.0,
)
P_fea_polar = _interp_fea(_R_fea, _Z_fea)          # (NR_FEA, NPHI_FEA)

# ── Analytical: Rayleigh integral on the coarse polar grid ────────────────────
# All NR_ANA × NPHI_ANA points are inside the domain; no masking needed.
_r_ana_flat = np.maximum(_R_ana.ravel(), 1e-9)      # floor avoids r=0 in quadrature
_z_ana_flat = np.maximum(_Z_ana.ravel(), 1e-9)      # floor avoids z=0 singularity
_CHUNK = 500
_p_ana_flat = np.zeros(NR_ANA * NPHI_ANA)
for _i0 in range(0, NR_ANA * NPHI_ANA, _CHUNK):
    _sl = slice(_i0, _i0 + _CHUNK)
    _p_ana_flat[_sl] = _rayleigh_chunk(_r_ana_flat[_sl], _z_ana_flat[_sl], k_hm)
_P_ana_coarse = _p_ana_flat.reshape(NR_ANA, NPHI_ANA)

# Upsample to the fine display grid by interpolating in (ρ, φ) space.
# RegularGridInterpolator works on the structured coarse grid without the
# duplicate-point degeneracy that would occur if we interpolated in (r, z).
_rgi_ana = RegularGridInterpolator(
    (_rho_ana, _phi_ana), _P_ana_coarse,
    method="linear", bounds_error=False, fill_value=0.0,
)
P_ana_polar = _rgi_ana(
    np.column_stack([_RHO_fea.ravel(), _PHI_fea.ravel()])
).reshape(NR_FEA, NPHI_FEA)

print("  Heatmap evaluation complete.")

# ── Convert to dB, normalised per-dataset to its own peak ─────────────────────
_HM_DYN = 40.0   # dynamic range [dB]

def _norm_db_arr(p_arr, dyn):
    """dB relative to array peak, clipped to [-dyn, 0]."""
    peak = np.nanmax(p_arr)
    db   = 20.0 * np.log10(np.maximum(p_arr, 1e-30) / max(peak, 1e-30))
    return np.clip(db, -dyn, 0.0)

P_fea_db  = _norm_db_arr(P_fea_polar, _HM_DYN)
P_ana_db  = _norm_db_arr(P_ana_polar, _HM_DYN)

# ── 1 × 2 heatmap figure — pcolormesh on curvilinear polar coordinates ─────────
# pcolormesh(X, Y, C) with 2D X, Y renders the curvilinear quadrilateral mesh
# directly.  The outer edge of the mesh IS the arc ρ = R_fluid, so there is
# nothing to clip — the boundary is geometrically exact.
_mm      = R_fluid * 1e3
_arc_ang = np.linspace(0.0, np.pi / 2.0, 400)

fig_hm, axes_hm = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

im = axes_hm[0].pcolormesh(
    _R_fea * 1e3, _Z_fea * 1e3, P_fea_db,
    shading="gouraud", cmap=HEATMAP_CMAP, vmin=-_HM_DYN, vmax=0.0,
)
axes_hm[1].pcolormesh(
    _R_fea * 1e3, _Z_fea * 1e3, P_ana_db,
    shading="gouraud", cmap=HEATMAP_CMAP, vmin=-_HM_DYN, vmax=0.0,
)

for ax, label in zip(
    axes_hm,
    [f"FEA  ({_f_hm/1e3:.0f} kHz)", f"Analytical  ({_f_hm/1e3:.0f} kHz)"],
):
    ax.plot(
        _mm * np.cos(_arc_ang), _mm * np.sin(_arc_ang),
        color="white", lw=0.8, alpha=0.5,
    )
    ax.set_xlim(0.0, _mm)
    ax.set_ylim(0.0, _mm)
    ax.set_aspect("equal")
    ax.set_xlabel("r  [mm]", fontsize=10)
    ax.set_ylabel("z  [mm]", fontsize=10)
    ax.set_title(label, fontsize=11)

fig_hm.colorbar(
    im, ax=axes_hm,
    label=f"SPL  (dB re peak,  {_HM_DYN:.0f} dB range)",
    shrink=0.75,
)
fig_hm.suptitle(
    f"Pressure field — fluid domain  "
    f"(f = {_f_hm/1e3:.0f} kHz,  ka = {k_hm*a:.2f})",
    fontsize=12,
)

_hm_path = OUTPUT_DIR / f"pressure_heatmap_{int(round(_f_hm))}Hz.png"
fig_hm.savefig(_hm_path, dpi=150, bbox_inches="tight")
print(f"Heatmap figure saved: {_hm_path}")
plt.show(block=True)
