import numpy as np
import bempp_cl.api as bempp
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# ── Parameters ────────────────────────────────────────────────────────────────
MESH_FILE  = os.path.join(os.path.dirname(__file__), "outputs", "sphere.msh")
PHI_FILE   = os.path.join(os.path.dirname(__file__), "outputs", "phi.npz")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
FREQ       = 10000.0
C          = 343.0
RADIUS     = 0.025
GRID_N     = 256
EXTENT     = 0.125
N_TERMS    = 40

# ── Load surface solution ─────────────────────────────────────────────────────
data       = np.load(PHI_FILE)
freqs      = data["frequencies"]
phi_coeffs = data["phi_coeffs"]

fi   = np.argmin(np.abs(freqs - FREQ))
freq = freqs[fi]
k    = 2 * np.pi * freq / C
print(f"Using frequency {freq:.1f} Hz (requested {FREQ:.1f} Hz), k={k:.3f}")

grid  = bempp.import_grid(MESH_FILE)
space = bempp.function_space(grid, "DP", 0)
phi   = bempp.GridFunction(space, coefficients=phi_coeffs[fi])

# ── Build evaluation grid in xz-plane (y=0) ──────────────────────────────────
x = np.linspace(-EXTENT, EXTENT, GRID_N)
z = np.linspace(-EXTENT, EXTENT, GRID_N)
XX, ZZ = np.meshgrid(x, z)
YY     = np.zeros_like(XX)

grid_points = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()])
inside      = np.sqrt(grid_points[0]**2 + grid_points[2]**2) < RADIUS
ext_idx     = np.where(~inside)[0]
ext_points  = grid_points[:, ext_idx]

# ── BEM scattered field ──────────────────────────────────────────────────────
print(f"Evaluating BEM at {ext_points.shape[1]} exterior points...")
dlp_pot  = bempp.operators.potential.helmholtz.double_layer(space, ext_points, k)
p_sc_bem = (dlp_pot * phi).flatten()

# ── Analytical scattered field ────────────────────────────────────────────────
def hankel(n, x):
    return spherical_jn(n, x) + 1j * spherical_yn(n, x)

def hankel_deriv(n, x):
    return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)

def legendre_poly(n, x):
    if n == 0: return np.ones_like(x)
    if n == 1: return np.asarray(x, dtype=float)
    P_prev, P_curr = np.ones_like(x), np.asarray(x, dtype=float)
    for m in range(2, n + 1):
        P_next = ((2*m - 1) * P_curr * x - (m - 1) * P_prev) / m
        P_prev, P_curr = P_curr, P_next
    return P_curr

print(f"Computing analytical solution ({N_TERMS} terms)...")
r_ext     = np.sqrt(ext_points[0]**2 + ext_points[2]**2)
cos_theta = ext_points[0] / r_ext  # angle from +x (incident direction)

p_sc_ana = np.zeros(len(r_ext), dtype=complex)
for n in range(N_TERMS + 1):
    A_n   = -spherical_jn(n, k * RADIUS, derivative=True) / hankel_deriv(n, k * RADIUS)
    p_sc_ana += (2*n + 1) * (1j**n) * A_n * hankel(n, k * r_ext) * legendre_poly(n, cos_theta)

# ── Assemble grids ────────────────────────────────────────────────────────────
def to_grid(values):
    g = np.full(GRID_N * GRID_N, np.nan, dtype=complex)
    g[ext_idx] = values
    return g.reshape(GRID_N, GRID_N)

p_sc_bem_grid = to_grid(p_sc_bem)
p_sc_ana_grid = to_grid(p_sc_ana)

p_inc_ext       = np.exp(1j * k * ext_points[0, :])
p_tot_bem_grid  = to_grid(p_inc_ext + p_sc_bem)
p_tot_ana_grid  = to_grid(p_inc_ext + p_sc_ana)

vmax_sc  = max(np.nanmax(np.abs(np.real(p_sc_bem_grid))),
               np.nanmax(np.abs(np.real(p_sc_ana_grid))))
vmax_tot = max(np.nanmax(np.abs(np.real(p_tot_bem_grid))),
               np.nanmax(np.abs(np.real(p_tot_ana_grid))))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

panels = [
    (axes[0, 0], np.real(p_sc_bem_grid),  vmax_sc,  f"BEM  —  Re(p_sc)  —  {freq:.0f} Hz"),
    (axes[0, 1], np.real(p_sc_ana_grid),  vmax_sc,  f"Analytical  —  Re(p_sc)  —  {freq:.0f} Hz"),
    (axes[1, 0], np.real(p_tot_bem_grid), vmax_tot, f"BEM  —  Re(p_tot)  —  {freq:.0f} Hz"),
    (axes[1, 1], np.real(p_tot_ana_grid), vmax_tot, f"Analytical  —  Re(p_tot)  —  {freq:.0f} Hz"),
]

for ax, field, vm, title in panels:
    im = ax.imshow(field, extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
                   origin="lower", cmap="RdBu_r",
                   vmin=-vm, vmax=vm, aspect="equal")
    ax.add_patch(Circle((0, 0), RADIUS, fc="white", ec="k", lw=1.5, zorder=2))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Pa")

plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, f"pressure_map_{int(freq)}Hz.png")
plt.savefig(outpath, dpi=150)
print(f"Saved {outpath}")
plt.show()
