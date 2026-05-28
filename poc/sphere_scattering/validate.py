import numpy as np
import bempp_cl.api as bempp
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt
import os

# ── Parameters ────────────────────────────────────────────────────────────────
MESH_FILE  = os.path.join(os.path.dirname(__file__), "outputs", "sphere.msh")
PHI_FILE   = os.path.join(os.path.dirname(__file__), "outputs", "phi.npz")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RADIUS     = 0.025
R_EVAL     = 0.1
C          = 343.0
N_TERMS    = 40

# Evaluation points: 36-point ring in xz-plane at r=0.1 m
ANGLES      = np.linspace(0, 2 * np.pi, 36, endpoint=False)
EVAL_POINTS = np.array([
    R_EVAL * np.cos(ANGLES),
    np.zeros(len(ANGLES)),
    R_EVAL * np.sin(ANGLES)
])

# ── Load surface solutions ────────────────────────────────────────────────────
data       = np.load(PHI_FILE)
freqs      = data["frequencies"]
phi_coeffs = data["phi_coeffs"]

grid  = bempp.import_grid(MESH_FILE)
space = bempp.function_space(grid, "DP", 0)
print(f"Loaded {len(freqs)} frequencies, {phi_coeffs.shape[1]} DOFs")

# ── Analytical solution ───────────────────────────────────────────────────────
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

def analytical_scattered(freq, r, theta, a=RADIUS, N=N_TERMS):
    k         = 2 * np.pi * freq / C
    cos_theta = np.cos(theta)
    p_sc      = 0j
    for n in range(N + 1):
        A_n   = -spherical_jn(n, k*a, derivative=True) / hankel_deriv(n, k*a)
        p_sc += (2*n + 1) * (1j**n) * A_n * hankel(n, k*r) * legendre_poly(n, cos_theta)
    return p_sc

p_analytical_fwd = np.array([analytical_scattered(f, R_EVAL, 0.0)   for f in freqs])
p_analytical_bck = np.array([analytical_scattered(f, R_EVAL, np.pi) for f in freqs])

# ── Post-solve evaluation ─────────────────────────────────────────────────────
p_sc_bem = np.zeros((len(freqs), len(ANGLES)), dtype=complex)

for i, freq in enumerate(freqs):
    k   = 2 * np.pi * freq / C
    phi = bempp.GridFunction(space, coefficients=phi_coeffs[i])

    dlp_pot          = bempp.operators.potential.helmholtz.double_layer(space, EVAL_POINTS, k)
    p_sc_bem[i, :]   = (dlp_pot * phi).flatten()

p_bem_fwd = p_sc_bem[:, 0]
p_bem_bck = p_sc_bem[:, 18]

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'Freq':>8}  {'|BEM fwd|':>10}  {'|Ana fwd|':>10}  {'ratio fwd':>10}  "
      f"{'|BEM bck|':>10}  {'|Ana bck|':>10}  {'ratio bck':>10}")
print("-" * 80)
for f, bf, af, bb, ab in zip(freqs, p_bem_fwd, p_analytical_fwd, p_bem_bck, p_analytical_bck):
    print(f"{f:8.1f}  {abs(bf):10.6f}  {abs(af):10.6f}  {abs(bf)/abs(af):10.4f}  "
          f"{abs(bb):10.6f}  {abs(ab):10.6f}  {abs(bb)/abs(ab):10.4f}")

# ── Plot: forward and back scatter ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for ax, p_bem, p_ana, label in [
    (axes[0, 0], p_bem_fwd, p_analytical_fwd, "Forward (θ=0°)"),
    (axes[0, 1], p_bem_bck, p_analytical_bck, "Back (θ=180°)")
]:
    ax.semilogx(freqs, np.abs(p_bem), label="BEM",        lw=2)
    ax.semilogx(freqs, np.abs(p_ana), label="Analytical", lw=1.5, ls="--")
    ax.set_title(label)
    ax.set_ylabel("|p_sc| (Pa)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

for ax, p_bem, p_ana in [
    (axes[1, 0], p_bem_fwd, p_analytical_fwd),
    (axes[1, 1], p_bem_bck, p_analytical_bck)
]:
    err = 20 * np.log10(np.abs(p_bem - p_ana) / (np.abs(p_ana) + 1e-12) + 1e-12)
    ax.semilogx(freqs, err, color="crimson", lw=1.5)
    ax.axhline(-40, color="gray", ls="--", lw=1, label="-40 dB")
    ax.set_ylabel("Error (dB re analytical)")
    ax.set_xlabel("Frequency (Hz)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

plt.suptitle("Rigid sphere scattering — r=0.1 m", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "validation_fwd_bck.png"), dpi=150)

# ── Plot: polar directivity ───────────────────────────────────────────────────
spot_indices = [np.argmin(np.abs(freqs - f)) for f in [500, 2000, 5000, 10000]]
fig2, axes2  = plt.subplots(1, len(spot_indices), figsize=(14, 4),
                             subplot_kw={"projection": "polar"})

for ax, fi in zip(axes2, spot_indices):
    freq       = freqs[fi]
    p_bem_ring = np.abs(p_sc_bem[fi, :])
    p_ana_ring = np.array([abs(analytical_scattered(freq, R_EVAL, th)) for th in ANGLES])
    ax.plot(ANGLES, p_bem_ring, lw=2,            label="BEM")
    ax.plot(ANGLES, p_ana_ring, lw=1.5, ls="--", label="Ana")
    ax.set_title(f"{freq:.0f} Hz\nka={2*np.pi*freq/C*RADIUS:.2f}", fontsize=9)
    ax.legend(fontsize=7)

plt.suptitle("Scattered pressure directivity |p_sc| at r=0.1 m", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "directivity_polar.png"), dpi=150)
plt.show()
print("\nDone.")