"""
HRTF POC — Validation
======================
Evaluate BEM surface solution at far-field points, compute HRTF,
compare to FABIAN SOFA reference.

Steps:
1. Load surface coefficients from phi.npz
2. Load SOFA reference HRIRs, FFT to HRTFs
3. Reconstruct BEM GridFunction, evaluate total field at far-field sphere
4. HRTF = p_total(x) / p_inc(x) at each evaluation direction
5. Plot horizontal-plane comparison (BEM vs measured vs simulated)
"""

import numpy as np
import bempp_cl.api as bempp
import trimesh
import sofar as sf
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MESH_STL = OUTPUT_DIR / "FABIAN_6k_HATO0_graded.stl"
PHI_FILE = OUTPUT_DIR / "phi_graded.npz"

FABIAN_ROOT = Path(
    "/Volumes/LPM02 storage/Datasets/Audio/HRTF/FABIAN/FABIAN_HRTF_DATABASE_v4"
)
HRIR_MEASURED = FABIAN_ROOT / "1 HRIRs" / "SOFA" / "FABIAN_HRIR_measured_HATO_0.sofa"
HRIR_SIMULATED = FABIAN_ROOT / "1 HRIRs" / "SOFA" / "FABIAN_HRIR_modeled_HATO_0.sofa"

C_AIR = 343.18
EVAL_RADIUS_M = 1.5  # meters — must match SOFA measurement radius

# ---------------------------------------------------------------------------
# Load BEM results
# ---------------------------------------------------------------------------
data = np.load(PHI_FILE)
frequencies = data["frequencies"]
phi_coeffs = data["phi_coeffs"]
source_m = data["source_m"]

print(f"BEM frequencies: {frequencies}")
print(f"Source (m): ({source_m[0]:.4f}, {source_m[1]:.4f}, {source_m[2]:.4f})")

# ---------------------------------------------------------------------------
# Load mesh, rebuild grid and space
# ---------------------------------------------------------------------------
tm = trimesh.load(str(MESH_STL), force="mesh")
vertices = (tm.vertices / 1000.0).astype(np.float64).T
elements = tm.faces.astype(np.int32).T

grid = bempp.Grid(vertices, elements)
space = bempp.function_space(grid, "P", 1)

# ---------------------------------------------------------------------------
# Load SOFA reference
# ---------------------------------------------------------------------------
sofa_meas = sf.read_sofa(str(HRIR_MEASURED))
sofa_sim = sf.read_sofa(str(HRIR_SIMULATED))

sofa_pos = sofa_meas.SourcePosition        # (N, 3): [az°, el°, r_m]
sofa_ir = sofa_meas.Data_IR                  # (N, 2, L): [directions, ears, samples]
sofa_fs = float(sofa_meas.Data_SamplingRate)

sofa_ir_sim = sofa_sim.Data_IR

print(f"SOFA: {sofa_pos.shape[0]} directions, {sofa_ir.shape[2]} samples, {sofa_fs} Hz")
print(f"SOFA radius: {np.unique(sofa_pos[:, 2])} m")

# Verify measurement radius matches our eval radius
sofa_radius = sofa_pos[0, 2]
if abs(sofa_radius - EVAL_RADIUS_M) > 0.01:
    print(f"WARNING: SOFA radius {sofa_radius} != eval radius {EVAL_RADIUS_M}")
    EVAL_RADIUS_M = sofa_radius
    print(f"  Using SOFA radius: {EVAL_RADIUS_M} m")

# ---------------------------------------------------------------------------
# Select horizontal plane directions from SOFA
# ---------------------------------------------------------------------------
horiz_mask = np.abs(sofa_pos[:, 1]) < 0.5  # elevation ≈ 0°
horiz_idx = np.where(horiz_mask)[0]
horiz_az = sofa_pos[horiz_idx, 0]  # degrees

# Sort by azimuth
sort_order = np.argsort(horiz_az)
horiz_idx = horiz_idx[sort_order]
horiz_az = horiz_az[sort_order]

print(f"Horizontal plane: {len(horiz_idx)} directions, "
      f"az=[{horiz_az[0]:.1f}°, {horiz_az[-1]:.1f}°]")

# Convert SOFA (az, el, r) to Cartesian evaluation points (meters)
# SOFA: az=0°→front(+X), az=90°→left(+Y), el=0°→horizontal
# This matches our mesh frame: X=front, Y=left, Z=up
az_rad = np.deg2rad(horiz_az)
el_rad = np.zeros_like(az_rad)
eval_points = np.zeros((3, len(horiz_idx)))  # (3, N_eval) for Bempp
eval_points[0, :] = EVAL_RADIUS_M * np.cos(el_rad) * np.cos(az_rad)
eval_points[1, :] = EVAL_RADIUS_M * np.cos(el_rad) * np.sin(az_rad)
eval_points[2, :] = EVAL_RADIUS_M * np.sin(el_rad)

# ---------------------------------------------------------------------------
# Compute SOFA reference HRTFs at BEM frequencies (left ear = index 0)
# ---------------------------------------------------------------------------
sofa_freqs = np.fft.rfftfreq(sofa_ir.shape[2], d=1.0 / sofa_fs)

# HRTFs for horizontal directions, left ear
hrtf_meas_horiz = np.fft.rfft(sofa_ir[horiz_idx, 0, :], axis=1)   # (N_horiz, N_fft)
hrtf_sim_horiz = np.fft.rfft(sofa_ir_sim[horiz_idx, 0, :], axis=1)

# Interpolate to our BEM frequencies
def interp_hrtf(hrtf_full, sofa_freqs, target_freqs):
    """Interpolate complex HRTF from SOFA FFT frequencies to target frequencies."""
    mag = np.abs(hrtf_full)
    phase = np.unwrap(np.angle(hrtf_full), axis=1)
    result = np.zeros((hrtf_full.shape[0], len(target_freqs)), dtype=complex)
    for i in range(hrtf_full.shape[0]):
        mag_interp = np.interp(target_freqs, sofa_freqs, mag[i, :])
        phase_interp = np.interp(target_freqs, sofa_freqs, phase[i, :])
        result[i, :] = mag_interp * np.exp(1j * phase_interp)
    return result

hrtf_meas_at_f = interp_hrtf(hrtf_meas_horiz, sofa_freqs, frequencies)
hrtf_sim_at_f = interp_hrtf(hrtf_sim_horiz, sofa_freqs, frequencies)

# ---------------------------------------------------------------------------
# Evaluate BEM HRTF at each frequency
# ---------------------------------------------------------------------------
x0 = source_m
hrtf_bem = np.zeros((len(horiz_idx), len(frequencies)), dtype=complex)

for i, freq in enumerate(frequencies):
    k = 2 * np.pi * freq / C_AIR

    # Reconstruct surface solution as GridFunction
    phi_fun = bempp.GridFunction(space, coefficients=phi_coeffs[i, :])

    # Evaluate scattered field at far-field points via double-layer potential
    dlp_pot = bempp.operators.potential.helmholtz.double_layer(
        space, eval_points, k
    )
    p_scattered = dlp_pot @ phi_fun

    # Incident field at evaluation points
    dx = eval_points[0, :] - x0[0]
    dy = eval_points[1, :] - x0[1]
    dz = eval_points[2, :] - x0[2]
    r_eval = np.sqrt(dx**2 + dy**2 + dz**2)
    p_inc_eval = np.exp(1j * k * r_eval) / (4 * np.pi * r_eval)

    # Total field and HRTF
    p_total = p_inc_eval + p_scattered.ravel()
    hrtf_bem[:, i] = p_total / p_inc_eval

    print(f"  f={freq:.0f} Hz: |HRTF| range = [{np.abs(hrtf_bem[:, i]).min():.3f}, "
          f"{np.abs(hrtf_bem[:, i]).max():.3f}]")

# ---------------------------------------------------------------------------
# Normalize — SOFA HRTFs are relative to free-field at head center;
# BEM reciprocal HRTF is p_total/p_inc which should equal 1 in free field.
# Both should be comparable directly in dB.
# ---------------------------------------------------------------------------
hrtf_bem_db = 20 * np.log10(np.abs(hrtf_bem) + 1e-30)

# Normalize SOFA: reference level = mean magnitude across all directions
# (SOFA HRIRs may have arbitrary gain from measurement chain)
for i in range(len(frequencies)):
    ref_meas = np.mean(np.abs(hrtf_meas_at_f[:, i]))
    ref_sim = np.mean(np.abs(hrtf_sim_at_f[:, i]))
    ref_bem = np.mean(np.abs(hrtf_bem[:, i]))

    hrtf_meas_at_f[:, i] *= ref_bem / ref_meas
    hrtf_sim_at_f[:, i] *= ref_bem / ref_sim

hrtf_meas_db = 20 * np.log10(np.abs(hrtf_meas_at_f) + 1e-30)
hrtf_sim_db = 20 * np.log10(np.abs(hrtf_sim_at_f) + 1e-30)

# ---------------------------------------------------------------------------
# Plot — horizontal plane HRTF comparison at each frequency
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={"projection": "polar"})
fig.suptitle("HRTF Horizontal Plane — BEM vs FABIAN Reference", fontsize=14, y=0.98)

for i, (ax, freq) in enumerate(zip(axes.ravel(), frequencies)):
    az_plot = np.deg2rad(horiz_az)

    ax.plot(az_plot, hrtf_meas_db[:, i], "k-", linewidth=0.8, alpha=0.6, label="Measured")
    ax.plot(az_plot, hrtf_sim_db[:, i], "b--", linewidth=0.8, alpha=0.6, label="Mesh2HRTF")
    ax.plot(az_plot, hrtf_bem_db[:, i], "r-", linewidth=1.2, label="Bempp (this)")

    ax.set_title(f"{freq:.0f} Hz", pad=12)
    ax.set_theta_zero_location("N")  # 0° = front = top of polar plot
    ax.set_theta_direction(-1)        # clockwise
    if i == 0:
        ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hrtf_validation_horizontal.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_DIR / 'hrtf_validation_horizontal.png'}")

# ---------------------------------------------------------------------------
# Error summary
# ---------------------------------------------------------------------------
print("\nError summary (dB RMSE, BEM vs measured, horizontal plane):")
for i, freq in enumerate(frequencies):
    rmse = np.sqrt(np.mean((hrtf_bem_db[:, i] - hrtf_meas_db[:, i]) ** 2))
    rmse_sim = np.sqrt(np.mean((hrtf_sim_db[:, i] - hrtf_meas_db[:, i]) ** 2))
    print(f"  {freq:6.0f} Hz:  Bempp={rmse:.2f} dB   Mesh2HRTF={rmse_sim:.2f} dB")


# Quick pattern diagnostic
for i, freq in enumerate(frequencies):
    h = np.abs(hrtf_bem[:, i])
    az_max = horiz_az[np.argmax(h)]
    az_min = horiz_az[np.argmin(h)]
    ild = 20 * np.log10(h.max() / h.min())
    print(f"  {freq:.0f} Hz: max at az={az_max:.0f}°  min at az={az_min:.0f}°  "
          f"ILD={ild:.1f} dB")
    
    # Same for measured reference
    h_m = np.abs(hrtf_meas_at_f[:, i])
    az_max_m = horiz_az[np.argmax(h_m)]
    az_min_m = horiz_az[np.argmin(h_m)]
    print(f"         ref: max at az={az_max_m:.0f}°  min at az={az_min_m:.0f}°")