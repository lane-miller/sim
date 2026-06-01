"""
validate.py — Compare FEM vs mode-matching for expansion chamber POC.

Produces two figures:
  1. TL vs frequency (FEM, mode-matching, plane-wave formula)
  2. On-axis |p(z)| at PLOT_FREQS (FEM vs mode-matching, 4-panel tile)
"""

import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from mode_matching import (PLOT_FREQS, compute_tl, compute_overlap,
                           compute_onaxis_pressure, solve_for_freq,
                           _pw_formula)

# mode_matching.py locks TkAgg at import time.  Force Agg so that the
# script is reliably non-interactive and saves PNG files without requiring
# a working display.  Remove this line (or switch to "TkAgg") if you want
# interactive windows.
plt.switch_backend("Agg")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ── Compute mode-matching data ─────────────────────────────────────────────────

print("Computing overlap integrals...")
C_ov = compute_overlap(cfg.INLET_W, cfg.INLET_H,
                       cfg.CHAMBER_W, cfg.CHAMBER_H, 8, 8)

print("Computing mode-matching TL sweep...")
TL_mm, _ = compute_tl(cfg.FREQS, C_ov, 8, 8)
TL_pw = _pw_formula(cfg.FREQS)

print("Computing mode-matching on-axis pressure at PLOT_FREQS...")
mm_onaxis = {}
mm_tl_at_plot = {}
for fq in PLOT_FREQS:
    coeffs = solve_for_freq(fq, C_ov, 8, 8)
    z_mm, p_mm = compute_onaxis_pressure(coeffs, fq, 8, 8)
    mm_onaxis[fq] = (z_mm, p_mm)
    # TL at this specific frequency via power-flux
    from mode_matching import _power_flux
    B1, A2, B2, A3, kz1, kz2, kz3, norm_s, norm_c = coeffs
    omega_fq = 2 * np.pi * fq
    W_inc = (cfg.INLET_W * cfg.INLET_H) / (2.0 * cfg.RHO * cfg.C)
    W_trans = _power_flux(A3, kz3, norm_s, omega_fq)
    mm_tl_at_plot[fq] = 10.0 * np.log10(W_inc / W_trans) if W_trans > 0 else np.inf

# ── Load FEM data ─────────────────────────────────────────────────────────────

print("Loading FEM results...")
data = np.load(os.path.join(OUTPUT_DIR, "fem_results.npz"), allow_pickle=False)
freqs_fem = data["freqs"]
TL_fem    = data["TL_fem"]

fem_onaxis = {}
for fq in PLOT_FREQS:
    key_z = f"onaxis_z_{int(fq)}"
    key_p = f"onaxis_p_{int(fq)}"
    fem_onaxis[fq] = (data[key_z], data[key_p])

# ── TL comparison summary ─────────────────────────────────────────────────────

# Interpolate TL_mm onto the FEM frequency grid for a fair comparison
TL_mm_interp = np.interp(freqs_fem, cfg.FREQS, TL_mm)
max_diff = np.max(np.abs(TL_fem - TL_mm_interp))
print(f"\nMax |TL_fem - TL_mm| across all frequencies: {max_diff:.2f} dB")

# ── Plot 1: TL vs frequency ───────────────────────────────────────────────────

print("\nGenerating TL comparison plot...")
fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.semilogx(freqs_fem, TL_fem,  "b-o",  ms=4, lw=1.5, label="FEM")
ax1.semilogx(cfg.FREQS, TL_mm,   "k-",         lw=1.5, label="Mode-matching")
ax1.semilogx(cfg.FREQS, TL_pw,   "r--",         lw=1.5, label="Plane-wave formula")

ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Transmission Loss (dB)")
ax1.set_title("Expansion Chamber TL — FEM vs Mode Matching")
ax1.legend()
ax1.grid(True, which="both", alpha=0.4)
fig1.tight_layout()

out_tl = os.path.join(OUTPUT_DIR, "validate_tl.png")
fig1.savefig(out_tl, dpi=150)
print(f"  Saved {out_tl}")
plt.show(block=False)

# ── Plot 2: On-axis pressure (2×2 tile) ──────────────────────────────────────

print("Generating on-axis pressure comparison plot...")
fig2, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

L1_mm = cfg.INLET_L * 1e3
L2_mm = cfg.CHAMBER_L * 1e3

for idx, (fq, ax) in enumerate(zip(PLOT_FREQS, axes.ravel())):
    z_fem, p_fem = fem_onaxis[fq]
    z_mm_arr,  p_mm_arr  = mm_onaxis[fq]

    tl_fem_fq = float(np.interp(fq, freqs_fem, TL_fem))
    tl_mm_fq  = mm_tl_at_plot[fq]

    ax.plot(z_fem,    np.abs(p_fem),    "b-",  lw=1.5,
            label="FEM" if idx == 0 else "_")
    ax.plot(z_mm_arr, np.abs(p_mm_arr), "k--", lw=1.5,
            label="Mode-matching" if idx == 0 else "_")

    ax.axvline(L1_mm,        color="gray", lw=1.0, ls="--")
    ax.axvline(L1_mm + L2_mm, color="gray", lw=1.0, ls="--")

    ax.set_ylabel("|p| (Pa)")
    ax.set_title(f"f = {fq:.0f} Hz  (TL_fem={tl_fem_fq:.1f}, TL_mm={tl_mm_fq:.1f} dB)")
    ax.grid(True, alpha=0.4)

    if idx == 0:
        ax.legend(loc="upper right")

# Shared axis labels
for ax in axes[1, :]:
    ax.set_xlabel("z (mm)")
for ax in axes[:, 0]:
    ax.set_ylabel("|p| (Pa)")

fig2.suptitle("On-axis Pressure — FEM vs Mode Matching", fontsize=13)
fig2.tight_layout()

out_onaxis = os.path.join(OUTPUT_DIR, "validate_onaxis.png")
fig2.savefig(out_onaxis, dpi=150)
print(f"  Saved {out_onaxis}")
plt.show(block=False)

# ── Final summary ─────────────────────────────────────────────────────────────

print("\nValidation complete.")
print(f"  Max TL difference (FEM vs MM): {max_diff:.2f} dB")
print("  Figures saved to outputs/")

plt.show(block=True)
