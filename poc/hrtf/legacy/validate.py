"""
HRTF POC — Validation (legacy)
================================
Single graded mesh BEM vs FABIAN SOFA reference.

Usage:
    python legacy/validate.py
"""

import sys
from pathlib import Path

HRTF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HRTF_ROOT))

import bempp_cl.api as bempp
import numpy as np

from common.bem import evaluate_hrtf, load_mesh_space
from common.plot import plot_polar_horizontal
from common.sofa import (
    horizontal_plane_from_sofa,
    interp_hrtf,
    load_sofa_pair,
    normalize_sofa_to_bem,
    to_db,
)
from config import C_AIR, EVAL_RADIUS_M, MESH_GRADED, OUTPUT_DIR

PHI_FILE = OUTPUT_DIR / "phi_graded.npz"

data = np.load(PHI_FILE)
frequencies = data["frequencies"]
phi_coeffs = data["phi_coeffs"]
source_m = data["source_m"]

print(f"BEM frequencies: {frequencies}")
print(f"Source (m): ({source_m[0]:.4f}, {source_m[1]:.4f}, {source_m[2]:.4f})")

_, _, space = load_mesh_space(MESH_GRADED)

sofa_meas, sofa_sim, sofa_fs = load_sofa_pair()
sofa_pos = sofa_meas.SourcePosition
sofa_ir = sofa_meas.Data_IR
sofa_ir_sim = sofa_sim.Data_IR

print(f"SOFA: {sofa_pos.shape[0]} directions, {sofa_ir.shape[2]} samples, {sofa_fs} Hz")

horiz_idx, horiz_az, eval_points, eval_radius = horizontal_plane_from_sofa(
    sofa_pos, EVAL_RADIUS_M,
)
print(f"Horizontal plane: {len(horiz_idx)} directions, "
      f"az=[{horiz_az[0]:.1f}°, {horiz_az[-1]:.1f}°]")

sofa_freqs = np.fft.rfftfreq(sofa_ir.shape[2], d=1.0 / sofa_fs)
hrtf_meas_horiz = np.fft.rfft(sofa_ir[horiz_idx, 0, :], axis=1)
hrtf_sim_horiz = np.fft.rfft(sofa_ir_sim[horiz_idx, 0, :], axis=1)

hrtf_meas_at_f = interp_hrtf(hrtf_meas_horiz, sofa_freqs, frequencies)
hrtf_sim_at_f = interp_hrtf(hrtf_sim_horiz, sofa_freqs, frequencies)

hrtf_bem = np.zeros((len(horiz_idx), len(frequencies)), dtype=complex)
for i, freq in enumerate(frequencies):
    hrtf_bem[:, i] = evaluate_hrtf(
        space, phi_coeffs[i, :], freq, source_m, eval_points,
    )
    print(f"  f={freq:.0f} Hz: |HRTF| range = [{np.abs(hrtf_bem[:, i]).min():.3f}, "
          f"{np.abs(hrtf_bem[:, i]).max():.3f}]")

hrtf_meas_at_f, hrtf_sim_at_f = normalize_sofa_to_bem(
    hrtf_meas_at_f, hrtf_sim_at_f, hrtf_bem,
)

hrtf_bem_db = to_db(hrtf_bem)
hrtf_meas_db = to_db(hrtf_meas_at_f)
hrtf_sim_db = to_db(hrtf_sim_at_f)

plot_polar_horizontal(
    frequencies, horiz_az, hrtf_meas_db, hrtf_sim_db, hrtf_bem_db,
    "HRTF Horizontal Plane — BEM vs FABIAN Reference",
    OUTPUT_DIR / "hrtf_validation_horizontal.png",
)

print("\nError summary (dB RMSE, BEM vs measured, horizontal plane):")
for i, freq in enumerate(frequencies):
    rmse = np.sqrt(np.mean((hrtf_bem_db[:, i] - hrtf_meas_db[:, i]) ** 2))
    rmse_sim = np.sqrt(np.mean((hrtf_sim_db[:, i] - hrtf_meas_db[:, i]) ** 2))
    print(f"  {freq:6.0f} Hz:  Bempp={rmse:.2f} dB   Mesh2HRTF={rmse_sim:.2f} dB")
