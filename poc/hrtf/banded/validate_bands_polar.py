"""
HRTF POC — Banded Validation (polar)
=====================================
Evaluate banded BEM surface solutions at far-field points, compare to SOFA.

Usage:
    python banded/validate_bands_polar.py
"""

import sys
from pathlib import Path

HRTF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HRTF_ROOT))

import numpy as np
from collections import defaultdict

from common.bem import evaluate_hrtf, load_mesh_space
from common.plot import plot_polar_horizontal
from common.sofa import (
    horizontal_plane_from_sofa,
    interp_hrtf,
    load_sofa_pair,
    normalize_sofa_to_bem,
    to_db,
)
from config import C_AIR, EVAL_RADIUS_M, OUTPUT_DIR, PHI_BANDED

# Frequencies to plot (must be present in solve output; empty = all solved freqs)
PLOT_FREQS = [1000.0, 5000.0, 10000.0]

data = np.load(PHI_BANDED)
frequencies = data["frequencies"]
phi_coeffs = data["phi_coeffs"]
n_dofs = data["n_dofs"]
meshes = data["meshes"].astype(str)
source_m = data["source_m"]

if PLOT_FREQS:
    plot_freqs = np.array([f for f in PLOT_FREQS if f in frequencies])
    if len(plot_freqs) == 0:
        raise ValueError(f"PLOT_FREQS {PLOT_FREQS} not found in solved frequencies {frequencies}")
    freq_indices = [int(np.where(frequencies == f)[0][0]) for f in plot_freqs]
else:
    plot_freqs = frequencies
    freq_indices = list(range(len(frequencies)))

print(f"BEM frequencies: {frequencies}")
print(f"Plot frequencies: {plot_freqs}")

sofa_meas, sofa_sim, sofa_fs = load_sofa_pair()
sofa_pos = sofa_meas.SourcePosition
sofa_ir = sofa_meas.Data_IR
sofa_ir_sim = sofa_sim.Data_IR

horiz_idx, horiz_az, eval_points, _ = horizontal_plane_from_sofa(
    sofa_pos, EVAL_RADIUS_M,
)
print(f"Horizontal plane: {len(horiz_idx)} directions")

sofa_freqs = np.fft.rfftfreq(sofa_ir.shape[2], d=1.0 / sofa_fs)
hrtf_meas_horiz = np.fft.rfft(sofa_ir[horiz_idx, 0, :], axis=1)
hrtf_sim_horiz = np.fft.rfft(sofa_ir_sim[horiz_idx, 0, :], axis=1)

hrtf_meas_at_f = interp_hrtf(hrtf_meas_horiz, sofa_freqs, plot_freqs)
hrtf_sim_at_f = interp_hrtf(hrtf_sim_horiz, sofa_freqs, plot_freqs)

hrtf_bem = np.zeros((len(horiz_idx), len(plot_freqs)), dtype=complex)
mesh_groups = defaultdict(list)
for i, m in enumerate(meshes):
    mesh_groups[m].append(i)

grid_cache = {}
for mesh_path, all_freq_indices in mesh_groups.items():
    if mesh_path not in grid_cache:
        _, _, space = load_mesh_space(mesh_path)
        grid_cache[mesh_path] = space
        print(f"Built grid for {Path(mesh_path).name} ({space.global_dof_count} DOFs)")
    else:
        space = grid_cache[mesh_path]

    for plot_i, freq_i in enumerate(freq_indices):
        if freq_i not in all_freq_indices:
            continue
        freq = frequencies[freq_i]
        hrtf_bem[:, plot_i] = evaluate_hrtf(
            space, phi_coeffs[freq_i, :n_dofs[freq_i]], freq, source_m, eval_points,
        )
        print(f"  f={freq:.0f} Hz ({Path(mesh_path).name}): "
              f"|HRTF| = [{np.abs(hrtf_bem[:, plot_i]).min():.3f}, "
              f"{np.abs(hrtf_bem[:, plot_i]).max():.3f}]")

hrtf_meas_at_f, hrtf_sim_at_f = normalize_sofa_to_bem(
    hrtf_meas_at_f, hrtf_sim_at_f, hrtf_bem,
)

hrtf_bem_db = to_db(hrtf_bem)
hrtf_meas_db = to_db(hrtf_meas_at_f)
hrtf_sim_db = to_db(hrtf_sim_at_f)

plot_polar_horizontal(
    plot_freqs, horiz_az, hrtf_meas_db, hrtf_sim_db, hrtf_bem_db,
    "HRTF Horizontal Plane — Banded BEM vs FABIAN Reference",
    OUTPUT_DIR / "hrtf_validation_banded.png",
)

print("\nError summary (dB RMSE, BEM vs measured, horizontal plane):")
for i, freq in enumerate(plot_freqs):
    rmse = np.sqrt(np.mean((hrtf_bem_db[:, i] - hrtf_meas_db[:, i]) ** 2))
    rmse_sim = np.sqrt(np.mean((hrtf_sim_db[:, i] - hrtf_meas_db[:, i]) ** 2))
    print(f"  {freq:6.0f} Hz:  Bempp={rmse:.2f} dB   Mesh2HRTF={rmse_sim:.2f} dB")
