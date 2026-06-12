"""
HRTF POC — Banded Frequency-Response Validation
================================================
Four canonical horizontal directions, magnitude vs frequency.

Usage:
    python banded/validate_bands_fr.py
"""

import sys
from pathlib import Path

HRTF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HRTF_ROOT))

import numpy as np
from collections import defaultdict

from common.bem import evaluate_hrtf, load_mesh_space
from common.plot import plot_frequency_response
from common.sofa import (
    canonical_directions,
    interp_hrtf,
    interp_scale,
    load_sofa_pair,
    to_db,
)
from config import EVAL_RADIUS_M, OUTPUT_DIR, PHI_BANDED

CANONICAL_AZ = [0.0, 90.0, 180.0, 270.0]
DIRECTION_NAMES = ["Front", "Left (ipsilateral)", "Back", "Right (contralateral)"]
FREQ_PLOT_MIN = 100.0
FREQ_PLOT_MAX = 15000.0

data = np.load(PHI_BANDED)
frequencies = data["frequencies"]
phi_coeffs = data["phi_coeffs"]
n_dofs = data["n_dofs"]
meshes = data["meshes"].astype(str)
source_m = data["source_m"]

print(f"BEM frequencies: {frequencies}")

sofa_meas, sofa_sim, sofa_fs = load_sofa_pair()
sofa_pos = sofa_meas.SourcePosition
sofa_ir = sofa_meas.Data_IR
sofa_ir_sim = sofa_sim.Data_IR

sofa_dir_idx, eval_points, _ = canonical_directions(
    sofa_pos, CANONICAL_AZ, EVAL_RADIUS_M,
)
for i, (name, az) in enumerate(zip(DIRECTION_NAMES, CANONICAL_AZ)):
    print(f"  {name}: target az={az:.0f}° → SOFA az={sofa_pos[sofa_dir_idx[i], 0]:.1f}°")

sofa_freqs = np.fft.rfftfreq(sofa_ir.shape[2], d=1.0 / sofa_fs)
hrtf_meas_full = np.fft.rfft(sofa_ir[sofa_dir_idx, 0, :], axis=1)
hrtf_sim_full = np.fft.rfft(sofa_ir_sim[sofa_dir_idx, 0, :], axis=1)

n_dirs = len(CANONICAL_AZ)
hrtf_bem = np.zeros((n_dirs, len(frequencies)), dtype=complex)

mesh_groups = defaultdict(list)
for i, m in enumerate(meshes):
    mesh_groups[m].append(i)

grid_cache = {}
for mesh_path, freq_indices in mesh_groups.items():
    if mesh_path not in grid_cache:
        _, _, space = load_mesh_space(mesh_path)
        grid_cache[mesh_path] = space
        print(f"Built grid for {Path(mesh_path).name} ({space.global_dof_count} DOFs)")
    else:
        space = grid_cache[mesh_path]

    for i in freq_indices:
        freq = frequencies[i]
        hrtf_bem[:, i] = evaluate_hrtf(
            space, phi_coeffs[i, :n_dofs[i]], freq, source_m, eval_points,
        )
        print(f"  f={freq:.0f} Hz ({Path(mesh_path).name})")

hrtf_meas_at_f = interp_hrtf(hrtf_meas_full, sofa_freqs, frequencies)
hrtf_sim_at_f = interp_hrtf(hrtf_sim_full, sofa_freqs, frequencies)

meas_scale_at_f = np.ones(len(frequencies))
sim_scale_at_f = np.ones(len(frequencies))

for i in range(len(frequencies)):
    ref_bem = np.mean(np.abs(hrtf_bem[:, i]))
    ref_meas = np.mean(np.abs(hrtf_meas_at_f[:, i]))
    ref_sim = np.mean(np.abs(hrtf_sim_at_f[:, i]))
    meas_scale_at_f[i] = ref_bem / ref_meas
    sim_scale_at_f[i] = ref_bem / ref_sim
    hrtf_meas_at_f[:, i] *= meas_scale_at_f[i]
    hrtf_sim_at_f[:, i] *= sim_scale_at_f[i]

meas_scale_full = interp_scale(meas_scale_at_f, frequencies, sofa_freqs)
sim_scale_full = interp_scale(sim_scale_at_f, frequencies, sofa_freqs)

hrtf_meas_plot = hrtf_meas_full * meas_scale_full[np.newaxis, :]
hrtf_sim_plot = hrtf_sim_full * sim_scale_full[np.newaxis, :]

plot_mask = (sofa_freqs >= FREQ_PLOT_MIN) & (sofa_freqs <= FREQ_PLOT_MAX)
sofa_freqs_plot = sofa_freqs[plot_mask]

hrtf_meas_db = to_db(hrtf_meas_plot[:, plot_mask])
hrtf_sim_db = to_db(hrtf_sim_plot[:, plot_mask])
hrtf_bem_db = to_db(hrtf_bem)
hrtf_meas_at_f_db = to_db(hrtf_meas_at_f)

plot_frequency_response(
    sofa_freqs_plot, hrtf_meas_db, hrtf_sim_db,
    frequencies, hrtf_bem_db,
    DIRECTION_NAMES, CANONICAL_AZ,
    FREQ_PLOT_MIN, FREQ_PLOT_MAX,
    "HRTF Frequency Response — Banded BEM vs FABIAN Reference",
    OUTPUT_DIR / "hrtf_validation_banded_fr.png",
)

print("\nError (dB, BEM vs measured) at each frequency and direction:")
for i, freq in enumerate(frequencies):
    print(f"  {freq:6.0f} Hz:")
    for d, name in enumerate(DIRECTION_NAMES):
        err = hrtf_bem_db[d, i] - hrtf_meas_at_f_db[d, i]
        print(f"    {name:24s}: {err:+.2f} dB")
