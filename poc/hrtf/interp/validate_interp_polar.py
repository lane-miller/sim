"""
HRTF POC — Interpolated HRTF Validation (polar)
=================================================
Compare dense interpolated HRTFs to FABIAN SOFA reference.

Usage:
    python interp/validate_interp_polar.py
"""

import sys
from pathlib import Path

HRTF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HRTF_ROOT))

import numpy as np

from common.plot import plot_polar_horizontal
from common.sofa import interp_hrtf, load_sofa_pair, normalize_sofa_to_bem, to_db
from config import HRTF_INTERP, OUTPUT_DIR

# Frequencies to plot (subset of target_freqs; empty = first 4 dense points)
PLOT_FREQS = [1000.0, 3000.0, 6000.0, 10000.0]

data = np.load(HRTF_INTERP)
target_freqs = data["target_freqs"]
expansion_freqs = data["expansion_freqs"]
hrtf_dense = data["hrtf_dense"]      # (n_freq, n_dir)
hrtf_sparse = data["hrtf_sparse"]    # (n_exp, n_dir)
horiz_az = data["azimuths"]

if PLOT_FREQS:
    plot_freqs = np.array([f for f in PLOT_FREQS if f in target_freqs])
    if len(plot_freqs) == 0:
        raise ValueError(f"PLOT_FREQS {PLOT_FREQS} not in target grid")
    plot_indices = [int(np.where(target_freqs == f)[0][0]) for f in plot_freqs]
else:
    plot_indices = list(range(min(4, len(target_freqs))))
    plot_freqs = target_freqs[plot_indices]

print(f"Target grid: {len(target_freqs)} frequencies")
print(f"Expansion points: {len(expansion_freqs)}")
print(f"Plot frequencies: {plot_freqs}")

sofa_meas, sofa_sim, sofa_fs = load_sofa_pair()
sofa_pos = sofa_meas.SourcePosition
sofa_ir = sofa_meas.Data_IR
sofa_ir_sim = sofa_sim.Data_IR

horiz_mask = np.abs(sofa_pos[:, 1]) < 0.5
horiz_sofa_idx = np.where(horiz_mask)[0]
sort_order = np.argsort(sofa_pos[horiz_sofa_idx, 0])
horiz_sofa_idx = horiz_sofa_idx[sort_order]

sofa_freqs = np.fft.rfftfreq(sofa_ir.shape[2], d=1.0 / sofa_fs)
hrtf_meas_horiz = np.fft.rfft(sofa_ir[horiz_sofa_idx, 0, :], axis=1)
hrtf_sim_horiz = np.fft.rfft(sofa_ir_sim[horiz_sofa_idx, 0, :], axis=1)

hrtf_meas_at_f = interp_hrtf(hrtf_meas_horiz, sofa_freqs, plot_freqs)
hrtf_sim_at_f = interp_hrtf(hrtf_sim_horiz, sofa_freqs, plot_freqs)

# (n_dir, n_freq) for plotting helpers
hrtf_bem = hrtf_dense[plot_indices, :].T
hrtf_sparse_at_plot = []
for f in plot_freqs:
    if f in expansion_freqs:
        exp_i = int(np.where(expansion_freqs == f)[0][0])
        hrtf_sparse_at_plot.append(hrtf_sparse[exp_i, :])
    else:
        hrtf_sparse_at_plot.append(None)

hrtf_meas_at_f, hrtf_sim_at_f = normalize_sofa_to_bem(
    hrtf_meas_at_f, hrtf_sim_at_f, hrtf_bem,
)

hrtf_bem_db = to_db(hrtf_bem)
hrtf_meas_db = to_db(hrtf_meas_at_f)
hrtf_sim_db = to_db(hrtf_sim_at_f)

plot_polar_horizontal(
    plot_freqs, horiz_az, hrtf_meas_db, hrtf_sim_db, hrtf_bem_db,
    "HRTF Horizontal Plane — Interpolated BEM vs FABIAN Reference",
    OUTPUT_DIR / "hrtf_validation_interp.png",
)

print("\nError summary (dB RMSE, dense interp vs measured):")
for i, freq in enumerate(plot_freqs):
    rmse = np.sqrt(np.mean((hrtf_bem_db[:, i] - hrtf_meas_db[:, i]) ** 2))
    rmse_sim = np.sqrt(np.mean((hrtf_sim_db[:, i] - hrtf_meas_db[:, i]) ** 2))
    print(f"  {freq:6.0f} Hz:  Interp={rmse:.2f} dB   Mesh2HRTF={rmse_sim:.2f} dB")

print("\nInterpolation self-check at expansion points on plot grid (dB max |Δ|):")
for i, freq in enumerate(plot_freqs):
    sparse = hrtf_sparse_at_plot[i]
    if sparse is None:
        continue
    dense_db = hrtf_bem_db[:, i]
    sparse_db = to_db(sparse)
    max_err = np.max(np.abs(dense_db - sparse_db))
    print(f"  {freq:6.0f} Hz:  max |dense − sparse| = {max_err:.3f} dB")
