"""
HRTF POC — Interpolated HRTF Frequency-Response Validation
===========================================================
Four canonical directions: dense interpolated curve + sparse solve markers.

Usage:
    python interp/validate_interp_fr.py
"""

import sys
from pathlib import Path

HRTF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HRTF_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, hilbert

from common.sofa import (
    canonical_directions,
    interp_hrtf,
    interp_scale,
    load_sofa_pair,
    to_db,
)
from config import EVAL_RADIUS_M, HRTF_INTERP, OUTPUT_DIR

CANONICAL_AZ = [0.0, 90.0, 180.0, 270.0]
DIRECTION_NAMES = ["Front", "Left (ipsilateral)", "Back", "Right (contralateral)"]
FREQ_PLOT_MIN = 100.0
FREQ_PLOT_MAX = 15000.0


def envelope_peak(hrir):
    """Sample index of the analytic-envelope peak."""
    return int(np.argmax(np.abs(hilbert(hrir))))


def circshift_align(ref, sig):
    """Circshift sig to maximize correlation with ref."""
    n = len(ref)
    lag = int(np.argmax(correlate(ref, sig, mode="full")) - (n - 1))
    return np.roll(sig, lag), lag


def hrtf_to_hrir(hrtf, source_freqs, n_samples, sample_rate):
    """Interpolate complex HRTF onto the SOFA rFFT grid and invert."""
    grid_freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    hrtf_on_grid = interp_hrtf(hrtf, source_freqs, grid_freqs)
    return np.fft.irfft(hrtf_on_grid, n=n_samples, axis=1)


data = np.load(HRTF_INTERP)
target_freqs = data["target_freqs"]
expansion_freqs = data["expansion_freqs"]
hrtf_dense = data["hrtf_dense"]    # (n_freq, n_dir)
hrtf_sparse = data["hrtf_sparse"]  # (n_exp, n_dir)
horiz_az = data["azimuths"]

sofa_meas, sofa_sim, sofa_fs = load_sofa_pair()
sofa_pos = sofa_meas.SourcePosition
sofa_ir = sofa_meas.Data_IR
sofa_ir_sim = sofa_sim.Data_IR

sofa_dir_idx, _, _ = canonical_directions(sofa_pos, CANONICAL_AZ, EVAL_RADIUS_M)

# Map canonical directions to indices in stored horizontal azimuth array
dir_indices = []
for az in CANONICAL_AZ:
    az_diff = np.abs((horiz_az - az + 180.0) % 360.0 - 180.0)
    dir_indices.append(int(np.argmin(az_diff)))

sofa_freqs = np.fft.rfftfreq(sofa_ir.shape[2], d=1.0 / sofa_fs)
hrtf_meas_full = np.fft.rfft(sofa_ir[sofa_dir_idx, 0, :], axis=1)
hrtf_sim_full = np.fft.rfft(sofa_ir_sim[sofa_dir_idx, 0, :], axis=1)

# BEM: (n_dir, n_freq)
hrtf_bem_dense = hrtf_dense[:, dir_indices].T
hrtf_bem_sparse = hrtf_sparse[:, dir_indices].T

hrtf_meas_at_f = interp_hrtf(hrtf_meas_full, sofa_freqs, target_freqs)
hrtf_sim_at_f = interp_hrtf(hrtf_sim_full, sofa_freqs, target_freqs)

meas_scale_at_f = np.ones(len(target_freqs))
sim_scale_at_f = np.ones(len(target_freqs))

for i in range(len(target_freqs)):
    ref_bem = np.mean(np.abs(hrtf_bem_dense[:, i]))
    ref_meas = np.mean(np.abs(hrtf_meas_at_f[:, i]))
    ref_sim = np.mean(np.abs(hrtf_sim_at_f[:, i]))
    meas_scale_at_f[i] = ref_bem / ref_meas
    sim_scale_at_f[i] = ref_bem / ref_sim
    hrtf_meas_at_f[:, i] *= meas_scale_at_f[i]
    hrtf_sim_at_f[:, i] *= sim_scale_at_f[i]

sim_scale_full = interp_scale(sim_scale_at_f, target_freqs, sofa_freqs)

hrtf_sim_plot = hrtf_sim_full * sim_scale_full[np.newaxis, :]

plot_mask = (sofa_freqs >= FREQ_PLOT_MIN) & (sofa_freqs <= FREQ_PLOT_MAX)
sofa_freqs_plot = sofa_freqs[plot_mask]

hrtf_sim_db = to_db(hrtf_sim_plot[:, plot_mask])
hrtf_bem_dense_db = to_db(hrtf_bem_dense)
hrtf_bem_sparse_db = to_db(hrtf_bem_sparse)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
fig.suptitle(
    "HRTF Frequency Response — Interpolated BEM vs Mesh2HRTF",
    fontsize=14, y=0.98,
)

for ax, dir_idx, name, az in zip(
    axes.ravel(), range(len(CANONICAL_AZ)), DIRECTION_NAMES, CANONICAL_AZ
):
    ax.semilogx(sofa_freqs_plot, hrtf_sim_db[dir_idx, :], "b--", linewidth=1.0, label="Mesh2HRTF")
    ax.plot(
        target_freqs, hrtf_bem_dense_db[dir_idx, :],
        "r-", linewidth=1.0, label="Bempp dense (interp)",
    )
    ax.plot(
        expansion_freqs, hrtf_bem_sparse_db[dir_idx, :],
        "ro", markersize=4, label="Bempp sparse (solved)",
    )
    ax.set_title(f"{name}  (az={az:.0f}°)")
    ax.set_xlim(FREQ_PLOT_MIN, FREQ_PLOT_MAX)
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which="both", alpha=0.3)

axes[0, 0].set_ylabel("HRTF magnitude (dB)")
axes[1, 0].set_ylabel("HRTF magnitude (dB)")

all_db = np.concatenate([
    hrtf_sim_db.ravel(),
    hrtf_bem_dense_db.ravel(), hrtf_bem_sparse_db.ravel(),
])
y_min = np.floor(all_db.min() / 5.0) * 5.0
y_max = np.ceil(all_db.max() / 5.0) * 5.0
for ax in axes.ravel():
    ax.set_ylim(y_min, y_max)

axes[0, 0].legend(loc="best", fontsize=7)
plt.tight_layout()
out_path = OUTPUT_DIR / "hrtf_validation_interp_fr.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")

n_samples = sofa_ir.shape[2]
hrir_sim = sofa_ir_sim[sofa_dir_idx, 0, :].astype(float)
hrir_bem = hrtf_to_hrir(hrtf_bem_dense, target_freqs, n_samples, sofa_fs)

hrir_sim_phase = np.zeros_like(hrir_sim)
hrir_bem_phase = np.zeros_like(hrir_bem)
print("\nBulk delay removal (circshift envelope peak → sample 0):")
for d, name in enumerate(DIRECTION_NAMES):
    hrir_bem_d, rel_lag = circshift_align(hrir_sim[d], hrir_bem[d])
    bulk = envelope_peak(hrir_sim[d])
    hrir_sim_phase[d] = np.roll(hrir_sim[d], -bulk)
    hrir_bem_phase[d] = np.roll(hrir_bem_d, -bulk)
    print(
        f"  {name:24s}: bulk circshift -{bulk:4d}  "
        f"bem rel circshift {rel_lag:+4d} ({1e3 * rel_lag / sofa_fs:+.3f} ms)"
    )

hrtf_sim_aligned = np.fft.rfft(hrir_sim_phase, axis=1)
hrtf_bem_aligned = np.fft.rfft(hrir_bem_phase, axis=1)

phase_sim = np.degrees(np.unwrap(np.angle(hrtf_sim_aligned), axis=1))
phase_bem = np.degrees(np.unwrap(np.angle(hrtf_bem_aligned), axis=1))

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
fig.suptitle(
    "HRTF Phase — Bulk-Delay Removed BEM vs Mesh2HRTF",
    fontsize=14, y=0.98,
)

for ax, dir_idx, name, az in zip(
    axes.ravel(), range(len(CANONICAL_AZ)), DIRECTION_NAMES, CANONICAL_AZ
):
    ax.semilogx(
        sofa_freqs_plot, phase_sim[dir_idx, plot_mask],
        "b--", linewidth=1.0, label="Mesh2HRTF",
    )
    ax.semilogx(
        sofa_freqs_plot, phase_bem[dir_idx, plot_mask],
        "r-", linewidth=1.0, label="Bempp dense (interp)",
    )
    ax.set_title(f"{name}  (az={az:.0f}°)")
    ax.set_xlim(FREQ_PLOT_MIN, FREQ_PLOT_MAX)
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, which="both", alpha=0.3)

axes[0, 0].set_ylabel("Unwrapped phase (deg)")
axes[1, 0].set_ylabel("Unwrapped phase (deg)")

all_phase = np.concatenate([
    phase_sim[:, plot_mask].ravel(),
    phase_bem[:, plot_mask].ravel(),
])
y_pad = max(45.0, 0.05 * (all_phase.max() - all_phase.min()))
y_min = np.floor((all_phase.min() - y_pad) / 45.0) * 45.0
y_max = np.ceil((all_phase.max() + y_pad) / 45.0) * 45.0
for ax in axes.ravel():
    ax.set_ylim(y_min, y_max)

axes[0, 0].legend(loc="best", fontsize=7)
plt.tight_layout()
phase_out_path = OUTPUT_DIR / "hrtf_validation_interp_fr_phase.png"
plt.savefig(phase_out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {phase_out_path}")

# Interpolation error at expansion frequencies
print("\nInterpolation error at expansion points (dB, dense − sparse):")
for d, name in enumerate(DIRECTION_NAMES):
    errs = []
    for i, f in enumerate(expansion_freqs):
        dense_i = int(np.where(target_freqs == f)[0][0])
        errs.append(hrtf_bem_dense_db[d, dense_i] - hrtf_bem_sparse_db[d, i])
    print(f"  {name:24s}: mean={np.mean(errs):+.3f} dB  max={np.max(np.abs(errs)):.3f} dB")
