"""Shared plotting helpers for HRTF validation."""

import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt


def plot_polar_horizontal(
    frequencies,
    horiz_az,
    hrtf_meas_db,
    hrtf_sim_db,
    hrtf_bem_db,
    title,
    out_path,
):
    """Polar horizontal-plane comparison, one subplot per frequency."""
    n_freq = len(frequencies)
    ncols = ceil(sqrt(n_freq))
    nrows = ceil(n_freq / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        subplot_kw={"projection": "polar"},
    )
    fig.suptitle(title, fontsize=14, y=0.98)

    axes_flat = np.atleast_1d(axes).ravel()
    az_plot = np.deg2rad(horiz_az)

    for i, freq in enumerate(frequencies):
        ax = axes_flat[i]
        ax.plot(az_plot, hrtf_meas_db[:, i], "k-", linewidth=0.8, alpha=0.6, label="Measured")
        ax.plot(az_plot, hrtf_sim_db[:, i], "b--", linewidth=0.8, alpha=0.6, label="Mesh2HRTF")
        ax.plot(az_plot, hrtf_bem_db[:, i], "r-", linewidth=1.2, label="Bempp (this)")
        ax.set_title(f"{freq:.0f} Hz", pad=12)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        if i == 0:
            ax.legend(loc="lower right", fontsize=8)

    for j in range(n_freq, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


def plot_frequency_response(
    sofa_freqs_plot,
    hrtf_meas_db,
    hrtf_sim_db,
    bem_freqs,
    hrtf_bem_db,
    direction_names,
    canonical_az,
    freq_plot_min,
    freq_plot_max,
    title,
    out_path,
    bem_marker="o",
    bem_linestyle="-",
):
    """2×2 frequency-response comparison, one subplot per direction."""
    n_dirs = len(direction_names)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    fig.suptitle(title, fontsize=14, y=0.98)

    for ax, dir_idx, name, az in zip(
        axes.ravel(), range(n_dirs), direction_names, canonical_az
    ):
        ax.semilogx(
            sofa_freqs_plot, hrtf_meas_db[dir_idx, :],
            "k-", linewidth=1.0, label="Measured",
        )
        ax.semilogx(
            sofa_freqs_plot, hrtf_sim_db[dir_idx, :],
            "b--", linewidth=1.0, label="Mesh2HRTF",
        )
        ax.plot(
            bem_freqs, hrtf_bem_db[dir_idx, :],
            f"r{bem_linestyle}{bem_marker}", linewidth=1.0,
            markersize=5, label="Bempp (this)",
        )
        ax.set_title(f"{name}  (az={az:.0f}°)")
        ax.set_xlim(freq_plot_min, freq_plot_max)
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, which="both", alpha=0.3)

    axes[0, 0].set_ylabel("HRTF magnitude (dB)")
    axes[1, 0].set_ylabel("HRTF magnitude (dB)")

    all_db = np.concatenate([hrtf_meas_db.ravel(), hrtf_sim_db.ravel(), hrtf_bem_db.ravel()])
    y_min = np.floor(all_db.min() / 5.0) * 5.0
    y_max = np.ceil(all_db.max() / 5.0) * 5.0
    for ax in axes.ravel():
        ax.set_ylim(y_min, y_max)

    axes[0, 0].legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")
