"""Compare baffled vs. unflanged circular-piston normalized radiation impedance."""

import os

import matplotlib.pyplot as plt
import numpy as np

from analytical import zrad_norm_analytical, zrad_norm_unflanged

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOT_FILE = os.path.join(RESULTS_DIR, "baffled_vs_unflanged.png")

# Silva Padé fit is accurate for ka < 3 (J. Sound Vib. 322, 2009, Section 5).
KA_LO = 0.05
KA_HI_UNFLANGED = 3.0
KA_HI_BAFFLED = 5.0


def make_plot(plot_file=PLOT_FILE):
    ka_baffled = np.geomspace(KA_LO, KA_HI_BAFFLED, 300)
    ka_unflanged = np.geomspace(KA_LO, KA_HI_UNFLANGED, 300)
    z_baffled = zrad_norm_analytical(ka_baffled)
    z_unflanged = zrad_norm_unflanged(ka_unflanged)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    axes[0].plot(ka_baffled, np.real(z_baffled), "k-", label="Infinite baffle")
    axes[0].plot(ka_unflanged, np.real(z_unflanged), "C1--", label="Unflanged")
    axes[0].set_ylabel(r"Re{$Z_\mathrm{rad}/(\rho c)$}")
    axes[0].set_title("Radiation Resistance")
    axes[0].set_xlabel(r"$ka$")
    axes[0].set_xlim(KA_LO, KA_HI_BAFFLED)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(ka_baffled, np.imag(z_baffled), "k-", label="Infinite baffle")
    axes[1].plot(ka_unflanged, np.imag(z_unflanged), "C1--", label="Unflanged")
    axes[1].set_ylabel(r"Im{$Z_\mathrm{rad}/(\rho c)$}")
    axes[1].set_title("Radiation Reactance")
    axes[1].set_xlabel(r"$ka$")
    axes[1].set_xlim(KA_LO, KA_HI_BAFFLED)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    for ax in axes:
        ax.set_xscale("log")

    fig.suptitle("Baffled vs. Unflanged Circular Piston: Radiation Impedance")
    fig.tight_layout()

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    fig.savefig(plot_file, dpi=150)
    print(f"Plot saved: {plot_file}")


def print_low_ka_check():
    """Verify unflanged Re(Z) ≈ ½ baffled Re(Z) at low ka (beta=1/2 asymptote)."""
    ka_samples = [0.01, 0.02, 0.05, 0.1]
    print("\nLow-ka resistance check (unflanged / baffled ratio, expect ≈ 0.5):")
    print(f"{'ka':>8}  {'Re baffled':>12}  {'Re unflanged':>14}  {'ratio':>8}")
    print("-" * 48)
    for ka in ka_samples:
        z_b = zrad_norm_analytical(ka)
        z_u = zrad_norm_unflanged(ka)
        ratio = np.real(z_u) / np.real(z_b)
        print(
            f"{ka:8.4g}  {np.real(z_b):12.6f}  {np.real(z_u):14.6f}  {ratio:8.4f}"
        )


if __name__ == "__main__":
    print_low_ka_check()
    make_plot()
