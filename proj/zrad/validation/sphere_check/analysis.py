"""Compare pulsating-sphere BEM results against the closed-form impedance."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from analytical import zrad_norm_analytical
from mesh import KA_VALUES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOT_FILE = os.path.join(RESULTS_DIR, "validation_plot.png")


def load_bem_results(results_dir=RESULTS_DIR):
    results = []
    for ka in KA_VALUES:
        tag = f"{ka:g}".replace(".", "p")
        path = os.path.join(results_dir, f"result_ka{tag}.json")
        if not os.path.isfile(path):
            print(f"  Missing: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))

    if not results:
        raise FileNotFoundError(
            f"No result JSON files found in {results_dir}. Run solve.py first."
        )

    results.sort(key=lambda r: r["ka"])
    return results


def print_comparison_table(bem_results):
    print("\nValidation comparison (Z_rad / (rho c))")
    print(
        f"{'ka':>6}  {'BEM Re':>10}  {'Ana Re':>10}  {'Err Re %':>9}  "
        f"{'BEM Im':>10}  {'Ana Im':>10}  {'Err Im %':>9}"
    )
    print("-" * 78)

    for row in bem_results:
        ka = row["ka"]
        z_bem = row["Z_rad_norm_real"] + 1j * row["Z_rad_norm_imag"]
        z_ana = zrad_norm_analytical(ka)

        err_re = 100.0 * abs(np.real(z_bem) - np.real(z_ana)) / max(abs(np.real(z_ana)), 1e-12)
        err_im = 100.0 * abs(np.imag(z_bem) - np.imag(z_ana)) / max(abs(np.imag(z_ana)), 1e-12)

        print(
            f"{ka:6.2f}  {np.real(z_bem):10.5f}  {np.real(z_ana):10.5f}  {err_re:9.2f}  "
            f"{np.imag(z_bem):10.5f}  {np.imag(z_ana):10.5f}  {err_im:9.2f}"
        )


def make_plot(bem_results, plot_file=PLOT_FILE):
    ka_bem = np.array([r["ka"] for r in bem_results])
    z_bem = np.array(
        [r["Z_rad_norm_real"] + 1j * r["Z_rad_norm_imag"] for r in bem_results]
    )

    ka_dense = np.geomspace(0.01, 10.0, 200)
    z_ana = zrad_norm_analytical(ka_dense)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    axes[0].plot(ka_dense, np.real(z_ana), "k-", label="Analytical")
    axes[0].plot(ka_bem, np.real(z_bem), "o", color="C0", label="BEM")
    axes[0].set_ylabel(r"Re{$Z_\mathrm{rad}/(\rho c)$}")
    axes[0].set_title("Radiation resistance")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(ka_dense, np.imag(z_ana), "k-", label="Analytical")
    axes[1].plot(ka_bem, np.imag(z_bem), "o", color="C1", label="BEM")
    axes[1].set_ylabel(r"Im{$Z_\mathrm{rad}/(\rho c)$}")
    axes[1].set_title("Radiation reactance")
    axes[1].set_xlabel(r"$ka$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Pulsating sphere: BEM vs analytical radiation impedance")
    fig.tight_layout()

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    fig.savefig(plot_file, dpi=150)
    print(f"\nPlot saved: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pulsating-sphere BEM validation results.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory containing result_ka*.json files",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=PLOT_FILE,
        help="Output path for validation plot",
    )
    args = parser.parse_args()

    bem_results = load_bem_results(args.results_dir)
    print_comparison_table(bem_results)
    make_plot(bem_results, plot_file=args.plot)


if __name__ == "__main__":
    main()
