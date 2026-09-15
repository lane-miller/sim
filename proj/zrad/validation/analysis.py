"""Compare BEM validation results against the analytical baffled-piston impedance."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from analytical import zrad_norm_analytical
from mesh import DEFAULT_RESULT_SUFFIX, KA_VALUES

DEFAULT_OUT_SUFFIX = DEFAULT_RESULT_SUFFIX

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOT_FILE = os.path.join(RESULTS_DIR, "validation_plot.png")


def load_bem_results(results_dir=RESULTS_DIR, out_suffix=DEFAULT_OUT_SUFFIX):
    """Load all per-ka JSON result files from results_dir."""
    results = []
    for ka in KA_VALUES:
        tag = f"{ka:g}".replace(".", "p")
        path = os.path.join(results_dir, f"result_ka{tag}{out_suffix}.json")
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


def _pct_err(bem_val, ana_val):
    return 100.0 * abs(bem_val - ana_val) / max(abs(ana_val), 1e-12)


def print_comparison_table(bem_results):
    """Print near-field vs far-field Re(Z) and near-field Im(Z) vs analytical."""
    has_ff = any("Z_rad_norm_real_ff" in row for row in bem_results)

    print("\nValidation comparison (Z_rad / (rho c))")
    if has_ff:
        print(
            f"{'ka':>6}  {'NF Re':>10}  {'FF Re':>10}  {'Ana Re':>10}  "
            f"{'Err NF Re%':>11}  {'Err FF Re%':>11}  "
            f"{'NF Im':>10}  {'Ana Im':>10}  {'Err Im %':>9}  {'Front/Total':>11}"
        )
        print("-" * 118)
    else:
        print(
            f"{'ka':>6}  {'BEM Re':>10}  {'Ana Re':>10}  {'Err Re %':>9}  "
            f"{'BEM Im':>10}  {'Ana Im':>10}  {'Err Im %':>9}"
        )
        print("-" * 78)

    for row in bem_results:
        ka = row["ka"]
        z_nf = row["Z_rad_norm_real"] + 1j * row["Z_rad_norm_imag"]
        z_ana = zrad_norm_analytical(ka)
        re_ana = float(np.real(z_ana))
        im_ana = float(np.imag(z_ana))

        if has_ff:
            re_ff = row.get("Z_rad_norm_real_ff", float("nan"))
            front_ratio = row.get("front_hemisphere_power_ratio", float("nan"))
            print(
                f"{ka:6.2f}  {np.real(z_nf):10.5f}  {re_ff:10.5f}  {re_ana:10.5f}  "
                f"{_pct_err(np.real(z_nf), re_ana):11.2f}  {_pct_err(re_ff, re_ana):11.2f}  "
                f"{np.imag(z_nf):10.5f}  {im_ana:10.5f}  {_pct_err(np.imag(z_nf), im_ana):9.2f}  "
                f"{front_ratio:11.4f}"
            )
        else:
            print(
                f"{ka:6.2f}  {np.real(z_nf):10.5f}  {re_ana:10.5f}  "
                f"{_pct_err(np.real(z_nf), re_ana):9.2f}  "
                f"{np.imag(z_nf):10.5f}  {im_ana:10.5f}  {_pct_err(np.imag(z_nf), im_ana):9.2f}"
            )


def make_plot(bem_results, plot_file=PLOT_FILE):
    """Overlay BEM points on the analytical curve."""
    ka_bem = np.array([r["ka"] for r in bem_results])
    z_bem = np.array(
        [r["Z_rad_norm_real"] + 1j * r["Z_rad_norm_imag"] for r in bem_results]
    )
    has_ff = any("Z_rad_norm_real_ff" in r for r in bem_results)
    re_ff = np.array([r.get("Z_rad_norm_real_ff", np.nan) for r in bem_results])

    ka_dense = np.geomspace(0.01, 10.0, 200)
    z_ana = zrad_norm_analytical(ka_dense)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    axes[0].plot(ka_dense, np.real(z_ana), "k-", label="Analytical")
    axes[0].plot(ka_bem, np.real(z_bem), "o", color="C0", label="Near-field Re")
    if has_ff:
        axes[0].plot(ka_bem, re_ff, "s", color="C2", label="Far-field power Re")
    axes[0].set_ylabel(r"Re{$Z_\mathrm{rad}/(\rho c)$}")
    axes[0].set_title("Radiation resistance")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(ka_dense, np.imag(z_ana), "k-", label="Analytical")
    axes[1].plot(ka_bem, np.imag(z_bem), "o", color="C1", label="Near-field Im")
    axes[1].set_ylabel(r"Im{$Z_\mathrm{rad}/(\rho c)$}")
    axes[1].set_title("Radiation reactance")
    axes[1].set_xlabel(r"$ka$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Baffled piston: BEM vs analytical radiation impedance")
    fig.tight_layout()

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    fig.savefig(plot_file, dpi=150)
    print(f"\nPlot saved: {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze baffled-piston BEM validation results.")
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
    parser.add_argument(
        "--out-suffix",
        type=str,
        default=DEFAULT_OUT_SUFFIX,
        help=f"Result filename suffix (default: {DEFAULT_OUT_SUFFIX})",
    )
    args = parser.parse_args()

    bem_results = load_bem_results(args.results_dir, out_suffix=args.out_suffix)
    print_comparison_table(bem_results)
    make_plot(bem_results, plot_file=args.plot)


if __name__ == "__main__":
    main()
