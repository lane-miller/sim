"""12-point dense validation: top-only far-field Re + near-field Im vs analytical."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from analytical import zrad_norm_analytical

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_SUFFIX = "_ff_top"
PLOT_FILE = os.path.join(RESULTS_DIR, "validation_plot_dense.png")

# Dense sweep ka grid (geomspace 0.05–5, 12 points) — duplicated here to avoid gmsh import.
KA_VALUES_DENSE = list(np.geomspace(0.05, 5.0, 12))


def load_bem_results(results_dir=RESULTS_DIR, out_suffix=OUT_SUFFIX):
    """Load per-ka JSON result files for the dense ka sweep."""
    results = []
    missing = []
    for ka in KA_VALUES_DENSE:
        tag = f"{ka:g}".replace(".", "p")
        path = os.path.join(results_dir, f"result_ka{tag}{out_suffix}.json")
        if not os.path.isfile(path):
            missing.append(ka)
            continue
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))

    if missing:
        print(f"  Missing results for ka: {missing}")

    if not results:
        raise FileNotFoundError(
            f"No result JSON files found in {results_dir} (suffix={out_suffix!r}). "
            "Run solve.py with --far-field-surfaces top first."
        )

    results.sort(key=lambda r: r["ka"])
    return results


def _pct_err(bem_val, ana_val):
    return 100.0 * abs(bem_val - ana_val) / max(abs(ana_val), 1e-12)


def print_comparison_table(bem_results):
    """Print adopted-method comparison: FF top Re, near-field Im."""
    print("\nDense validation comparison (Z_rad / (rho c))")
    print(
        f"{'ka':>8}  {'BEM Re':>10}  {'Ana Re':>10}  {'Err Re%':>9}  "
        f"{'BEM Im':>10}  {'Ana Im':>10}  {'Err Im%':>9}"
    )
    print("-" * 78)

    max_re_err = 0.0
    max_im_err = 0.0
    for row in bem_results:
        ka = row["ka"]
        re_bem = row["Z_rad_norm_real_ff"]
        im_bem = row["Z_rad_norm_imag"]
        z_ana = zrad_norm_analytical(ka)
        re_ana = float(np.real(z_ana))
        im_ana = float(np.imag(z_ana))

        re_err = _pct_err(re_bem, re_ana)
        im_err = _pct_err(im_bem, im_ana)
        max_re_err = max(max_re_err, re_err)
        max_im_err = max(max_im_err, im_err)

        print(
            f"{ka:8.4g}  {re_bem:10.5f}  {re_ana:10.5f}  {re_err:9.2f}  "
            f"{im_bem:10.5f}  {im_ana:10.5f}  {im_err:9.2f}"
        )

    print(f"\nMax error: Re {max_re_err:.2f}%, Im {max_im_err:.2f}%")
    return max_re_err, max_im_err


def make_plot(bem_results, plot_file=PLOT_FILE):
    """Plot adopted BEM markers (FF top Re, NF Im) on the analytical curve."""
    ka_bem = np.array([r["ka"] for r in bem_results])
    re_bem = np.array([r["Z_rad_norm_real_ff"] for r in bem_results])
    im_bem = np.array([r["Z_rad_norm_imag"] for r in bem_results])

    ka_margin = 1.15
    ka_lo = ka_bem.min() / ka_margin
    ka_hi = ka_bem.max() * ka_margin
    ka_dense = np.geomspace(ka_lo, ka_hi, 200)
    z_ana = zrad_norm_analytical(ka_dense)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    axes[0].plot(ka_dense, np.real(z_ana), "k-", label="Analytical")
    axes[0].plot(ka_bem, re_bem, "s", color="C2", label="BEM")
    axes[0].set_ylabel(r"Re{$Z_\mathrm{rad}/(\rho c)$}")
    axes[0].set_title("Radiation Resistance")
    axes[0].set_xlabel(r"$ka$")
    axes[0].set_xlim(ka_lo, ka_hi)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(ka_dense, np.imag(z_ana), "k-", label="Analytical")
    axes[1].plot(ka_bem, im_bem, "o", color="C1", label="BEM")
    axes[1].set_ylabel(r"Im{$Z_\mathrm{rad}/(\rho c)$}")
    axes[1].set_title("Radiation Reactance")
    axes[1].set_xlabel(r"$ka$")
    axes[1].set_xlim(ka_lo, ka_hi)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Baffled Piston Validation")
    fig.tight_layout()

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    fig.savefig(plot_file, dpi=150)
    print(f"\nPlot saved: {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dense (12-point) baffled-piston BEM validation results."
    )
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
        help="Output path for dense validation plot",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default=OUT_SUFFIX,
        help=f"Result filename suffix (default: {OUT_SUFFIX})",
    )
    args = parser.parse_args()

    bem_results = load_bem_results(args.results_dir, out_suffix=args.out_suffix)
    print_comparison_table(bem_results)
    make_plot(bem_results, plot_file=args.plot)


if __name__ == "__main__":
    main()
