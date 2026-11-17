"""B&W slide plot for preliminary off-center dy × ka sweep (hybrid extraction)."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from full_sweep import (
    DY_VALUES,
    RESULTS_DIR,
    build_ka_grid,
    dy_tag,
    is_valid_result,
    ka_tag,
)

PLOT_FILE = os.path.join(os.path.dirname(__file__), "results", "offcenter_sweep_plot_bw.png")

# Grayscale-safe trace differentiation: gray level, linestyle, hollow marker shape.
TRACE_STYLES = [
    {"gray": "0.0", "linestyle": "-", "marker": "o"},
    {"gray": "0.2", "linestyle": "--", "marker": "s"},
    {"gray": "0.4", "linestyle": "-.", "marker": "^"},
    {"gray": "0.6", "linestyle": ":", "marker": "D"},
    {"gray": "0.75", "linestyle": (0, (5, 2, 1, 2)), "marker": "P"},
]

MARKER_EVERY = 3  # sparse hollow markers (~every 3rd ka point on 27-pt grid)


def _style_axes_bw(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def dy_legend_label(dy):
    """Short legend text for each piston offset."""
    dy_max = DY_VALUES[-1]
    if abs(dy) < 1e-12:
        return r"dy = 0 (centered)"
    if abs(dy - dy_max) < 1e-9:
        return f"dy = {dy:.3f} (near-edge)"
    return f"dy = {dy:.3f}"


def _result_path(results_dir, dy, ka):
    return os.path.join(results_dir, f"result_dy{dy_tag(dy)}_ka{ka_tag(ka)}.json")


def load_sweep_results(results_dir=RESULTS_DIR):
    """Load canonical per-(dy, ka) JSON results (hybrid: FF top Re, NF Im)."""
    ka_values = build_ka_grid()
    by_dy = {}
    missing = []

    for dy in DY_VALUES:
        rows = []
        for ka in ka_values:
            path = _result_path(results_dir, dy, ka)
            if not is_valid_result(path):
                missing.append((dy, ka, path))
                continue
            with open(path, encoding="utf-8") as f:
                row = json.load(f)
            rows.append(row)
        rows.sort(key=lambda r: r["ka"])
        by_dy[dy] = rows

    if missing:
        print(f"  Missing or incomplete results ({len(missing)} pairs):")
        for dy, ka, path in missing:
            print(f"    dy={dy:g}  ka={ka:g}  ({os.path.basename(path)})")

    loaded = sum(len(v) for v in by_dy.values())
    expected = len(DY_VALUES) * len(ka_values)
    print(f"Loaded {loaded}/{expected} (dy, ka) pairs from {results_dir}")

    if loaded == 0:
        raise FileNotFoundError(
            f"No valid sweep results in {results_dir}. Run full_sweep.py first."
        )

    return by_dy, ka_values


def _plot_trace(ax, ka, values, style, label):
    """Line trace with sparse hollow markers."""
    color = style["gray"]
    ax.plot(
        ka,
        values,
        color=color,
        linestyle=style["linestyle"],
        linewidth=1.6,
        label=label,
        zorder=2,
    )
    ax.plot(
        ka,
        values,
        linestyle="none",
        marker=style["marker"],
        markevery=MARKER_EVERY,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.1,
        markersize=5.5,
        color=color,
        zorder=3,
    )


def make_plot(by_dy, plot_file=PLOT_FILE):
    """Two-panel Re/Im plot for all dy offsets."""
    ka_hi = max(row["ka"] for rows in by_dy.values() for row in rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    fig.patch.set_facecolor("white")

    for dy, style in zip(DY_VALUES, TRACE_STYLES):
        rows = by_dy.get(dy, [])
        if not rows:
            continue
        ka = np.array([r["ka"] for r in rows])
        re_ff = np.array([r["Z_rad_norm_real_ff"] for r in rows])
        im_nf = np.array([r["Z_rad_norm_imag"] for r in rows])
        label = dy_legend_label(dy)

        _plot_trace(axes[0], ka, re_ff, style, label)
        _plot_trace(axes[1], ka, im_nf, style, label)

    axes[0].set_ylabel(r"Re{$Z_\mathrm{rad}/(\rho c)$}")
    axes[0].set_title("Radiation Resistance")
    axes[0].set_xlabel(r"$ka$")
    axes[0].set_xlim(0.0, ka_hi)
    _style_axes_bw(axes[0])

    axes[1].set_ylabel(r"Im{$Z_\mathrm{rad}/(\rho c)$}")
    axes[1].set_title("Radiation Reactance")
    axes[1].set_xlabel(r"$ka$")
    axes[1].set_xlim(0.0, ka_hi)
    _style_axes_bw(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)

    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    fig.savefig(plot_file, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot off-center dy × ka sweep in deck B&W style."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory with result_dy*_ka*.json files",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=PLOT_FILE,
        help="Output PNG path",
    )
    args = parser.parse_args()

    by_dy, _ = load_sweep_results(args.results_dir)
    make_plot(by_dy, plot_file=args.plot)


if __name__ == "__main__":
    main()
