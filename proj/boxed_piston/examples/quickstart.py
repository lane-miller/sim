#!/usr/bin/env python3
"""boxrad quickstart: dry-run, run, summarize, and plot one boxed-piston system.

Usage (boxrad installed; any working directory)::

    python proj/boxed_piston/examples/quickstart.py [CONFIG] [-o OUT.h5] [--no-show]

CONFIG defaults to ``configs/default.json`` (the C9 box geometry with h from the
default rules; no slower than ``configs/c09_box.json``, about 6–7 s on an M1). The figure
shows |z| and arg z, on-axis SPL at 1 m for 1 m/s peak radiator velocity, and
xz-plane polar patterns at kL = 2, 5, 10 (those inside the band). It is saved
next to the HDF5 file as PNG. Box-mode configs only. Not imported by tests.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boxrad import Result, run

ROOT = Path(__file__).resolve().parents[1]
P_REF = 20e-6  # Pa
V_PEAK = 1.0  # m/s
KL_POLAR = (2.0, 5.0, 10.0)
POLAR_FLOOR_DB = -40.0


def _polar_frequencies(res: Result) -> list[float]:
    """Frequencies at kL = 2, 5, 10 (L = largest realized box side) inside the band."""
    f_min, f_max = res.band
    d = res.derived
    L = max(d.realized_W, d.realized_H, d.realized_D)
    c = float(res.config.air.c)
    f_kl = [kl * c / (2.0 * math.pi * L) for kl in KL_POLAR]
    inside = [f for f in f_kl if f_min <= f <= f_max]
    return inside or list(np.geomspace(f_min, f_max, 5)[1:4])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="boxrad quickstart")
    parser.add_argument(
        "config", nargs="?", type=Path, default=ROOT / "configs" / "default.json"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=ROOT / "outputs" / "quickstart.h5"
    )
    parser.add_argument("--no-show", action="store_true", help="Save the PNG only")
    args = parser.parse_args(argv)

    summary = run(args.config, dry_run=True)
    if summary["geometry"] != "box":
        raise SystemExit("quickstart.py expects a box-mode config")
    h5_path = run(args.config, out_path=args.output)

    with Result(h5_path) as res:
        print()
        print(res.summary())
        f_min, f_max = res.band

        f = np.geomspace(f_min, f_max, 300)
        _, z = res.z(f)

        f_spl = np.geomspace(f_min, f_max, 40)
        h_axis = res.field([[0.0, 0.0, 1.0]], f_spl)[:, 0]  # Pa/(m/s) at 1 m on axis
        spl = 20.0 * np.log10(np.abs(h_axis) * V_PEAK / (math.sqrt(2.0) * P_REF))

        f_pol = _polar_frequencies(res)
        theta = np.linspace(0.0, math.pi, 181)
        pat_px = np.abs(res.directivity(theta, 0.0, f_pol, normalize=True))  # x > 0 half
        pat_nx = np.abs(res.directivity(theta, math.pi, f_pol, normalize=True))  # x < 0 half

        print("\n    f [Hz]       Re z       Im z   W [W at 1 m/s]")
        for f_chk in np.geomspace(f_min, f_max, 4):
            _, z_chk = res.z(f_chk)
            w_chk = res.power(f_chk)[0]
            print(
                f"  {f_chk:8.1f}  {z_chk[0].real:9.4f}  {z_chk[0].imag:9.4f}  {w_chk:12.4e}"
            )

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    ax_mag = fig.add_subplot(2, 2, 1)
    ax_arg = fig.add_subplot(2, 2, 2)
    ax_spl = fig.add_subplot(2, 2, 3)
    ax_pol = fig.add_subplot(2, 2, 4, projection="polar")

    ax_mag.loglog(f, np.abs(z))
    ax_mag.set(xlabel="f [Hz]", ylabel="|z|", title="Radiation impedance magnitude")
    ax_arg.semilogx(f, np.degrees(np.angle(z)))
    ax_arg.set(xlabel="f [Hz]", ylabel="arg z [deg]", title="Radiation impedance phase")
    ax_spl.semilogx(f_spl, spl, "o-", ms=3)
    ax_spl.set(
        xlabel="f [Hz]",
        ylabel="SPL [dB re 20 µPa]",
        title=f"On-axis SPL at 1 m, {V_PEAK:g} m/s peak",
    )
    for ax in (ax_mag, ax_arg, ax_spl):
        ax.grid(True, which="both", alpha=0.3)

    # Full xz-plane circle: angle from +z, positive toward +x.
    psi = np.concatenate([-theta[::-1], theta[1:]])
    for i, f_hz in enumerate(f_pol):
        mag = np.concatenate([pat_nx[i, ::-1], pat_px[i, 1:]])
        db = np.maximum(20.0 * np.log10(np.maximum(mag, 1e-12)), POLAR_FLOOR_DB)
        ax_pol.plot(psi, db, label=f"{f_hz:.0f} Hz")
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_rlim(POLAR_FLOOR_DB, 5.0)
    ax_pol.set_title("xz-plane directivity [dB re on-axis]")
    ax_pol.legend(loc="lower left", fontsize=8)

    png = Path(h5_path).with_suffix(".png")
    fig.savefig(png, dpi=150)
    print(f"\nwrote {png}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
