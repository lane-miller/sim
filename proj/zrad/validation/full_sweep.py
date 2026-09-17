"""Full off-center parametric sweep: dy × ka with hybrid extraction.

5 piston-y offsets × ~25 ka values. One mesh per (dy, ka) pair; resumable
via per-pair JSON in results/offcenter_sweep/. Hybrid everywhere:
far-field top-only Re(Z), near-field Im(Z).
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

from mesh import generate_fixed_baffle_mesh
from solve import RTOL, SOLVER_CHOICES, solve_one

# ── Geometry (matches offcenter_pilot.py) ────────────────────────────────────
A = 0.0125
LX = 8.0 * A  # 0.1 m
LY = 16.0 * A  # 0.2 m
THICKNESS_PILOT = A / 30.0
L_BAFFLE = max(LX, LY) / 2.0

# Linear dy grid: centered → near-tangent (pilot edge_margin = a/10)
DY_VALUES = list(np.linspace(0.0, 0.08625, 5))

# Dense ka: geomspace 0.05–2.5 (22 pts, same spacing family as validation dense
# 12-pt grid but extended/adjusted). High-ka checkpoints at 0.5 spacing.
DENSE_KA_N = 22
KA_CHECKPOINTS = [3.0, 3.5, 4.0, 4.5, 5.0]

MESH_DIR = os.path.join(os.path.dirname(__file__), "meshes", "offcenter_sweep")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "offcenter_sweep")

REQUIRED_RESULT_KEYS = ("Z_rad_norm_real_ff", "Z_rad_norm_imag")

# Band-4 reference from pilot ka=5 centered (solve_time_s ≈ 222 s, ~14.4k elements)
BAND4_SOLVE_S = 222.0


def build_ka_grid():
    """Return sorted unique ka list: dense region + high-ka checkpoints."""
    dense = list(np.geomspace(0.05, 2.5, DENSE_KA_N))
    extra = [k for k in KA_CHECKPOINTS if k > dense[-1] + 1e-12]
    return dense + extra


def _float_tag(x):
    return f"{x:g}".replace(".", "p")


def dy_tag(dy):
    return _float_tag(dy)


def ka_tag(ka):
    return _float_tag(ka)


def position_label(dy):
    return f"dy{dy_tag(dy)}"


def result_path(dy, ka):
    return os.path.join(
        RESULTS_DIR, f"result_dy{dy_tag(dy)}_ka{ka_tag(ka)}.json"
    )


def is_valid_result(path):
    """True if path holds a complete hybrid-extraction result."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    for key in REQUIRED_RESULT_KEYS:
        val = data.get(key)
        if val is None or not np.isfinite(val):
            return False
    return True


def save_sweep_result(dy, ka, solve_result, mesh_info):
    """Write one (dy, ka) result immediately after the solve."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = {
        **solve_result,
        "piston_dx_m": 0.0,
        "piston_dy_m": dy,
        "dy_tag": dy_tag(dy),
        "ka_tag": ka_tag(ka),
        "position_label": position_label(dy),
        "geometry": {
            "a_m": A,
            "Lx_m": LX,
            "Ly_m": LY,
            "piston_dx_m": 0.0,
            "piston_dy_m": dy,
        },
        "mesh_info": mesh_info,
        "hybrid_extraction": {
            "Re": "far_field_top",
            "Im": "near_field",
        },
        "Z_rad_norm_real_ff": solve_result["Z_rad_norm_real_ff"],
        "Z_rad_norm_imag": solve_result["Z_rad_norm_imag"],
    }
    path = result_path(dy, ka)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {path}")
    return path


def check_failure(dy, ka, solve_result):
    """Return a failure description string, or None if OK."""
    info = solve_result.get("linear_solve_info", 0)
    re_ff = solve_result.get("Z_rad_norm_real_ff")
    im_nf = solve_result.get("Z_rad_norm_imag")

    flags = []
    if info != 0:
        flags.append(f"solver info={info}")
    if re_ff is not None and re_ff < 0:
        flags.append("Re<0")
    if im_nf is not None and im_nf < 0:
        flags.append("Im<0")
    if not np.isfinite(re_ff) or not np.isfinite(im_nf):
        flags.append("NaN/Inf")

    if flags:
        return f"dy={dy:g} ka={ka:g}: {', '.join(flags)}"
    return None


def _format_duration(seconds):
    if seconds < 0 or not np.isfinite(seconds):
        return "?"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


class CaffeinateGuard:
    """Spawn ``caffeinate -i`` for the sweep lifetime; always terminate in finally."""

    def __init__(self):
        self._proc = None

    def start(self):
        print("Starting caffeinate -i (prevent system sleep during sweep)...")
        self._proc = subprocess.Popen(["caffeinate", "-i"])
        print(f"  caffeinate PID={self._proc.pid}")

    def stop(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            print("Terminated caffeinate.")
        else:
            print(f"caffeinate already exited (code={self._proc.returncode}).")
        self._proc = None


def _print_preflight(ka_values, pairs):
    n_high = sum(1 for _, ka in pairs if ka >= 3.0)
    n_low = len(pairs) - n_high
    est_low_s = 45.0  # typical mid-band including mesh gen
    est_total_s = n_high * BAND4_SOLVE_S + n_low * est_low_s

    print("=" * 72)
    print("Off-center full parametric sweep")
    print(f"  a={A} m  Lx={LX} m  Ly={LY} m  dx=0")
    print(f"  dy values ({len(DY_VALUES)}): {[f'{d:g}' for d in DY_VALUES]}")
    print(f"  ka grid ({len(ka_values)} points):")
    for ka in ka_values:
        print(f"    {ka:.8g}")
    print(f"  Total (dy, ka) pairs: {len(pairs)}")
    print(f"  High-ka checkpoints (ka≥3): {n_high} pairs  (~{BAND4_SOLVE_S:.0f} s/solve)")
    print(f"  Lower-ka pairs: {n_low}  (est ~{est_low_s:.0f} s/solve avg)")
    print(
        f"  Rough runtime estimate: {_format_duration(est_total_s)}"
        f"  ({est_total_s / 3600:.1f} h) — dominated by band-4 class solves"
    )
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Meshes:  {MESH_DIR}")
    print("=" * 72)


def run_sweep(solver="auto"):
    os.makedirs(MESH_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ka_values = build_ka_grid()
    pairs = [(dy, ka) for dy in DY_VALUES for ka in ka_values]
    total = len(pairs)

    _print_preflight(ka_values, pairs)

    stats = {
        "attempted": 0,
        "skipped": 0,
        "solved": 0,
        "failures": [],
    }
    solve_durations = []
    t_sweep = time.perf_counter()

    caffeinate = CaffeinateGuard()
    caffeinate.start()

    try:
        for idx, (dy, ka) in enumerate(pairs, start=1):
            stats["attempted"] += 1
            out_path = result_path(dy, ka)
            prefix = f"[{idx}/{total}] dy={dy:g}  ka={ka:g}"

            if is_valid_result(out_path):
                stats["skipped"] += 1
                elapsed = time.perf_counter() - t_sweep
                avg = np.mean(solve_durations) if solve_durations else 0.0
                remaining = (total - idx) * avg
                print(
                    f"{prefix}  SKIP (valid result exists)  "
                    f"elapsed={_format_duration(elapsed)}  "
                    f"ETA≈{_format_duration(remaining)}"
                )
                continue

            print(f"\n{'─' * 72}")
            print(f"{prefix}  SOLVE")
            elapsed = time.perf_counter() - t_sweep
            avg = np.mean(solve_durations) if solve_durations else 0.0
            remaining = (total - idx) * avg
            print(
                f"  elapsed={_format_duration(elapsed)}  "
                f"ETA≈{_format_duration(remaining)}  "
                f"(based on {len(solve_durations)} completed solves)"
            )
            print(f"{'─' * 72}")

            t_pair = time.perf_counter()

            pos_label = position_label(dy)
            mesh_info = generate_fixed_baffle_mesh(
                ka,
                a=A,
                lx=LX,
                ly=LY,
                piston_dx=0.0,
                piston_dy=dy,
                position_label=pos_label,
                thickness=THICKNESS_PILOT,
                mesh_dir=MESH_DIR,
            )

            result = solve_one(
                ka,
                mesh_file=mesh_info["mesh_path"],
                solver=solver,
                rtol=RTOL,
                far_field_surfaces="top",
                a=A,
                L_baffle_m=L_BAFFLE,
            )

            save_sweep_result(dy, ka, result, mesh_info)

            dt_pair = time.perf_counter() - t_pair
            solve_durations.append(dt_pair)
            stats["solved"] += 1

            fail = check_failure(dy, ka, result)
            if fail:
                stats["failures"].append(fail)
                print(f"  *** WARNING: {fail} ***")
            else:
                print(
                    f"  OK  wall={dt_pair:.1f} s  "
                    f"elements={result['n_elements']}  DOFs={result['n_dofs']}  "
                    f"Z/(ρc)={result['Z_rad_norm_real_ff']:.6f}"
                    f"+{result['Z_rad_norm_imag']:.6f}j  "
                    f"(Re=ff_top, Im=near-field)"
                )

    finally:
        caffeinate.stop()

    _print_summary(stats, time.perf_counter() - t_sweep)
    return 0 if not stats["failures"] else 1


def _print_summary(stats, elapsed_s):
    print(f"\n{'=' * 72}")
    print("SWEEP SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total pairs attempted : {stats['attempted']}")
    print(f"  Skipped (existing)    : {stats['skipped']}")
    print(f"  Newly solved          : {stats['solved']}")
    print(f"  Elapsed               : {_format_duration(elapsed_s)}")
    if stats["failures"]:
        print(f"  Failures flagged      : {len(stats['failures'])}")
        for msg in stats["failures"]:
            print(f"    - {msg}")
    else:
        print("  Failures flagged      : 0")
    print(f"{'=' * 72}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Full off-center dy × ka parametric sweep (resumable)."
    )
    parser.add_argument(
        "--solver",
        choices=SOLVER_CHOICES,
        default="auto",
        help="Linear solver (default: auto).",
    )
    args = parser.parse_args()
    return run_sweep(solver=args.solver)


if __name__ == "__main__":
    sys.exit(main())
