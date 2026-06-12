"""
HRTF POC — Banded BEM Solve with Frequency Interpolation
=========================================================
Solve at sparse expansion frequencies, interpolate HRTFs to dense grid.

Usage:
    python interp/solve_interp.py
"""

import sys
import time
from pathlib import Path

HRTF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HRTF_ROOT))

import numpy as np

from common.bem import (
    compute_source,
    evaluate_hrtf,
    load_mesh_space,
    mass_matrix_preconditioner,
    solve_burton_miller,
)
from common.sofa import horizontal_plane_from_sofa, interp_hrtf_spline, load_sofa_pair
from config import BAND_LIMITS, HRTF_INTERP

TARGET_FREQS = np.arange(200.0, 12200.0, 200.0)
N_EXPANSION = 20
BAND_BOUNDARIES = [2000.0, 6000.0]


def select_expansion_points(target_freqs, n_points, band_boundaries):
    f_min, f_max = target_freqs[0], target_freqs[-1]
    base = np.geomspace(f_min, f_max, n_points)
    required = np.array([f_min, f_max] + list(band_boundaries))
    combined = np.unique(np.concatenate([base, required]))
    snapped = np.array([
        target_freqs[np.argmin(np.abs(target_freqs - f))] for f in combined
    ])
    return np.unique(snapped)


def solve_band_and_evaluate(band_cfg, eval_points):
    mesh_path = band_cfg["mesh"]
    freqs = band_cfg["freqs"]

    _, _, space = load_mesh_space(mesh_path)
    n_dofs = space.global_dof_count
    M = mass_matrix_preconditioner(space)
    source_m, _ = compute_source(mesh_path)

    print(f"\n  Mesh: {mesh_path.name}  ({n_dofs} DOFs, "
          f"{(n_dofs**2 * 16) / 1e9:.2f} GB dense)")

    band_results = []
    for j, freq in enumerate(freqs):
        print(f"\n    [{j + 1}/{len(freqs)}] f={freq:.1f} Hz")

        t_asm = time.perf_counter()
        iter_state = [0, time.perf_counter()]

        def gmres_callback(residual_norm):
            iter_state[0] += 1
            if iter_state[0] % 50 == 0 or iter_state[0] == 1:
                elapsed = time.perf_counter() - iter_state[1]
                print(f"      iter {iter_state[0]:4d}  |r|={residual_norm:.3e}  "
                      f"elapsed={elapsed:.1f}s", flush=True)

        x, info = solve_burton_miller(
            space, freq, source_m, rtol=1e-4, M=M, callback=gmres_callback,
        )
        dt_asm = time.perf_counter() - t_asm
        dt_solve = time.perf_counter() - iter_state[1]
        status = "OK" if info == 0 else f"FAIL({info})"
        print(f"    Asm: {dt_asm:.1f}s  Solve: {dt_solve:.1f}s  "
              f"{iter_state[0]} iters  {status}")

        hrtf = evaluate_hrtf(space, x, freq, source_m, eval_points)
        print(f"    HRTF |H| range: [{np.abs(hrtf).min():.3f}, {np.abs(hrtf).max():.3f}]")

        band_results.append({"freq": freq, "hrtf": hrtf, "source_m": source_m})

    return band_results


if __name__ == "__main__":
    expansion_freqs = select_expansion_points(TARGET_FREQS, N_EXPANSION, BAND_BOUNDARIES)

    BANDS = []
    for f_lo, f_hi, mesh_path in BAND_LIMITS:
        freqs = expansion_freqs[(expansion_freqs >= f_lo) & (expansion_freqs <= f_hi)]
        if len(freqs) > 0:
            BANDS.append({"mesh": mesh_path, "freqs": freqs})

    sofa_meas, _, _ = load_sofa_pair()
    horiz_idx, horiz_az, eval_points, eval_radius = horizontal_plane_from_sofa(
        sofa_meas.SourcePosition,
    )
    n_directions = eval_points.shape[1]

    print(f"Target grid: {len(TARGET_FREQS)} frequencies "
          f"({TARGET_FREQS[0]:.0f}–{TARGET_FREQS[-1]:.0f} Hz)")
    print(f"Expansion points: {len(expansion_freqs)} (requested {N_EXPANSION})")
    print(f"  {expansion_freqs}")
    print(f"Horizontal plane: {n_directions} directions")

    t_total = time.perf_counter()
    all_results = []

    for i, band in enumerate(BANDS):
        f_lo, f_hi = band["freqs"][0], band["freqs"][-1]
        print(f"\n{'='*60}")
        print(f"Band {i + 1}/{len(BANDS)}: {f_lo:.0f}–{f_hi:.0f} Hz  "
              f"({len(band['freqs'])} freqs)")
        print(f"{'='*60}")
        all_results.extend(solve_band_and_evaluate(band, eval_points))

    dt_total = time.perf_counter() - t_total

    all_results.sort(key=lambda r: r["freq"])
    expansion_freqs_out = np.array([r["freq"] for r in all_results])
    hrtf_sparse = np.array([r["hrtf"] for r in all_results])
    source_m = all_results[0]["source_m"]

    print(f"\n{'='*60}")
    print("Interpolating to dense grid...")
    hrtf_dense = interp_hrtf_spline(expansion_freqs_out, hrtf_sparse, TARGET_FREQS)

    np.savez(
        HRTF_INTERP,
        target_freqs=TARGET_FREQS,
        expansion_freqs=expansion_freqs_out,
        hrtf_dense=hrtf_dense,
        hrtf_sparse=hrtf_sparse,
        azimuths=horiz_az,
        eval_radius=eval_radius,
        source_m=source_m,
    )

    n_dense = len(TARGET_FREQS)
    n_exp = len(expansion_freqs_out)
    pct_saved = 100.0 * (1.0 - n_exp / n_dense)
    avg_per_freq = dt_total / n_exp

    print(f"\n{'='*60}")
    print(f"Expansion solves: {n_exp}  |  Dense grid: {n_dense}")
    print(f"Total solve time: {dt_total:.1f}s  ({avg_per_freq:.1f}s avg/freq)")
    print(f"Estimated time saved vs full solve: ~{pct_saved:.0f}%")
    print(f"Saved: {HRTF_INTERP}")
