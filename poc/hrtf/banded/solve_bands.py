"""
HRTF POC — Banded BEM Solve
==============================
Solve across three frequency bands, each with its own graded mesh.

Usage:
    python banded/solve_bands.py
"""

import sys
import time
from pathlib import Path

HRTF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HRTF_ROOT))

import numpy as np

from common.bem import compute_source, load_mesh_space, mass_matrix_preconditioner, solve_burton_miller
from config import BAND_LIMITS, PHI_BANDED

FREQUENCIES = np.array([1000.0, 5000.0, 10000.0])

BANDS = []
for f_lo, f_hi, mesh_path in BAND_LIMITS:
    freqs = FREQUENCIES[(FREQUENCIES >= f_lo) & (FREQUENCIES <= f_hi)]
    if len(freqs) > 0:
        BANDS.append({"mesh": mesh_path, "freqs": freqs})


def solve_band(band_cfg):
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

        band_results.append({
            "freq": freq,
            "coeffs": x,
            "n_dofs": n_dofs,
            "mesh": str(mesh_path),
            "source_m": source_m,
        })

    return band_results


if __name__ == "__main__":
    all_freqs = np.concatenate([b["freqs"] for b in BANDS])
    print(f"Total: {len(all_freqs)} frequencies across {len(BANDS)} bands")
    print(f"Range: {all_freqs.min():.0f}–{all_freqs.max():.0f} Hz")

    t_total = time.perf_counter()
    all_results = []

    for i, band in enumerate(BANDS):
        f_lo, f_hi = band["freqs"][0], band["freqs"][-1]
        print(f"\n{'='*60}")
        print(f"Band {i + 1}/{len(BANDS)}: {f_lo:.0f}–{f_hi:.0f} Hz  "
              f"({len(band['freqs'])} freqs)")
        print(f"{'='*60}")
        all_results.extend(solve_band(band))

    dt_total = time.perf_counter() - t_total

    out_freqs = np.array([r["freq"] for r in all_results])
    out_sources = np.array([r["source_m"] for r in all_results])
    out_meshes = [r["mesh"] for r in all_results]
    out_n_dofs = np.array([r["n_dofs"] for r in all_results])

    max_dofs = max(r["n_dofs"] for r in all_results)
    out_coeffs = np.zeros((len(all_results), max_dofs), dtype=complex)
    for i, r in enumerate(all_results):
        out_coeffs[i, :r["n_dofs"]] = r["coeffs"]

    np.savez(
        PHI_BANDED,
        frequencies=out_freqs,
        phi_coeffs=out_coeffs,
        n_dofs=out_n_dofs,
        meshes=np.array(out_meshes),
        source_m=out_sources[0],
        source_normal=np.zeros(3),
    )

    print(f"\n{'='*60}")
    print(f"Total: {dt_total:.1f}s  ({dt_total / len(all_results):.1f}s avg/freq)")
    print(f"Saved: {PHI_BANDED}")
