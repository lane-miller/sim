"""
HRTF POC — BEM Solve (legacy)
==============================
Single graded mesh, Burton-Miller formulation.

Usage:
    python legacy/solve.py
"""

import sys
import time
from pathlib import Path

HRTF_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HRTF_ROOT))

import numpy as np

from common.bem import compute_source, get_mem_gb, load_mesh_space, solve_burton_miller
from config import C_AIR, MESH_GRADED, OUTPUT_DIR, SOURCE_LEFT_MM

OUTPUT_FILE = OUTPUT_DIR / "phi_graded.npz"
FREQUENCIES = np.array([1000.0, 4000.0])

_, grid, space = load_mesh_space(MESH_GRADED)
n_dofs = space.global_dof_count
dense_matrix_gb = (n_dofs ** 2 * 16) / 1e9
print(f"Grid: {grid.number_of_elements} elements, {n_dofs} DOFs (P1 vertices)")
print(f"Dense matrix estimate: {dense_matrix_gb:.2f} GB")

source_m, local_normal = compute_source(MESH_GRADED)
print(f"Source (m): ({source_m[0]:.4f}, {source_m[1]:.4f}, {source_m[2]:.4f})")

phi_coeffs = np.zeros((len(FREQUENCIES), n_dofs), dtype=complex)
t_total_start = time.perf_counter()

print(f"\nSolving {len(FREQUENCIES)} frequencies (Burton-Miller, P1)...")

for i, freq in enumerate(FREQUENCIES):
    k = 2 * np.pi * freq / C_AIR
    print(f"\n[{i + 1}/{len(FREQUENCIES)}] f={freq:.1f} Hz  k={k:.4f}  "
          f"λ={C_AIR / freq * 1000:.1f} mm")

    t_asm = time.perf_counter()
    iter_state = [0, time.perf_counter()]

    def gmres_callback(residual_norm):
        iter_state[0] += 1
        if iter_state[0] % 10 == 0 or iter_state[0] == 1:
            elapsed = time.perf_counter() - iter_state[1]
            print(
                f"    iter {iter_state[0]:4d}  |r|={residual_norm:.3e}  "
                f"elapsed={elapsed:.1f}s  mem={get_mem_gb():.2f} GB"
            )

    x, info = solve_burton_miller(
        space, freq, source_m, rtol=1e-5, callback=gmres_callback,
    )
    dt_asm = time.perf_counter() - t_asm
    dt_solve = time.perf_counter() - iter_state[1]

    if info != 0:
        print(f"  GMRES did not converge (info={info})")

    phi_coeffs[i, :] = x
    print(f"  Assembly+solve: {dt_asm:.1f}s  {iter_state[0]} iters  "
          f"|phi|={np.linalg.norm(x):.6f}  solve={dt_solve:.1f}s")

t_total = time.perf_counter() - t_total_start
print(f"\nTotal: {t_total:.1f}s  ({t_total / len(FREQUENCIES):.1f}s avg/freq)")

np.savez(
    OUTPUT_FILE,
    frequencies=FREQUENCIES,
    phi_coeffs=phi_coeffs,
    source_m=source_m,
    source_normal=local_normal,
)
print(f"Saved: {OUTPUT_FILE}")
