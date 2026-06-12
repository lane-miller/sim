"""
HRTF POC — BEM Solve
=====================
Exterior rigid scattering via Burton-Miller formulation.
Point monopole source at left ear canal entrance (reciprocal HRTF).

Burton-Miller formulation:
    (D - 0.5I - α*hyp) φ = -(p_inc + α * ∂p_inc/∂n)

Representation formula (exterior, rigid):
    p_total(x) = p_inc(x) + D_pot[φ](x)

Bempp sign convention: hyp_bempp = -H_standard.
"""

import numpy as np
import bempp_cl.api as bempp
import trimesh
import time
import platform
import resource
from pathlib import Path
from scipy.sparse.linalg import gmres as scipy_gmres

bempp.DEFAULT_DEVICE_INTERFACE = "opencl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MESH_STL = OUTPUT_DIR / "FABIAN_6k_HATO0_graded.stl"
OUTPUT_FILE = OUTPUT_DIR / "phi_graded.npz"

C_AIR = 343.18
SOURCE_LEFT_MM = np.array([-2.22, 66.23, -2.00])
SOURCE_OFFSET_MM = 1.0
FREQUENCIES = np.array([1000.0, 4000.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_mem_gb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if platform.system() == "Darwin":
        return rss / 1e9
    else:
        return rss / 1e6


# ---------------------------------------------------------------------------
# Load mesh, convert mm → m
# ---------------------------------------------------------------------------
tm = trimesh.load(str(MESH_STL), force="mesh")

vertices = (tm.vertices / 1000.0).astype(np.float64).T
elements = tm.faces.astype(np.int32).T

grid = bempp.Grid(vertices, elements)
space = bempp.function_space(grid, "P", 1)
identity = bempp.operators.boundary.sparse.identity(space, space, space)

n_dofs = space.global_dof_count
dense_matrix_gb = (n_dofs ** 2 * 16) / 1e9
print(f"Grid: {grid.number_of_elements} elements, {n_dofs} DOFs (P1 vertices)")
print(f"Dense matrix estimate: {dense_matrix_gb:.2f} GB")

# ---------------------------------------------------------------------------
# Source position: ear canal + outward normal offset
# ---------------------------------------------------------------------------
dists = np.linalg.norm(tm.vertices - SOURCE_LEFT_MM, axis=1)
nearest_vidx = np.argmin(dists)

face_mask = np.any(tm.faces == nearest_vidx, axis=1)
local_normal = tm.face_normals[face_mask].mean(axis=0)
local_normal /= np.linalg.norm(local_normal)

source_mm = SOURCE_LEFT_MM + SOURCE_OFFSET_MM * local_normal
source_m = source_mm / 1000.0

print(f"Source (mm): ({source_mm[0]:.2f}, {source_mm[1]:.2f}, {source_mm[2]:.2f})")
print(f"Normal:      ({local_normal[0]:.3f}, {local_normal[1]:.3f}, {local_normal[2]:.3f})")

# ---------------------------------------------------------------------------
# BEM solve — frequency loop
# ---------------------------------------------------------------------------
phi_coeffs = np.zeros((len(FREQUENCIES), n_dofs), dtype=complex)
x0 = source_m

print(f"\nSolving {len(FREQUENCIES)} frequencies (Burton-Miller, P1)...")
t_total_start = time.perf_counter()

for i, freq in enumerate(FREQUENCIES):
    k = 2 * np.pi * freq / C_AIR
    alpha = 1j / k

    print(f"\n[{i + 1}/{len(FREQUENCIES)}] f={freq:.1f} Hz  k={k:.4f}  "
          f"λ={C_AIR / freq * 1000:.1f} mm")

    t_asm = time.perf_counter()

    @bempp.complex_callable
    def p_inc_callable(point, normal, domain_index, result):
        dx = point[0] - x0[0]
        dy = point[1] - x0[1]
        dz = point[2] - x0[2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        result[0] = np.exp(1j * k * r) / (4 * np.pi * r)

    @bempp.complex_callable
    def dp_inc_dn_callable(point, normal, domain_index, result):
        dx = point[0] - x0[0]
        dy = point[1] - x0[1]
        dz = point[2] - x0[2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        drdn = (dx * normal[0] + dy * normal[1] + dz * normal[2]) / r
        result[0] = np.exp(1j * k * r) / (4 * np.pi * r) * (1j * k - 1.0 / r) * drdn

    p_inc_fun = bempp.GridFunction(space, fun=p_inc_callable)
    dp_inc_dn_fun = bempp.GridFunction(space, fun=dp_inc_dn_callable)

    dlp = bempp.operators.boundary.helmholtz.double_layer(space, space, space, k)
    hyp = bempp.operators.boundary.helmholtz.hypersingular(space, space, space, k)

    lhs = dlp - 0.5 * identity - alpha * hyp
    rhs = -(p_inc_fun + alpha * dp_inc_dn_fun)

    A_discrete = lhs.weak_form()
    b_vec = rhs.projections(lhs.dual_to_range)

    dt_asm = time.perf_counter() - t_asm
    print(f"  Assembly: {dt_asm:.1f}s  mem={get_mem_gb():.2f} GB")

    # -- GMRES solve --
    iter_state = [0, time.perf_counter()]

    def gmres_callback(residual_norm):
        iter_state[0] += 1
        if iter_state[0] % 10 == 0 or iter_state[0] == 1:
            elapsed = time.perf_counter() - iter_state[1]
            print(
                f"    iter {iter_state[0]:4d}  "
                f"|r|={residual_norm:.3e}  "
                f"elapsed={elapsed:.1f}s  "
                f"mem={get_mem_gb():.2f} GB"
            )

    x, info = scipy_gmres(A_discrete, b_vec, rtol=1e-5, callback=gmres_callback,
                           callback_type='legacy')

    dt_solve = time.perf_counter() - iter_state[1]

    if info != 0:
        print(f"  GMRES did not converge (info={info})")

    phi_coeffs[i, :] = x
    print(f"  Solve: {dt_solve:.1f}s  {iter_state[0]} iters  "
          f"|phi|={np.linalg.norm(x):.6f}")

t_total = time.perf_counter() - t_total_start
print(f"\nTotal: {t_total:.1f}s  ({t_total / len(FREQUENCIES):.1f}s avg/freq)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.savez(
    OUTPUT_FILE,
    frequencies=FREQUENCIES,
    phi_coeffs=phi_coeffs,
    source_m=source_m,
    source_normal=local_normal,
)
print(f"Saved: {OUTPUT_FILE}")