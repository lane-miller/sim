import numpy as np
import bempp_cl.api as bempp
import os

MESH_FILE   = os.path.join(os.path.dirname(__file__), "outputs", "sphere.msh")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "outputs", "phi.npz")
FREQUENCIES = np.geomspace(100, 10000, 20)
C           = 343.0

grid     = bempp.import_grid(MESH_FILE)
space    = bempp.function_space(grid, "DP", 0)
identity = bempp.operators.boundary.sparse.identity(space, space, space)
n_dofs   = space.global_dof_count
print(f"Mesh loaded: {grid.number_of_elements} elements, {n_dofs} DOFs")

phi_coeffs = np.zeros((len(FREQUENCIES), n_dofs), dtype=complex)

for i, freq in enumerate(FREQUENCIES):
    k = 2 * np.pi * freq / C

    @bempp.complex_callable
    def p_inc_callable(point, normal, domain_index, result):
        result[0] = np.exp(1j * k * point[0])

    p_inc_fun = bempp.GridFunction(space, fun=p_inc_callable)
    dlp       = bempp.operators.boundary.helmholtz.double_layer(space, space, space, k)
    lhs       = dlp - 0.5 * identity
    rhs       = -p_inc_fun

    phi, info = bempp.linalg.gmres(lhs, rhs, tol=1e-5)
    if info != 0:
        print(f"  WARNING: GMRES did not converge at f={freq:.1f} Hz (info={info})")

    phi_coeffs[i, :] = phi.coefficients
    print(f"  f={freq:8.1f} Hz  k={k:.4f}  norm(phi)={np.linalg.norm(phi.coefficients):.6f}")

np.savez(OUTPUT_FILE, frequencies=FREQUENCIES, phi_coeffs=phi_coeffs)
print(f"\nSurface solutions saved to {OUTPUT_FILE}")