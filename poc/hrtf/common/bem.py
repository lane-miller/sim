"""BEM mesh loading, source placement, and far-field evaluation."""

import platform
import resource

import bempp_cl.api as bempp
import numpy as np
import trimesh
from scipy.sparse.linalg import LinearOperator, gmres as scipy_gmres, splu

from config import C_AIR, SOURCE_LEFT_MM, SOURCE_OFFSET_MM

bempp.DEFAULT_DEVICE_INTERFACE = "opencl"


def get_mem_gb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if platform.system() == "Darwin":
        return rss / 1e9
    return rss / 1e6


def load_mesh_space(mesh_path):
    """Load STL mesh and build Bempp grid + P1 space."""
    tm = trimesh.load(str(mesh_path), force="mesh")
    vertices = (tm.vertices / 1000.0).astype(np.float64).T
    elements = tm.faces.astype(np.int32).T
    grid = bempp.Grid(vertices, elements)
    space = bempp.function_space(grid, "P", 1)
    return tm, grid, space


def compute_source(mesh_path, source_left_mm=SOURCE_LEFT_MM, offset_mm=SOURCE_OFFSET_MM):
    """Compute source position from mesh (ear canal + normal offset)."""
    tm = trimesh.load(str(mesh_path), force="mesh")
    dists = np.linalg.norm(tm.vertices - source_left_mm, axis=1)
    nearest_vidx = np.argmin(dists)
    face_mask = np.any(tm.faces == nearest_vidx, axis=1)
    local_normal = tm.face_normals[face_mask].mean(axis=0)
    local_normal /= np.linalg.norm(local_normal)
    source_mm = source_left_mm + offset_mm * local_normal
    return source_mm / 1000.0, local_normal


def mass_matrix_preconditioner(space):
    """Build mass-matrix LU preconditioner for GMRES."""
    n_dofs = space.global_dof_count
    mass = bempp.operators.boundary.sparse.identity(space, space, space)
    mass_mat = mass.weak_form().A.astype(complex)
    mass_lu = splu(mass_mat.tocsc())
    M_precond = lambda x: mass_lu.solve(x)
    return LinearOperator(shape=(n_dofs, n_dofs), matvec=M_precond, dtype=complex)


def evaluate_hrtf(space, coeffs, freq, source_m, eval_points):
    """Evaluate HRTF = p_total / p_inc at far-field points."""
    k = 2 * np.pi * freq / C_AIR
    x0 = source_m

    phi_fun = bempp.GridFunction(space, coefficients=coeffs)

    dlp_pot = bempp.operators.potential.helmholtz.double_layer(
        space, eval_points, k
    )
    p_scattered = dlp_pot @ phi_fun

    dx = eval_points[0, :] - x0[0]
    dy = eval_points[1, :] - x0[1]
    dz = eval_points[2, :] - x0[2]
    r_eval = np.sqrt(dx**2 + dy**2 + dz**2)
    p_inc_eval = np.exp(1j * k * r_eval) / (4 * np.pi * r_eval)

    p_total = p_inc_eval + p_scattered.ravel()
    return p_total / p_inc_eval


def solve_burton_miller(space, freq, source_m, rtol=1e-4, M=None, callback=None):
    """Solve Burton-Miller BEM system; return surface coefficients."""
    k = 2 * np.pi * freq / C_AIR
    alpha = 1j / k
    x0 = source_m

    identity = bempp.operators.boundary.sparse.identity(space, space, space)

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

    kwargs = {"rtol": rtol, "callback_type": "legacy"}
    if M is not None:
        kwargs["M"] = M
    if callback is not None:
        kwargs["callback"] = callback

    x, info = scipy_gmres(A_discrete, b_vec, **kwargs)
    return x, info
