"""Exterior Neumann Burton–Miller BEM reference (bempp-cl).

FDTD coordinates: radiator center at the origin, front face at *z* = 0, box interior
in −*z*. Time-harmonic data from bempp-cl use e^{−iωt}; map to the project
convention e^{+jωt} via :func:`to_plus_jwt` (complex conjugation) in one place
before any comparison with FDTD or published references.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import bempp_cl.api as bempp
import numpy as np
from bempp_cl.api.linalg import gmres as bempp_gmres
from scipy.sparse.linalg import gmres as scipy_gmres

RADIATOR_TAG = 1
BOX_TAG = 2
_DENSE_MAX_GB = 3.0
_GMRES_RTOL = 1e-5
_DEFAULT_EPW = 10
_DEFAULT_H_MAX_M = 0.02
_F64 = np.float64
_C128 = np.complex128


def h_target_from_frequency(
    f_hz: float,
    c: float,
    *,
    epw: float = _DEFAULT_EPW,
    h_max: float = _DEFAULT_H_MAX_M,
) -> float:
    """Mesh target length min(λ/epw, h_max) for time-harmonic frequency *f_hz*."""
    f_hz = float(f_hz)
    if f_hz <= 0.0:
        raise ValueError("f_hz must be positive")
    wavelength = float(c) / f_hz
    return min(wavelength / float(epw), float(h_max))


def h_target_sphere(ka: float, a: float, *, epw: float = _DEFAULT_EPW) -> float:
    """Sphere mesh target: min(λ/epw, a/4) with λ = 2πa/ka."""
    ka = float(ka)
    a = float(a)
    if ka <= 0.0 or a <= 0.0:
        raise ValueError("ka and a must be positive")
    wavelength = 2.0 * math.pi * a / ka
    return min(wavelength / float(epw), 0.25 * a)


def dedupe_frequencies_hz(freqs: Sequence[float], *, sep_hz: float = 1.0) -> np.ndarray:
    """Sort ascending; drop entries within *sep_hz* of the previous kept frequency."""
    if len(freqs) == 0:
        return np.asarray([], dtype=_F64)
    arr = np.sort(np.asarray(freqs, dtype=_F64))
    out: list[float] = [float(arr[0])]
    for f in arr[1:]:
        if f - out[-1] > float(sep_hz):
            out.append(float(f))
    return np.asarray(out, dtype=_F64)


class BoxMeshCache:
    """Reuse :func:`box_mesh` grids keyed by rounded *h_target* (m)."""

    def __init__(self) -> None:
        self._grids: dict[float, bempp.Grid] = {}

    def get(
        self,
        W: float,
        H: float,
        D: float,
        rad_rect: tuple[float, float],
        origin: tuple[float, float, float],
        h_target: float,
    ) -> bempp.Grid:
        key = round(float(h_target), 12)
        if key not in self._grids:
            self._grids[key] = box_mesh(W, H, D, rad_rect, origin, h_target)
        return self._grids[key]


@dataclass(frozen=True)
class BmSolveStats:
    n_dofs: int
    assemble_s: float
    solve_s: float
    gmres_iters: int
    dense_gb: float


def format_bem_progress_line(
    *,
    index: int,
    total: int,
    f_hz: float,
    h_m: float,
    stats: BmSolveStats,
    elapsed_s: float,
    eta_s: float | None,
    prefix: str = "bem",
) -> str:
    eta = "—" if eta_s is None else f"{eta_s:.1f}"
    return (
        f"[{prefix} {index}/{total}] f={f_hz:.2f} Hz h={h_m * 1e3:.2f} mm "
        f"DOF={stats.n_dofs} assemble={stats.assemble_s:.2f}s "
        f"solve={stats.solve_s:.2f}s iters={stats.gmres_iters} "
        f"elapsed={elapsed_s:.1f}s ETA={eta}s"
    )


def to_plus_jwt(z: np.ndarray | complex) -> np.ndarray | complex:
    """Map bempp e^{−iωt} phasors to this project's e^{+jωt} convention."""
    if np.isscalar(z):
        return np.conj(z)
    return np.conj(np.asarray(z, dtype=_C128))


def _estimate_dense_gb(n_dofs: int) -> float:
    return (n_dofs**2 * 16) / 1e9


def _subdivide_axis(edges: Sequence[float], h_target: float) -> np.ndarray:
    """Sorted 1-D nodes with every interval length ≤ *h_target*."""
    h_target = float(h_target)
    uniq = sorted(set(float(x) for x in edges))
    if len(uniq) < 2:
        return np.asarray(uniq, dtype=_F64)
    out: list[float] = [uniq[0]]
    for a, b in zip(uniq[:-1], uniq[1:]):
        length = b - a
        if length <= 0.0:
            continue
        n_seg = max(1, int(math.ceil(length / h_target)))
        for s in range(1, n_seg):
            out.append(a + length * s / n_seg)
        out.append(b)
    return np.asarray(out, dtype=_F64)


def box_mesh(
    W: float,
    H: float,
    D: float,
    rad_rect: tuple[float, float],
    origin: tuple[float, float, float],
    h_target: float,
) -> bempp.Grid:
    """Structured triangle mesh of a closed box surface (outward normals).

    *origin* is the minimum corner (x₀, y₀, z₀); the box is
    [x₀, x₀+W]×[y₀, y₀+H]×[z₀, z₀+D] with the front face at z = z₀+D (typically 0).
    *rad_rect* is (width, height) of the radiator rectangle on the front face,
    centered at the origin in the horizontal plane. Triangles on the radiator
    carry :data:`RADIATOR_TAG`; all others use :data:`BOX_TAG`.
    """
    W, H, D = float(W), float(H), float(D)
    rw, rh = float(rad_rect[0]), float(rad_rect[1])
    ox, oy, oz = (float(origin[0]), float(origin[1]), float(origin[2]))
    x0, x1 = ox, ox + W
    y0, y1 = oy, oy + H
    z0, z1 = oz, oz + D
    rx0, rx1 = -0.5 * rw, 0.5 * rw
    ry0, ry1 = -0.5 * rh, 0.5 * rh

    xs = _subdivide_axis([x0, x1, rx0, rx1], h_target)
    ys = _subdivide_axis([y0, y1, ry0, ry1], h_target)
    zs = _subdivide_axis([z0, z1], h_target)

    vertices: list[np.ndarray] = []
    vmap: dict[tuple[float, float, float], int] = {}
    elements: list[list[int]] = []
    domain: list[int] = []

    def _radiator_element(i: int, j: int, u: np.ndarray, v: np.ndarray) -> int:
        uc = 0.5 * (u[i] + u[i + 1])
        vc = 0.5 * (v[j] + v[j + 1])
        if abs(uc) <= 0.5 * rw + 1e-15 and abs(vc) <= 0.5 * rh + 1e-15:
            return RADIATOR_TAG
        return BOX_TAG

    def _add_face_tagged(u, v, fixed, flip: bool) -> None:
        axis, coord = fixed
        nu, nv = len(u), len(v)
        tang = [t for t in range(3) if t != axis]
        u_arr, v_arr = u, v

        def _vid(x: float, y: float, z: float) -> int:
            key = (x, y, z)
            if key not in vmap:
                vmap[key] = len(vertices)
                vertices.append(np.array([x, y, z], dtype=_F64))
            return vmap[key]

        for i in range(nu - 1):
            for j in range(nv - 1):
                corners = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    pt = [0.0, 0.0, 0.0]
                    pt[axis] = coord
                    pt[tang[0]] = float(u_arr[i + du])
                    pt[tang[1]] = float(v_arr[j + dv])
                    corners.append(_vid(*pt))
                a, b, c, d = corners
                tag = _radiator_element(i, j, u_arr, v_arr) if axis == 2 and coord == z1 else BOX_TAG
                if flip:
                    elements.append([a, c, b])
                    elements.append([a, d, c])
                else:
                    elements.append([a, b, c])
                    elements.append([a, c, d])
                domain.extend([tag, tag])

    # x = x0 (−x normal), x = x1 (+x)
    _add_face_tagged(ys, zs, (0, x0), flip=True)
    _add_face_tagged(ys, zs, (0, x1), flip=False)
    # y = y0 (−y), y = y1 (+y); tangents (x, z) ⇒ flip=False is −y, flip=True is +y
    _add_face_tagged(xs, zs, (1, y0), flip=False)
    _add_face_tagged(xs, zs, (1, y1), flip=True)
    # z = z0 (rear, −z), z = z1 (front, +z)
    _add_face_tagged(xs, ys, (2, z0), flip=True)
    _add_face_tagged(xs, ys, (2, z1), flip=False)

    verts = np.array(vertices, dtype=_F64).T
    elems = np.array(elements, dtype=np.int32).T
    tags = np.array(domain, dtype=np.int32)
    return bempp.Grid(verts, elems, domain_indices=tags)


@dataclass
class NeumannBmResult:
    """Exterior Neumann Burton–Miller solution (P1 pressure, DP0 Neumann data)."""

    grid: object
    space_p: object
    space_q: object
    p_gf: object
    neumann_fun: object
    k: float
    rho: float
    c: float
    vn_P0: float
    stats: BmSolveStats | None = None

    def zrad(self) -> complex:
        """Dimensionless z = ⟨p⟩_rad / (ρ c v_n) in e^{+jωt}."""
        domain = self.grid.domain_indices.astype(int)
        areas = self.grid.volumes.astype(float)
        rad = domain == RADIATOR_TAG
        if not np.any(rad):
            rad = np.ones(domain.shape, dtype=bool)
        p_el = self.p_gf.evaluate_on_element_centers().ravel()
        num = np.sum(p_el[rad] * areas[rad])
        den = np.sum(areas[rad])
        z_bem = num / (den * self.rho * self.c * self.vn_P0)
        return complex(to_plus_jwt(z_bem).item())

    def field_at(self, points: np.ndarray) -> np.ndarray:
        """Exterior pressure at *points* (N,3) via full single- plus double-layer potentials."""
        pts = np.asarray(points, dtype=_F64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        eval_pts = pts.T
        k = self.k
        slp_pot = bempp.operators.potential.helmholtz.single_layer(
            self.space_q, eval_pts, k
        )
        dlp_pot = bempp.operators.potential.helmholtz.double_layer(
            self.space_p, eval_pts, k
        )
        p_ext = (slp_pot @ self.neumann_fun - dlp_pot @ self.p_gf).ravel()
        return np.asarray(to_plus_jwt(p_ext), dtype=_C128)


def solve_neumann_bm(
    grid: bempp.Grid,
    k: float,
    vn_P0: float,
    rho: float,
    c: float,
    *,
    rtol: float = _GMRES_RTOL,
    use_scipy_gmres: bool = True,
    pulsating_sphere: bool = False,
    log_dof: bool = False,
) -> NeumannBmResult:
    """Solve exterior Neumann (Burton–Miller): prescribed normal velocity on the radiator."""
    k = float(k)
    omega = k * c
    space_p = bempp.function_space(grid, "P", 1)
    space_q = bempp.function_space(grid, "DP", 0)
    id_pp = bempp.operators.boundary.sparse.identity(space_p, space_p, space_p)
    id_qp = bempp.operators.boundary.sparse.identity(space_q, space_p, space_p)

    n_dofs = space_p.global_dof_count
    dense_gb = _estimate_dense_gb(n_dofs)
    if log_dof:
        print(
            f"  BEM DOFs={n_dofs}  est. dense={dense_gb:.2f} GB",
            flush=True,
        )
    if dense_gb > _DENSE_MAX_GB:
        raise RuntimeError(
            f"BEM dense matrix estimate {dense_gb:.2f} GB exceeds {_DENSE_MAX_GB} GB limit"
        )

    alpha = 1j / k

    @bempp.complex_callable
    def neumann_callable(point, normal, domain_index, result):
        if pulsating_sphere or int(domain_index) == RADIATOR_TAG:
            result[0] = 1j * omega * rho * vn_P0
        else:
            result[0] = 0.0

    neumann_fun = bempp.GridFunction(space_q, fun=neumann_callable)

    slp = bempp.operators.boundary.helmholtz.single_layer(space_q, space_p, space_p, k)
    dlp = bempp.operators.boundary.helmholtz.double_layer(space_p, space_p, space_p, k)
    adlp = bempp.operators.boundary.helmholtz.adjoint_double_layer(space_q, space_p, space_p, k)
    hyp = bempp.operators.boundary.helmholtz.hypersingular(space_p, space_p, space_p, k)

    # bempp-cl hypersingular sign: +α W (not −α W).
    lhs = 0.5 * id_pp - dlp + alpha * hyp
    rhs_op = -slp - alpha * (adlp + 0.5 * id_qp)

    gmres_iters = 0
    if use_scipy_gmres:
        t_asm = time.perf_counter()
        lhs_w = lhs.weak_form()
        rhs_proj = (rhs_op * neumann_fun).projections(lhs.dual_to_range)
        assemble_s = time.perf_counter() - t_asm

        def _iter_cb(_: object) -> None:
            nonlocal gmres_iters
            gmres_iters += 1

        t_slv = time.perf_counter()
        x, info = scipy_gmres(
            lhs_w,
            rhs_proj,
            rtol=rtol,
            callback=_iter_cb,
            callback_type="legacy",
        )
        solve_s = time.perf_counter() - t_slv
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")
        p_gf = bempp.GridFunction(space_p, coefficients=x)
    else:
        assemble_s = 0.0
        t_slv = time.perf_counter()
        p_gf, info = bempp_gmres(
            lhs,
            rhs_op * neumann_fun,
            rtol=rtol,
            use_strong_form=True,
            callback_type="legacy",
        )
        solve_s = time.perf_counter() - t_slv
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info})")

    stats = BmSolveStats(
        n_dofs=n_dofs,
        assemble_s=assemble_s,
        solve_s=solve_s,
        gmres_iters=gmres_iters,
        dense_gb=dense_gb,
    )

    return NeumannBmResult(
        grid=grid,
        space_p=space_p,
        space_q=space_q,
        p_gf=p_gf,
        neumann_fun=neumann_fun,
        k=k,
        rho=rho,
        c=c,
        vn_P0=vn_P0,
        stats=stats,
    )


def zrad_bem(sol: NeumannBmResult) -> complex:
    """Dimensionless radiation impedance from a :class:`NeumannBmResult` (e^{+jωt})."""
    return sol.zrad()


def zrad_sphere_analytic(ka: float) -> complex:
    """Pulsating sphere z = j ka / (1 + j ka), e^{+jωt}, unit normal velocity."""
    jka = 1j * float(ka)
    return jka / (1.0 + jka)