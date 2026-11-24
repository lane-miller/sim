"""Cut-cell FDTD kernel path (v0.3 step 6a); float32 numba, gated vs v0.2 ``kernels.step``.

Index layout matches :mod:`boxrad.grid` / :mod:`boxrad.cutcell` (``alpha_*`` on ``vx/vy/vz`` faces).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from boxrad.cutcell import Fractions
from boxrad.grid import CpmlSlabs, Fields
from boxrad.kernels import UpdateCoeffs, update_p, zero_rigid_face_velocities, update_v

_F32 = np.float32


@dataclass(frozen=True)
class FastRadiator:
    """Axis-aligned flat radiator: face indices into ``vx``, ``vy``, or ``vz``."""

    axis: int
    idx: np.ndarray
    sw: np.ndarray


@dataclass(frozen=True)
class PatchRadiator:
    """Patch-flux radiator: pressure-cell flat indices and ``a/h²`` coefficients."""

    cells: np.ndarray
    coef: np.ndarray


@dataclass(frozen=True)
class CutOperator:
    afl_x: np.ndarray
    afl_y: np.ndarray
    afl_z: np.ndarray
    open_x: np.ndarray
    open_y: np.ndarray
    open_z: np.ndarray
    inv_beta: np.ndarray
    radiators_fast: tuple[FastRadiator, ...]
    radiators_patch: tuple[PatchRadiator, ...]
    closed_faces_x: np.ndarray
    closed_faces_y: np.ndarray
    closed_faces_z: np.ndarray
    special_cells: np.ndarray


def _closed_faces_touching_fluid(
    open_f: np.ndarray, inv_beta: np.ndarray, axis: int
) -> np.ndarray:
    """Flat indices of faces with open==0 adjacent to at least one cell with inv_beta>0."""
    nx, ny, nz = inv_beta.shape
    fluid = inv_beta > _F32(0)
    if axis == 0:
        touch = np.zeros(open_f.shape, dtype=bool)
        touch[1:, :, :] |= fluid
        touch[:-1, :, :] |= fluid
    elif axis == 1:
        touch = np.zeros(open_f.shape, dtype=bool)
        touch[:, 1:, :] |= fluid
        touch[:, :-1, :] |= fluid
    else:
        touch = np.zeros(open_f.shape, dtype=bool)
        touch[:, :, 1:] |= fluid
        touch[:, :, :-1] |= fluid
    closed = (open_f == _F32(0)) & touch
    return np.flatnonzero(closed).astype(np.int64)


def _special_cell_indices(
    inv_beta: np.ndarray,
    afl_x: np.ndarray,
    afl_y: np.ndarray,
    afl_z: np.ndarray,
) -> np.ndarray:
    """Cells needing cut-form pressure overwrite after legacy ``update_p``."""
    nx, ny, nz = inv_beta.shape
    fluid = inv_beta > _F32(0)
    one = _F32(1.0)
    not_beta_one = fluid & (inv_beta != one)
    any_afl_not_one = fluid & (
        (afl_x[:nx] != one)
        | (afl_x[1:] != one)
        | (afl_y[:, :ny, :] != one)
        | (afl_y[:, 1:, :] != one)
        | (afl_z[:, :, :nz] != one)
        | (afl_z[:, :, 1:] != one)
    )
    special = not_beta_one | any_afl_not_one
    return np.flatnonzero(special.reshape(-1)).astype(np.int64)


def build_operator(
    fractions: Fractions,
    beta_tilde: np.ndarray,
    radiators_fast: list[FastRadiator],
    radiators_patch: list[PatchRadiator],
    h: float,
) -> CutOperator:
    """Build float32 cut-operator arrays (snapped ``α``, floored ``β̃``)."""
    ax, ay, az = fractions.alpha
    afl_x = np.asarray(ax, dtype=_F32).copy()
    afl_y = np.asarray(ay, dtype=_F32).copy()
    afl_z = np.asarray(az, dtype=_F32).copy()
    # fast-path radiator faces use unit α for the face update
    afls = (afl_x, afl_y, afl_z)
    for rd in radiators_fast:
        arr = afls[rd.axis]
        arr.reshape(-1)[rd.idx] = _F32(1.0)
    open_x = (afl_x > _F32(0)).astype(_F32)
    open_y = (afl_y > _F32(0)).astype(_F32)
    open_z = (afl_z > _F32(0)).astype(_F32)
    bt = np.asarray(beta_tilde, dtype=_F32)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_beta = np.where(bt > _F32(0), _F32(1.0) / bt, _F32(0.0)).astype(_F32)
    _ = float(h)  # reserved for patch coef scaling at build sites
    cfx = _closed_faces_touching_fluid(open_x, inv_beta, 0)
    cfy = _closed_faces_touching_fluid(open_y, inv_beta, 1)
    cfz = _closed_faces_touching_fluid(open_z, inv_beta, 2)
    sp = _special_cell_indices(inv_beta, afl_x, afl_y, afl_z)
    return CutOperator(
        afl_x=afl_x,
        afl_y=afl_y,
        afl_z=afl_z,
        open_x=open_x,
        open_y=open_y,
        open_z=open_z,
        inv_beta=inv_beta,
        radiators_fast=tuple(radiators_fast),
        radiators_patch=tuple(radiators_patch),
        closed_faces_x=cfx,
        closed_faces_y=cfy,
        closed_faces_z=cfz,
        special_cells=sp,
    )


def assert_geometry_clear_of_cpml(op: CutOperator, slabs: CpmlSlabs) -> None:
    """Raise if cut geometry intrudes into CPML (require open=1, afl=1, inv_beta=1 there)."""
    c = slabs.cpml_cells
    nx = op.inv_beta.shape[0]
    ny = op.inv_beta.shape[1]
    nz = op.inv_beta.shape[2]
    tol = _F32(1.0 - 1e-5)

    def _check_p(lo: int, hi: int, axis: int) -> None:
        sl = [slice(None)] * 3
        sl[axis] = slice(lo, hi)
        block = op.inv_beta[tuple(sl)]
        if np.any(block < tol):
            raise ValueError("CPML slab: inv_beta != 1")

    _check_p(0, c, 0)
    _check_p(nx - c, nx, 0)
    _check_p(0, c, 1)
    _check_p(ny - c, ny, 1)
    _check_p(0, c, 2)
    _check_p(nz - c, nz, 2)

    def _check_face(afl: np.ndarray, lo: int, hi: int, axis: int) -> None:
        sl = [slice(None)] * 3
        sl[axis] = slice(lo, hi)
        block = afl[tuple(sl)]
        if np.any(block < tol):
            raise ValueError("CPML slab: face afl != 1")

    _check_face(op.afl_x, 0, c + 1, 0)
    _check_face(op.afl_x, nx - c, nx + 1, 0)
    _check_face(op.afl_y, 0, c + 1, 1)
    _check_face(op.afl_y, ny - c, ny + 1, 1)
    _check_face(op.afl_z, 0, c + 1, 2)
    _check_face(op.afl_z, nz - c, nz + 1, 2)


@njit(parallel=True, fastmath=False, cache=True)
def apply_open_mask(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    open_x: np.ndarray,
    open_y: np.ndarray,
    open_z: np.ndarray,
) -> None:
    nx, ny, nz = vx.shape[0] - 1, vy.shape[1] - 1, vz.shape[2] - 1
    for j in prange(vx.shape[1]):
        for k in range(vx.shape[2]):
            for i in range(vx.shape[0]):
                vx[i, j, k] *= open_x[i, j, k]
    for i in prange(vy.shape[0]):
        for k in range(vy.shape[2]):
            for j in range(vy.shape[1]):
                vy[i, j, k] *= open_y[i, j, k]
    for i in prange(vz.shape[0]):
        for j in range(vz.shape[1]):
            for k in range(vz.shape[2]):
                vz[i, j, k] *= open_z[i, j, k]
    _ = nx, ny, nz


@njit(cache=True)
def zero_face_indices(v_flat: np.ndarray, idx: np.ndarray) -> None:
    for q in range(idx.shape[0]):
        v_flat[idx[q]] = _F32(0.0)


@njit(parallel=True, fastmath=False, cache=True)
def overwrite_special_cells(
    p: np.ndarray,
    p_old: np.ndarray,
    cells: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    afl_x: np.ndarray,
    afl_y: np.ndarray,
    afl_z: np.ndarray,
    cp: np.float32,
    inv_beta: np.ndarray,
) -> None:
    nx, ny, nz = p.shape
    pflat = p.reshape(-1)
    ibflat = inv_beta.reshape(-1)
    for q in prange(cells.shape[0]):
        c = cells[q]
        k = c % nz
        j = (c // nz) % ny
        i = c // (ny * nz)
        ib = ibflat[c]
        div_v = (
            afl_x[i + 1, j, k] * vx[i + 1, j, k]
            - afl_x[i, j, k] * vx[i, j, k]
            + afl_y[i, j + 1, k] * vy[i, j + 1, k]
            - afl_y[i, j, k] * vy[i, j, k]
            + afl_z[i, j, k + 1] * vz[i, j, k + 1]
            - afl_z[i, j, k] * vz[i, j, k]
        )
        pflat[c] = p_old[q] - (cp * ib) * div_v


@njit(cache=True)
def set_fastpath_faces(
    v_flat: np.ndarray,
    idx: np.ndarray,
    sw: np.ndarray,
    v_now: np.float32,
) -> None:
    for q in range(idx.shape[0]):
        v_flat[idx[q]] = sw[q] * v_now


@njit(parallel=True, fastmath=False, cache=True)
def update_p_cut(
    p: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    afl_x: np.ndarray,
    afl_y: np.ndarray,
    afl_z: np.ndarray,
    cp: np.float32,
    inv_beta: np.ndarray,
) -> None:
    nx, ny, nz = p.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                ib = inv_beta[i, j, k]
                if ib == _F32(0):
                    continue
                div_v = (
                    afl_x[i + 1, j, k] * vx[i + 1, j, k]
                    - afl_x[i, j, k] * vx[i, j, k]
                    + afl_y[i, j + 1, k] * vy[i, j + 1, k]
                    - afl_y[i, j, k] * vy[i, j, k]
                    + afl_z[i, j, k + 1] * vz[i, j, k + 1]
                    - afl_z[i, j, k] * vz[i, j, k]
                )
                p[i, j, k] -= (cp * ib) * div_v


@njit(cache=True)
def add_patch_flux(
    p: np.ndarray,
    cells: np.ndarray,
    coef: np.ndarray,
    cp: np.float32,
    inv_beta: np.ndarray,
    v_now: np.float32,
) -> None:
    pflat = p.reshape(-1)
    ibflat = inv_beta.reshape(-1)
    for q in range(cells.shape[0]):
        c = cells[q]
        pflat[c] -= (cp * ibflat[c]) * (-coef[q] * v_now)


def _domain_args(p: np.ndarray) -> tuple[int, int, int, int, int, int]:
    nx, ny, nz = p.shape
    return 0, nx, 0, ny, 0, nz


def _inject_test_source(
    fields: Fields,
    flat_index: int,
    coeff: np.float32,
    v_q: np.float32,
) -> None:
    p_flat = fields.p.reshape(-1)
    p_flat[int(flat_index)] += coeff * v_q


def step_cut_dense(
    fields: Fields,
    coeffs: UpdateCoeffs,
    op: CutOperator,
    cpml: object | None,
    v_now_list: list[np.float32],
    *,
    test_source: tuple[int, np.float32, np.float32] | None = None,
) -> None:
    """Dense cut-cell leapfrog step (full-domain ``update_p_cut``; reference path)."""
    i_lo, i_hi, j_lo, j_hi, k_lo, k_hi = _domain_args(fields.p)
    zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
    update_v(
        fields.p,
        fields.vx,
        fields.vy,
        fields.vz,
        coeffs.cv,
        i_lo,
        i_hi,
        j_lo,
        j_hi,
        k_lo,
        k_hi,
    )
    if cpml is not None:
        from boxrad.cpml import apply_v as cpml_apply_v

        cpml_apply_v(fields, cpml, coeffs)
    zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
    apply_open_mask(fields.vx, fields.vy, fields.vz, op.open_x, op.open_y, op.open_z)
    nf = len(op.radiators_fast)
    npatch = len(op.radiators_patch)
    if len(v_now_list) != nf + npatch:
        raise ValueError("v_now_list length must match fast + patch radiators")
    vxs = fields.vx.reshape(-1)
    vys = fields.vy.reshape(-1)
    vzs = fields.vz.reshape(-1)
    flats = (vxs, vys, vzs)
    for qi, rd in enumerate(op.radiators_fast):
        set_fastpath_faces(flats[rd.axis], rd.idx, rd.sw, v_now_list[qi])
    update_p_cut(
        fields.p,
        fields.vx,
        fields.vy,
        fields.vz,
        op.afl_x,
        op.afl_y,
        op.afl_z,
        coeffs.cp,
        op.inv_beta,
    )
    if test_source is not None:
        fi, coeff, v_q = test_source
        _inject_test_source(fields, fi, coeff, v_q)
    for pi, rd in enumerate(op.radiators_patch):
        add_patch_flux(
            fields.p,
            rd.cells,
            rd.coef,
            coeffs.cp,
            op.inv_beta,
            v_now_list[nf + pi],
        )
    if cpml is not None:
        from boxrad.cpml import apply_p as cpml_apply_p

        cpml_apply_p(fields, cpml, coeffs)


def step_cut(
    fields: Fields,
    coeffs: UpdateCoeffs,
    op: CutOperator,
    cpml: object | None,
    v_now_list: list[np.float32],
    *,
    test_source: tuple[int, np.float32, np.float32] | None = None,
) -> None:
    """Sparse cut-cell leapfrog step (precomputed closed faces and special cells)."""
    i_lo, i_hi, j_lo, j_hi, k_lo, k_hi = _domain_args(fields.p)
    zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
    update_v(
        fields.p,
        fields.vx,
        fields.vy,
        fields.vz,
        coeffs.cv,
        i_lo,
        i_hi,
        j_lo,
        j_hi,
        k_lo,
        k_hi,
    )
    if cpml is not None:
        from boxrad.cpml import apply_v as cpml_apply_v

        cpml_apply_v(fields, cpml, coeffs)
    zero_rigid_face_velocities(fields.vx, fields.vy, fields.vz)
    zero_face_indices(fields.vx.reshape(-1), op.closed_faces_x)
    zero_face_indices(fields.vy.reshape(-1), op.closed_faces_y)
    zero_face_indices(fields.vz.reshape(-1), op.closed_faces_z)
    nf = len(op.radiators_fast)
    npatch = len(op.radiators_patch)
    if len(v_now_list) != nf + npatch:
        raise ValueError("v_now_list length must match fast + patch radiators")
    vxs = fields.vx.reshape(-1)
    vys = fields.vy.reshape(-1)
    vzs = fields.vz.reshape(-1)
    flats = (vxs, vys, vzs)
    for qi, rd in enumerate(op.radiators_fast):
        set_fastpath_faces(flats[rd.axis], rd.idx, rd.sw, v_now_list[qi])
    p_flat = fields.p.reshape(-1)
    p_old = p_flat[op.special_cells].copy()
    update_p(
        fields.p,
        fields.vx,
        fields.vy,
        fields.vz,
        coeffs.cp,
        i_lo,
        i_hi,
        j_lo,
        j_hi,
        k_lo,
        k_hi,
    )
    overwrite_special_cells(
        fields.p,
        p_old,
        op.special_cells,
        fields.vx,
        fields.vy,
        fields.vz,
        op.afl_x,
        op.afl_y,
        op.afl_z,
        coeffs.cp,
        op.inv_beta,
    )
    if test_source is not None:
        fi, coeff, v_q = test_source
        _inject_test_source(fields, fi, coeff, v_q)
    for pi, rd in enumerate(op.radiators_patch):
        add_patch_flux(
            fields.p,
            rd.cells,
            rd.coef,
            coeffs.cp,
            op.inv_beta,
            v_now_list[nf + pi],
        )
    if cpml is not None:
        from boxrad.cpml import apply_p as cpml_apply_p

        cpml_apply_p(fields, cpml, coeffs)