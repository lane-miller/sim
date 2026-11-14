"""Yee grid geometry, coordinates, CPML slab indices, and field allocation (Stage A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fdtd.config import Config, Derived

_FLOAT32 = np.dtype(np.float32)


@dataclass(frozen=True)
class SlabRange:
    """Half-open index intervals ``[lo, hi)`` along one axis for a CPML face slab."""

    lo: int
    hi: int

    def thickness(self) -> int:
        return self.hi - self.lo


@dataclass(frozen=True)
class CpmlSlabs:
    """CPML slab index ranges on the staggered acoustic Yee grid.

    **Pressure** ``p[i, j, k]`` uses the same six face slabs on ``i``, ``j``, and
    ``k`` (cell-center indices).

    **Velocity** components live on faces; the normal axis uses one extra index
    at the outer boundary:

    * ``vx[i, j, k]``: ``i ∈ [0, nx]`` — x-normal CPML slabs are
      ``x_neg: i ∈ [0, cpml]``, ``x_pos: i ∈ [nx - cpml, nx + 1)`` (``cpml + 1``
      faces thick).
    * ``vy[i, j, k]``: ``j ∈ [0, ny]`` — y slabs ``[0, cpml]`` and
      ``[ny - cpml, ny + 1)``.
    * ``vz[i, j, k]``: ``k ∈ [0, nz]`` — z slabs ``[0, cpml]`` and
      ``[nz - cpml, nz + 1)``.

    Tangential axes for a given component span the full staggered extent
    ``(nx or nx±1, …)`` inside each slab kernel; only the documented normal-axis
    ranges differ from ``p``.

    **Interior** (non-CPML cell centers): ``i ∈ [cpml, nx - cpml)`` (and likewise
    for ``j``, ``k``).
    """

    cpml_cells: int
    p_x_neg: SlabRange
    p_x_pos: SlabRange
    p_y_neg: SlabRange
    p_y_pos: SlabRange
    p_z_neg: SlabRange
    p_z_pos: SlabRange
    vx_x_neg: SlabRange
    vx_x_pos: SlabRange
    vy_y_neg: SlabRange
    vy_y_pos: SlabRange
    vz_z_neg: SlabRange
    vz_z_pos: SlabRange
    i_inner: SlabRange
    j_inner: SlabRange
    k_inner: SlabRange


@dataclass
class Fields:
    """Allocated FDTD arrays (float32, C-contiguous)."""

    p: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    psi_p_x_neg: np.ndarray
    psi_p_x_pos: np.ndarray
    psi_p_y_neg: np.ndarray
    psi_p_y_pos: np.ndarray
    psi_p_z_neg: np.ndarray
    psi_p_z_pos: np.ndarray
    psi_vx_neg: np.ndarray
    psi_vx_pos: np.ndarray
    psi_vy_neg: np.ndarray
    psi_vy_pos: np.ndarray
    psi_vz_neg: np.ndarray
    psi_vz_pos: np.ndarray


def _slab(lo: int, hi: int) -> SlabRange:
    return SlabRange(lo, hi)


def build_cpml_slabs(nx: int, ny: int, nz: int, cpml: int) -> CpmlSlabs:
    c = cpml
    return CpmlSlabs(
        cpml_cells=c,
        p_x_neg=_slab(0, c),
        p_x_pos=_slab(nx - c, nx),
        p_y_neg=_slab(0, c),
        p_y_pos=_slab(ny - c, ny),
        p_z_neg=_slab(0, c),
        p_z_pos=_slab(nz - c, nz),
        vx_x_neg=_slab(0, c + 1),
        vx_x_pos=_slab(nx - c, nx + 1),
        vy_y_neg=_slab(0, c + 1),
        vy_y_pos=_slab(ny - c, ny + 1),
        vz_z_neg=_slab(0, c + 1),
        vz_z_pos=_slab(nz - c, nz + 1),
        i_inner=_slab(c, nx - c),
        j_inner=_slab(c, ny - c),
        k_inner=_slab(c, nz - c),
    )


class Grid:
    """Structured Yee grid for linear acoustics FDTD."""

    def __init__(self, cfg: Config, derived: Derived) -> None:
        self.cfg = cfg
        self.derived = derived
        self.h = float(derived.h)
        self.nx = derived.nx
        self.ny = derived.ny
        self.nz = derived.nz
        self.cpml = derived.cpml_cells
        self.cpml_slabs = build_cpml_slabs(
            self.nx, self.ny, self.nz, self.cpml
        )

        # Stage A free-field: origin at the domain center (meters).
        # TODO(Stage C): when cfg.box is set, shift so the origin is the radiator center
        # on the front face (+z outward normal), per brief §1.
        self.x_p = self._cell_centers(self.nx)
        self.y_p = self._cell_centers(self.ny)
        self.z_p = self._cell_centers(self.nz)
        self.x_vx = self._faces(self.nx)
        self.y_vy = self._faces(self.ny)
        self.z_vz = self._faces(self.nz)

    def _cell_centers(self, n: int) -> np.ndarray:
        i = np.arange(n, dtype=_FLOAT32)
        return (i + 0.5 - 0.5 * n) * np.float32(self.h)

    def _faces(self, n: int) -> np.ndarray:
        i = np.arange(n + 1, dtype=_FLOAT32)
        return (i - 0.5 * n) * np.float32(self.h)

    def coords_of(self, i: int, j: int, k: int) -> tuple[float, float, float]:
        """Pressure cell-center coordinates (m) at indices ``(i, j, k)``."""
        return (
            float(self.x_p[i]),
            float(self.y_p[j]),
            float(self.z_p[k]),
        )

    def index_of(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Nearest pressure cell indices to physical coordinates ``(x, y, z)``."""
        h = self.h
        i = int(round(x / h + 0.5 * self.nx - 0.5))
        j = int(round(y / h + 0.5 * self.ny - 0.5))
        k = int(round(z / h + 0.5 * self.nz - 0.5))
        i = max(0, min(self.nx - 1, i))
        j = max(0, min(self.ny - 1, j))
        k = max(0, min(self.nz - 1, k))
        return i, j, k


def _zeros(shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    return np.zeros(shape, dtype=dtype, order="C")


def allocate_fields(
    grid: Grid, dtype: np.dtype[Any] = _FLOAT32
) -> Fields:
    """Allocate zeroed field and CPML auxiliary arrays (slab-local shapes only)."""
    if dtype != _FLOAT32:
        dtype = np.dtype(dtype)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    c = grid.cpml
    p = _zeros((nx, ny, nz), dtype)
    vx = _zeros((nx + 1, ny, nz), dtype)
    vy = _zeros((nx, ny + 1, nz), dtype)
    vz = _zeros((nx, ny, nz + 1), dtype)

    psi_p_x_neg = _zeros((c, ny, nz), dtype)
    psi_p_x_pos = _zeros((c, ny, nz), dtype)
    psi_p_y_neg = _zeros((nx, c, nz), dtype)
    psi_p_y_pos = _zeros((nx, c, nz), dtype)
    psi_p_z_neg = _zeros((nx, ny, c), dtype)
    psi_p_z_pos = _zeros((nx, ny, c), dtype)

    psi_vx_neg = _zeros((c + 1, ny, nz), dtype)
    psi_vx_pos = _zeros((c + 1, ny, nz), dtype)
    psi_vy_neg = _zeros((nx, c + 1, nz), dtype)
    psi_vy_pos = _zeros((nx, c + 1, nz), dtype)
    psi_vz_neg = _zeros((nx, ny, c + 1), dtype)
    psi_vz_pos = _zeros((nx, ny, c + 1), dtype)

    return Fields(
        p=p,
        vx=vx,
        vy=vy,
        vz=vz,
        psi_p_x_neg=psi_p_x_neg,
        psi_p_x_pos=psi_p_x_pos,
        psi_p_y_neg=psi_p_y_neg,
        psi_p_y_pos=psi_p_y_pos,
        psi_p_z_neg=psi_p_z_neg,
        psi_p_z_pos=psi_p_z_pos,
        psi_vx_neg=psi_vx_neg,
        psi_vx_pos=psi_vx_pos,
        psi_vy_neg=psi_vy_neg,
        psi_vy_pos=psi_vy_pos,
        psi_vz_neg=psi_vz_neg,
        psi_vz_pos=psi_vz_pos,
    )


def _nbytes_array(a: np.ndarray) -> int:
    return int(a.nbytes)


def memory_bytes(fields: Fields) -> int:
    """Total bytes of all arrays in *fields*."""
    total = 0
    for f in fields.__dataclass_fields__:
        total += _nbytes_array(getattr(fields, f))
    return total
