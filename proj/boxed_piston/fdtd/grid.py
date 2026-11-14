"""Yee grid geometry, coordinates, CPML slab indices, and field allocation (Stage A)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from fdtd.config import Config, Derived, Radiator

_FLOAT32 = np.dtype(np.float32)
_F64 = np.float64
DEFAULT_RADIATOR_SUPERSAMPLE = 32


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

    def __init__(
        self,
        cfg: Config,
        derived: Derived,
        *,
        radiator_supersample: int = DEFAULT_RADIATOR_SUPERSAMPLE,
    ) -> None:
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
        self.radiator_supersample = int(radiator_supersample)

        self.realized_W = float(derived.realized_W)
        self.realized_H = float(derived.realized_H)
        self.realized_D = float(derived.realized_D)

        # Box / radiator geometry (``None`` in free-field mode).
        self.i_box: int | None = None
        self.j_box: int | None = None
        self.k_box: int | None = None
        self.nbx = derived.box_nx
        self.nby = derived.box_ny
        self.nbz = derived.box_nz
        self.k_face_z: int | None = None
        self.box_cells: np.ndarray | None = None
        self.rigid_vx: np.ndarray | None = None
        self.rigid_vy: np.ndarray | None = None
        self.rigid_vz: np.ndarray | None = None
        self.radiator_face_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = (
            None
        )
        self.radiator_weights: np.ndarray | None = None
        self.radiator_area_exact: float | None = None
        self.radiator_area_grid: float | None = None

        if cfg.box is None:
            self.x_p = self._cell_centers_domain_centered(self.nx)
            self.y_p = self._cell_centers_domain_centered(self.ny)
            self.z_p = self._cell_centers_domain_centered(self.nz)
            self.x_vx = self._faces_domain_centered(self.nx)
            self.y_vy = self._faces_domain_centered(self.ny)
            self.z_vz = self._faces_domain_centered(self.nz)
        else:
            self._build_box_geometry(cfg)

    def _cell_centers_domain_centered(self, n: int) -> np.ndarray:
        i = np.arange(n, dtype=_FLOAT32)
        return (i + 0.5 - 0.5 * n) * np.float32(self.h)

    def _faces_domain_centered(self, n: int) -> np.ndarray:
        i = np.arange(n + 1, dtype=_FLOAT32)
        return (i - 0.5 * n) * np.float32(self.h)

    def _build_box_geometry(self, cfg: Config) -> None:
        """Origin at the radiator center; +z outward; box extends in −z (brief §1)."""
        d = self.derived
        gap = d.gap_cells
        c = self.cpml
        self.i_box = ib = c + gap
        self.j_box = jb = c + gap
        self.k_box = kb = c + gap
        nbx, nby, nbz = self.nbx, self.nby, self.nbz
        self.k_face_z = kb + nbz

        xc = float(cfg.radiator.x_c)
        yc = float(cfg.radiator.y_c)
        rw, rh, rd = self.realized_W, self.realized_H, self.realized_D
        h = self.h

        ii = np.arange(self.nx, dtype=_FLOAT32)
        jj = np.arange(self.ny, dtype=_FLOAT32)
        kk = np.arange(self.nz, dtype=_FLOAT32)
        self.x_p = (-xc - 0.5 * rw + (ii - float(ib) + 0.5) * h).astype(_FLOAT32)
        self.y_p = (-yc - 0.5 * rh + (jj - float(jb) + 0.5) * h).astype(_FLOAT32)
        self.z_p = (-rd + (kk - float(kb) + 0.5) * h).astype(_FLOAT32)
        self.x_vx = (-xc - 0.5 * rw + (np.arange(self.nx + 1, dtype=_FLOAT32) - float(ib)) * h)
        self.y_vy = (-yc - 0.5 * rh + (np.arange(self.ny + 1, dtype=_FLOAT32) - float(jb)) * h)
        self.z_vz = (-rd + (np.arange(self.nz + 1, dtype=_FLOAT32) - float(kb)) * h)

        p_shape = (self.nx, self.ny, self.nz)
        self.box_cells = np.zeros(p_shape, dtype=bool)
        self.box_cells[ib : ib + nbx, jb : jb + nby, kb : kb + nbz] = True

        self.rigid_vx = np.zeros((self.nx + 1, self.ny, self.nz), dtype=bool)
        self.rigid_vy = np.zeros((self.nx, self.ny + 1, self.nz), dtype=bool)
        self.rigid_vz = np.zeros((self.nx, self.ny, self.nz + 1), dtype=bool)

        j0, j1 = jb, jb + nby
        k0, k1 = kb, kb + nbz
        i0, i1 = ib, ib + nbx

        self.rigid_vx[ib, j0:j1, k0:k1] = True
        self.rigid_vx[ib + nbx, j0:j1, k0:k1] = True
        self.rigid_vy[i0:i1, jb, k0:k1] = True
        self.rigid_vy[i0:i1, jb + nby, k0:k1] = True
        self.rigid_vz[i0:i1, j0:j1, kb] = True
        self.rigid_vz[i0:i1, j0:j1, kb + nbz] = True

        w_plane = _radiator_face_weights(
            cfg.radiator,
            self.x_p[i0:i1],
            self.y_p[j0:j1],
            h,
            self.radiator_supersample,
        )
        kf = self.k_face_z
        self.rigid_vz[i0:i1, j0:j1, kf] &= w_plane == 0.0

        ii_sub, jj_sub = np.nonzero(w_plane > 0.0)
        ni = len(ii_sub)
        if ni:
            i_rad = ii_sub + i0
            j_rad = jj_sub + j0
            k_rad = np.full(ni, kf, dtype=np.int64)
            self.radiator_face_indices = (i_rad, j_rad, k_rad)
            self.radiator_weights = w_plane[w_plane > 0.0].astype(_F64)
        else:
            self.radiator_face_indices = (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
            self.radiator_weights = np.array([], dtype=_F64)

        self.radiator_area_exact = radiator_area_exact(cfg.radiator)
        self.radiator_area_grid = float(np.sum(w_plane) * h * h)

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
        if self.cfg.box is None:
            i = int(round(x / h + 0.5 * self.nx - 0.5))
            j = int(round(y / h + 0.5 * self.ny - 0.5))
            k = int(round(z / h + 0.5 * self.nz - 0.5))
        else:
            ib, jb, kb = self.i_box, self.j_box, self.k_box
            xc = float(self.cfg.radiator.x_c)
            yc = float(self.cfg.radiator.y_c)
            rw, rd = self.realized_W, self.realized_D
            i = int(round((x + xc + 0.5 * rw) / h + float(ib) - 0.5))
            j = int(round((y + yc + 0.5 * self.realized_H) / h + float(jb) - 0.5))
            k = int(round((z + rd) / h + float(kb) - 0.5))
        i = max(0, min(self.nx - 1, i))
        j = max(0, min(self.ny - 1, j))
        k = max(0, min(self.nz - 1, k))
        return i, j, k


def _radiator_half_extents_xy(rad: Radiator) -> tuple[float, float]:
    if rad.shape == "circle":
        r = 0.5 * rad.dims[0]
        return r, r
    if rad.shape in ("rect", "racetrack"):
        return 0.5 * rad.dims[0], 0.5 * rad.dims[1]
    raise ValueError(f"unknown radiator shape: {rad.shape!r}")


def point_inside_radiator(x: float, y: float, rad: Radiator) -> bool:
    """True if ``(x, y)`` lies inside the radiator outline (radiator-centered frame)."""
    if rad.shape == "circle":
        r = 0.5 * rad.dims[0]
        return x * x + y * y <= r * r
    if rad.shape == "rect":
        hw, hh = 0.5 * rad.dims[0], 0.5 * rad.dims[1]
        return abs(x) <= hw and abs(y) <= hh
    if rad.shape == "racetrack":
        w, h = rad.dims[0], rad.dims[1]
        d = min(w, h)
        r = 0.5 * d
        if w >= h:
            if abs(y) > r:
                return False
            if x < -(0.5 * w - r):
                return (x + 0.5 * w - r) ** 2 + y * y <= r * r
            if x > 0.5 * w - r:
                return (x - (0.5 * w - r)) ** 2 + y * y <= r * r
            return True
        if abs(x) > r:
            return False
        if y < -(0.5 * h - r):
            return x * x + (y + 0.5 * h - r) ** 2 <= r * r
        if y > 0.5 * h - r:
            return x * x + (y - (0.5 * h - r)) ** 2 <= r * r
        return True
    raise ValueError(f"unknown radiator shape: {rad.shape!r}")


def radiator_area_exact(rad: Radiator) -> float:
    """Analytic area (m²) of the radiator outline."""
    if rad.shape == "circle":
        r = 0.5 * rad.dims[0]
        return math.pi * r * r
    if rad.shape == "rect":
        return rad.dims[0] * rad.dims[1]
    if rad.shape == "racetrack":
        w, h = rad.dims[0], rad.dims[1]
        d = min(w, h)
        r = 0.5 * d
        if w >= h:
            return h * (w - d) + math.pi * r * r
        return w * (h - d) + math.pi * r * r
    raise ValueError(f"unknown radiator shape: {rad.shape!r}")


def _rect_face_weights(
    rad: Radiator,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    h: float,
) -> np.ndarray:
    """Exact axis-aligned overlap of each h×h face with the radiator rectangle."""
    hw, hh = 0.5 * rad.dims[0], 0.5 * rad.dims[1]
    nx, ny = len(x_centers), len(y_centers)
    half = 0.5 * h
    h2 = h * h
    w = np.zeros((nx, ny), dtype=_F64)
    for i in range(nx):
        xc = float(x_centers[i])
        ox = max(
            0.0,
            min(xc + half, hw) - max(xc - half, -hw),
        )
        for j in range(ny):
            yc = float(y_centers[j])
            oy = max(
                0.0,
                min(yc + half, hh) - max(yc - half, -hh),
            )
            w[i, j] = ox * oy / h2
    return w


def _radiator_face_weights(
    rad: Radiator,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    h: float,
    n_sub: int,
) -> np.ndarray:
    """Area fraction (float64) on each front-face ``vz`` patch, shape ``(nx_box, ny_box)``."""
    if rad.shape == "rect":
        return _rect_face_weights(rad, x_centers, y_centers, h)
    n_sub = max(1, int(n_sub))
    nx, ny = len(x_centers), len(y_centers)
    w = np.zeros((nx, ny), dtype=_F64)
    offsets = (np.arange(n_sub, dtype=_F64) + 0.5) / float(n_sub) - 0.5
    ox, oy = np.meshgrid(offsets * h, offsets * h, indexing="ij")
    flat_x = ox.ravel()
    flat_y = oy.ravel()
    n_pts = n_sub * n_sub
    for i in range(nx):
        xc = float(x_centers[i])
        for j in range(ny):
            yc = float(y_centers[j])
            inside = 0
            for px, py in zip(flat_x, flat_y):
                if point_inside_radiator(xc + px, yc + py, rad):
                    inside += 1
            w[i, j] = inside / float(n_pts)
    return w


def validate_box_radiator_on_grid(
    cfg: Config, derived: Derived, grid: Grid | None = None
) -> list[str]:
    """Errors if the radiator does not fit on the realized front face with edge clearance."""
    if cfg.box is None:
        return []
    errors: list[str] = []
    h = derived.h
    rw = derived.realized_W
    rh = derived.realized_H
    min_clear = float(cfg.numerics.min_edge_cells) * h
    hx, hy = _radiator_half_extents_xy(cfg.radiator)
    xc, yc = cfg.radiator.x_c, cfg.radiator.y_c
    half_w = 0.5 * rw
    half_h = 0.5 * rh
    clear_x_neg = (xc - hx) + half_w
    clear_x_pos = half_w - (xc + hx)
    clear_y_neg = (yc - hy) + half_h
    clear_y_pos = half_h - (yc + hy)
    min_c = min(clear_x_neg, clear_x_pos, clear_y_neg, clear_y_pos)
    if min_c < 0.0:
        errors.append("radiator does not fit on the box front face")
    elif min_c < min_clear:
        errors.append(
            f"radiator edge clearance {min_c / h:.2f} cells "
            f"< min_edge_cells={cfg.numerics.min_edge_cells}"
        )
    return errors


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
