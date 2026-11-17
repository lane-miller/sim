"""Chunked HDF5 recording for box / radiator FDTD runs (Stage C)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from boxrad.config import Config, Derived, decimation_m_cap
from boxrad.grid import Fields, Grid
from boxrad.post.kirchhoff import (
    box_surface_patches,
    huygens_box_patches,
    huygens_open_patches,
)
from boxrad.post.finalize import compute_finalize_results, write_finalize_meta
from boxrad.source import (
    BoxBoundary,
    velocity_at_half_step,
    velocity_pulse,
    velocity_pulse_derivative,
)

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

_F64 = np.float64
_F32 = np.float32
_BUFFER_SAMPLES = 256
_CONVENTIONS = "e^{+jωt}; p at integer steps, v at half steps"
_PULSE_BANDLIMIT_NOTE = (
    "Huygens decimation m from alias rejection vs analytic pulse spectrum (see derived_json)"
)
_DEFAULT_HUYGENS_OFFSET_CELLS = 3
_STENCIL_W4 = np.array([-1.0, 9.0, 9.0, -1.0], dtype=_F64) / 16.0


def decimation_factor(f_max: float, dt: float) -> int:
    """Legacy cap ``floor(1/(2.5·f_max·Δt))`` (tests / upper bound only)."""
    return decimation_m_cap(f_max, dt)


def resolve_decimation_factor(cfg: Config, derived: Derived) -> int:
    """Recorder decimation ``M`` from :func:`~boxrad.config.derive` (override wins)."""
    if cfg.numerics.decimation_override is not None:
        return max(1, int(cfg.numerics.decimation_override))
    return int(derived.decimation_m)


def resolve_huygens_offset(cfg: Config) -> int:
    """Huygens surface offset in cells (config override with default 3)."""
    num = cfg.numerics
    val = getattr(num, "huygens_offset_cells", None)
    return _DEFAULT_HUYGENS_OFFSET_CELLS if val is None else int(val)


def validate_huygens_clearance(
    derived: Derived,
    offset_cells: int,
    *,
    geometry: str = "box",
) -> None:
    """Raise ``ValueError`` if Huygens offset is too small or enters the CPML.

    Box mode: outer Huygens faces sit ``offset_cells`` into the gap beyond the
    rigid box. Baffle mode: lateral faces match box mode; the top face is at
    ``offset_cells·h`` above the baffle (``D = 0`` in :func:`huygens_open_patches`).
    In both cases the top/outward stencil must stay ``≥ 2`` cells inside the
    CPML inner edge, so require ``gap_cells − offset_cells ≥ 2``.
    """
    if offset_cells < 2:
        raise ValueError(
            f"huygens offset {offset_cells} cells < 2 (stencil needs two p cells each side)"
        )
    clearance = derived.gap_cells - offset_cells
    if clearance < 2:
        mode = "baffle" if geometry == "baffle" else "box"
        raise ValueError(
            f"huygens offset {offset_cells} leaves {clearance} cells to CPML/gap "
            f"(need ≥ 2; gap_cells={derived.gap_cells}, geometry={mode})"
        )


def validate_huygens_patches_interior(grid: Grid, patches) -> None:
    """Raise ``ValueError`` if any Huygens face sample lies in CPML or outside the grid."""
    if grid.i_box is None or grid.j_box is None or grid.k_box is None:
        raise ValueError("grid has no geometry origin indices")
    slabs = grid.cpml_slabs
    i0, i1 = slabs.i_inner.lo, slabs.i_inner.hi
    j0, j1 = slabs.j_inner.lo, slabs.j_inner.hi
    k0, k1 = slabs.k_inner.lo, slabs.k_inner.hi
    ib, jb, kb = grid.i_box, grid.j_box, grid.k_box
    h = grid.h
    n_p = patches.centers.shape[0]
    for q in range(n_p):
        center = patches.centers[q]
        n_hat = patches.normals[q]
        n_u = n_hat / max(float(np.linalg.norm(n_hat)), 1e-30)
        for mul in (-1.5, -0.5, 0.5, 1.5):
            pos = center + mul * h * n_u
            i, j, k = _p_index_from_xyz(
                grid, float(pos[0]), float(pos[1]), float(pos[2]), ib=ib, jb=jb, kb=kb
            )
            if not (i0 <= i < i1 and j0 <= j < j1 and k0 <= k < k1):
                raise ValueError(
                    f"Huygens patch {q} order-4 stencil at p index ({i},{j},{k}) "
                    f"outside non-CPML interior (i∈[{i0},{i1}), j∈[{j0},{j1}), k∈[{k0},{k1}))"
                )


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError(
            "h5py is required for boxrad.recorder but is not installed in this environment"
        )
    return h5py


def _geometry_origin(grid: Grid) -> tuple[float, float, float]:
    cfg = grid.cfg
    xc = float(cfg.radiator.x_c)
    yc = float(cfg.radiator.y_c)
    if cfg.geometry == "baffle":
        return (
            -xc - 0.5 * grid.realized_W,
            -yc - 0.5 * grid.realized_H,
            0.0,
        )
    return (
        -xc - 0.5 * grid.realized_W,
        -yc - 0.5 * grid.realized_H,
        -grid.realized_D,
    )


def _huygens_depth(grid: Grid, derived: Derived) -> float:
    if grid.cfg.geometry == "baffle":
        return 0.0
    return grid.realized_D


def _resolve_probe_targets(cfg: Config, grid: Grid) -> list[tuple[float, float, float]]:
    """Physical probe targets (m) from ``Output.probes`` and optional baffle z cell list."""
    targets: list[tuple[float, float, float]] = []
    out = cfg.output
    if out.probes:
        for row in out.probes:
            if len(row) != 3:
                raise ValueError(f"output.probes entries must be [x,y,z], got {row!r}")
            targets.append((float(row[0]), float(row[1]), float(row[2])))
    if cfg.geometry == "baffle" and out.probes_baffle_z_cells:
        h = grid.h
        xc = float(cfg.radiator.x_c)
        yc = float(cfg.radiator.y_c)
        for k in out.probes_baffle_z_cells:
            targets.append((xc, yc, (int(k) + 0.5) * h))
    return targets


def _probe_indices(grid: Grid, targets: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest pressure cell center for each target (documented probe rule).

    Returns flat ``p`` indices and actual sample coordinates (N, 3).
    """
    n = len(targets)
    idx = np.empty(n, dtype=np.int64)
    centers = np.empty((n, 3), dtype=_F64)
    shape = (grid.nx, grid.ny, grid.nz)
    for q, (x, y, z) in enumerate(targets):
        i, j, k = grid.index_of(x, y, z)
        idx[q] = int(np.ravel_multi_index((i, j, k), shape))
        centers[q, 0], centers[q, 1], centers[q, 2] = grid.coords_of(i, j, k)
    return idx, centers


def _p_index_from_xyz(
    grid: Grid,
    x: float,
    y: float,
    z: float,
    *,
    ib: int,
    jb: int,
    kb: int,
) -> tuple[int, int, int]:
    ox, oy, oz = _geometry_origin(grid)
    h = grid.h
    i = ib + int(round((float(x) - ox) / h - 0.5))
    j = jb + int(round((float(y) - oy) / h - 0.5))
    k = kb + int(round((float(z) - oz) / h - 0.5))
    return i, j, k


def _assert_in_p(grid: Grid, i: int, j: int, k: int) -> int:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        raise ValueError(f"p index ({i},{j},{k}) out of bounds for grid {nx}×{ny}×{nz}")
    return int(np.ravel_multi_index((i, j, k), (nx, ny, nz)))


def _v_index_from_normal(
    grid: Grid,
    center: np.ndarray,
    normal: np.ndarray,
    *,
    ib: int,
    jb: int,
    kb: int,
) -> tuple[int, float]:
    """Flat staggered velocity index and sign so v_n is positive outward."""
    ox, oy, oz = _geometry_origin(grid)
    h = grid.h
    n = normal / max(np.linalg.norm(normal), 1e-30)
    axis = int(np.argmax(np.abs(n)))
    sgn = 1.0 if n[axis] > 0.0 else -1.0
    x, y, z = float(center[0]), float(center[1]), float(center[2])
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if axis == 0:
        i = ib + int(round((x - ox) / h))
        j = jb + int(round((y - oy) / h - 0.5))
        k = kb + int(round((z - oz) / h - 0.5))
        i = max(0, min(nx, i))
        j = max(0, min(ny - 1, j))
        k = max(0, min(nz - 1, k))
        flat = int(np.ravel_multi_index((i, j, k), (nx + 1, ny, nz)))
        return flat, sgn
    if axis == 1:
        i = ib + int(round((x - ox) / h - 0.5))
        j = jb + int(round((y - oy) / h))
        k = kb + int(round((z - oz) / h - 0.5))
        i = max(0, min(nx - 1, i))
        j = max(0, min(ny, j))
        k = max(0, min(nz - 1, k))
        flat = int(np.ravel_multi_index((i, j, k), (nx, ny + 1, nz)))
        return flat, sgn
    i = ib + int(round((x - ox) / h - 0.5))
    j = jb + int(round((y - oy) / h - 0.5))
    k = kb + int(round((z - oz) / h))
    i = max(0, min(nx - 1, i))
    j = max(0, min(ny - 1, j))
    k = max(0, min(nz, k))
    flat = int(np.ravel_multi_index((i, j, k), (nx, ny, nz + 1)))
    return flat, sgn


def _p_stencil_indices(
    grid: Grid,
    center: np.ndarray,
    normal: np.ndarray,
    *,
    ib: int,
    jb: int,
    kb: int,
) -> np.ndarray:
    """Four flat ``p`` indices at ±3h/2, ±h/2 along outward normal (order 4 rule)."""
    n = normal / max(np.linalg.norm(normal), 1e-30)
    h = grid.h
    offs = np.array([-1.5, -0.5, 0.5, 1.5], dtype=_F64) * h
    flat = np.empty(4, dtype=np.int64)
    for q, d_along in enumerate(offs):
        pos = center + d_along * n
        i, j, k = _p_index_from_xyz(grid, pos[0], pos[1], pos[2], ib=ib, jb=jb, kb=kb)
        flat[q] = _assert_in_p(grid, i, j, k)
    return flat


def _build_huygens_gathers(
    grid: Grid,
    patches,
    *,
    ib: int,
    jb: int,
    kb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = patches.centers.shape[0]
    idx4 = np.empty((n, 4), dtype=np.int64)
    idx_v = np.empty(n, dtype=np.int64)
    sgn_v = np.empty(n, dtype=_F32)
    v_axis = np.empty(n, dtype=np.int8)
    for q in range(n):
        c = patches.centers[q]
        n_hat = patches.normals[q]
        idx4[q] = _p_stencil_indices(grid, c, n_hat, ib=ib, jb=jb, kb=kb)
        iv, sv = _v_index_from_normal(grid, c, n_hat, ib=ib, jb=jb, kb=kb)
        idx_v[q] = iv
        sgn_v[q] = np.float32(sv)
        v_axis[q] = np.int8(np.argmax(np.abs(n_hat)))
    return idx4, idx_v, sgn_v, v_axis


def _build_box_surface_gathers(
    grid: Grid,
    boundary: BoxBoundary,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per box-surface patch: flat ``p`` index, radiator local index (−1 if rigid)."""
    assert grid.i_box is not None and grid.j_box is not None and grid.k_box is not None
    assert grid.k_face_z is not None
    ib, jb, kb = grid.i_box, grid.j_box, grid.k_box
    kf = grid.k_face_z
    patches = box_surface_patches(
        grid.realized_W,
        grid.realized_H,
        grid.realized_D,
        grid.h,
        _geometry_origin(grid),
    )
    n = patches.centers.shape[0]
    n_rad = boundary.idx_radiator.size
    rad_ij: dict[tuple[int, int], int] = {}
    if n_rad and grid.radiator_face_indices is not None:
        ir, jr, _ = grid.radiator_face_indices
        for q in range(n_rad):
            rad_ij[(int(ir[q]), int(jr[q]))] = q

    idx_p = np.empty(n, dtype=np.int64)
    rad_local = np.full(n, -1, dtype=np.int64)
    h = grid.h

    for q in range(n):
        c = patches.centers[q]
        n_hat = patches.normals[q]
        n_u = n_hat / max(np.linalg.norm(n_hat), 1e-30)
        pos = c + 0.5 * h * n_u
        i, j, k = _p_index_from_xyz(grid, pos[0], pos[1], pos[2], ib=ib, jb=jb, kb=kb)
        idx_p[q] = _assert_in_p(grid, i, j, k)
        axis = int(np.argmax(np.abs(n_u)))
        if axis == 2 and n_u[2] > 0.0:
            key = (i, j)
            if key in rad_ij:
                rad_local[q] = rad_ij[key]

    return patches, idx_p, rad_local


class _ChunkWriter:
    """Append rows to a 2-D HDF5 dataset in chunks of at most ``buf_rows``."""

    def __init__(self, dset, buf_rows: int) -> None:
        self._dset = dset
        self._buf_rows = buf_rows
        self._buf = np.empty((buf_rows,) + dset.shape[1:], dtype=dset.dtype)
        self._n = 0
        self._written = 0

    @property
    def max_buffer_bytes(self) -> int:
        return int(self._buf.nbytes)

    def append(self, row: np.ndarray | float) -> None:
        if np.isscalar(row):
            self._buf[self._n] = row
        else:
            self._buf[self._n] = row
        self._n += 1
        if self._n >= self._buf_rows:
            self.flush()

    def flush(self) -> None:
        if self._n == 0:
            return
        start = self._written
        new_len = start + self._n
        if new_len > self._dset.shape[0]:
            self._dset.resize(new_len, axis=0)
        self._dset[start:new_len] = self._buf[: self._n]
        self._written = new_len
        self._n = 0

    def close(self) -> None:
        self.flush()


class Recorder:
    """Incremental HDF5 writer; call :meth:`sample` once per integer step after :func:`step`."""

    def __init__(
        self,
        cfg: Config,
        derived: Derived,
        grid: Grid,
        boundary: BoxBoundary,
        path: str | Path,
        *,
        offset_cells: int | None = None,
        buffer_samples: int = _BUFFER_SAMPLES,
    ) -> None:
        h5 = _require_h5py()
        if cfg.geometry not in ("box", "baffle"):
            raise ValueError("Recorder requires box or baffle geometry")
        if cfg.geometry == "box" and grid.cfg.box is None:
            raise ValueError("Recorder requires cfg.box in box geometry mode")
        offset = resolve_huygens_offset(cfg) if offset_cells is None else int(offset_cells)
        validate_huygens_clearance(derived, offset, geometry=cfg.geometry)

        self.cfg = cfg
        self._is_baffle = cfg.geometry == "baffle"
        self.derived = derived
        self.grid = grid
        self.boundary = boundary
        self.h = float(derived.h)
        self.dt = float(derived.dt)
        self.rho = float(cfg.air.rho)
        self.sigma = float(derived.pulse_sigma)
        self.t0 = float(derived.pulse_t0)
        self.decim = resolve_decimation_factor(cfg, derived)
        self._offset = offset
        self._zrad_source = "fdtd"

        assert grid.i_box is not None and grid.j_box is not None and grid.k_box is not None
        ib, jb, kb = grid.i_box, grid.j_box, grid.k_box
        origin = _geometry_origin(grid)
        depth = _huygens_depth(grid, derived)
        if self._is_baffle:
            huygens_patches = huygens_open_patches(
                grid.realized_W,
                grid.realized_H,
                depth,
                grid.h,
                origin,
                offset_cells=offset,
            )
        else:
            huygens_patches = huygens_box_patches(
                grid.realized_W,
                grid.realized_H,
                grid.realized_D,
                grid.h,
                origin,
                offset_cells=offset,
            )
        (
            self._huygens_idx4,
            self._huygens_iv,
            self._huygens_vsgn,
            self._huygens_vaxis,
        ) = _build_huygens_gathers(grid, huygens_patches, ib=ib, jb=jb, kb=kb)
        validate_huygens_patches_interior(grid, huygens_patches)
        if self._is_baffle:
            box_patches = None
            self._box_idx_p = np.empty(0, dtype=np.int64)
            self._box_rad_local = np.empty(0, dtype=np.int64)
            self._n_box = 0
        else:
            box_patches, self._box_idx_p, self._box_rad_local = _build_box_surface_gathers(
                grid, boundary
            )
            self._n_box = box_patches.centers.shape[0]
        self._n_huy = self._huygens_idx4.shape[0]

        self._rad_p_idx = np.empty(boundary.idx_radiator.size, dtype=np.int64)
        if boundary.idx_radiator.size:
            vz_shape = (grid.nx, grid.ny, grid.nz + 1)
            ir, jr, kr = np.unravel_index(boundary.idx_radiator, vz_shape)
            kf = int(grid.k_face_z)
            self._rad_p_idx = np.ravel_multi_index(
                (ir, jr, np.full_like(ir, kf)), (grid.nx, grid.ny, grid.nz)
            )
        self._rad_w = boundary.w_radiator.astype(_F64)
        self._rad_w_sum = float(np.sum(self._rad_w)) if self._rad_w.size else 1.0

        probe_targets = _resolve_probe_targets(cfg, grid)
        self._n_probes = len(probe_targets)
        self._probe_requested = (
            np.asarray(probe_targets, dtype=_F64) if probe_targets else np.empty((0, 3), _F64)
        )
        if self._n_probes:
            self._probe_idx, self._probe_centers = _probe_indices(grid, probe_targets)
        else:
            self._probe_idx = np.empty(0, dtype=np.int64)
            self._probe_centers = np.empty((0, 3), dtype=_F64)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = h5.File(path, "w")
        self._step = 0
        self._buf_samples = int(buffer_samples)
        self._tail_db: float | None = None
        self._closed = False

        self._write_static_meta(box_patches, huygens_patches)
        self._init_datasets()

    @property
    def decimation(self) -> int:
        return self.decim

    @property
    def max_buffer_bytes(self) -> int:
        """Upper bound on simultaneous sample buffering (for memory tests)."""
        parts = [
            self._w_v_int.max_buffer_bytes,
            self._w_v_half.max_buffer_bytes,
            self._w_p_avg.max_buffer_bytes,
            self._w_rad_tp.max_buffer_bytes,
            self._w_rad_tv.max_buffer_bytes,
            self._w_h_p.max_buffer_bytes,
            self._w_h_vn.max_buffer_bytes,
            self._w_h_tp.max_buffer_bytes,
            self._w_h_tvn.max_buffer_bytes,
            *(
                w.max_buffer_bytes
                for w in (self._w_b_p, self._w_b_t)
                if w is not None
            ),
        ]
        return int(sum(parts))

    def _write_static_meta(self, box_patches, huygens_patches) -> None:
        meta = self._file.create_group("meta")
        meta.attrs["conventions"] = _CONVENTIONS
        meta.attrs["zrad_source"] = self._zrad_source
        meta.attrs["decimation_factor"] = self.decim
        meta.attrs["decimation_note"] = _PULSE_BANDLIMIT_NOTE
        meta.attrs["h"] = self.h
        meta.attrs["dt"] = self.dt
        meta.attrs["huygens_offset_cells"] = self._offset
        meta.create_dataset(
            "config_json",
            data=np.asarray(self.cfg.to_json(), dtype="S"),
        )
        derived_blob = json.dumps(asdict(self.derived))
        meta.create_dataset("derived_json", data=np.asarray(derived_blob, dtype="S"))

        rad = self._file.create_group("radiator")
        rad.attrs["description"] = "Unit pulse v(t); area-weighted radiator face pressure p_avg"
        rad.attrs["times_p_note"] = "t_p = (n+1)·Δt after step(n) (p^{n+1} on grid)"
        rad.attrs["times_v_half_note"] = "t_v = (n+½)·Δt after step(n) (v^{n+½} on grid)"
        rad.attrs["v_pulse_integer_note"] = (
            "velocity_pulse(t_p); same pressure time level as p_avg (audit: not drive time)"
        )

        huy = self._file.create_group("huygens")
        for name, arr in (
            ("centers", huygens_patches.centers),
            ("normals", huygens_patches.normals),
            ("areas", huygens_patches.areas),
        ):
            huy.create_dataset(name, data=arr.astype(_F64), compression="gzip")
        huy.attrs["p_times"] = "t_p = (n+1)·Δt at decimated steps n ≡ 0 (mod M)"
        huy.attrs["vn_times"] = "t_v = (n+½)·Δt at decimated steps n ≡ 0 (mod M)"

        if self._n_probes:
            pr = self._file.create_group("probes")
            pr.attrs["sampling"] = (
                "nearest pressure cell center to requested (x,y,z); "
                "p at time level (n+1)Δt after step(n)"
            )
            pr.create_dataset("requested", data=self._probe_requested, compression="gzip")
            pr.create_dataset("centers", data=self._probe_centers, compression="gzip")

        if not self._is_baffle and box_patches is not None:
            box = self._file.create_group("box_surface")
            for name, arr in (
                ("centers", box_patches.centers),
                ("normals", box_patches.normals),
                ("areas", box_patches.areas),
            ):
                box.create_dataset(name, data=arr.astype(_F64), compression="gzip")
            box.attrs["p_times"] = "t_p = (n+1)·Δt, decimated n ≡ 0 (mod M)"
        meta.attrs["geometry"] = self.cfg.geometry
        meta.attrs["time_after_step_n"] = (
            "sample() is called after step(n); fields.p is p^{n+1}, staggered v is v^{n+½}"
        )

    def _create_extensible(
        self,
        group,
        name: str,
        tail_shape: tuple[int, ...],
        dtype,
        chunk0: int,
    ):
        shape = (0,) + tail_shape
        maxshape = (None,) + tail_shape
        chunks = (min(chunk0, 64),) + tail_shape
        return group.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression="gzip",
        )

    def _init_datasets(self) -> None:
        rad = self._file["radiator"]
        buf = self._buf_samples
        self._ds_v_int = self._create_extensible(rad, "v_pulse_integer", (), _F64, buf)
        self._ds_v_half = self._create_extensible(rad, "v_pulse_half", (), _F64, buf)
        self._ds_p_avg = self._create_extensible(rad, "p_avg", (), _F64, buf)
        self._ds_rad_tp = self._create_extensible(rad, "times_p", (), _F64, buf)
        self._ds_rad_tv = self._create_extensible(rad, "times_v_half", (), _F64, buf)

        huy = self._file["huygens"]
        nh = self._n_huy
        self._ds_h_p = self._create_extensible(huy, "p_face", (nh,), _F32, buf)
        self._ds_h_vn = self._create_extensible(huy, "vn", (nh,), _F32, buf)
        self._ds_h_tp = self._create_extensible(huy, "times_p", (), _F64, buf)
        self._ds_h_tvn = self._create_extensible(huy, "times_vn", (), _F64, buf)

        if self._n_box:
            box = self._file["box_surface"]
            nb = self._n_box
            self._ds_b_p = self._create_extensible(box, "p", (nb,), _F32, buf)
            self._ds_b_t = self._create_extensible(box, "times", (), _F64, buf)
        else:
            self._ds_b_p = None
            self._ds_b_t = None

        if self._n_probes:
            pr = self._file["probes"]
            np_ = self._n_probes
            self._ds_pr_p = self._create_extensible(pr, "p", (np_,), _F32, buf)
            self._ds_pr_tp = self._create_extensible(pr, "times_p", (), _F64, buf)
        else:
            self._ds_pr_p = None
            self._ds_pr_tp = None

        self._w_v_int = _ChunkWriter(self._ds_v_int, buf)
        self._w_v_half = _ChunkWriter(self._ds_v_half, buf)
        self._w_p_avg = _ChunkWriter(self._ds_p_avg, buf)
        self._w_rad_tp = _ChunkWriter(self._ds_rad_tp, buf)
        self._w_rad_tv = _ChunkWriter(self._ds_rad_tv, buf)
        self._w_h_p = _ChunkWriter(self._ds_h_p, buf)
        self._w_h_vn = _ChunkWriter(self._ds_h_vn, buf)
        self._w_h_tp = _ChunkWriter(self._ds_h_tp, buf)
        self._w_h_tvn = _ChunkWriter(self._ds_h_tvn, buf)
        if self._ds_b_p is not None and self._ds_b_t is not None:
            self._w_b_p = _ChunkWriter(self._ds_b_p, buf)
            self._w_b_t = _ChunkWriter(self._ds_b_t, buf)
        else:
            self._w_b_p = None
            self._w_b_t = None
        if self._ds_pr_p is not None and self._ds_pr_tp is not None:
            self._w_pr_p = _ChunkWriter(self._ds_pr_p, buf)
            self._w_pr_tp = _ChunkWriter(self._ds_pr_tp, buf)
        else:
            self._w_pr_p = None
            self._w_pr_tp = None

    def _time_p_after_step(self, n: int) -> float:
        """Pressure time level after :func:`boxrad.kernels.step` with index *n*."""
        return (float(n) + 1.0) * self.dt

    def _time_v_after_step(self, n: int) -> float:
        """Staggered velocity time level after :func:`boxrad.kernels.step` with index *n*."""
        return (float(n) + 0.5) * self.dt

    def _radiator_p_avg(self, p_flat: np.ndarray, n: int) -> float:
        if self._rad_p_idx.size == 0:
            return 0.0
        t_p = self._time_p_after_step(n)
        dvdt = velocity_pulse_derivative(t_p, self.sigma, self.t0)
        pc = p_flat[self._rad_p_idx].astype(_F64)
        corr = 0.5 * self.h * self.rho * self._rad_w * dvdt
        p_face = pc + corr
        return float(np.sum(self._rad_w * p_face) / self._rad_w_sum)

    def _huygens_p(self, p_flat: np.ndarray) -> np.ndarray:
        samples = p_flat[self._huygens_idx4].astype(_F64)
        return np.dot(samples, _STENCIL_W4)

    def _huygens_vn(self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
        vx_f = vx.reshape(-1)
        vy_f = vy.reshape(-1)
        vz_f = vz.reshape(-1)
        iv = self._huygens_iv
        sgn = self._huygens_vsgn.astype(_F64)
        axis = self._huygens_vaxis
        v_pick = np.where(axis == 0, vx_f[iv], np.where(axis == 1, vy_f[iv], vz_f[iv]))
        return (sgn * v_pick).astype(_F32)

    def _box_surface_p(self, p_flat: np.ndarray, n: int) -> np.ndarray:
        n_p = self._n_box
        out = p_flat[self._box_idx_p].astype(_F32)
        if self._rad_p_idx.size == 0:
            return out
        t_p = self._time_p_after_step(n)
        dvdt = velocity_pulse_derivative(t_p, self.sigma, self.t0)
        corr_scale = np.float32(0.5 * self.h * self.rho * dvdt)
        for q in range(n_p):
            loc = int(self._box_rad_local[q])
            if loc < 0:
                continue
            w = self._rad_w[loc]
            out[q] = np.float32(
                float(out[q]) + float(corr_scale * w)
            )
        return out

    def sample(self, fields: Fields, n: int) -> None:
        """Record fields after :func:`boxrad.kernels.step` with index *n* (``p^{n+1}``, ``v^{n+½}``)."""
        if n != self._step:
            raise ValueError(f"expected step {self._step}, got {n}")
        t_p = self._time_p_after_step(n)
        t_v = self._time_v_after_step(n)
        v_at_p = velocity_pulse(t_p, self.sigma, self.t0)
        v_half = velocity_at_half_step(n, self.dt, self.sigma, self.t0)
        p_flat = fields.p.reshape(-1)

        self._w_v_int.append(v_at_p)
        self._w_v_half.append(float(v_half))
        self._w_p_avg.append(self._radiator_p_avg(p_flat, n))
        self._w_rad_tp.append(t_p)
        self._w_rad_tv.append(t_v)

        if n % self.decim == 0:
            self._w_h_p.append(self._huygens_p(p_flat).astype(_F32))
            self._w_h_vn.append(
                self._huygens_vn(fields.vx, fields.vy, fields.vz)
            )
            self._w_h_tp.append(t_p)
            self._w_h_tvn.append(t_v)
            if self._w_b_p is not None and self._w_b_t is not None:
                self._w_b_p.append(self._box_surface_p(p_flat, n))
                self._w_b_t.append(t_p)
            if self._w_pr_p is not None and self._w_pr_tp is not None:
                self._w_pr_p.append(p_flat[self._probe_idx].astype(_F32))
                self._w_pr_tp.append(t_p)

        self._step += 1

    def close(self) -> None:
        if self._closed:
            return
        h5 = _require_h5py()
        for w in (
            self._w_v_int,
            self._w_v_half,
            self._w_p_avg,
            self._w_rad_tp,
            self._w_rad_tv,
            self._w_h_p,
            self._w_h_vn,
            self._w_h_tp,
            self._w_h_tvn,
            *(
                w
                for w in (self._w_b_p, self._w_b_t, self._w_pr_p, self._w_pr_tp)
                if w is not None
            ),
        ):
            w.close()
        p_avg_full: np.ndarray
        if self._file is not None:
            n_full = self._step
            n_dec = (n_full + self.decim - 1) // self.decim if n_full else 0
            meta = self._file["meta"]
            meta.attrs["n_steps"] = n_full
            meta.attrs["n_samples_decimated"] = n_dec
            if n_dec:
                steps = np.arange(n_dec, dtype=_F64) * float(self.decim)
                times_dec_p = (steps + 1.0) * self.dt
                times_dec_v = (steps + 0.5) * self.dt
            else:
                times_dec_p = times_dec_v = np.empty(0, _F64)
            meta.create_dataset("sample_times_decimated_p", data=times_dec_p, compression="gzip")
            meta.create_dataset("sample_times_decimated_v", data=times_dec_v, compression="gzip")
            p_avg_full = np.asarray(self._ds_p_avg[()], dtype=_F64)
            self._file.flush()
            self._file.close()
            self._file = None  # type: ignore[assignment]
        else:
            with h5.File(self._path, "r") as f_read:
                p_avg_full = np.asarray(f_read["radiator/p_avg"][()], dtype=_F64)

        with h5.File(self._path, "r") as f_read:
            tail_db, f_log, z_log = compute_finalize_results(
                f_read,
                self.cfg,
                self.derived,
                p_avg=p_avg_full,
            )
        with h5.File(self._path, "a") as f_append:
            write_finalize_meta(
                f_append,
                self.cfg,
                self.derived,
                tail_db=tail_db,
                f_log=f_log,
                z_log=z_log,
            )
        self._tail_db = float(tail_db)
        self._closed = True