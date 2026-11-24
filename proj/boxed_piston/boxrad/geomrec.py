"""HDF5 recording for parametric (cut-kernel) FDTD runs (v0.3 step 6b2a)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from boxrad.config import Config, Derived, test_source_active
from boxrad.geom.geometry import Geometry
from boxrad.geomderive import GeomGrid
from boxrad.grid import Fields
from boxrad.cutcell import wetted_wall_arrays
from boxrad.post.finalize import (
    compute_finalize_results,
    write_finalize_meta,
    write_surface_transfer,
)
from boxrad.post.zrad import log_frequency_grid
from boxrad.post.zrow import write_zrad_row_dataset, zrad_row_at_frequencies
from boxrad.post.kirchhoff import SurfacePatches, huygens_box_patches
from boxrad.recorder import (
    _CONVENTIONS,
    _ChunkWriter,
    _STENCIL_W4,
    _assert_probe_not_in_cpml,
    _cpml_free_center_bounds,
    resolve_decimation_factor,
    resolve_huygens_offset,
    validate_huygens_clearance,
)
from boxrad.source import velocity_at_half_step, velocity_pulse, velocity_pulse_derivative

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

_F64 = np.float64
_F32 = np.float32
_BUFFER_SAMPLES = 256
_WORLD_CONVENTIONS = (
    "e^{+jωt}; z = <p>_rad/(ρc v); H = p/v [Pa/(m/s)]; "
    "coordinates: world frame of the config; θ from +z, φ from +x; "
    "directivity phase referenced to the world origin"
)


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError(
            "h5py is required for boxrad.geomrec but is not installed in this environment"
        )
    return h5py


def _p_index_from_xyz(gg: GeomGrid, x: float, y: float, z: float) -> tuple[int, int, int]:
    h = gg.h
    x0 = gg.x0
    i = int(math.floor((float(x) - float(x0[0])) / h))
    j = int(math.floor((float(y) - float(x0[1])) / h))
    k = int(math.floor((float(z) - float(x0[2])) / h))
    return i, j, k


def _assert_in_p(gg: GeomGrid, i: int, j: int, k: int) -> int:
    nx, ny, nz = gg.nx, gg.ny, gg.nz
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        raise ValueError(f"p index ({i},{j},{k}) out of bounds for grid {nx}×{ny}×{nz}")
    return int(np.ravel_multi_index((i, j, k), (nx, ny, nz)))


def _v_index_from_normal(
    gg: GeomGrid,
    center: np.ndarray,
    normal: np.ndarray,
) -> tuple[int, float]:
    """Flat staggered velocity index and sign so v_n is positive outward."""
    h = gg.h
    x0 = gg.x0
    n = normal / max(np.linalg.norm(normal), 1e-30)
    axis = int(np.argmax(np.abs(n)))
    sgn = 1.0 if n[axis] > 0.0 else -1.0
    x, y, z = float(center[0]), float(center[1]), float(center[2])
    nx, ny, nz = gg.nx, gg.ny, gg.nz
    if axis == 0:
        i = int(round((x - float(x0[0])) / h))
        j = int(round((y - float(x0[1])) / h - 0.5))
        k = int(round((z - float(x0[2])) / h - 0.5))
        i = max(0, min(nx, i))
        j = max(0, min(ny - 1, j))
        k = max(0, min(nz - 1, k))
        flat = int(np.ravel_multi_index((i, j, k), (nx + 1, ny, nz)))
        return flat, sgn
    if axis == 1:
        i = int(round((x - float(x0[0])) / h - 0.5))
        j = int(round((y - float(x0[1])) / h))
        k = int(round((z - float(x0[2])) / h - 0.5))
        i = max(0, min(nx - 1, i))
        j = max(0, min(ny, j))
        k = max(0, min(nz - 1, k))
        flat = int(np.ravel_multi_index((i, j, k), (nx, ny + 1, nz)))
        return flat, sgn
    i = int(round((x - float(x0[0])) / h - 0.5))
    j = int(round((y - float(x0[1])) / h - 0.5))
    k = int(round((z - float(x0[2])) / h))
    i = max(0, min(nx - 1, i))
    j = max(0, min(ny - 1, j))
    k = max(0, min(nz, k))
    flat = int(np.ravel_multi_index((i, j, k), (nx, ny, nz + 1)))
    return flat, sgn


def _p_stencil_indices(
    gg: GeomGrid,
    center: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Four flat ``p`` indices at ±3h/2, ±h/2 along outward normal (order 4 rule)."""
    n = normal / max(np.linalg.norm(normal), 1e-30)
    h = gg.h
    offs = np.array([-1.5, -0.5, 0.5, 1.5], dtype=_F64) * h
    flat = np.empty(4, dtype=np.int64)
    for q, d_along in enumerate(offs):
        pos = center + d_along * n
        i, j, k = _p_index_from_xyz(gg, pos[0], pos[1], pos[2])
        flat[q] = _assert_in_p(gg, i, j, k)
    return flat


def _build_huygens_gathers(
    gg: GeomGrid,
    patches,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = patches.centers.shape[0]
    idx4 = np.empty((n, 4), dtype=np.int64)
    idx_v = np.empty(n, dtype=np.int64)
    sgn_v = np.empty(n, dtype=_F32)
    v_axis = np.empty(n, dtype=np.int8)
    for q in range(n):
        c = patches.centers[q]
        n_hat = patches.normals[q]
        idx4[q] = _p_stencil_indices(gg, c, n_hat)
        iv, sv = _v_index_from_normal(gg, c, n_hat)
        idx_v[q] = iv
        sgn_v[q] = np.float32(sv)
        v_axis[q] = np.int8(np.argmax(np.abs(n_hat)))
    return idx4, idx_v, sgn_v, v_axis


def validate_huygens_clearance_geom(
    derived: Derived,
    offset_cells: int,
    rigid_sides: dict[str, bool],
) -> None:
    """Require ``gap_cells − offset ≥ 2`` on each absorbing side only."""
    if offset_cells < 2:
        raise ValueError(
            f"huygens offset {offset_cells} cells < 2 (stencil needs two p cells each side)"
        )
    gap = derived.gap_cells
    for axis, (neg, pos) in enumerate((("x", "x"), ("y", "y"), ("z", "z"))):
        if not rigid_sides.get(f"{neg}-", False):
            clearance = gap - offset_cells
            if clearance < 2:
                raise ValueError(
                    f"huygens offset {offset_cells} leaves {clearance} cells to CPML/gap "
                    f"on {neg}- (need ≥ 2; gap_cells={gap})"
                )
        if not rigid_sides.get(f"{pos}+", False):
            clearance = gap - offset_cells
            if clearance < 2:
                raise ValueError(
                    f"huygens offset {offset_cells} leaves {clearance} cells to CPML/gap "
                    f"on {pos}+ (need ≥ 2; gap_cells={gap})"
                )


def huygens_patches_planes(
    lo_s: np.ndarray,
    hi_s: np.ndarray,
    h: float,
    offset_cells: int,
    rigid_sides: dict[str, bool],
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
) -> SurfacePatches:
    """Open or closed Huygens patches; rigid sides omit a face and extend laterals to the plane."""
    _ = domain_lo, domain_hi
    lo_s = np.asarray(lo_s, dtype=_F64).reshape(3)
    hi_s = np.asarray(hi_s, dtype=_F64).reshape(3)
    pad = float(offset_cells) * float(h)
    w = hi_s - lo_s

    if not any(rigid_sides.get(k, False) for k in rigid_sides):
        return huygens_box_patches(
            float(w[0]),
            float(w[1]),
            float(w[2]),
            h,
            tuple(lo_s.tolist()),
            offset_cells=offset_cells,
        )

    def _face_lo(axis: int) -> float:
        key = f"{('x', 'y', 'z')[axis]}-"
        if rigid_sides.get(key, False):
            return float(lo_s[axis])
        return float(lo_s[axis]) - pad

    def _face_hi(axis: int) -> float:
        key = f"{('x', 'y', 'z')[axis]}+"
        if rigid_sides.get(key, False):
            return float(hi_s[axis])
        return float(hi_s[axis]) + pad

    ox, ox2 = _face_lo(0), _face_hi(0)
    oy, oy2 = _face_lo(1), _face_hi(1)
    oz, oz2 = _face_lo(2), _face_hi(2)
    wx, wy, wz = ox2 - ox, oy2 - oy, oz2 - oz
    h = float(h)

    def _axis_centers(x0: float, length: float) -> np.ndarray:
        n = max(1, int(round(length / h)))
        return x0 + (np.arange(n, dtype=_F64) + 0.5) * h

    xs_in = _axis_centers(ox, wx)
    ys_in = _axis_centers(oy, wy)
    zs_in = _axis_centers(oz, wz)

    centers_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    areas_list: list[float] = []
    ex = np.array([1.0, 0.0, 0.0], dtype=_F64)
    ey = np.array([0.0, 1.0, 0.0], dtype=_F64)
    ez = np.array([0.0, 0.0, 1.0], dtype=_F64)

    def _add_face(
        fixed_coord: float,
        axis_fixed: int,
        u_centers: np.ndarray,
        v_centers: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        nu, nv = len(u_centers), len(v_centers)
        pts = np.zeros((nu * nv, 3), dtype=_F64)
        tang = [i for i in range(3) if i != axis_fixed]
        uu, vv = np.meshgrid(u_centers, v_centers, indexing="ij")
        pts[:, axis_fixed] = fixed_coord
        pts[:, tang[0]] = uu.ravel()
        pts[:, tang[1]] = vv.ravel()
        centers_list.append(pts)
        normals_list.append(np.tile(normal, (nu * nv, 1)))
        areas_list.extend([h * h] * (nu * nv))

    if not rigid_sides.get("x-", False):
        _add_face(ox, 0, ys_in, zs_in, -ex)
    if not rigid_sides.get("x+", False):
        _add_face(ox2, 0, ys_in, zs_in, ex)
    if not rigid_sides.get("y-", False):
        _add_face(oy, 1, xs_in, zs_in, -ey)
    if not rigid_sides.get("y+", False):
        _add_face(oy2, 1, xs_in, zs_in, ey)
    if not rigid_sides.get("z-", False):
        _add_face(oz, 2, xs_in, ys_in, -ez)
    if not rigid_sides.get("z+", False):
        _add_face(oz2, 2, xs_in, ys_in, ez)

    centers = np.vstack(centers_list)
    normals = np.vstack(normals_list)
    areas = np.asarray(areas_list, dtype=_F64)
    return SurfacePatches(centers=centers, normals=normals, areas=areas, h=h)


def validate_huygens_patches_geom(gg: GeomGrid, patches) -> None:
    """Raise ``ValueError`` if any Huygens stencil lies in CPML or outside the grid."""
    slabs = gg.cpml_slabs
    i0, i1 = slabs.i_inner.lo, slabs.i_inner.hi
    j0, j1 = slabs.j_inner.lo, slabs.j_inner.hi
    k0, k1 = slabs.k_inner.lo, slabs.k_inner.hi
    h = gg.h
    n_p = patches.centers.shape[0]
    for q in range(n_p):
        center = patches.centers[q]
        n_hat = patches.normals[q]
        n_u = n_hat / max(float(np.linalg.norm(n_hat)), 1e-30)
        for mul in (-1.5, -0.5, 0.5, 1.5):
            pos = center + mul * h * n_u
            i, j, k = _p_index_from_xyz(gg, float(pos[0]), float(pos[1]), float(pos[2]))
            if not (i0 <= i < i1 and j0 <= j < j1 and k0 <= k < k1):
                raise ValueError(
                    f"Huygens patch {q} order-4 stencil at p index ({i},{j},{k}) "
                    f"outside non-CPML interior (i∈[{i0},{i1}), j∈[{j0},{j1}), k∈[{k0},{k1}))"
                )


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _geom_stats_json(op_info: dict[str, Any]) -> str:
    fr = op_info["fractions"]
    blob = {
        "fractions_stats": fr.stats,
        "floor_stats": op_info.get("floor_stats", {}),
        "thin_wall_stats": op_info.get("thin_wall_stats", {}),
    }
    return json.dumps(blob, default=_json_default)


def _cell_center(gg: GeomGrid, flat: int) -> np.ndarray:
    ny, nz = gg.ny, gg.nz
    h = gg.h
    x0 = gg.x0.astype(_F64)
    i = int(flat) // (ny * nz)
    rem = int(flat) % (ny * nz)
    j = rem // nz
    k = rem % nz
    return x0 + (np.array([i, j, k], dtype=_F64) + 0.5) * h


class GeomRecorder:
    """Incremental HDF5 writer for parametric geometry; call :meth:`sample` after :func:`step_geom`.

    Area-weighted radiator pressure uses centroids from cut-cell clipping. For an
    axis-aligned radiator on a grid face, the normal-offset term reduces to the
    v0.2 face correction (d_c = h/2). For tilted radiators, rho*d_c*w_c*dv/dt is an
    O(h) reactance approximation; Re z is unaffected at leading order.
    """

    def __init__(
        self,
        cfg: Config,
        derived: Derived,
        gg: GeomGrid,
        geometry: Geometry,
        op_info: dict[str, Any],
        path: str | Path,
        *,
        offset_cells: int | None = None,
        buffer_samples: int = _BUFFER_SAMPLES,
        driven: int = 0,
    ) -> None:
        h5 = _require_h5py()
        offset = resolve_huygens_offset(cfg) if offset_cells is None else int(offset_cells)
        rigid_sides = dict(gg.rigid_sides or {})
        if rigid_sides:
            validate_huygens_clearance_geom(derived, offset, rigid_sides)
        else:
            validate_huygens_clearance(derived, offset, geometry="box")

        self.cfg = cfg
        self.derived = derived
        self.gg = gg
        self.geometry = geometry
        self.op_info = op_info
        self.h = float(derived.h)
        self.dt = float(derived.dt)
        self.rho = float(cfg.air.rho)
        self.sigma = float(derived.pulse_sigma)
        self.t0 = float(derived.pulse_t0)
        self.decim = resolve_decimation_factor(cfg, derived)
        self._offset = offset
        n_rad = len(geometry.radiators)
        self._test_source_only = bool(test_source_active(cfg) and n_rad == 0)
        if n_rad == 0:
            if not self._test_source_only:
                raise ValueError("GeomRecorder requires at least one radiator")
            self._driven = 0
            self._zrad_source = "test_source"
        else:
            if not (0 <= driven < n_rad):
                raise ValueError(f"driven index {driven} out of range for {n_rad} radiators")
            self._driven = int(driven)
            self._zrad_source = "fdtd"
        self._n_radiators = n_rad
        self._rad_names = [r.name for r in geometry.radiators]

        h = self.h
        gap_cpml = (derived.gap_cells + derived.cpml_cells) * h
        lo_s = np.asarray(gg.x0, dtype=_F64) + gap_cpml * (
            np.array(
                [
                    0.0 if rigid_sides.get("x-", False) else 1.0,
                    0.0 if rigid_sides.get("y-", False) else 1.0,
                    0.0 if rigid_sides.get("z-", False) else 1.0,
                ],
                dtype=_F64,
            )
        )
        hi_s = lo_s + np.array([derived.realized_W, derived.realized_H, derived.realized_D], dtype=_F64)
        domain_lo = np.asarray(gg.x0, dtype=_F64)
        domain_hi = domain_lo + np.array([gg.nx, gg.ny, gg.nz], dtype=_F64) * h
        huygens_patches = huygens_patches_planes(
            lo_s,
            hi_s,
            h,
            offset,
            rigid_sides,
            domain_lo,
            domain_hi,
        )
        (
            self._huygens_idx4,
            self._huygens_iv,
            self._huygens_vsgn,
            self._huygens_vaxis,
        ) = _build_huygens_gathers(gg, huygens_patches)
        validate_huygens_patches_geom(gg, huygens_patches)
        self._n_huy = self._huygens_idx4.shape[0]

        fractions = op_info["fractions"]
        beta_tilde = op_info["beta_tilde"]
        self._rad_S = [float(np.sum(fractions.radiator_a[i])) for i in range(n_rad)]
        self._all_p_idx: list[np.ndarray] = []
        self._all_w: list[np.ndarray] = []
        self._all_w_sum: list[float] = []
        self._all_d: list[np.ndarray] = []
        self._all_n_hat: list[np.ndarray] = []
        for ir in range(n_rad):
            rd_i = geometry.radiators[ir]
            n_hat_i = rd_i.pose.R[:, 2].astype(_F64)
            n_hat_i = n_hat_i / max(float(np.linalg.norm(n_hat_i)), 1e-30)
            cells_i = fractions.radiator_cell[ir]
            areas_i = fractions.radiator_a[ir].astype(_F64)
            centroids_i = fractions.radiator_centroid[ir].astype(_F64)
            h2 = h * h
            w_c_i = areas_i / h2
            self._all_p_idx.append(cells_i.astype(np.int64))
            self._all_w.append(w_c_i)
            self._all_w_sum.append(float(np.sum(w_c_i)) if w_c_i.size else 1.0)
            self._all_n_hat.append(n_hat_i)
            d_c_i = np.empty(cells_i.shape[0], dtype=_F64)
            for q, flat in enumerate(cells_i):
                xc = _cell_center(gg, int(flat))
                d_c_i[q] = float(np.dot(xc - centroids_i[q], n_hat_i))
            self._all_d.append(d_c_i)
        if not self._test_source_only:
            self._n_hat = self._all_n_hat[driven]
            self._rad_p_idx = self._all_p_idx[driven]
            self._rad_w = self._all_w[driven]
            self._rad_w_sum = self._all_w_sum[driven]
            self._rad_d = self._all_d[driven]

        self._record_surface = bool(cfg.output.surface_p)
        cells_w, areas_w, cents_w, avec_w = wetted_wall_arrays(fractions, beta_tilde, h)
        self._n_wet = int(cells_w.shape[0])
        self._sf_p_idx = cells_w.astype(np.int64) if self._record_surface else np.empty(0, dtype=np.int64)
        self._sf_areas = areas_w
        self._sf_centroids = cents_w
        self._sf_area_vec = avec_w

        probe_targets = self._resolve_probe_targets(cfg)
        self._n_probes = len(probe_targets)
        self._probe_requested = (
            np.asarray(probe_targets, dtype=_F64) if probe_targets else np.empty((0, 3), _F64)
        )
        if self._n_probes:
            self._probe_idx, self._probe_centers = self._probe_indices(gg, probe_targets, beta_tilde)
        else:
            self._probe_idx = np.empty(0, dtype=np.int64)
            self._probe_centers = np.empty((0, 3), _F64)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = h5.File(path, "w")
        self._step = 0
        self._buf_samples = int(buffer_samples)
        self._tail_db: float | None = None
        self._closed = False

        self._write_static_meta(huygens_patches)
        self._init_datasets()

    @property
    def decimation(self) -> int:
        return self.decim

    def _resolve_probe_targets(self, cfg: Config) -> list[tuple[float, float, float]]:
        targets: list[tuple[float, float, float]] = []
        out = cfg.output
        if out.probes:
            for row in out.probes:
                if len(row) != 3:
                    raise ValueError(f"output.probes entries must be [x,y,z], got {row!r}")
                targets.append((float(row[0]), float(row[1]), float(row[2])))
        return targets

    def _probe_indices(
        self,
        gg: GeomGrid,
        targets: list[tuple[float, float, float]],
        beta_tilde: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(targets)
        idx = np.empty(n, dtype=np.int64)
        centers = np.empty((n, 3), dtype=_F64)
        shape = (gg.nx, gg.ny, gg.nz)
        beta_flat = beta_tilde.reshape(-1)
        x0 = gg.x0.astype(_F64)
        h = gg.h
        domain_hi = x0 + np.array([gg.nx, gg.ny, gg.nz], dtype=_F64) * h
        slabs = gg.cpml_slabs
        bounds_lo, bounds_hi = _cpml_free_center_bounds(slabs, x0=x0, h=h)
        planes = list(self.geometry.planes)

        def _behind_plane(xv: float, yv: float, zv: float) -> str | None:
            pxyz = (xv, yv, zv)
            for pl in planes:
                ax = {"x": 0, "y": 1, "z": 2}[pl.axis]
                if pl.side == "-" and pxyz[ax] < float(pl.offset) - 0.5 * h:
                    return f"{pl.axis}{pl.side} at offset {pl.offset}"
                if pl.side == "+" and pxyz[ax] > float(pl.offset) + 0.5 * h:
                    return f"{pl.axis}{pl.side} at offset {pl.offset}"
            return None

        for q, (x, y, z) in enumerate(targets):
            behind = _behind_plane(x, y, z)
            if behind is not None:
                raise ValueError(
                    f"probe {q} at ({x},{y},{z}) is behind rigid plane {behind}"
                )
            if (
                x < float(x0[0])
                or y < float(x0[1])
                or z < float(x0[2])
                or x >= float(domain_hi[0])
                or y >= float(domain_hi[1])
                or z >= float(domain_hi[2])
            ):
                raise ValueError(
                    f"probe {q} at ({x},{y},{z}) lies outside the FDTD domain "
                    f"[{x0[0]},{domain_hi[0]})×[{x0[1]},{domain_hi[1]})×[{x0[2]},{domain_hi[2]})"
                )
            i, j, k = _p_index_from_xyz(gg, x, y, z)
            flat = _assert_in_p(gg, i, j, k)
            _assert_probe_not_in_cpml(q, x, y, z, i, j, k, slabs, bounds_lo, bounds_hi)
            if beta_flat[flat] == 0.0:
                raise ValueError(
                    f"probe {q} at ({x},{y},{z}) maps to solid cell ({i},{j},{k})"
                )
            idx[q] = flat
            centers[q] = _cell_center(gg, flat)
        return idx, centers

    def _write_static_meta(self, huygens_patches) -> None:
        meta = self._file.create_group("meta")
        meta.attrs["schema_version"] = 2
        meta.attrs["n_radiators"] = int(self._n_radiators)
        meta.attrs["radiator_names_json"] = json.dumps(self._rad_names)
        meta.attrs["radiator_S_json"] = json.dumps(self._rad_S)
        meta.attrs["driven_index"] = int(self._driven)
        if self._rad_names:
            meta.attrs["driven_name"] = str(self._rad_names[self._driven])
        else:
            meta.attrs["driven_name"] = "test_source"
        meta.attrs["mutual_status"] = "recorded_unvalidated"
        meta.attrs["conventions"] = _CONVENTIONS
        meta.attrs["zrad_source"] = self._zrad_source
        if test_source_active(self.cfg):
            meta.attrs["drive_kind"] = "test_source"
            meta.attrs["transfer_units"] = "Pa/(m^3/s)"
            ts = self.cfg.numerics.test_source
            assert ts is not None
            blob = {
                "position": list(ts.position),
                "cell_center": list(self.derived.test_source_cell_center or (0.0, 0.0, 0.0)),
                "index": list(self.derived.test_source_cell_index or (0, 0, 0)),
            }
            meta.attrs["test_source_json"] = json.dumps(blob)
        meta.attrs["decimation_factor"] = self.decim
        meta.attrs["decimation_note"] = (
            "Huygens decimation m from alias rejection vs analytic pulse spectrum (see derived_json)"
        )
        meta.attrs["h"] = self.h
        meta.attrs["dt"] = self.dt
        meta.attrs["huygens_offset_cells"] = self._offset
        meta.attrs["geometry"] = "parametric"
        meta.attrs["S_grid"] = float(self.op_info["S_grid"]) if not self._test_source_only else 1.0
        meta.attrs["frame"] = "world"
        if self.geometry.radiators:
            pose = self.geometry.radiators[self._driven].pose
            meta.attrs["radiator0_position"] = pose.position.astype(_F64)
            meta.attrs["radiator0_normal"] = pose.R[:, 2].astype(_F64)
            meta.attrs["radiator0_x_axis"] = pose.R[:, 0].astype(_F64)
        meta.attrs["kernel_path"] = str(self.op_info["path"])
        meta.attrs["geom_stats_json"] = _geom_stats_json(self.op_info)
        planes = list(self.gg.planes) if self.gg.planes else []
        meta.attrs["environment_json"] = json.dumps(
            [{"axis": p.axis, "side": p.side, "offset": float(p.offset)} for p in planes]
        )
        meta.attrs["n_planes"] = int(len(planes))
        meta.attrs["h_nominal"] = float(self.gg.h_nominal)
        meta.create_dataset(
            "config_json",
            data=np.asarray(self.cfg.to_json(), dtype="S"),
        )
        derived_blob = json.dumps(asdict(self.derived))
        meta.create_dataset("derived_json", data=np.asarray(derived_blob, dtype="S"))

        if not self._test_source_only:
            rad = self._file.create_group("radiator")
            rad.attrs["description"] = "Unit pulse v(t); area-weighted radiator face pressure p_avg"
            rad.attrs["p_avg_all_note"] = (
                "columns in config radiator order; velocity correction on driven column only"
            )
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

        if self._record_surface and self._n_wet > 0:
            surf = self._file.create_group("surface")
            surf.attrs["surface_kind"] = "wall_patches"
            surf.attrs["n_wet"] = int(self._n_wet)
            surf.create_dataset("points", data=self._sf_centroids.astype(_F64), compression="gzip")
            avec = self._sf_area_vec.astype(_F64)
            areas = self._sf_areas.astype(_F64)
            norms = avec / np.linalg.norm(avec, axis=1, keepdims=True)
            surf.create_dataset("normals", data=norms, compression="gzip")
            surf.create_dataset("areas", data=areas, compression="gzip")
            surf.create_dataset("area_vectors", data=avec, compression="gzip")
            surf.attrs["p_times"] = "t_p = (n+1)Δt at decimated steps n ≡ 0 (mod M)"
            meta.attrs["surface_kind"] = "wall_patches"
        elif not self._record_surface:
            meta.attrs["surface_kind"] = "none"

        meta.attrs["time_after_step_n"] = (
            "sample() is called after step_geom(n); fields.p is p^{n+1}, staggered v is v^{n+½}"
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
        buf = self._buf_samples
        if self._test_source_only:
            self._ds_v_int = self._ds_v_half = self._ds_p_avg = None  # type: ignore[assignment]
            self._ds_p_avg_all = self._ds_rad_tp = self._ds_rad_tv = None  # type: ignore[assignment]
            self._w_v_int = self._w_v_half = self._w_p_avg = None  # type: ignore[assignment]
            self._w_p_avg_all = self._w_rad_tp = self._w_rad_tv = None  # type: ignore[assignment]
        else:
            rad = self._file["radiator"]
            self._ds_v_int = self._create_extensible(rad, "v_pulse_integer", (), _F64, buf)
            self._ds_v_half = self._create_extensible(rad, "v_pulse_half", (), _F64, buf)
            self._ds_p_avg = self._create_extensible(rad, "p_avg", (), _F64, buf)
            self._ds_p_avg_all = self._create_extensible(
                rad, "p_avg_all", (self._n_radiators,), _F64, buf
            )
            self._ds_rad_tp = self._create_extensible(rad, "times_p", (), _F64, buf)
            self._ds_rad_tv = self._create_extensible(rad, "times_v_half", (), _F64, buf)
            self._w_v_int = _ChunkWriter(self._ds_v_int, buf)
            self._w_v_half = _ChunkWriter(self._ds_v_half, buf)
            self._w_p_avg = _ChunkWriter(self._ds_p_avg, buf)
            self._w_p_avg_all = _ChunkWriter(self._ds_p_avg_all, buf)
            self._w_rad_tp = _ChunkWriter(self._ds_rad_tp, buf)
            self._w_rad_tv = _ChunkWriter(self._ds_rad_tv, buf)

        huy = self._file["huygens"]
        nh = self._n_huy
        self._ds_h_p = self._create_extensible(huy, "p_face", (nh,), _F32, buf)
        self._ds_h_vn = self._create_extensible(huy, "vn", (nh,), _F32, buf)
        self._ds_h_tp = self._create_extensible(huy, "times_p", (), _F64, buf)
        self._ds_h_tvn = self._create_extensible(huy, "times_vn", (), _F64, buf)

        if self._n_probes:
            pr = self._file["probes"]
            np_ = self._n_probes
            self._ds_pr_p = self._create_extensible(pr, "p", (np_,), _F32, buf)
            self._ds_pr_tp = self._create_extensible(pr, "times_p", (), _F64, buf)
        else:
            self._ds_pr_p = None
            self._ds_pr_tp = None

        if self._record_surface and self._n_wet > 0:
            surf = self._file["surface"]
            nw = self._n_wet
            self._ds_sf_p = self._create_extensible(surf, "p_raw", (nw,), _F32, buf)
            self._ds_sf_tp = self._create_extensible(surf, "times_p", (), _F64, buf)
            self._w_sf_p = _ChunkWriter(self._ds_sf_p, buf)
            self._w_sf_tp = _ChunkWriter(self._ds_sf_tp, buf)
        else:
            self._ds_sf_p = None
            self._ds_sf_tp = None
            self._w_sf_p = None
            self._w_sf_tp = None

        self._w_h_p = _ChunkWriter(self._ds_h_p, buf)
        self._w_h_vn = _ChunkWriter(self._ds_h_vn, buf)
        self._w_h_tp = _ChunkWriter(self._ds_h_tp, buf)
        self._w_h_tvn = _ChunkWriter(self._ds_h_tvn, buf)
        if self._ds_pr_p is not None and self._ds_pr_tp is not None:
            self._w_pr_p = _ChunkWriter(self._ds_pr_p, buf)
            self._w_pr_tp = _ChunkWriter(self._ds_pr_tp, buf)
        else:
            self._w_pr_p = None
            self._w_pr_tp = None

    def _time_p_after_step(self, n: int) -> float:
        return (float(n) + 1.0) * self.dt

    def _time_v_after_step(self, n: int) -> float:
        return (float(n) + 0.5) * self.dt

    def _radiator_p_avg_one(self, p_flat: np.ndarray, n: int, ir: int) -> float:
        idx = self._all_p_idx[ir]
        if idx.size == 0:
            return 0.0
        w_c = self._all_w[ir]
        w_sum = self._all_w_sum[ir]
        pc = p_flat[idx].astype(_F64)
        if ir == self._driven:
            t_p = self._time_p_after_step(n)
            dvdt = velocity_pulse_derivative(t_p, self.sigma, self.t0)
            corr = self.rho * self._all_d[ir] * w_c * dvdt
            p_face = pc + corr
        else:
            p_face = pc
        return float(np.sum(w_c * p_face) / w_sum)

    def _radiator_p_avg(self, p_flat: np.ndarray, n: int) -> float:
        return self._radiator_p_avg_one(p_flat, n, self._driven)

    def _radiator_p_avg_all(self, p_flat: np.ndarray, n: int) -> np.ndarray:
        out = np.empty(self._n_radiators, dtype=_F64)
        for ir in range(self._n_radiators):
            out[ir] = self._radiator_p_avg_one(p_flat, n, ir)
        return out

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

    def sample(self, fields: Fields, n: int) -> None:
        """Record fields after :func:`boxrad.geomrun.step_geom` with index *n*."""
        if n != self._step:
            raise ValueError(f"expected step {self._step}, got {n}")
        t_p = self._time_p_after_step(n)
        t_v = self._time_v_after_step(n)
        v_at_p = velocity_pulse(t_p, self.sigma, self.t0)
        v_half = velocity_at_half_step(n, self.dt, self.sigma, self.t0)
        p_flat = fields.p.reshape(-1)

        if not self._test_source_only:
            self._w_v_int.append(v_at_p)  # type: ignore[union-attr]
            self._w_v_half.append(float(v_half))  # type: ignore[union-attr]
            self._w_p_avg.append(self._radiator_p_avg(p_flat, n))  # type: ignore[union-attr]
            self._w_p_avg_all.append(self._radiator_p_avg_all(p_flat, n))  # type: ignore[union-attr]
            self._w_rad_tp.append(t_p)  # type: ignore[union-attr]
            self._w_rad_tv.append(t_v)  # type: ignore[union-attr]

        if n % self.decim == 0:
            self._w_h_p.append(self._huygens_p(p_flat).astype(_F32))
            self._w_h_vn.append(self._huygens_vn(fields.vx, fields.vy, fields.vz))
            self._w_h_tp.append(t_p)
            self._w_h_tvn.append(t_v)
            if self._w_pr_p is not None and self._w_pr_tp is not None:
                self._w_pr_p.append(p_flat[self._probe_idx].astype(_F32))
                self._w_pr_tp.append(t_p)
            if self._w_sf_p is not None and self._w_sf_tp is not None:
                self._w_sf_p.append(p_flat[self._sf_p_idx].astype(_F32))
                self._w_sf_tp.append(t_p)

        self._step += 1

    def close(self) -> None:
        if self._closed:
            return
        h5 = _require_h5py()
        writers = [
            self._w_h_p,
            self._w_h_vn,
            self._w_h_tp,
            self._w_h_tvn,
            *(w for w in (self._w_pr_p, self._w_pr_tp, self._w_sf_p, self._w_sf_tp) if w is not None),
        ]
        if not self._test_source_only:
            writers = [
                self._w_v_int,
                self._w_v_half,
                self._w_p_avg,
                self._w_p_avg_all,
                self._w_rad_tp,
                self._w_rad_tv,
                *writers,
            ]
        for w in writers:
            if w is not None:
                w.close()
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
            self._file.flush()
            self._file.close()
            self._file = None  # type: ignore[assignment]

        f_log = log_frequency_grid(
            float(self.cfg.band.f_min), float(self.derived.f_max), points_per_decade=100
        )
        if self._ds_sf_p is not None:
            with h5.File(self._path, "r") as f_read:
                if "surface" in f_read and "p_raw" in f_read["surface"]:
                    p_sf = np.asarray(f_read["surface/p_raw"][()], dtype=_F64)
                    t_sf = np.asarray(f_read["surface/times_p"][()], dtype=_F64)
                else:
                    p_sf = np.empty((0, 0), _F64)
                    t_sf = np.empty(0, _F64)
            with h5.File(self._path, "a") as f_append:
                write_surface_transfer(
                    f_append,
                    self.cfg,
                    self.derived,
                    p_ts=p_sf,
                    times_p=t_sf,
                    f_log=f_log,
                )

        if self._test_source_only:
            with h5.File(self._path, "a") as f_append:
                f_append["meta"].attrs["conventions"] = _WORLD_CONVENTIONS
            self._tail_db = float("nan")
            self._closed = True
            return

        p_avg_full: np.ndarray
        with h5.File(self._path, "r") as f_read:
            p_avg_full = np.asarray(f_read["radiator/p_avg"][()], dtype=_F64)
            p_avg_all_full = np.asarray(f_read["radiator/p_avg_all"][()], dtype=_F64)
            times_p = np.asarray(f_read["radiator/times_p"][()], dtype=_F64)
            derived_blob = json.loads(
                bytes(f_read["meta/derived_json"][()]).decode("utf-8")
            )
            tail_db, f_log, z_log = compute_finalize_results(
                f_read,
                self.cfg,
                self.derived,
                p_avg=p_avg_full,
            )
            z_row = zrad_row_at_frequencies(
                p_avg_all_full,
                times_p,
                f_log,
                self.cfg,
                derived_blob,
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
            write_zrad_row_dataset(f_append["results"], f_log, z_row)
            f_append["meta"].attrs["conventions"] = _WORLD_CONVENTIONS
        self._tail_db = float(tail_db)
        self._closed = True