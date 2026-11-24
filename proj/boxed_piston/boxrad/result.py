"""Lazy-loading user API over boxed-piston HDF5 results."""

from __future__ import annotations

import json
import math
import re
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np

from boxrad import __version__
from boxrad.config import Config, Derived
from boxrad.grid import Grid
from boxrad.post.directivity import di_from_pattern
from boxrad.post.kirchhoff import (
    SurfacePatches,
    distance_to_patches,
    dpdn_from_vn,
    kh_far_field,
    kh_pressure,
)
from boxrad.post.images import (
    direction_open,
    expand_patches,
    first_plane_violation,
    in_fluid,
)
from boxrad.post.transfer import (
    kh_huygens_pressure_at_points,
    load_box_surface_spectrum,
    load_huygens_spectra,
    probe_fdtd_pressure_spectrum,
)
from boxrad.post.zrow import zrow_at_frequencies
from boxrad.post.zrad import zrad_at_frequencies

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]

_F64 = np.float64
_C128 = np.complex128
_DI_THETA = 24
_DI_PHI = 48
_V_REF = 1.0
_BAND_RTOL = 1e-9
_POSE_RTOL = 1e-9


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError("h5py is required for boxrad.Result")
    return h5py


class Result:
    """User-facing, lazy view of a completed boxrad HDF5 file.

    All transfer functions are normalized to the radiator's uniform normal
    velocity ``v`` (time convention ``e^{+jωt}``, peak-amplitude phasors, SI
    units). Spectra are evaluated exactly at the requested frequencies from the
    stored time records (no interpolation). Every method that takes ``f`` raises
    ``ValueError`` for frequencies outside :attr:`band` unless called with
    ``allow_out_of_band=True`` (then it warns and evaluates anyway).

    Coordinates depend on the ``frame`` keyword on :meth:`field`,
    :meth:`directivity`, and :meth:`directivity_index` (default
    :attr:`frame_default` = ``"local"``). Parametric files use the radiator-0
    local frame by default: origin at the radiator center, ``+z`` outward along
    the radiator normal, ``+x`` along the stored pose x-axis; phase is referenced
    to that origin. ``frame="world"`` uses the config/world frame (stored
    Huygens data and probe positions). Legacy box/baffle files have identity pose
    so both frames agree. In box mode the box occupies ``z ∈ [−D, 0]`` in the
    legacy radiator-local frame.

    The HDF5 file is opened read-only on first access and kept open until
    :meth:`close` (or the end of a ``with`` block).

    Args:
        path: Path to an HDF5 file written by :func:`boxrad.run`.
        frame: Default coordinate frame for :meth:`field`, :meth:`directivity`, and
            :meth:`directivity_index` (per-call ``frame=`` overrides).

    Raises:
        FileNotFoundError: ``path`` is not an existing file.
        ImportError: h5py is not installed (raised on first data access).
    """

    frame_default: Literal["local", "world"] = "local"

    def __init__(
        self,
        path: str | Path,
        *,
        frame: Literal["local", "world"] = "local",
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if frame not in ("local", "world"):
            raise ValueError(f"frame must be 'local' or 'world', got {frame!r}")
        self.frame_default = frame
        self._h5: Any | None = None
        self._spectra_cache: dict[float, tuple[SurfacePatches, complex, np.ndarray, np.ndarray]] = {}
        self._geometry_cache: Any | None = None
        self._radiator_open: dict[int, Result] = {}

    def _file(self) -> Any:
        if self._h5 is None:
            self._h5 = _require_h5py().File(self.path, "r")
        return self._h5

    def close(self) -> None:
        """Close the underlying HDF5 handle (safe to call more than once)."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @cached_property
    def config(self) -> Config:
        """Run configuration as stored in ``/meta/config_json``."""
        raw = bytes(self._file()["meta/config_json"][()]).decode("utf-8")
        return Config.from_json(raw)

    @cached_property
    def derived(self) -> Derived:
        """Derived run parameters (h, Δt, grid size, pulse, …) from ``/meta/derived_json``."""
        blob = json.loads(bytes(self._file()["meta/derived_json"][()]).decode("utf-8"))
        return Derived.from_dict(blob)

    @cached_property
    def meta(self) -> dict[str, Any]:
        """All ``/meta`` attributes as plain Python scalars/strings (e.g. ``tail_db``, ``n_steps``)."""
        g = self._file()["meta"]
        out: dict[str, Any] = {}
        for key, val in g.attrs.items():
            if isinstance(val, np.ndarray) and val.size != 1:
                out[key] = val.tolist()
            else:
                out[key] = val.item() if hasattr(val, "item") else val
        return out

    @cached_property
    def accuracy(self) -> dict[str, Any]:
        """``/meta/accuracy`` attributes (``ppw_realized``, ``cells_across_radiator``,
        ``h`` [m], ``notes``); empty dict if the group is missing."""
        f = self._file()
        if "accuracy" not in f["meta"]:
            return {}
        g = f["meta/accuracy"]
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in g.attrs.items()}

    @property
    def band(self) -> tuple[float, float]:
        """Supported frequency band ``(f_min, f_max)`` in Hz (config ``f_min``, derived ``f_max``)."""
        d = self.derived
        f_max = float(d.f_max)
        return float(self.config.band.f_min), f_max

    @cached_property
    def _grid(self) -> Grid:
        return Grid(self.config, self.derived)

    @cached_property
    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Radiator-0 rigid rotation ``R`` (local → world, float64 (3, 3)) and translation ``t`` (3,).

        Parametric files store ``radiator0_x_axis``, ``radiator0_normal``, and
        ``radiator0_position`` in ``/meta``; legacy box/baffle files return
        identity rotation and zero translation.

        Raises:
            ValueError: Stored axes do not form a proper rotation (det ``R`` ≤ 0
                or ``R`` is not orthonormal within ``1e-9``).
        """
        m = self.meta
        keys = ("radiator0_x_axis", "radiator0_normal", "radiator0_position")
        if not all(k in m for k in keys):
            return np.eye(3, dtype=_F64), np.zeros(3, dtype=_F64)
        x_hat = np.asarray(m["radiator0_x_axis"], dtype=_F64).reshape(3)
        n_hat = np.asarray(m["radiator0_normal"], dtype=_F64).reshape(3)
        t = np.asarray(m["radiator0_position"], dtype=_F64).reshape(3)
        y_hat = np.cross(n_hat, x_hat)
        r_mat = np.column_stack([x_hat, y_hat, n_hat])
        err = float(np.max(np.abs(r_mat.T @ r_mat - np.eye(3, dtype=_F64))))
        if err > _POSE_RTOL:
            raise ValueError(
                f"radiator0 pose is not orthonormal (max |RᵀR − I| = {err:.3g}, limit {_POSE_RTOL})"
            )
        if float(np.linalg.det(r_mat)) <= 0.0:
            raise ValueError("radiator0 pose has non-positive determinant (not a rotation)")
        return r_mat.astype(_F64, copy=False), t.astype(_F64, copy=False)

    @cached_property
    def test_source(self) -> dict[str, Any] | None:
        """Validation monopole metadata from ``meta.test_source_json``, or ``None``."""
        raw = self.meta.get("test_source_json")
        if raw is None or raw == "":
            return None
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return dict(raw)
        return None

    def _resolve_frame(self, frame: Literal["local", "world"] | None) -> Literal["local", "world"]:
        if frame is None:
            if int(self.meta.get("n_radiators", 1)) == 0 or self.meta.get("drive_kind") == "test_source":
                return "world"
            return self.frame_default
        if frame not in ("local", "world"):
            raise ValueError(f"frame must be 'local' or 'world', got {frame!r}")
        return frame

    def _points_local_to_world(self, pts: np.ndarray) -> np.ndarray:
        r_mat, t = self.pose
        return pts @ r_mat.T + t

    def _dirs_local_to_world(self, dirs: np.ndarray) -> np.ndarray:
        r_mat, _ = self.pose
        return dirs @ r_mat.T

    @cached_property
    def planes(self) -> list[dict[str, Any]]:
        """Environment rigid planes from ``meta.environment_json`` (empty if absent)."""
        raw = self.meta.get("environment_json")
        if raw is None or raw == "":
            return []
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, (list, tuple)):
            return list(raw)
        return []

    def _expand_huygens(
        self,
        patches: SurfacePatches,
        p_face: np.ndarray,
        vn_face: np.ndarray,
    ) -> tuple[SurfacePatches, np.ndarray, np.ndarray]:
        if not self.planes:
            return patches, p_face, vn_face
        return expand_patches(patches, self.planes, p_face, vn_face)  # type: ignore[return-value]

    def _on_axis_world(self, frame_sel: Literal["local", "world"]) -> np.ndarray:
        """Unit +z in the evaluation frame, expressed in world coordinates."""
        if frame_sel == "world":
            return np.array([0.0, 0.0, 1.0], dtype=_F64)
        r_mat, _ = self.pose
        return (r_mat @ np.array([0.0, 0.0, 1.0], dtype=_F64)).astype(_F64, copy=False)

    def _apply_rigid_plane_bounds(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lo = lo.copy()
        hi = hi.copy()
        for pl in self.planes:
            ax = {"x": 0, "y": 1, "z": 2}[str(pl["axis"])]
            off = float(pl["offset"])
            if pl["side"] == "-":
                lo[ax] = off
            else:
                hi[ax] = off
        return lo, hi

    @cached_property
    def S_grid(self) -> float:
        """Discretized radiator area ``h² Σ w`` in m² (used for every z ↔ Z conversion)."""
        if "S_grid" in self.meta:
            return float(self.meta["S_grid"])
        return float(self._grid.radiator_area_grid)

    @cached_property
    def _schema_version(self) -> int:
        return int(self.meta.get("schema_version", 1))

    @cached_property
    def names(self) -> list[str]:
        """Radiator names in config order (schema-1: ``[\"radiator\"]``)."""
        if self._schema_version >= 2:
            raw = self.meta.get("radiator_names_json")
            if raw is None:
                return ["radiator"]
            if isinstance(raw, str):
                return list(json.loads(raw))
            return list(raw)
        return ["radiator"]

    @cached_property
    def driven_index(self) -> int:
        """Index of the radiator driven in this HDF5 file (schema-1: 0)."""
        if self._schema_version >= 2:
            return int(self.meta.get("driven_index", 0))
        return 0

    @cached_property
    def S_list(self) -> np.ndarray:
        """Discretized radiator areas ``(N,)`` in m² (schema-1: ``[S_grid]``)."""
        if self._schema_version >= 2:
            raw = self.meta.get("radiator_S_json")
            if raw is None:
                return np.array([self.S_grid], dtype=_F64)
            if isinstance(raw, str):
                return np.asarray(json.loads(raw), dtype=_F64)
            return np.asarray(raw, dtype=_F64)
        return np.array([self.S_grid], dtype=_F64)

    def _primary_path(self) -> Path:
        if self.driven_index == 0:
            return self.path
        raw = self.meta.get("run_files_json")
        if raw is not None:
            files = json.loads(raw) if isinstance(raw, str) else list(raw)
            if files and files[0]:
                return self.path.parent / str(files[0])
        mobj = re.match(r"^(.+)\.r\d+_", self.path.name)
        if mobj is not None:
            return self.path.parent / f"{mobj.group(1)}{self.path.suffix}"
        raise ValueError(
            f"cannot resolve primary HDF5 for multi-radiator set from {self.path.name}"
        )

    def _run_files_basenames(self) -> list[str | None]:
        if self._schema_version < 2:
            return [self.path.name]
        primary = self._primary_path()
        if primary == self.path:
            raw = self.meta.get("run_files_json")
        else:
            with Result(primary, frame=self.frame_default) as pr:
                raw = pr.meta.get("run_files_json")
        if raw is None:
            out: list[str | None] = [None] * len(self.names)
            out[0] = primary.name
            if self.driven_index < len(out):
                out[self.driven_index] = self.path.name
            return out
        if isinstance(raw, str):
            return list(json.loads(raw))
        return list(raw)

    def _resolve_radiator_index(self, key: int | str) -> int:
        names = self.names
        if isinstance(key, str):
            if key not in names:
                raise ValueError(f"unknown radiator name {key!r}")
            return names.index(key)
        j = int(key)
        if j < 0 or j >= len(names):
            raise ValueError(f"radiator index {j} out of range for {len(names)} radiators")
        return j

    def radiator(self, key: int | str) -> Result:
        """Open the HDF5 file that drives radiator ``key`` (index or name).

        Index 0 or radiator 0's name on the primary file returns ``self`` when
        this file drives radiator 0. Sibling files resolve ``run_files_json`` via
        the primary file in the same directory.

        Raises:
            ValueError: Radiator was not driven in this run set.
            FileNotFoundError: Expected sibling HDF5 is missing on disk.
        """
        j = self._resolve_radiator_index(key)
        if j == self.driven_index:
            return self
        files = self._run_files_basenames()
        if j >= len(files) or files[j] is None:
            raise ValueError(f"radiator {self.names[j]} was not driven in this run set")
        sibling = self.path.parent / files[j]
        if not sibling.is_file():
            raise FileNotFoundError(str(sibling))
        cached = self._radiator_open.get(j)
        if cached is not None:
            return cached
        out = Result(sibling, frame=self.frame_default)
        self._radiator_open[j] = out
        return out

    def _built_geometry(self):
        if self._geometry_cache is None:
            from boxrad.geom.geometry import build_geometry

            geom = build_geometry(self.config)
            if geom is None:
                raise ValueError("inside() requires a buildable solid geometry")
            self._geometry_cache = geom
        return self._geometry_cache

    @cached_property
    def poses(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Rigid rotation and translation ``(R, t)`` for each radiator (world frame)."""
        geom = self._built_geometry()
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for rd in geom.radiators:
            r_mat = np.asarray(rd.pose.R, dtype=_F64)
            t = np.asarray(rd.pose.position, dtype=_F64).reshape(3)
            out.append((r_mat, t))
        return out

    def inside(
        self,
        points: np.ndarray | list,
        *,
        frame: Literal["local", "world"] | None = None,
    ) -> np.ndarray:
        """True where a point lies inside the solid or behind a rigid environment plane."""
        frame_sel = self._resolve_frame(frame)
        pts = np.atleast_2d(np.asarray(points, dtype=_F64))
        if frame_sel == "local":
            r0, t0 = self.poses[0]
            pts_world = pts @ r0.T + t0
        else:
            pts_world = pts
        geom = self._built_geometry()
        in_solid = geom.inside(pts_world)
        if self.planes:
            behind = ~in_fluid(pts_world, self.planes)
            return np.asarray(in_solid | behind, dtype=bool)
        return np.asarray(in_solid, dtype=bool)

    def Z_matrix(
        self,
        f: np.ndarray | float,
        *,
        allow_out_of_band: bool = False,
    ) -> np.ndarray:
        """Acoustic mutual impedance matrix ``Z_ij(f) = ρ c z_ij / S_j`` in Pa·s/m³.

        Shape ``(n_f, N, N)``. Column ``j`` uses the file that drives radiator ``j``;
        undriven columns are NaN. Raw mutual data (``mutual_status = recorded_unvalidated``).
        """
        f_arr = self._check_f(f, allow_out_of_band)
        n_f = f_arr.size
        n_rad = len(self.names)
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        s_list = self.S_list
        z_mat = np.full((n_f, n_rad, n_rad), np.nan + 0j, dtype=_C128)
        files = self._run_files_basenames()
        for j in range(n_rad):
            if j >= len(files) or files[j] is None:
                continue
            path_j = self.path.parent / files[j]
            z_row = zrow_at_frequencies(path_j, f_arr)
            sj = float(s_list[j])
            for i in range(n_rad):
                z_mat[:, i, j] = (rho * c * z_row[:, i] / sj).astype(_C128, copy=False)
        return z_mat

    def field_sum(
        self,
        points: np.ndarray | list,
        f: np.ndarray | float,
        U: np.ndarray | None = None,
        *,
        frame: Literal["local", "world"] | None = None,
        allow_out_of_band: bool = False,
    ) -> np.ndarray:
        """Superposed pressure ``Σ_j (U_j/S_j) H_j(x, f)`` with ``H_j = p/v`` (Pa/(m/s)).

        ``frame=\"local\"`` is radiator 0's local frame (siblings are evaluated in world).
        """
        frame_sel = self._resolve_frame(frame)
        f_arr = self._check_f(f, allow_out_of_band)
        pts = np.atleast_2d(np.asarray(points, dtype=_F64))
        if frame_sel == "local":
            r0, t0 = self.poses[0]
            pts_world = pts @ r0.T + t0
        else:
            pts_world = pts

        n_rad = len(self.names)
        n_f = f_arr.size
        if U is None:
            u_arr = np.ones((n_f, n_rad), dtype=_F64)
        else:
            u_in = np.asarray(U, dtype=_F64)
            if u_in.ndim == 1:
                if u_in.shape[0] != n_rad:
                    raise ValueError(f"U length {u_in.shape[0]} != n_radiators {n_rad}")
                u_arr = np.broadcast_to(u_in.reshape(1, n_rad), (n_f, n_rad)).copy()
            elif u_in.ndim == 2:
                if u_in.shape != (n_f, n_rad):
                    raise ValueError(f"U shape {u_in.shape} != ({n_f}, {n_rad})")
                u_arr = u_in
            else:
                raise ValueError("U must have shape (N,) or (F, N)")
        files = self._run_files_basenames()
        for j in range(n_rad):
            driven = j < len(files) and files[j] is not None
            if not driven and np.any(u_arr[:, j] != 0.0):
                raise ValueError(f"radiator {self.names[j]} was not driven in this run set")
        s_list = self.S_list
        out = np.zeros((n_f, pts_world.shape[0]), dtype=_C128)
        for j in range(n_rad):
            if j >= len(files) or files[j] is None:
                continue
            hj = self.radiator(j).field(
                pts_world,
                f_arr,
                frame="world",
                allow_out_of_band=allow_out_of_band,
            )
            scale = (u_arr[:, j] / float(s_list[j])).reshape(n_f, 1)
            out += scale * hj
        return out

    def summary(self) -> str:
        """Multi-line human-readable summary (version, geometry, band, h, radiator, tail).

        Returns:
            Text with one item group per line; no trailing newline.
        """
        c = self.config
        d = self.derived
        rad = c.radiator
        dim_s = "×".join(f"{x:.4g}" for x in rad.dims)
        lines = [
            f"boxrad {self.meta.get('boxrad_version', __version__)}  {self.path.name}",
            f"geometry={c.geometry}  band={self.band[0]:.4g}–{self.band[1]:.4g} Hz",
            f"h={d.h:.4g} m  ppw={d.ppw_realized:.3g}  steps={self.meta.get('n_steps', '?')}",
            f"radiator {rad.shape} {dim_s} m at ({rad.x_c}, {rad.y_c})",
            f"S_grid={self.S_grid:.6g} m²  tail_db={self.meta.get('tail_db', float('nan')):.1f} dB",
        ]
        if self.accuracy:
            lines.append(
                f"accuracy: cells_across_radiator="
                f"{self.accuracy.get('cells_across_radiator', float('nan')):.3g}"
            )
        return "\n".join(lines)

    def _check_f(self, f: np.ndarray | float, allow_out_of_band: bool) -> np.ndarray:
        """Flatten *f* to float64 ``(n_f,)``; raise (or warn) if any value is outside :attr:`band`."""
        f_arr = np.atleast_1d(np.asarray(f, dtype=_F64)).ravel()
        f_min, f_max = self.band
        bad = (
            ~np.isfinite(f_arr)
            | (f_arr < f_min * (1.0 - _BAND_RTOL))
            | (f_arr > f_max * (1.0 + _BAND_RTOL))
        )
        if np.any(bad):
            n_bad = int(np.count_nonzero(bad))
            first = f_arr[bad][0].item()
            msg = (
                f"{n_bad} frequenc{'y' if n_bad == 1 else 'ies'} outside the supported band "
                f"[{f_min:.6g}, {f_max:.6g}] Hz of {self.path.name} (first: {first:.6g} Hz)"
            )
            if not allow_out_of_band:
                raise ValueError(msg + "; pass allow_out_of_band=True to evaluate anyway")
            warnings.warn(msg + "; results there are not supported", UserWarning, stacklevel=3)
        return f_arr

    def _z_at(self, f_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f_out, z_out = zrad_at_frequencies(self.path, f_arr)
        return np.atleast_1d(f_out), np.atleast_1d(z_out)

    def z(
        self,
        f: np.ndarray | float | None = None,
        *,
        allow_out_of_band: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dimensionless radiation impedance ``z = ⟨p⟩_rad / (ρ c v)``.

        Args:
            f: Frequencies in Hz (scalar or array-like) inside :attr:`band`. ``None``
                returns the stored ``/results/zrad`` table (log-spaced, 100 points
                per decade over the band; never band-checked).
            allow_out_of_band: Evaluate frequencies outside :attr:`band` with a
                ``UserWarning`` instead of raising.

        Returns:
            ``(f_hz, z)``: float64 ``(n_f,)`` in Hz and complex128 ``(n_f,)``
            (dimensionless). Out-of-band values (only with ``allow_out_of_band``)
            are NaN where the drive spectrum is more than 40 dB below its peak
            (with the default pulse, above about ``1.2·f_max``) and at ``f ≤ 0``.

        Raises:
            ValueError: A frequency is outside :attr:`band` and
                ``allow_out_of_band`` is false.

        Notes:
            Exact discrete spectra at each ``f`` (no interpolation).
        """
        if f is None:
            f_file = self._file()
            if "results" not in f_file or "zrad" not in f_file["results"]:
                raise ValueError("test-source run: no radiator impedance")
            dset = f_file["results/zrad"]
            f_arr = np.atleast_1d(np.asarray(dset.attrs["f"], dtype=_F64))
            z_arr = np.atleast_1d(np.asarray(dset[()], dtype=_C128))
            return f_arr, z_arr
        if int(self.meta.get("n_radiators", 1)) == 0:
            raise ValueError("test-source run: no radiator impedance")
        return self._z_at(self._check_f(f, allow_out_of_band))

    def Z_acoustic(self, f: np.ndarray | float, *, allow_out_of_band: bool = False) -> np.ndarray:
        """Acoustic radiation impedance ``Z_a = ρ c z / S_grid``.

        Args:
            f: Frequencies in Hz (scalar or array-like; required) inside :attr:`band`.
            allow_out_of_band: Warn instead of raising for out-of-band ``f``.

        Returns:
            Complex128 ``(n_f,)`` in Pa·s/m³ (pressure per volume velocity).

        Raises:
            ValueError: Out-of-band ``f`` without ``allow_out_of_band``.
        """
        if int(self.meta.get("n_radiators", 1)) == 0:
            raise ValueError("test-source run: no radiator impedance")
        _, z = self._z_at(self._check_f(f, allow_out_of_band))
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        S = self.S_grid
        return np.atleast_1d((rho * c * z / S).astype(_C128, copy=False))

    def Z_mechanical(self, f: np.ndarray | float, *, allow_out_of_band: bool = False) -> np.ndarray:
        """Mechanical radiation impedance ``Z_m = ρ c S_grid z`` (front side only).

        Args:
            f: Frequencies in Hz (scalar or array-like; required) inside :attr:`band`.
            allow_out_of_band: Warn instead of raising for out-of-band ``f``.

        Returns:
            Complex128 ``(n_f,)`` in N·s/m (force per radiator velocity).

        Raises:
            ValueError: Out-of-band ``f`` without ``allow_out_of_band``.
        """
        if int(self.meta.get("n_radiators", 1)) == 0:
            raise ValueError("test-source run: no radiator impedance")
        _, z = self._z_at(self._check_f(f, allow_out_of_band))
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        S = self.S_grid
        return np.atleast_1d((rho * c * S * z).astype(_C128, copy=False))

    def _huygens_at_f(self, f_hz: float) -> tuple[SurfacePatches, complex, np.ndarray, np.ndarray]:
        key = float(f_hz)
        if key not in self._spectra_cache:
            self._spectra_cache[key] = load_huygens_spectra(self.path, self.derived, key)
        return self._spectra_cache[key]

    def _radiator_v_at_f(self, f_hz: float) -> complex:
        return self._huygens_at_f(f_hz)[1]

    def _image_plane_z(self) -> float | None:
        if self.config.geometry == "baffle":
            return 0.0
        return None

    def _probe_hit(
        self,
        points: np.ndarray,
        h: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-point probe index or −1 when no probe within ``h/2``."""
        f = self._file()
        if "probes" not in f:
            return np.full(points.shape[0], -1, dtype=np.int64), np.empty((0, 3), _F64)
        centers = np.asarray(f["probes/centers"][()], dtype=_F64)
        requested = (
            np.asarray(f["probes/requested"][()], dtype=_F64)
            if "probes/requested" in f
            else centers
        )
        idx = np.full(points.shape[0], -1, dtype=np.int64)
        half = 0.5 * float(h)
        for m, pt in enumerate(points):
            d_req = np.linalg.norm(requested - pt, axis=1)
            j = int(np.argmin(d_req))
            if d_req[j] <= 1e-9:
                idx[m] = j
                continue
            d = np.linalg.norm(centers - pt, axis=1)
            j = int(np.argmin(d))
            if d[j] <= half:
                idx[m] = j
        return idx, centers

    def _too_close_huygens(self, points: np.ndarray, patches: SurfacePatches) -> np.ndarray:
        """True where a point is enclosed by the Huygens surface or within ``h/2`` of a patch.

        The enclosed region is the box spanned by the patch centers (box mode),
        or that box with its bottom on the baffle plane ``z = 0`` (baffle mode).
        Parametric runs with ``environment`` planes extend the enclosed box to each
        rigid plane on the fluid side (as baffle mode does for ``z = 0``).
        """
        half = 0.5 * float(patches.h)
        lo = patches.centers.min(axis=0)
        hi = patches.centers.max(axis=0)
        if self.config.geometry == "baffle":
            lo = lo.copy()
            lo[2] = 0.0
        elif self.planes:
            lo, hi = self._apply_rigid_plane_bounds(lo, hi)
        inside = np.all((points > lo) & (points < hi), axis=1)
        near = distance_to_patches(points, patches) < half
        return inside | near

    def field(
        self,
        points: np.ndarray | list,
        f: np.ndarray | float,
        *,
        frame: Literal["local", "world"] | None = None,
        per: Literal["velocity", "volume_velocity"] = "velocity",
        allow_out_of_band: bool = False,
    ) -> np.ndarray:
        """Pressure transfer function at exterior points (Kirchhoff–Helmholtz).

        Points outside the Huygens surface use KH on the stored Huygens ``p`` and
        ``v_n`` (box mode: closed six-face surface; baffle mode: open five-face
        surface plus its image in ``z = 0``; parametric files with
        :attr:`planes` expand the open Huygens data over all plane-reflection
        copies and integrate as a closed surface with no separate image Green's
        function). Points enclosed by the Huygens surface, or within ``h/2`` of
        one of its finite patches, use the FDTD record of an ``output.probes``
        probe whose cell center is within ``h/2`` (probe coordinates in the file
        are always world-frame). Points in the plane of a Huygens face but beyond
        its edges are ordinary exterior points. World-frame points behind a rigid
        environment plane raise ``ValueError`` naming that plane.

        Args:
            points: ``(n_pts, 3)`` or ``(3,)`` coordinates in m in the frame
                given by ``frame`` (default :attr:`frame_default`).
            f: Frequencies in Hz (scalar or array-like) inside :attr:`band`.
            frame: ``"local"`` (radiator-0 frame) or ``"world"``; ``None`` uses
                :attr:`frame_default`.
            per: ``"velocity"`` returns ``p/v`` in Pa/(m/s);
                ``"volume_velocity"`` returns ``p/(S_grid v)`` in Pa/(m³/s).
            allow_out_of_band: Warn instead of raising for out-of-band ``f``.

        Returns:
            Complex128 ``(n_f, n_pts)``; phase relative to the radiator velocity.

        Raises:
            ValueError: Out-of-band ``f`` without ``allow_out_of_band``; ``per``
                not recognized; or a point is enclosed by or within ``h/2`` of the
                Huygens surface and no probe is within ``h/2`` of it.
        """
        if per not in ("velocity", "volume_velocity"):
            raise ValueError(f"per must be 'velocity' or 'volume_velocity', got {per!r}")
        frame_sel = self._resolve_frame(frame)
        f_arr = self._check_f(f, allow_out_of_band)
        pts = np.atleast_2d(np.asarray(points, dtype=_F64))
        if frame_sel == "local":
            pts_world = self._points_local_to_world(pts)
        else:
            pts_world = pts
        if self.planes:
            fluid = in_fluid(pts_world, self.planes)
            if not np.all(fluid):
                for m in np.nonzero(~fluid)[0]:
                    pt = pts_world[m]
                    viol = first_plane_violation(pt, self.planes)
                    raise ValueError(
                        f"field point {m} at ({pt[0]},{pt[1]},{pt[2]}) is behind rigid plane {viol}"
                    )
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        h = float(self.derived.h)
        out = np.zeros((f_arr.size, pts_world.shape[0]), dtype=_C128)
        probe_idx, _ = self._probe_hit(pts_world, h)

        file = self._file()
        has_probes = "probes" in file
        if has_probes:
            t_pr = np.asarray(file["probes/times_p"][()], dtype=_F64)
            p_ts = np.asarray(file["probes/p"][()], dtype=_F64)

        geom = self.config.geometry
        kh_geom = "box" if geom == "parametric" or self.planes else geom
        close: np.ndarray | None = None
        for fi, f_hz in enumerate(f_arr):
            patches, v_ref, p_face, vn_face = self._huygens_at_f(float(f_hz))
            if close is None:
                close = self._too_close_huygens(pts_world, patches)
                blocked = close & (probe_idx < 0)
                if np.any(blocked):
                    d_hy = distance_to_patches(pts_world[blocked], patches)
                    msg_parts = [
                        "field evaluation inside or within h/2 of the Huygens surface; "
                        "add output.probes within h/2 of the target point or move outward"
                    ]
                    for m, d_m in zip(np.nonzero(blocked)[0], d_hy):
                        msg_parts.append(
                            f"  point {m}: dist_to_Huygens={d_m:.4g} m (need ≥ {0.5 * h:.4g} m "
                            f"and outside the surface)"
                        )
                    raise ValueError("\n".join(msg_parts))
            free = ~close
            if np.any(free):
                patches_kh, p_kh_face, vn_kh_face = self._expand_huygens(
                    patches, p_face, vn_face
                )
                p_kh = kh_huygens_pressure_at_points(
                    pts_world[free],
                    patches_kh,
                    p_kh_face,
                    vn_kh_face,
                    float(f_hz),
                    rho,
                    c,
                    geometry=kh_geom,
                )
                out[fi, free] = p_kh / v_ref
            if np.any(close) and has_probes:
                for m in np.nonzero(close)[0]:
                    q = int(probe_idx[m])
                    p_q = probe_fdtd_pressure_spectrum(p_ts, t_pr, float(f_hz), q)
                    out[fi, m] = p_q / v_ref
        if per == "volume_velocity":
            out = out / self.S_grid
        return out

    def directivity(
        self,
        theta: np.ndarray | float,
        phi: np.ndarray | float,
        f: np.ndarray | float,
        *,
        frame: Literal["local", "world"] | None = None,
        normalize: bool = False,
        allow_out_of_band: bool = False,
        allow_masked: bool = False,
    ) -> np.ndarray:
        """Complex far-field pattern ``D(θ, φ, f) = lim_{r→∞} r e^{+jkr} p/v``.

        Evaluated as the Fraunhofer (r → ∞) KH amplitude of the stored Huygens
        data. For ``frame="local"`` (default), θ and φ define directions in the
        radiator-0 frame and the pattern includes the phase
        ``exp(−jk (R r̂_ℓ)·t)`` so that :meth:`field` at
        ``r r̂_ℓ`` matches ``D e^{−jkr}/r`` with ``r`` from the local origin. For
        ``frame="world"``, θ and φ are in the world frame and phase is referenced
        to the world origin. Parametric files with :attr:`planes` expand the
        Huygens data over all reflection copies; directions not in the open
        acoustic region raise ``ValueError``, or become NaN when
        ``allow_masked=True`` (ignored when there are no planes).

        Args:
            theta: Polar angle(s) from ``+z`` in radians (in the chosen frame).
            phi: Azimuth(s) from ``+x`` toward ``+y`` in radians. Broadcast with
                ``theta`` and flattened to ``n_dir`` directions.
            f: Frequencies in Hz (scalar or array-like) inside :attr:`band`.
            frame: ``"local"`` or ``"world"``; ``None`` uses :attr:`frame_default`.
            normalize: If true, divide each frequency row by the on-axis value
                (θ = 0 in the chosen frame: local ``+z`` or world ``+z``).
            allow_out_of_band: Warn instead of raising for out-of-band ``f``.
            allow_masked: With :attr:`planes`, write NaN for directions behind a
                rigid plane instead of raising (ignored when there are no planes).

        Returns:
            Complex128 ``(n_f, n_dir)`` in Pa·m/(m/s), or dimensionless when
            ``normalize`` is true. In baffle mode only ``θ ≤ π/2`` is physical
            (``θ > π/2`` returns the mirror image).

        Raises:
            ValueError: Out-of-band ``f`` without ``allow_out_of_band``; a
                direction or on-axis (with ``normalize=True``) is not in the open
                region when :attr:`planes` is non-empty and ``allow_masked`` is
                false.
        """
        frame_sel = self._resolve_frame(frame)
        f_arr = self._check_f(f, allow_out_of_band)
        th = np.atleast_1d(np.asarray(theta, dtype=_F64))
        ph = np.atleast_1d(np.asarray(phi, dtype=_F64))
        th, ph = np.broadcast_arrays(th, ph)
        th = np.ravel(th)
        ph = np.ravel(ph)
        st = np.sin(th)
        dirs_loc = np.column_stack([st * np.cos(ph), st * np.sin(ph), np.cos(th)])
        n_dir = dirs_loc.shape[0]
        on_axis_loc = np.array([[0.0, 0.0, 1.0]], dtype=_F64)
        if normalize:
            dirs_loc = np.vstack([dirs_loc, on_axis_loc])
        if frame_sel == "local":
            dirs_kh = self._dirs_local_to_world(dirs_loc)
        else:
            dirs_kh = dirs_loc
        open_mask: np.ndarray | None = None
        if self.planes:
            open_mask = direction_open(dirs_kh, self.planes)
            if normalize and not bool(open_mask[-1]):
                raise ValueError(
                    "normalize=True but the on-axis direction is not in the open "
                    "acoustic region defined by environment planes"
                )
            if not allow_masked and not np.all(open_mask):
                raise ValueError(
                    "directivity direction(s) not in the open acoustic region "
                    "defined by environment planes"
                )
        _, t_pose = self.pose
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        image_z = None if self.planes else self._image_plane_z()
        out = np.zeros((f_arr.size, n_dir), dtype=_C128)
        for fi, f_hz in enumerate(f_arr):
            patches, v_ref, p_face, vn_face = self._huygens_at_f(float(f_hz))
            patches, p_face, vn_face = self._expand_huygens(patches, p_face, vn_face)
            omega = np.asarray([2.0 * math.pi * float(f_hz)], dtype=_F64)
            k_sel = np.asarray([2.0 * math.pi * float(f_hz) / c], dtype=_F64)
            dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
            amp = kh_far_field(
                dirs_kh,
                patches,
                p_face[np.newaxis, :],
                dpdn,
                k_sel,
                image_plane_z=image_z,
            )[0]
            pat = amp / v_ref
            if frame_sel == "local":
                k = float(k_sel[0])
                phase = np.exp(-1j * k * (dirs_kh @ t_pose))
                pat = pat * phase
            if open_mask is not None and allow_masked:
                pat = pat.copy()
                pat[~open_mask] = np.nan + 0j
            if normalize:
                axis_val = pat[-1]
                if abs(axis_val) < 1e-300:
                    axis_val = 1e-300 + 0j
                pat = pat[:n_dir] / axis_val
            elif open_mask is not None and allow_masked:
                pat = pat[:n_dir]
            out[fi] = pat[:n_dir]
        return out

    def surface_p(
        self,
        points: np.ndarray | list,
        f: np.ndarray | float,
        *,
        allow_out_of_band: bool = False,
    ) -> np.ndarray:
        """Pressure transfer ``H = p/v`` on the rigid box surface (box mode only).

        Each point maps to the nearest stored ``h × h`` box-surface patch (no
        interpolation). Rigid patches carry the adjacent cell-center pressure
        (``h/2`` outside the face); radiator patches add the ``(h/2)ρ ∂v/∂t``
        face correction. Pressure is singular at box edges and the radiator rim.

        Args:
            points: ``(n_pts, 3)`` or ``(3,)`` coordinates in m on or near the box.
            f: Frequencies in Hz (scalar or array-like) inside :attr:`band`.
            allow_out_of_band: Warn instead of raising for out-of-band ``f``.

        Returns:
            Complex128 ``(n_f, n_pts)`` in Pa/(m/s).

        Raises:
            ValueError: Out-of-band ``f`` without ``allow_out_of_band``; the file
                has no ``box_surface`` group (baffle mode); or a point is more than
                ``h`` from the plane of its nearest patch.
        """
        f_arr = self._check_f(f, allow_out_of_band)
        pts = np.atleast_2d(np.asarray(points, dtype=_F64))
        h = float(self.derived.h)
        out = np.zeros((f_arr.size, pts.shape[0]), dtype=_C128)
        f_h5 = self._file()
        if "box_surface" not in f_h5:
            raise ValueError("HDF5 has no box_surface group (baffle geometry)")
        normals = np.asarray(f_h5["box_surface/normals"][()], dtype=_F64)
        centers_st = np.asarray(f_h5["box_surface/centers"][()], dtype=_F64)
        for fi, f_hz in enumerate(f_arr):
            _, p_spec = load_box_surface_spectrum(self.path, float(f_hz))
            v_ref = self._radiator_v_at_f(float(f_hz))
            for m, pt in enumerate(pts):
                dist = np.linalg.norm(centers_st - pt, axis=1)
                j = int(np.argmin(dist))
                rel = pt - centers_st[j]
                dist_plane = abs(float(np.dot(rel, normals[j])))
                if dist_plane > h:
                    raise ValueError(
                        f"surface_p point {m} is {dist_plane:.4g} m from the nearest "
                        f"box patch (limit h={h:.4g} m)"
                    )
                out[fi, m] = p_spec[j] / v_ref
        return out

    def power(self, f: np.ndarray | float, *, allow_out_of_band: bool = False) -> np.ndarray:
        """Radiated power (front side) for a 1 m/s peak radiator velocity.

        ``W = ½ ρ c S_grid Re z``; scale by ``|v|²`` for other velocities.

        Args:
            f: Frequencies in Hz (scalar or array-like; required) inside :attr:`band`.
            allow_out_of_band: Warn instead of raising for out-of-band ``f``.

        Returns:
            Float64 ``(n_f,)`` in W.

        Raises:
            ValueError: Out-of-band ``f`` without ``allow_out_of_band``.
        """
        _, z = self._z_at(self._check_f(f, allow_out_of_band))
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        return np.atleast_1d(0.5 * rho * c * self.S_grid * np.real(z))

    def directivity_index(
        self,
        f: np.ndarray | float,
        *,
        frame: Literal["local", "world"] | None = None,
        allow_out_of_band: bool = False,
    ) -> np.ndarray:
        """On-axis directivity index ``DI = 10 log₁₀(4π |p_axis|² / ∫ |p|² dΩ)``.

        ``p`` is the KH pressure on a sphere of radius ``50·max(W, H, D, h)``
        centered at the radiator origin in the chosen frame (local: :attr:`pose`
        translation; world: coordinate origin). Quadrature uses directions in
        that frame (Gauss–Legendre 24 in cos θ × 48 uniform in φ, via
        :func:`boxrad.post.directivity.di_from_pattern`). Box mode integrates the
        full sphere; baffle mode integrates only the physical half space
        ``z ≥ 0`` (conventional half-space DI, e.g. 3.01 dB for a small piston).
        Parametric files with :attr:`planes` expand the Huygens data, integrate
        over the full sphere (``half_space=False``), then add
        ``10 log₁₀ 2^{n_p}`` (e.g. 6.02 dB in a quarter space with two planes).
        The on-axis direction must lie in the open region or ``ValueError`` is
        raised.

        Args:
            f: Frequencies in Hz (scalar or array-like) inside :attr:`band`.
            frame: ``"local"`` or ``"world"``; ``None`` uses :attr:`frame_default``.
            allow_out_of_band: Warn instead of raising for out-of-band ``f``.

        Returns:
            Float64 ``(n_f,)`` in dB.

        Raises:
            ValueError: Out-of-band ``f`` without ``allow_out_of_band``; or the
                on-axis direction is not in the open acoustic region when
                :attr:`planes` is non-empty.
        """
        frame_sel = self._resolve_frame(frame)
        f_arr = self._check_f(f, allow_out_of_band)
        if self.planes:
            axis_w = self._on_axis_world(frame_sel)
            if not bool(direction_open(axis_w.reshape(1, 3), self.planes)[0]):
                raise ValueError(
                    "directivity_index on-axis direction is not in the open "
                    "acoustic region defined by environment planes"
                )
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        image_z = None if self.planes else self._image_plane_z()
        half_space = False if self.planes else self.config.geometry == "baffle"
        di_plane = (
            10.0 * math.log10(float(1 << len(self.planes))) if self.planes else 0.0
        )
        d = self.derived
        r_far = 50.0 * max(d.realized_W, d.realized_H, d.realized_D, d.h)
        r_mat, t_pose = self.pose
        out = np.zeros(f_arr.size, dtype=_F64)
        for fi, f_hz in enumerate(f_arr):
            patches, v_ref, p_face, vn_face = self._huygens_at_f(float(f_hz))
            patches, p_face, vn_face = self._expand_huygens(patches, p_face, vn_face)
            omega = np.asarray([2.0 * math.pi * float(f_hz)], dtype=_F64)
            k_sel = np.asarray([2.0 * math.pi * float(f_hz) / c], dtype=_F64)
            dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
            p_row = p_face[np.newaxis, :]

            def pattern(dirs: np.ndarray) -> np.ndarray:
                if frame_sel == "local":
                    dirs_w = dirs @ r_mat.T
                    pts = t_pose + r_far * dirs_w
                else:
                    pts = r_far * dirs
                return kh_pressure(
                    pts,
                    patches,
                    p_row,
                    dpdn,
                    k_sel,
                    image_plane_z=image_z,
                    near_subdivide=False,
                )[0] / v_ref

            out[fi] = di_from_pattern(pattern, half_space, _DI_THETA, _DI_PHI) + di_plane
        return np.atleast_1d(out)
