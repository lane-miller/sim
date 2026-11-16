"""Lazy-loading user API over boxed-piston HDF5 results."""

from __future__ import annotations

import json
import math
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np

from boxrad import __version__
from boxrad.config import Config, Derived
from boxrad.grid import Grid
from boxrad.post.kirchhoff import (
    SurfacePatches,
    _dist_to_box,
    dpdn_from_vn,
    far_field_at_radius,
    kh_far_field,
    kh_pressure,
)
from boxrad.post.transfer import (
    kh_huygens_pressure_at_points,
    load_box_surface_spectrum,
    load_huygens_spectra,
    min_distance_to_huygens_surface,
    probe_fdtd_pressure_spectrum,
)
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


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError("h5py is required for boxrad.Result")
    return h5py


class Result:
    """User-facing view of a completed FDTD HDF5 file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._h5: Any | None = None
        self._spectra_cache: dict[float, tuple[SurfacePatches, complex, np.ndarray, np.ndarray]] = {}

    def _file(self) -> Any:
        if self._h5 is None:
            self._h5 = _require_h5py().File(self.path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> Result:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @cached_property
    def config(self) -> Config:
        raw = bytes(self._file()["meta/config_json"][()]).decode("utf-8")
        return Config.from_json(raw)

    @cached_property
    def derived(self) -> Derived:
        blob = json.loads(bytes(self._file()["meta/derived_json"][()]).decode("utf-8"))
        return Derived(**blob)

    @cached_property
    def meta(self) -> dict[str, Any]:
        g = self._file()["meta"]
        out: dict[str, Any] = {}
        for key, val in g.attrs.items():
            out[key] = val.item() if hasattr(val, "item") else val
        return out

    @cached_property
    def accuracy(self) -> dict[str, Any]:
        f = self._file()
        if "accuracy" not in f["meta"]:
            return {}
        g = f["meta/accuracy"]
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in g.attrs.items()}

    @property
    def band(self) -> tuple[float, float]:
        d = self.derived
        f_max = float(d.f_max)
        return float(self.config.band.f_min), f_max

    @cached_property
    def _grid(self) -> Grid:
        return Grid(self.config, self.derived)

    @cached_property
    def S_grid(self) -> float:
        return float(self._grid.radiator_area_grid)

    def summary(self) -> str:
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

    def z(self, f: np.ndarray | float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Radiation impedance ``z_rad`` (Pa·s/m³ per m/s) on the radiator.

        Returns ``f_hz`` with shape ``(n_f,)`` and complex ``z_rad`` with shape ``(n_f,)``.
        With ``f is None``, uses the log-spaced ``results/zrad`` table stored in the HDF5 file.
        """
        if f is None:
            dset = self._file()["results/zrad"]
            f_arr = np.atleast_1d(np.asarray(dset.attrs["f"], dtype=_F64))
            z_arr = np.atleast_1d(np.asarray(dset[()], dtype=_C128))
            return f_arr, z_arr
        f_arr = np.atleast_1d(np.asarray(f, dtype=_F64))
        f_out, z_out = zrad_at_frequencies(self.path, f_arr)
        return np.atleast_1d(f_out), np.atleast_1d(z_out)

    def Z_acoustic(self, f: np.ndarray | float) -> np.ndarray:
        """Acoustic impedance ``Z_a = ρ c z_rad / S`` with shape ``(n_f,)`` (Pa·s/m³ per m³/s)."""
        _, z = self.z(f)
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        S = self.S_grid
        return np.atleast_1d((rho * c * z / S).astype(_C128, copy=False))

    def Z_mechanical(self, f: np.ndarray | float) -> np.ndarray:
        """Mechanical impedance ``Z_m = ρ c S z_rad`` with shape ``(n_f,)`` (N·s/m per m/s)."""
        _, z = self.z(f)
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
        return 0.0 if self.config.geometry == "baffle" else None

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
        idx = np.full(points.shape[0], -1, dtype=np.int64)
        half = 0.5 * float(h)
        for m, pt in enumerate(points):
            d = np.linalg.norm(centers - pt, axis=1)
            j = int(np.argmin(d))
            if d[j] <= half:
                idx[m] = j
        return idx, centers

    def _too_close_huygens(self, points: np.ndarray, patches: SurfacePatches) -> np.ndarray:
        h = float(patches.h)
        half = 0.5 * h
        ymin = patches.centers.min(axis=0)
        ymax = patches.centers.max(axis=0)
        out = np.zeros(points.shape[0], dtype=bool)
        for m, pt in enumerate(points):
            if _dist_to_box(pt, ymin, ymax) < half:
                out[m] = True
                continue
            rel = pt - patches.centers
            dist_plane = np.abs(np.sum(rel * patches.normals, axis=1))
            if float(np.min(dist_plane)) < half:
                out[m] = True
        return out

    def field(
        self,
        points: np.ndarray | list,
        f: np.ndarray | float,
        *,
        per: Literal["velocity", "volume_velocity"] = "velocity",
    ) -> np.ndarray:
        """KH pressure transfer ``H(f)`` shape ``(n_f, n_pts)`` in Pa/(m/s) or Pa/(m³/s)."""
        pts = np.atleast_2d(np.asarray(points, dtype=_F64))
        f_arr = np.atleast_1d(np.asarray(f, dtype=_F64))
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        h = float(self.derived.h)
        out = np.zeros((f_arr.size, pts.shape[0]), dtype=_C128)
        probe_idx, _ = self._probe_hit(pts, h)

        file = self._file()
        has_probes = "probes" in file
        if has_probes:
            t_pr = np.asarray(file["probes/times_p"][()], dtype=_F64)
            p_ts = np.asarray(file["probes/p"][()], dtype=_F64)

        geom = self.config.geometry
        for fi, f_hz in enumerate(f_arr):
            patches, v_ref, p_face, vn_face = self._huygens_at_f(float(f_hz))
            close = self._too_close_huygens(pts, patches)
            free = ~(close & (probe_idx < 0))
            if np.any(free):
                p_kh = kh_huygens_pressure_at_points(
                    pts[free],
                    patches,
                    p_face,
                    vn_face,
                    float(f_hz),
                    rho,
                    c,
                    geometry=geom,
                )
                out[fi, free] = p_kh / v_ref
            blocked = close & (probe_idx < 0)
            if np.any(blocked):
                msg_parts = [
                    "field evaluation inside or within h/2 of the Huygens surface; "
                    "add output.probes within h/2 of the target point or move outward"
                ]
                for m in np.nonzero(blocked)[0]:
                    d_hy = min_distance_to_huygens_surface(pts[m], patches)
                    msg_parts.append(
                        f"  point {m}: dist_to_Huygens={d_hy:.4g} m (need ≥ {0.5 * h:.4g} m)"
                    )
                raise ValueError("\n".join(msg_parts))
            use_pr = close & (probe_idx >= 0)
            if np.any(use_pr) and has_probes:
                for m in np.nonzero(use_pr)[0]:
                    q = int(probe_idx[m])
                    p_q = probe_fdtd_pressure_spectrum(p_ts, t_pr, float(f_hz), q)
                    out[fi, m] = p_q / v_ref
        if per == "volume_velocity":
            out = out * self.S_grid
        return out

    def directivity(
        self,
        theta: np.ndarray | float,
        phi: np.ndarray | float,
        f: np.ndarray | float,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Far-field magnitude ``|H(θ, φ)|`` with shape ``(n_f, n_dir)``.

        ``theta`` and ``phi`` are broadcast to 1-D length ``n_dir`` (radians).
        When ``normalize`` is true, each frequency row is divided by its on-axis value.
        """
        th = np.atleast_1d(np.asarray(theta, dtype=_F64))
        ph = np.atleast_1d(np.asarray(phi, dtype=_F64))
        th, ph = np.broadcast_arrays(th, ph)
        th = np.ravel(th)
        ph = np.ravel(ph)
        f_arr = np.atleast_1d(np.asarray(f, dtype=_F64))
        st = np.sin(th)
        dirs = np.column_stack([st * np.cos(ph), st * np.sin(ph), np.cos(th)])
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        image_z = self._image_plane_z()
        r_ref = 1.0
        out = np.zeros((f_arr.size, dirs.shape[0]), dtype=_F64)
        for fi, f_hz in enumerate(f_arr):
            patches, v_ref, p_face, vn_face = self._huygens_at_f(float(f_hz))
            omega = np.asarray([2.0 * math.pi * float(f_hz)], dtype=_F64)
            k_sel = np.asarray([2.0 * math.pi * float(f_hz) / c], dtype=_F64)
            dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
            amp = kh_far_field(
                dirs,
                patches,
                p_face[np.newaxis, :],
                dpdn,
                k_sel,
                image_plane_z=image_z,
            )[0]
            p_ff = far_field_at_radius(amp[np.newaxis, :], k_sel, R_ref=r_ref)[0]
            h_pat = p_ff / v_ref
            mag = np.abs(h_pat)
            if normalize:
                axis = np.array([[0.0, 0.0, 1.0]], dtype=_F64)
                amp0 = kh_far_field(
                    axis,
                    patches,
                    p_face[np.newaxis, :],
                    dpdn,
                    k_sel,
                    image_plane_z=image_z,
                )[0, 0]
                p0 = far_field_at_radius(np.asarray([[amp0]]), k_sel, R_ref=r_ref)[0, 0]
                mag = mag / max(abs(p0 / v_ref), 1e-30)
            out[fi] = mag
        return out

    def surface_p(
        self,
        points: np.ndarray | list,
        f: np.ndarray | float,
    ) -> np.ndarray:
        """Box-face pressure transfer ``H`` with shape ``(n_f, n_pts)`` (Pa/(m/s)).

        ``points`` are promoted to ``(n_pts, 3)``; each point maps to the nearest patch center.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=_F64))
        f_arr = np.atleast_1d(np.asarray(f, dtype=_F64))
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

    def power(self, f: np.ndarray | float) -> np.ndarray:
        """Acoustic power ``½ ρ c S Re(z_rad)`` with shape ``(n_f,)`` (W)."""
        _, z = self.z(f)
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        return np.atleast_1d(0.5 * rho * c * self.S_grid * np.real(z))

    def _sphere_dirs_weights(self) -> tuple[np.ndarray, np.ndarray]:
        from scipy.special import roots_legendre

        n_theta, n_phi = _DI_THETA, _DI_PHI
        u_nodes, w_u = roots_legendre(n_theta)
        phi = (np.arange(n_phi, dtype=_F64) + 0.5) * (2.0 * math.pi / n_phi)
        w_phi = 2.0 * math.pi / n_phi
        u_grid, phi_grid = np.meshgrid(u_nodes, phi, indexing="ij")
        w_u_grid, _ = np.meshgrid(w_u, phi, indexing="ij")
        sin_theta = np.sqrt(np.maximum(1.0 - u_grid * u_grid, 0.0))
        dirs = np.stack(
            [sin_theta * np.cos(phi_grid), sin_theta * np.sin(phi_grid), u_grid],
            axis=-1,
        ).reshape(-1, 3)
        weights = (w_u_grid * w_phi).reshape(-1)
        weights *= (4.0 * math.pi) / float(np.sum(weights))
        return dirs.astype(_F64), weights.astype(_F64)

    def directivity_index(self, f: np.ndarray | float) -> np.ndarray:
        """Directivity index (dB) via far-field sphere quadrature; shape ``(n_f,)``."""
        f_arr = np.atleast_1d(np.asarray(f, dtype=_F64))
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        image_z = self._image_plane_z()
        d = self.derived
        r_far = 50.0 * max(d.realized_W, d.realized_H, d.realized_D, d.h)
        dirs, w_omega = self._sphere_dirs_weights()
        pts = r_far * dirs
        out = np.zeros(f_arr.size, dtype=_F64)
        axis_pt = np.array([[0.0, 0.0, r_far]], dtype=_F64)
        for fi, f_hz in enumerate(f_arr):
            patches, v_ref, p_face, vn_face = self._huygens_at_f(float(f_hz))
            omega = np.asarray([2.0 * math.pi * float(f_hz)], dtype=_F64)
            k_sel = np.asarray([2.0 * math.pi * float(f_hz) / c], dtype=_F64)
            dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
            p_row = p_face[np.newaxis, :]
            p_obs = kh_pressure(
                pts,
                patches,
                p_row,
                dpdn,
                k_sel,
                image_plane_z=image_z,
                near_subdivide=False,
            )[0]
            h_ff = p_obs / v_ref
            w_ff = float(
                np.sum((np.abs(h_ff) ** 2) / (2.0 * rho * c) * (r_far**2) * w_omega)
            )
            p_ax = kh_pressure(
                axis_pt,
                patches,
                p_row,
                dpdn,
                k_sel,
                image_plane_z=image_z,
                near_subdivide=False,
            )[0, 0]
            h_axis = complex(p_ax / v_ref)
            numer = 4.0 * math.pi * (r_far**2) * (abs(h_axis) ** 2)
            denom = 2.0 * rho * c * max(w_ff, 1e-30)
            out[fi] = 10.0 * math.log10(max(numer / denom, 1e-30))
        return np.atleast_1d(out)