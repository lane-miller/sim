"""Lazy-loading user API over boxed-piston HDF5 results."""

from __future__ import annotations

import json
import math
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
from boxrad.post.transfer import (
    kh_huygens_pressure_at_points,
    load_box_surface_spectrum,
    load_huygens_spectra,
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
_BAND_RTOL = 1e-9


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

    Coordinates: origin at the radiator center on the radiating face, ``+z``
    outward; in box mode the box occupies ``z ∈ [−D, 0]``.

    The HDF5 file is opened read-only on first access and kept open until
    :meth:`close` (or the end of a ``with`` block).

    Args:
        path: Path to an HDF5 file written by :func:`boxrad.run`.

    Raises:
        FileNotFoundError: ``path`` is not an existing file.
        ImportError: h5py is not installed (raised on first data access).
    """

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
    def S_grid(self) -> float:
        """Discretized radiator area ``h² Σ w`` in m² (used for every z ↔ Z conversion)."""
        if "S_grid" in self.meta:
            return float(self.meta["S_grid"])
        return float(self._grid.radiator_area_grid)

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
            dset = self._file()["results/zrad"]
            f_arr = np.atleast_1d(np.asarray(dset.attrs["f"], dtype=_F64))
            z_arr = np.atleast_1d(np.asarray(dset[()], dtype=_C128))
            return f_arr, z_arr
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
        idx = np.full(points.shape[0], -1, dtype=np.int64)
        half = 0.5 * float(h)
        for m, pt in enumerate(points):
            d = np.linalg.norm(centers - pt, axis=1)
            j = int(np.argmin(d))
            if d[j] <= half:
                idx[m] = j
        return idx, centers

    def _too_close_huygens(self, points: np.ndarray, patches: SurfacePatches) -> np.ndarray:
        """True where a point is enclosed by the Huygens surface or within ``h/2`` of a patch.

        The enclosed region is the box spanned by the patch centers (box mode),
        or that box with its bottom on the baffle plane ``z = 0`` (baffle mode).
        """
        half = 0.5 * float(patches.h)
        lo = patches.centers.min(axis=0)
        hi = patches.centers.max(axis=0)
        if self.config.geometry == "baffle":
            lo = lo.copy()
            lo[2] = 0.0
        inside = np.all((points > lo) & (points < hi), axis=1)
        near = distance_to_patches(points, patches) < half
        return inside | near

    def field(
        self,
        points: np.ndarray | list,
        f: np.ndarray | float,
        *,
        per: Literal["velocity", "volume_velocity"] = "velocity",
        allow_out_of_band: bool = False,
    ) -> np.ndarray:
        """Pressure transfer function at exterior points (Kirchhoff–Helmholtz).

        Points outside the Huygens surface use KH on the stored Huygens ``p`` and
        ``v_n`` (box mode: closed six-face surface; baffle mode: open five-face
        surface plus its image in ``z = 0``), evaluated with a chunked,
        vectorized kernel. Points enclosed by the Huygens surface, or within
        ``h/2`` of one of its finite patches, use the FDTD record of an
        ``output.probes`` probe whose cell center is within ``h/2``. Points in
        the plane of a Huygens face but beyond its edges are ordinary exterior
        points.

        Args:
            points: ``(n_pts, 3)`` or ``(3,)`` coordinates in m.
            f: Frequencies in Hz (scalar or array-like) inside :attr:`band`.
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
        f_arr = self._check_f(f, allow_out_of_band)
        pts = np.atleast_2d(np.asarray(points, dtype=_F64))
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
        kh_geom = "box" if geom == "parametric" else geom
        close: np.ndarray | None = None
        for fi, f_hz in enumerate(f_arr):
            patches, v_ref, p_face, vn_face = self._huygens_at_f(float(f_hz))
            if close is None:
                close = self._too_close_huygens(pts, patches)
                blocked = close & (probe_idx < 0)
                if np.any(blocked):
                    d_hy = distance_to_patches(pts[blocked], patches)
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
                p_kh = kh_huygens_pressure_at_points(
                    pts[free],
                    patches,
                    p_face,
                    vn_face,
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
        normalize: bool = False,
        allow_out_of_band: bool = False,
    ) -> np.ndarray:
        """Complex far-field pattern ``D(θ, φ, f) = lim_{r→∞} r e^{+jkr} p/v``.

        Evaluated as the Fraunhofer (r → ∞) KH amplitude of the stored Huygens
        data. The phase is referenced to the coordinate origin (the radiator
        center on the radiating face): at large ``r``,
        ``p/v ≈ D(θ, φ) e^{−jkr}/r`` with ``r`` measured from the origin
        (``e^{+jωt}`` convention). Use ``np.abs`` for the magnitude.

        Args:
            theta: Polar angle(s) from ``+z`` in radians.
            phi: Azimuth(s) from ``+x`` toward ``+y`` in radians. Broadcast with
                ``theta`` and flattened to ``n_dir`` directions.
            f: Frequencies in Hz (scalar or array-like) inside :attr:`band`.
            normalize: If true, divide each frequency row by the complex on-axis
                (θ = 0) value, so the on-axis entry is exactly ``1 + 0j``.
            allow_out_of_band: Warn instead of raising for out-of-band ``f``.

        Returns:
            Complex128 ``(n_f, n_dir)`` in Pa·m/(m/s), or dimensionless when
            ``normalize`` is true. In baffle mode only ``θ ≤ π/2`` is physical
            (``θ > π/2`` returns the mirror image).

        Raises:
            ValueError: Out-of-band ``f`` without ``allow_out_of_band``.
        """
        f_arr = self._check_f(f, allow_out_of_band)
        th = np.atleast_1d(np.asarray(theta, dtype=_F64))
        ph = np.atleast_1d(np.asarray(phi, dtype=_F64))
        th, ph = np.broadcast_arrays(th, ph)
        th = np.ravel(th)
        ph = np.ravel(ph)
        st = np.sin(th)
        dirs = np.column_stack([st * np.cos(ph), st * np.sin(ph), np.cos(th)])
        n_dir = dirs.shape[0]
        if normalize:
            dirs = np.vstack([dirs, np.array([[0.0, 0.0, 1.0]], dtype=_F64)])
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        image_z = self._image_plane_z()
        out = np.zeros((f_arr.size, n_dir), dtype=_C128)
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
            pat = amp / v_ref
            if normalize:
                axis_val = pat[-1]
                if abs(axis_val) < 1e-300:
                    axis_val = 1e-300 + 0j
                pat = pat[:n_dir] / axis_val
            out[fi] = pat
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
        allow_out_of_band: bool = False,
    ) -> np.ndarray:
        """On-axis directivity index ``DI = 10 log₁₀(4π |p_axis|² / ∫ |p|² dΩ)``.

        ``p`` is the KH pressure on a sphere of radius ``50·max(W, H, D, h)``
        (Gauss–Legendre 24 in cos θ × 48 uniform in φ, via
        :func:`boxrad.post.directivity.di_from_pattern`). Box mode integrates the
        full sphere; baffle mode integrates only the physical half space
        ``z ≥ 0`` (conventional half-space DI, e.g. 3.01 dB for a small piston).

        Args:
            f: Frequencies in Hz (scalar or array-like) inside :attr:`band`.
            allow_out_of_band: Warn instead of raising for out-of-band ``f``.

        Returns:
            Float64 ``(n_f,)`` in dB.

        Raises:
            ValueError: Out-of-band ``f`` without ``allow_out_of_band``.
        """
        f_arr = self._check_f(f, allow_out_of_band)
        rho = float(self.config.air.rho)
        c = float(self.config.air.c)
        image_z = self._image_plane_z()
        half_space = self.config.geometry == "baffle"
        d = self.derived
        r_far = 50.0 * max(d.realized_W, d.realized_H, d.realized_D, d.h)
        out = np.zeros(f_arr.size, dtype=_F64)
        for fi, f_hz in enumerate(f_arr):
            patches, v_ref, p_face, vn_face = self._huygens_at_f(float(f_hz))
            omega = np.asarray([2.0 * math.pi * float(f_hz)], dtype=_F64)
            k_sel = np.asarray([2.0 * math.pi * float(f_hz) / c], dtype=_F64)
            dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, rho)
            p_row = p_face[np.newaxis, :]

            def pattern(dirs: np.ndarray) -> np.ndarray:
                return kh_pressure(
                    r_far * dirs,
                    patches,
                    p_row,
                    dpdn,
                    k_sel,
                    image_plane_z=image_z,
                    near_subdivide=False,
                )[0] / v_ref

            out[fi] = di_from_pattern(pattern, half_space, _DI_THETA, _DI_PHI)
        return np.atleast_1d(out)
