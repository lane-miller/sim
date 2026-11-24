"""HDF5 transfer-function helpers (Huygens / radiator / box surface spectra)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from boxrad.post.kirchhoff import (
    SurfacePatches,
    distance_to_patches,
    dpdn_from_vn,
    kh_pressure,
)
from boxrad.post.spectra import spectrum, spectrum_rows
from boxrad.post.zrad import causal_end_taper
from boxrad.source import velocity_pulse

_F64 = np.float64
_C128 = np.complex128


def min_distance_to_huygens_surface(
    point: np.ndarray, patches: SurfacePatches
) -> float:
    """Unsigned distance (m) from a point to the nearest finite Huygens patch."""
    pt = np.asarray(point, dtype=_F64).ravel()[:3]
    return distance_to_patches(pt[np.newaxis, :], patches)[0].item()


def probe_fdtd_pressure_spectrum(
    probe_p_ts: np.ndarray,
    t_probe: np.ndarray,
    f_hz: float,
    probe_index: int,
) -> complex:
    """Probe pressure spectrum ``P(f)`` (validation C6/C8/C9 recipe)."""
    win = causal_end_taper(int(probe_p_ts.shape[0]))
    col = np.asarray(probe_p_ts[:, int(probe_index)], dtype=_F64)
    t = np.asarray(t_probe, dtype=_F64)
    return complex(spectrum(col * win, t, float(f_hz)))


def kh_huygens_pressure_at_points(
    points: np.ndarray,
    patches: SurfacePatches,
    p_face: np.ndarray,
    vn_face: np.ndarray,
    f_hz: float,
    rho: float,
    c: float,
    *,
    geometry: str,
) -> np.ndarray:
    """Exterior KH pressure at points; matches validated box/baffle probe scripts.

    Both modes use the chunked, vectorized :func:`kh_pressure` without near-field
    subdivision; in box mode this equals the reference
    :func:`~boxrad.post.kirchhoff.kh_pressure_loop` to rounding.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=_F64))
    omega = np.asarray([2.0 * math.pi * float(f_hz)], dtype=_F64)
    k_sel = np.asarray([2.0 * math.pi * float(f_hz) / float(c)], dtype=_F64)
    p_face = np.asarray(p_face, dtype=_C128).ravel()
    vn_face = np.asarray(vn_face, dtype=_C128).ravel()
    dpdn = dpdn_from_vn(vn_face[np.newaxis, :], omega, float(rho))
    p_row = p_face[np.newaxis, :]
    return kh_pressure(
        pts,
        patches,
        p_row,
        dpdn,
        k_sel,
        image_plane_z=0.0 if geometry == "baffle" else None,
        near_subdivide=False,
    )[0]


def h_transfer_err(h_ref: complex, h: complex) -> tuple[float, float]:
    """Compare transfer phasors H (dB and phase deg relative to *h_ref*)."""
    db = 20.0 * math.log10(abs(h) / max(abs(h_ref), 1e-30))
    ph = float(np.rad2deg(np.angle(h) - np.angle(h_ref)))
    while ph > 180.0:
        ph -= 360.0
    while ph < -180.0:
        ph += 360.0
    return db, ph


def load_radiator_velocity_spectrum(source: str | Path | Any, derived: object, f_hz: float) -> complex:
    """Radiator velocity spectrum ``V(f)`` at pressure sample times."""
    import h5py

    owned = False
    if isinstance(source, h5py.File):
        f = source
    else:
        f = h5py.File(Path(source), "r")
        owned = True
    try:
        t_rad = np.asarray(f["radiator/times_p"][()], dtype=_F64)
        v_rad = np.array(
            [
                velocity_pulse(float(t), derived.pulse_sigma, derived.pulse_t0)  # type: ignore[attr-defined]
                for t in t_rad
            ],
            dtype=_F64,
        )
    finally:
        if owned:
            f.close()
    return complex(spectrum(v_rad, t_rad, f_hz))


def load_huygens_spectra(
    path: Path,
    derived: object,
    f_hz: float,
) -> tuple[SurfacePatches, complex, np.ndarray, np.ndarray]:
    """Huygens face spectra and radiator ``V(f)`` (dimensionless ``z`` uses the same ``V``)."""
    import h5py

    with h5py.File(path, "r") as f:
        centers = np.asarray(f["huygens/centers"][()], dtype=_F64)
        normals = np.asarray(f["huygens/normals"][()], dtype=_F64)
        areas = np.asarray(f["huygens/areas"][()], dtype=_F64)
        patches = SurfacePatches(centers, normals, areas, float(derived.h))  # type: ignore[attr-defined]
        t_p = np.asarray(f["huygens/times_p"][()], dtype=_F64)
        t_v = np.asarray(f["huygens/times_vn"][()], dtype=_F64)
        p_ts = np.asarray(f["huygens/p_face"][()], dtype=_F64)
        vn_ts = np.asarray(f["huygens/vn"][()], dtype=_F64)
        if "radiator" in f:
            t_rad = np.asarray(f["radiator/times_p"][()], dtype=_F64)
        else:
            t_rad = np.asarray(f["huygens/times_p"][()], dtype=_F64)
        v_rad = np.array(
            [
                velocity_pulse(float(t), derived.pulse_sigma, derived.pulse_t0)  # type: ignore[attr-defined]
                for t in t_rad
            ],
            dtype=_F64,
        )
    n_samp = p_ts.shape[0]
    win = causal_end_taper(n_samp)
    p_win = p_ts * win[:, np.newaxis]
    vn_win = vn_ts * win[:, np.newaxis]
    V = complex(spectrum(v_rad, t_rad, f_hz))
    p_face = np.asarray(spectrum_rows(p_win, t_p, f_hz), dtype=_C128)
    vn_face = np.asarray(spectrum_rows(vn_win, t_v, f_hz), dtype=_C128)
    return patches, V, p_face, vn_face


def load_box_surface_spectrum(path: Path, f_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Box-face pressure row spectrum at stored sample times."""
    import h5py

    with h5py.File(path, "r") as f:
        if "box_surface" not in f:
            raise RuntimeError("HDF5 missing box_surface group")
        centers = np.asarray(f["box_surface/centers"][()], dtype=_F64)
        t = np.asarray(f["box_surface/times"][()], dtype=_F64)
        p_ts = np.asarray(f["box_surface/p"][()], dtype=_F64)
    win = causal_end_taper(p_ts.shape[0])
    p_win = p_ts * win[:, np.newaxis]
    p_spec = np.asarray(spectrum_rows(p_win, t, f_hz), dtype=_C128)
    return centers, p_spec.ravel()


def p_avg_tail_db(p_avg: np.ndarray) -> float:
    """Peak level of ``p_avg`` in the last 5% of samples, relative to run peak (dB)."""
    p = np.asarray(p_avg, dtype=_F64).ravel()
    if p.size == 0:
        return float("nan")
    pk = float(np.max(np.abs(p)))
    if pk <= 0.0:
        return float("-inf")
    n_tail = max(1, int(math.floor(0.05 * p.size)))
    tail = float(np.max(np.abs(p[-n_tail:])))
    return 20.0 * math.log10(tail / pk)