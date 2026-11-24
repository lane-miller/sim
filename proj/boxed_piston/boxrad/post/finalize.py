"""Post-run HDF5 self-checks and stored ``/results/zrad`` (called from :class:`Recorder`)."""

from __future__ import annotations

from typing import Any

import numpy as np

from boxrad import __version__
from boxrad.config import Config, Derived, test_source_active
from boxrad.post.spectra import spectrum_rows
from boxrad.post.transfer import load_radiator_velocity_spectrum, p_avg_tail_db
from boxrad.post.zrad import causal_end_taper, log_frequency_grid, zrad_at_frequencies

_ACCURACY_NOTES = (
    "dispersion: ≈1% |Δz| at f_max for ppw 20, scales ≈ "
    "(f/f_max)²(20/ppw)² (C8); "
    "radiator-edge reactance excess ≈1.5–2.6% at 10–12 cells across the short "
    "side, decreasing with resolution (C9)"
)
_HUYGENS_POWER_NOTE = (
    "direct Huygens flux ½Re Σ p v_n* carries O((kh)²) discrete-product error: "
    "≈0.5 % at f_max at ~23 ppw (C10), ≈0.03 % at ~71 ppw (C11); pre-0.2.0 1.04 % "
    "included decimation aliasing (fixed by alias-aware decimation); "
    "far-field power (primary check) ≤ 0.43 %; "
    "Huygens/surface decimation alias level ≤ −alias_rejection_db dB re |S(f_max)| (C11)"
)


def cells_across_radiator(cfg: Config, derived: Derived) -> float:
    if cfg.geometry_spec is not None:
        if test_source_active(cfg) and not cfg.geometry_spec.radiators:
            return float("nan")
        from boxrad.geomderive import _radiator_D_min

        return _radiator_D_min(cfg.geometry_spec) / derived.h
    rad = cfg.radiator
    if rad.shape == "circle":
        d = rad.dims[0]
    else:
        d = min(rad.dims[0], rad.dims[1])
    return d / derived.h


_CONVENTIONS_PREFIX = "e^{+jωt}; z = <p>_rad/(ρc v); H = p/v [Pa/(m/s)]; "
_COORDINATES = (
    "coordinates: origin = radiator center on the front face; +z outward "
    "(radiating axis); box occupies z ∈ [−D, 0] (box mode) or baffle plane "
    "z = 0 (baffle mode); θ from +z, φ from +x; directivity phase referenced "
    "to the origin"
)


def conventions_text(cfg: Config) -> str:
    del cfg
    return _CONVENTIONS_PREFIX + _COORDINATES


def compute_finalize_results(
    source: Any,
    cfg: Config,
    derived: Derived,
    *,
    p_avg: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Read-only pass: ``tail_db`` and log-grid ``zrad`` from an open HDF5 file or path."""
    tail_db = float(p_avg_tail_db(p_avg))
    f_min = float(cfg.band.f_min)
    f_max = float(derived.f_max)
    f_log = log_frequency_grid(f_min, f_max, points_per_decade=100)
    _, z_log = zrad_at_frequencies(source, f_log)
    return tail_db, f_log, z_log


def write_finalize_meta(
    file: Any,
    cfg: Config,
    derived: Derived,
    *,
    tail_db: float,
    f_log: np.ndarray,
    z_log: np.ndarray,
) -> None:
    """Append-mode write of tail, accuracy, conventions, version, and ``/results/zrad``."""
    meta = file["meta"]
    meta.attrs["tail_db"] = float(tail_db)

    acc = meta.require_group("accuracy")
    acc.attrs["ppw_realized"] = float(derived.ppw_realized)
    acc.attrs["cfl_safety_resolved"] = float(derived.cfl_safety_resolved)
    acc.attrs["cfl_safety_source"] = str(derived.cfl_safety_source)
    acc.attrs["cells_across_radiator"] = float(cells_across_radiator(cfg, derived))
    acc.attrs["h"] = float(derived.h)
    acc.attrs["alias_level_db"] = float(derived.alias_level_db)
    acc.attrs["notes"] = f"{_ACCURACY_NOTES} {_HUYGENS_POWER_NOTE}"

    meta.attrs["conventions"] = conventions_text(cfg)
    meta.attrs["boxrad_version"] = __version__

    results = file.require_group("results")
    if "zrad" in results:
        del results["zrad"]
    dset = results.create_dataset(
        "zrad",
        data=z_log.astype(np.complex128),
        compression="gzip",
    )
    dset.attrs["f"] = f_log.astype(np.float64)
    dset.attrs["f_min_hz"] = float(cfg.band.f_min)
    dset.attrs["f_max_hz"] = float(derived.f_max)
    dset.attrs["points_per_decade"] = 100
    dset.attrs["convention"] = (
        "z(f) = ⟨p⟩_rad(f) / (ρ c V(f)); e^{+jωt}; dimensionless; "
        "Δt-weighted discrete spectra at stored radiator times"
    )


def finalize_wall_surface_transfer(
    p_ts: np.ndarray,
    times_p: np.ndarray,
    f_log: np.ndarray,
    *,
    path: Any,
    derived: Derived,
    test_source: bool,
) -> np.ndarray:
    """Frequency transfer ``(n_f, n_wet)``: Pa/(m/s) per unit ``v`` or Pa/(m³/s) for test source."""
    p_ts = np.asarray(p_ts, dtype=np.float64)
    times_p = np.asarray(times_p, dtype=np.float64).ravel()
    f_log = np.asarray(f_log, dtype=np.float64).ravel()
    if p_ts.size == 0 or times_p.size == 0:
        return np.zeros((f_log.size, p_ts.shape[1] if p_ts.ndim > 1 else 0), dtype=np.complex128)
    win = causal_end_taper(int(p_ts.shape[0]))
    p_win = p_ts * win[:, np.newaxis]
    p_f = np.asarray(spectrum_rows(p_win, times_p, f_log), dtype=np.complex128)
    if test_source:
        return p_f
    out = np.empty_like(p_f)
    for i, f_hz in enumerate(f_log):
        v_ref = load_radiator_velocity_spectrum(path, derived, float(f_hz))
        out[i] = p_f[i] / v_ref
    return out


def write_surface_transfer(
    file: Any,
    cfg: Config,
    derived: Derived,
    *,
    p_ts: np.ndarray,
    times_p: np.ndarray,
    f_log: np.ndarray | None = None,
) -> None:
    """Replace decimated time series in ``/surface`` with complex ``surface/p`` on ``f_log``."""
    if "surface" not in file:
        return
    surf = file["surface"]
    if f_log is None:
        f_log = log_frequency_grid(float(cfg.band.f_min), float(derived.f_max), points_per_decade=100)
    test_source = bool(test_source_active(cfg) and int(file["meta"].attrs.get("n_radiators", 0)) == 0)
    h_trans = finalize_wall_surface_transfer(
        p_ts,
        times_p,
        f_log,
        path=file,
        derived=derived,
        test_source=test_source,
    )
    for key in ("p_raw", "times_p"):
        if key in surf:
            del surf[key]
    if "p" in surf:
        del surf["p"]
    dset = surf.create_dataset(
        "p",
        data=h_trans.astype(np.complex128),
        compression="gzip",
    )
    dset.attrs["f"] = f_log.astype(np.float64)
    dset.attrs["convention"] = (
        "Pa/(m³/s) per unit monopole strength (test source)"
        if test_source
        else "H = p/v at fluid cell center adjacent to patch; e^{+jωt}"
    )


def finalize_recording(
    file: Any,
    cfg: Config,
    derived: Derived,
    *,
    p_avg: np.ndarray,
) -> float:
    """Legacy helper: compute and write finalize metadata on an open HDF5 file."""
    tail_db, f_log, z_log = compute_finalize_results(file, cfg, derived, p_avg=p_avg)
    write_finalize_meta(
        file, cfg, derived, tail_db=tail_db, f_log=f_log, z_log=z_log
    )
    return float(tail_db)