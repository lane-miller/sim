"""Mutual radiation impedance row z_ij(f) from ``radiator/p_avg_all`` (step 8a)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from boxrad.config import Config
from boxrad.post.zrad import _z_from_traces

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError("h5py is required for boxrad.post.zrow")
    return h5py

_F64 = np.float64
_C128 = np.complex128
_ROW_CONVENTION = (
    "z_ij = <p>_i / (rho c v_j), driven j = meta.driven_index; Z_ij = rho c z_ij / S_j"
)


def zrad_row_at_frequencies(
    p_avg_all: np.ndarray,
    times: np.ndarray,
    f_hz: np.ndarray,
    cfg: Config,
    derived_pulse: dict[str, Any],
    *,
    use_taper: bool = True,
    time_shift_s: float = 0.0,
) -> np.ndarray:
    """``(F, N)`` complex128; column *i* uses the same discrete spectra as :mod:`boxrad.post.zrad`."""
    p_avg_all = np.asarray(p_avg_all, dtype=_F64)
    if p_avg_all.ndim != 2:
        raise ValueError(f"p_avg_all must be 2-D (T, N); got shape {p_avg_all.shape!r}")
    n_t, n_rad = p_avg_all.shape
    times = np.asarray(times, dtype=_F64).reshape(-1)
    if times.shape[0] != n_t:
        raise ValueError(
            f"times length {times.shape[0]} != p_avg_all rows {n_t}"
        )
    f_hz = np.asarray(f_hz, dtype=_F64).ravel()
    rho = float(cfg.air.rho)
    c = float(cfg.air.c)
    sigma = float(derived_pulse["pulse_sigma"])
    t0 = float(derived_pulse["pulse_t0"])
    out = np.empty((f_hz.size, n_rad), dtype=_C128)
    for col in range(n_rad):
        _, z_col = _z_from_traces(
            rho=rho,
            c=c,
            p_avg=p_avg_all[:, col],
            times=times,
            sigma=sigma,
            t0=t0,
            f_hz=f_hz,
            use_taper=use_taper,
            time_shift_s=time_shift_s,
        )
        out[:, col] = z_col
    return out


def zrow_at_frequencies(
    path: str | Path,
    f_hz: np.ndarray,
    *,
    use_taper: bool = True,
    time_shift_s: float = 0.0,
) -> np.ndarray:
    """Dimensionless mutual row ``z_ij(f)`` shape ``(F, N)`` at explicit frequencies.

    Column *i* is the pickup on radiator *i* while the file's ``driven_index`` drives.
    Uses the same discrete spectral estimate as :mod:`boxrad.post.zrad`.
    """
    path = Path(path)
    h5 = _require_h5py()
    with h5.File(path, "r") as f:
        p_avg_all = np.asarray(f["radiator/p_avg_all"][()], dtype=_F64)
        times = np.asarray(f["radiator/times_p"][()], dtype=_F64)
        cfg = Config.from_json(bytes(f["meta/config_json"][()]).decode("utf-8"))
        derived = json.loads(bytes(f["meta/derived_json"][()]).decode("utf-8"))
    return zrad_row_at_frequencies(
        p_avg_all,
        times,
        f_hz,
        cfg,
        derived,
        use_taper=use_taper,
        time_shift_s=time_shift_s,
    )


def write_zrad_row_dataset(
    results_group: Any,
    f_log: np.ndarray,
    z_row: np.ndarray,
) -> None:
    """Create ``results/zrad_row`` with attrs (caller supplies open HDF5 group)."""
    if "zrad_row" in results_group:
        del results_group["zrad_row"]
    dset = results_group.create_dataset(
        "zrad_row",
        data=z_row.astype(_C128),
        compression="gzip",
    )
    dset.attrs["f"] = f_log.astype(_F64)
    dset.attrs["convention"] = _ROW_CONVENTION