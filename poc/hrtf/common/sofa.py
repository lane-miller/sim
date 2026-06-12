"""SOFA reference loading and HRTF interpolation."""

import numpy as np
import sofar as sf
from scipy.interpolate import CubicSpline

from config import EVAL_RADIUS_M, HRIR_MEASURED, HRIR_SIMULATED


def load_sofa_pair():
    """Load measured and simulated FABIAN SOFA files."""
    sofa_meas = sf.read_sofa(str(HRIR_MEASURED))
    sofa_sim = sf.read_sofa(str(HRIR_SIMULATED))
    sofa_fs = float(sofa_meas.Data_SamplingRate)
    return sofa_meas, sofa_sim, sofa_fs


def resolve_eval_radius(sofa_pos, eval_radius=EVAL_RADIUS_M):
    """Return eval radius, falling back to SOFA value if mismatched."""
    sofa_radius = sofa_pos[0, 2]
    if abs(sofa_radius - eval_radius) > 0.01:
        print(f"WARNING: SOFA radius {sofa_radius} != eval radius {eval_radius}")
        print(f"  Using SOFA radius: {sofa_radius} m")
        return sofa_radius
    return eval_radius


def horizontal_plane_from_sofa(sofa_pos, eval_radius=EVAL_RADIUS_M):
    """Select horizontal-plane directions and Cartesian eval points."""
    eval_radius = resolve_eval_radius(sofa_pos, eval_radius)

    horiz_mask = np.abs(sofa_pos[:, 1]) < 0.5
    horiz_idx = np.where(horiz_mask)[0]
    horiz_az = sofa_pos[horiz_idx, 0]

    sort_order = np.argsort(horiz_az)
    horiz_idx = horiz_idx[sort_order]
    horiz_az = horiz_az[sort_order]

    az_rad = np.deg2rad(horiz_az)
    el_rad = np.zeros_like(az_rad)
    eval_points = np.zeros((3, len(horiz_idx)))
    eval_points[0, :] = eval_radius * np.cos(el_rad) * np.cos(az_rad)
    eval_points[1, :] = eval_radius * np.cos(el_rad) * np.sin(az_rad)
    eval_points[2, :] = eval_radius * np.sin(el_rad)

    return horiz_idx, horiz_az, eval_points, eval_radius


def canonical_directions(sofa_pos, canonical_az, eval_radius=EVAL_RADIUS_M):
    """Map canonical azimuths to nearest SOFA horizontal directions."""
    eval_radius = resolve_eval_radius(sofa_pos, eval_radius)

    horiz_mask = np.abs(sofa_pos[:, 1]) < 0.5
    horiz_sofa_idx = np.where(horiz_mask)[0]
    horiz_az = sofa_pos[horiz_sofa_idx, 0]

    sofa_dir_idx = []
    for az_target in canonical_az:
        az_diff = np.abs((horiz_az - az_target + 180.0) % 360.0 - 180.0)
        best_local = np.argmin(az_diff)
        sofa_dir_idx.append(horiz_sofa_idx[best_local])

    sofa_dir_idx = np.array(sofa_dir_idx)

    eval_points = np.zeros((3, len(canonical_az)))
    for i, az_deg in enumerate(canonical_az):
        az_rad = np.deg2rad(az_deg)
        eval_points[0, i] = eval_radius * np.cos(az_rad)
        eval_points[1, i] = eval_radius * np.sin(az_rad)
        eval_points[2, i] = 0.0

    return sofa_dir_idx, eval_points, eval_radius


def interp_hrtf(hrtf_full, source_freqs, target_freqs):
    """Interpolate complex HRTF from source to target frequencies (linear)."""
    mag = np.abs(hrtf_full)
    phase = np.unwrap(np.angle(hrtf_full), axis=1)
    result = np.zeros((hrtf_full.shape[0], len(target_freqs)), dtype=complex)
    for i in range(hrtf_full.shape[0]):
        mag_interp = np.interp(target_freqs, source_freqs, mag[i, :])
        phase_interp = np.interp(target_freqs, source_freqs, phase[i, :])
        result[i, :] = mag_interp * np.exp(1j * phase_interp)
    return result


def interp_hrtf_spline(expansion_freqs, hrtf_sparse, target_freqs):
    """Interpolate magnitude (dB) and unwrapped phase via cubic spline."""
    n_expansion, n_directions = hrtf_sparse.shape
    n_target = len(target_freqs)
    hrtf_dense = np.zeros((n_target, n_directions), dtype=complex)

    for j in range(n_directions):
        mag_db = 20 * np.log10(np.abs(hrtf_sparse[:, j]) + 1e-30)
        phase = np.unwrap(np.angle(hrtf_sparse[:, j]))

        mag_spline = CubicSpline(expansion_freqs, mag_db)
        phase_spline = CubicSpline(expansion_freqs, phase)

        mag_interp = mag_spline(target_freqs)
        phase_interp = phase_spline(target_freqs)

        hrtf_dense[:, j] = 10 ** (mag_interp / 20) * np.exp(1j * phase_interp)

    return hrtf_dense


def interp_scale(scale_at_freqs, source_freqs, target_freqs):
    """Interpolate real scale factors onto a target frequency axis."""
    return np.interp(target_freqs, source_freqs, scale_at_freqs)


def normalize_sofa_to_bem(hrtf_meas, hrtf_sim, hrtf_bem):
    """Scale SOFA HRTFs to match BEM mean magnitude per frequency."""
    n_freq = hrtf_bem.shape[1]
    hrtf_meas = hrtf_meas.copy()
    hrtf_sim = hrtf_sim.copy()
    for i in range(n_freq):
        ref_bem = np.mean(np.abs(hrtf_bem[:, i]))
        ref_meas = np.mean(np.abs(hrtf_meas[:, i]))
        ref_sim = np.mean(np.abs(hrtf_sim[:, i]))
        if ref_meas > 0:
            hrtf_meas[:, i] *= ref_bem / ref_meas
        if ref_sim > 0:
            hrtf_sim[:, i] *= ref_bem / ref_sim
    return hrtf_meas, hrtf_sim


def to_db(hrtf):
    return 20 * np.log10(np.abs(hrtf) + 1e-30)
