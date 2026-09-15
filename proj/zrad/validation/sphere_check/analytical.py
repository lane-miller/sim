"""Closed-form normalized radiation impedance for a pulsating sphere."""

import numpy as np


def zrad_norm_analytical(ka):
    """Normalized radiation impedance Z_rad / (rho*c) for ka array.

    Uniform radial surface velocity on a sphere of radius a, with ka = k * a:
        Z_rad_norm = (ka)^2 / (1 + (ka)^2) + j * ka / (1 + (ka)^2)
    """
    ka = np.asarray(ka, dtype=float)
    denom = 1.0 + ka**2
    return (ka**2 / denom) + 1j * (ka / denom)
