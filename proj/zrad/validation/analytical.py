"""Closed-form normalized radiation impedance for a circular piston in an infinite baffle."""

import numpy as np
from scipy.special import jv, struve


def zrad_norm_analytical(ka):
    """Normalized radiation impedance Z_rad / (rho*c) for ka array.

    Blackstock / Kinsler-Frey (infinite rigid baffle, circular piston):
        Z_rad_norm = [1 - J1(2ka)/(ka)] + j * [H1(2ka)/(ka)]

    where J1 is the Bessel function of the first kind (order 1) and H1 is the
    Struve function of order 1 (scipy.special.struve, NOT modified Bessel K1).
    """
    ka = np.asarray(ka, dtype=float)
    x = 2.0 * ka
    # Avoid division by zero at ka=0 (limit is 0 + j*0 for the leading term).
    with np.errstate(divide="ignore", invalid="ignore"):
        real = 1.0 - jv(1, x) / ka
        imag = struve(1, x) / ka
    real = np.where(ka == 0, 0.0, real)
    imag = np.where(ka == 0, 0.0, imag)
    return real + 1j * imag


# Silva et al. (2009) Padé(2,6) coefficients — unflanged circular piston (beta=0.5, eta=0.6133).
_SILVA_UNFLANGED = dict(
    beta=0.5,
    eta=0.6133,
    a1=0.800,
    a2=0.266,
    a3=0.0263,
    b1=0.0599,
    b2=0.238,
    b3=-0.0153,
    b4=0.00150,
)


def zrad_norm_unflanged(ka):
    """Normalized radiation impedance Z_rad / (rho*c) for an unflanged circular piston.

    Silva, Guillemain, Kergomard, Mallaroni & Norris (2009), J. Sound Vib. 322(1-2),
    Section 5, Eqs. 21-23 (unflanged coefficients). Valid and accurate for ka < 3.

    Silva's Eqs. 21-23 use exp(-j*omega*t) (Table 1 caption); this codebase uses
    exp(+j*omega*t) like zrad_norm_analytical(). The result is complex-conjugated
    before return so both functions share the same time convention.
    """
    ka = np.asarray(ka, dtype=float)
    ka2 = ka * ka
    ka4 = ka2 * ka2
    ka6 = ka4 * ka2
    c = _SILVA_UNFLANGED

    abs_r = (1.0 + c["a1"] * ka2) / (
        1.0 + (c["beta"] + c["a1"]) * ka2 + c["a2"] * ka4 + c["a3"] * ka6
    )
    end_correction = c["eta"] * (1.0 + c["b1"] * ka2) / (
        1.0 + c["b2"] * ka2 + c["b3"] * ka4 + c["b4"] * ka6
    )

    # Eq. 2: R(ka) = -|R(ka)| * exp(2j * ka * (L/a));  kL = ka * (L/a).
    r_ka = -abs_r * np.exp(2j * ka * end_correction)

    # Eq. 3: Z_rad_norm = (1 + R) / (1 - R).
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (1.0 + r_ka) / (1.0 - r_ka)

    # At ka=0: |R|=1, L/a=eta, phase=0 => R=-1 => Z=0.
    z = np.where(ka == 0, 0.0 + 0.0j, z)
    return np.conj(z)
