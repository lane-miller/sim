# Using boxrad from acs-lpm

This guide explains how acs-lpm (the lumped-parameter / MNA model) should
consume boxrad results. It provides the **front-side radiation load** and the
**radiated field** of a rigid piston in a rigid box. For units, the API, and
validity limits, see the [user guide](user_guide.md).

## Workflow

1. **One boxrad run per enclosure geometry and band.** The geometry is the
   box $W\times H\times D$ plus the radiator shape, size, and position. Run
   it before any MNA solve:
   `boxrad-run enclosure.json` writes one `.h5` file.
2. **Reuse that `.h5` across every driver and circuit variant.** boxrad
   results are normalized per unit radiator velocity and do not depend on
   the driver, crossover, or amplifier. Change any of those without
   rerunning boxrad. Rerun only when the box, the radiator geometry, the air
   properties, or the band change.
3. **Dependency direction:** acs-lpm depends on boxrad, and boxrad never
   imports acs-lpm. Install boxrad into the acs-lpm environment, from the
   acs-lpm checkout:

   ```bash
   python -m pip install -e ../sim/proj/boxed_piston
   ```

4. In acs-lpm, open the file with `boxrad.Result`, evaluate on the MNA
   frequency grid, and cache the arrays (see the sketch below).

Set the boxrad radiator to the driver's effective radiating area. boxrad
uses the discretized area `Result.S_grid` ($h^2\sum w$) in every conversion.
It is printed by `boxrad-info` and plotted by `boxrad-plot --face` (title
`S_grid = …, S = …`). Use `S_grid`, not the nominal $S_d$, when converting
between $z$, $Z_A$, $Z_M$, and volume velocity, so the results stay
consistent.

## Radiation load (impedance)

At each MNA frequency, evaluate the radiation impedance and insert it on the
front side of the diaphragm.

- **Mechanical domain:** $Z_M(f) = \rho c\,S\,z(f)$ [N·s/m], via
  `Result.Z_mechanical(f)`. Put it in series with the driver's mechanical
  impedance, as the force needed per unit diaphragm velocity.
- **Acoustic domain:** $Z_A(f) = \rho c\,z(f)/S$ [Pa·s/m³], via
  `Result.Z_acoustic(f)`. Use it as the load seen by the volume velocity
  $U = S v$.

`Result` evaluates the spectra **exactly at the requested frequencies**,
directly from the stored time records. No interpolation is needed, as long
as the MNA grid lies inside `Result.band`. The stored `/results/zrad` table
(100 points/decade) is for plotting; pass your own grid to
`Z_mechanical(f)` instead of interpolating it.

**Front side only.** boxrad models the exterior radiation of the piston's
front face on a rigid, sealed box. The box interior (cavity compliance,
standing waves, damping), rear radiation, ports, passive radiators, and
wall vibration are **not** in the result. acs-lpm must model them.

## Radiated pressure, SPL, directivity, and power

All boxrad transfer functions are per unit radiator velocity, with time
convention $e^{+j\omega t}$ and peak-amplitude phasors. If acs-lpm uses
$e^{-j\omega t}$, conjugate the boxrad values. Take the diaphragm velocity
$v(f)$ (or volume velocity $U(f)$) from the MNA solution:

$$p(\mathbf r, f) = H(\mathbf r, f)\,v(f),\qquad H = \texttt{Result.field(points, f)}\ [\mathrm{Pa/(m/s)}]$$

$$p(\mathbf r, f) = H_U(\mathbf r, f)\,U(f),\qquad H_U = \frac{H}{S}\ [\mathrm{Pa/(m^3/s)}]$$

where $H_U$ is `Result.field(points, f, per="volume_velocity")`.

- **SPL:** $L_p = 20\log_{10}(|p|/(\sqrt2\cdot 20\,\mu\mathrm{Pa}))$ for a
  peak phasor $p$. Drop the $\sqrt2$ if acs-lpm's phasors are RMS.
- **Directivity / polar plots:** `Result.directivity(theta, phi, f)` returns
  the complex far-field pattern
  $D(\theta,\varphi,f) = \lim_{r\to\infty} r\,e^{+jkr}\,p/v$
  [Pa·m/(m/s)], with $\theta$ from $+z$ and $\varphi$ from $+x$, in
  radians. **The phase is referenced to the origin, the radiator center on
  the front face**: at large $r$ (measured from that point),
  $p \approx D\,v\,e^{-jkr}/r$. With `normalize=True` each row is divided by
  the complex on-axis value (on-axis $= 1 + 0j$). Use `np.abs(D)` for
  magnitudes; the far-field level at $r$ is $|D|\,|v|/r$.
- **Power:** $W(f) = \tfrac12\rho c\,S\,\mathrm{Re}\{z\}\,|v|^2$, which is
  `Result.power(f) * abs(v)**2`. For the DI, use
  `Result.directivity_index(f)`: full sphere in box mode, half space
  $z \ge 0$ in baffle mode.
- **Points:** `field()` is valid outside the Huygens surface, which lies 3
  cells outside the box, and is exact at any distance there, so the near
  field is included. The origin is the radiator center on the front face,
  and the box lies in $z \le 0$. `field()` uses a vectorized KH kernel; its
  cost still grows with points × frequencies × Huygens patches, so
  precompute $H$ at the observation points once per enclosure and frequency
  grid, then reuse it for all circuit variants.

## Out-of-band behavior

Every `Result` method that takes `f` raises `ValueError` for any frequency
outside `Result.band` $= [f_\mathrm{min}, f_\mathrm{max}]$. Passing
`allow_out_of_band=True` evaluates anyway and issues a `UserWarning`;
results there are unsupported:

- above $f_\mathrm{max}$ the grid is under-resolved. Above about
  $1.2 f_\mathrm{max}$ the drive spectrum is below −40 dB: `z` returns NaN
  there, while `field`, `directivity`, and `surface_p` return unmasked noise;
- below $f_\mathrm{min}$ the values are computed but not validated.

acs-lpm should not pass `allow_out_of_band=True` in production paths; let
the `ValueError` surface or catch it and handle the range explicitly.
Low-frequency handling is an acs-lpm decision. The options are:

- run boxrad with `band.f_min` at or below the lowest MNA frequency
  (cheap: $f_\mathrm{min}$ only sets the CPML $\alpha$ and the lower table
  edge);
- or use a fitted low-frequency asymptote of the form validated in C5,
  $\mathrm{Re}\,z = A k^2$ and $\mathrm{Im}\,z = k\,\delta$, with
  $A \to S/(4\pi)$ for a box in free space.

For frequencies above $f_\mathrm{max}$, rerun boxrad with a higher
`band.f_max`. A rigid piston is not physical above $ka \approx 10$.

## Sketch: thin wrappers (illustrative)

> **Illustrative only.** This is not part of boxrad or acs-lpm and has not
> been tested. Adapt the names and error handling to acs-lpm's element API.

```python
# ILLUSTRATIVE SKETCH — not shipped code.
from __future__ import annotations

import numpy as np
from boxrad import Result

# Result methods raise ValueError for f outside res.band (no wrapper check needed).


class RadiationLoad:
    """Front-side radiation load of one enclosure geometry (boxrad .h5)."""

    def __init__(self, h5_path: str) -> None:
        self.res = Result(h5_path)
        self.S = self.res.S_grid                      # m², use for all conversions
        self.accuracy = dict(self.res.accuracy)       # propagate to acs-lpm outputs

    def Z_mechanical(self, f) -> np.ndarray:          # N·s/m, shape (n_f,)
        return self.res.Z_mechanical(f)

    def Z_acoustic(self, f) -> np.ndarray:            # Pa·s/m³, shape (n_f,)
        return self.res.Z_acoustic(f)


class Radiation:
    """Pressure at fixed observation points; H is cached per frequency grid."""

    def __init__(self, h5_path: str, points) -> None:
        self.res = Result(h5_path)
        self.points = np.atleast_2d(np.asarray(points, dtype=float))   # (n_pts, 3) m
        self._H: dict[bytes, np.ndarray] = {}

    def H(self, f) -> np.ndarray:                     # Pa/(m/s), (n_f, n_pts)
        f = np.atleast_1d(np.asarray(f, dtype=float))
        key = f.tobytes()
        if key not in self._H:
            self._H[key] = self.res.field(self.points, f)
        return self._H[key]

    def pressure(self, f, v) -> np.ndarray:           # v: diaphragm velocity (n_f,), peak
        return self.H(f) * np.asarray(v)[:, None]

    def pressure_from_U(self, f, U) -> np.ndarray:    # U: volume velocity (n_f,)
        return self.H(f) / self.res.S_grid * np.asarray(U)[:, None]

    def spl(self, f, v) -> np.ndarray:                # dB SPL, peak phasors
        return 20 * np.log10(np.abs(self.pressure(f, v)) / (np.sqrt(2) * 20e-6))

    def power(self, f, v) -> np.ndarray:              # W
        return self.res.power(f) * np.abs(np.asarray(v)) ** 2

    def directivity(self, theta, phi, f) -> np.ndarray:   # complex, relative to on-axis
        return self.res.directivity(theta, phi, f, normalize=True)
```

## Accuracy notes to propagate

Every result file carries its own accuracy metadata. acs-lpm should record
these values with any fit or report built on a boxrad run, and warn on the
conditions listed:

| Source | Content | Suggested acs-lpm check |
|---|---|---|
| `Result.accuracy["notes"]` | Validated error summary: ≈1 % $\lvert\Delta z\rvert$ at $f_\mathrm{max}$ for ppw 20, scaling ≈ $(f/f_\mathrm{max})^2(20/\mathrm{ppw})^2$ (C8); radiator-edge reactance excess ≈1.5–2.6 % at 10–12 cells across the short side, decreasing with resolution (C9) | Attach verbatim |
| `Result.accuracy["ppw_realized"]` | Points per wavelength at $f_\mathrm{max}$ | Warn if < 20 |
| `Result.accuracy["cells_across_radiator"]` | Radiator resolution | Warn if < 10; raise it if Im z fits are sensitive |
| `Result.accuracy["h"]` | Grid spacing [m] | — |
| `Result.meta["tail_db"]` | Record decay [dB re peak] | Warn if > −60 dB |
| `Result.band` | Supported band | Enforced by `Result` (`ValueError` outside) |
| `Result.meta["boxrad_version"]`, `["git_commit"]` | Provenance | Record |

For C9-type geometries, the default-resolution $|z_\mathrm{FDTD} - z_\mathrm{BEM}|$
grows from 0.0014 at 100 Hz to 0.025 at 3 kHz. Most of it is the
radiator-edge excess in Im z. At $h$ = 5 mm, FDTD Im z is 2.5–2.7 % above
BEM from 100 Hz to 1.6 kHz, rising to 3.2 % at 2.2 kHz and 5.8 % at 3 kHz,
where dispersion adds. Re z agrees within 1.3 % across the band. The far-field
directivity agrees with BEM within 0.26 dB (see [validation.md](validation.md)).
