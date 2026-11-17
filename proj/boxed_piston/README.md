# boxrad: boxed-piston radiation by FDTD

boxrad is a config-driven 3-D acoustic FDTD solver: a staggered (Yee) grid,
second order in space and time, with float32 fields and a CPML absorber. It
models a rigid piston flush in one face of a rigid rectangular box, or in an
infinite rigid baffle. The piston moves with uniform normal velocity, and
$v_n = 0$ everywhere else. One impulse run covers the whole band and writes an
HDF5 file. The `Result` API reads that file and returns:

- the radiation impedance $z_\mathrm{rad}(f)$, plus acoustic and mechanical $Z$;
- the complex far-field directivity (phase referenced to the radiator
  center), plus complex pressure at any point outside the Huygens surface via
  Kirchhoff–Helmholtz;
- near-field probe pressure and pressure on the box surface.

All transfer functions are per unit radiator velocity, with time convention
$e^{+j\omega t}$.

## Install

From `sim/`, with Python ≥ 3.11:

```bash
python -m pip install -e proj/boxed_piston            # numpy, scipy, numba, h5py, matplotlib
python -m pip install -e "proj/boxed_piston[bem]"     # + bempp-cl, only for validation C9
```

This installs three console commands: `boxrad-run`, `boxrad-info`, and
`boxrad-plot`.

## Quickstart

From `proj/boxed_piston/`, using `configs/default.json`: a 0.20×0.15×0.25 m
box with a 0.08×0.06 m rectangular radiator off center, 50–3000 Hz. This is
the C9 validation geometry with $h$ taken from the default rules
($h = 343/3000/20 \approx 5.72$ mm) instead of C9's fixed 5 mm. The C9 config
at 5 mm takes about 6–7 s on an M1 with 8 GB and writes about 16 MB; the
default config has fewer cells and steps, so it is no slower.

```bash
boxrad-run configs/default.json --dry-run   # derived grid/time/RAM/disk summary + geometry window
boxrad-run configs/default.json             # writes outputs/default.h5
boxrad-info outputs/default.h5              # summary, provenance, conventions, dataset list
```

The default config produces no pre-flight warnings.

```python
import numpy as np
from boxrad import Result

with Result("outputs/default.h5") as res:
    print(res.summary())
    f = np.geomspace(*res.band, 40)                 # Hz; out-of-band f raises ValueError
    _, z = res.z(f)                                 # dimensionless, complex
    Zm = res.Z_mechanical(f)                        # N·s/m, front-side load
    H = res.field([[0.0, 0.0, 1.0]], f)[:, 0]       # Pa/(m/s) at 1 m on axis
    spl = 20 * np.log10(np.abs(H) / np.sqrt(2) / 20e-6)   # dB SPL at 1 m/s peak
    D = res.directivity(np.radians([0, 30, 60, 90]), 0.0, 1000.0, normalize=True)
    pattern_db = 20 * np.log10(np.abs(D))           # complex D; on-axis entry is 1+0j
```

For a fuller script (plots of $z$, SPL, and polar patterns), see
[`examples/quickstart.py`](examples/quickstart.py).

## Documentation

- [User guide](docs/user_guide.md): concepts, coordinates, the config
  reference, choosing numerics, CLI, HDF5 layout, the `Result` API,
  conventions, warnings, and limits.
- [acs-lpm integration](docs/acs_lpm.md): using a boxrad run as the radiation
  load and radiation model in lumped-parameter circuits.
- [Validation](docs/validation.md): checks C1–C10, with their references,
  gates, and results, and how to rerun them.
- [Developer brief](docs/brief.md): the design spec and method details.

## Layout

| Path | Contents |
|---|---|
| `boxrad/` | Installable package (solver, recorder, post-processing, `Result`, CLIs) |
| `configs/` | `default.json` (quickstart example), validation configs (`c05_*` … `c09_*`), and demos (`configs/demo/`) |
| `validation/` | Check scripts C1–C10 (not part of the wheel); results in `validation/results/` |
| `tests/` | pytest unit tests (`-m "not slow"` for the fast set) |
| `examples/` | `quickstart.py` |
| `outputs/` | Default HDF5 destination (created on first run) |
