# boxrad user guide

This guide covers boxrad 0.1.0 and is written for engineers who will set up
runs and use the results. For design rationale and method detail, see the
[developer brief](brief.md). For validation evidence, see
[validation.md](validation.md). For coupling to lumped-parameter models, see
[acs_lpm.md](acs_lpm.md).

Contents: [Concepts](#a-concepts) · [Coordinates](#b-coordinate-system) ·
[Config reference](#c-config-reference) · [Choosing numerics](#d-choosing-numerics) ·
[Running](#e-running) · [Outputs (HDF5)](#f-outputs-hdf5-layout) ·
[Result API](#g-result-api-reference) · [Conventions](#h-conventions) ·
[Warnings and troubleshooting](#i-warnings-and-troubleshooting) · [Limits](#j-limits)

---

## a. Concepts

**Rigid piston.** The radiator is a flat region of one box face whose whole
surface moves with the same normal velocity $v(t)$. That region is a circle, a
rectangle, or a racetrack. Every other surface is rigid ($v_n = 0$). Air is
linear and lossless. The radiator is not a cone: there is no breakup, no
surround, and no velocity profile.

**Box vs. baffle mode** (`geometry`):

- `"box"`: the radiator is flush in the **+z face** of a rigid rectangular box
  of size $W \times H \times D$. The box sits in free space, surrounded on all
  six sides by an air gap and an absorbing layer (CPML).
- `"baffle"`: the radiator is flush in an **infinite rigid plane** at $z = 0$,
  and only the half space $z > 0$ is simulated. This is the classic baffled
  piston. Analytic references exist for it, so it is used for validation.

**Huygens surface.** The solver does not store the volume field. It records
pressure $p$ and outward normal velocity $v_n$ on a closed box-shaped surface 3
cells outside the rigid box, inside the air gap. In baffle mode this surface
is an open five-sided box over the plane. The **Kirchhoff–Helmholtz (KH)
integral** over that surface gives the exact exterior field (up to
discretization) at any point outside it, including the far field. The grid
therefore never has to reach the far field.

**Why one impulse run gives all frequencies.** The system is linear and time
invariant. The radiator is driven with one Gaussian velocity pulse,
$v(t) = e^{-(t-t_0)^2/(2\sigma^2)}$, whose spectrum is smooth from DC to about
$1.2\,f_\mathrm{max}$, where it is $-40$ dB down. Every output is a transfer
function: the output spectrum divided by the input spectrum $V(f)$, evaluated
exactly at any requested frequency from the stored time records. A box in free
space does not ring, so a record a few box transits long resolves the low
frequencies too.

## b. Coordinate system

![Box geometry for configs/c09_box.json](img/geometry_c09.png)

![Front-face mesh and radiator weights for the demo racetrack](img/face_racetrack.png)

To regenerate these images (from `proj/boxed_piston/`):

```bash
boxrad-plot configs/c09_box.json --no-show -o docs/img/geometry_c09.png
boxrad-plot configs/demo/demo_racetrack_box.json --face --no-show -o docs/img/face_racetrack.png
```

- **Origin:** the radiator center, in the plane of the front (radiating) face.
- **+z:** the outward normal of the radiating face. In box mode the box extends
  toward $-z$ and occupies $z \in [-D, 0]$. In baffle mode the baffle is the
  plane $z = 0$ and the field exists for $z > 0$.
- **Angles:** $\theta$ is the polar angle from $+z$, and $\varphi$ is the
  azimuth from $+x$ toward $+y$. The unit direction is
  $\hat{d} = (\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta)$.
  On axis is $\theta = 0$. In box mode, $\theta > 90^\circ$ is behind the
  front-face plane.
- **Radiator offset:** `(x_c, y_c)` is the radiator center measured from the
  **front-face center**. Because the origin is at the radiator, the face
  center is at $(-x_c, -y_c, 0)$. The box occupies
  $x \in [-x_c - W/2,\ -x_c + W/2]$,
  $y \in [-y_c - H/2,\ -y_c + H/2]$, $z \in [-D, 0]$.
- **Snapping to h:** each box side is rounded to a whole number of cells:
  $W_\mathrm{real} = \mathrm{round}(W/h)\,h$, and likewise for $H$ and $D$.
  The dry run and the plot title show the realized size, with
  `(requested …, snapped to h)` when it differs. The radiator is **not**
  snapped. Its outline stays exact, and each face cell is weighted by the
  fraction of its area inside the outline, so the radiator can sit at any
  sub-cell position. Rectangles use exact overlap; circles and racetracks use
  32×32 supersampling per cell. The discretized area is
  $S_\mathrm{grid} = h^2 \sum w$, and every $z \leftrightarrow Z$ conversion
  uses it.
- In baffle mode, keep `x_c = y_c = 0` or use only sub-cell offsets, as C7
  does. The radiator mask is the radiator's bounding box, rounded to whole
  cells and shifted by $-(x_c, y_c)$, so an offset clips the outline by up to
  $\max(|x_c| + w/2 - W_\mathrm{bbox}/2,\ |y_c| + h_r/2 - H_\mathrm{bbox}/2)$,
  where $w \times h_r$ is the radiator's bounding box and
  $W_\mathrm{bbox} \times H_\mathrm{bbox}$ its rounded size. `validate()`
  warns when this clip is positive and up to one cell, and it is an error
  above one cell (see [§i](#i-warnings-and-troubleshooting)).

## c. Config reference

A config is one JSON object. Sections map to the dataclasses in
`boxrad/config.py`. Omitted sections or fields take the defaults below. **An
unknown key inside a section raises `TypeError`**; unknown top-level keys
(such as the `"_doc"` notes in `configs/default.json`) are ignored.
`radiator` is required in JSON. All units are SI.

### Top level

| Field | Type | Default | Meaning | Valid values |
|---|---|---|---|---|
| `geometry` | string | `"box"` | Box or infinite-baffle mode | `"box"`, `"baffle"` |

### `air`

| Field | Type | Unit | Default | Meaning | Valid range |
|---|---|---|---|---|---|
| `c` | float | m/s | 343.0 | Speed of sound | > 0 |
| `rho` | float | kg/m³ | 1.204 | Air density | > 0 |

### `box` (required for `geometry: "box"`; omit or set `null` for baffle)

| Field | Type | Unit | Default | Meaning | Valid range |
|---|---|---|---|---|---|
| `W` | float | m | required | Front-face width (x) | Intended 10–600 mm (warning outside) |
| `H` | float | m | required | Front-face height (y) | Intended 10–600 mm (warning outside) |
| `D` | float | m | required | Depth behind the front face (−z) | Intended 10–600 mm (warning outside); max/min side ≤ 4 (warning otherwise) |

In baffle mode a non-null `box` is ignored, with a warning.

### `radiator` (required)

| Field | Type | Unit | Default | Meaning | Valid range |
|---|---|---|---|---|---|
| `shape` | string | — | required | Outline shape | `"circle"`, `"rect"`, `"racetrack"` |
| `dims` | list of float | m | required | Size (see below) | Non-empty; positive |
| `x_c` | float | m | 0.0 | Radiator center minus front-face center, along x | Box mode: the outline must clear every face edge by ≥ `min_edge_cells`·h (error). Baffle mode: keep 0; an offset that clips the radiator mask warns (≤ 1 cell) or errors (> 1 cell) |
| `y_c` | float | m | 0.0 | The same, along y | Same as `x_c` |

Shapes and `dims` semantics:

| Shape | `dims` | Outline | Characteristic size $D_r$ | Exact area $S$ |
|---|---|---|---|---|
| `circle` | `[d]` | Circle of diameter `d` | $d$ | $\pi d^2/4$ |
| `rect` | `[w, h]` | Axis-aligned $w \times h$ rectangle | $\min(w, h)$ | $w h$ |
| `racetrack` | `[w, h]` | $w \times h$ bounding box; semicircular ends of diameter $d=\min(w,h)$ on the longer axis | $\min(w, h)$ | $h(w-d) + \pi d^2/4$ for $w \ge h$ ($w$ and $h$ swap otherwise) |

$D_r$ sets the default $f_\mathrm{max}$ ($ka = 10$ with $a = D_r/2$) and the
radiator-resolution limit on $h$.

### `band`

| Field | Type | Unit | Default | Meaning | Valid range |
|---|---|---|---|---|---|
| `f_min` | float | Hz | 20.0 | Lower band edge. Sets the CPML $\alpha = $ `cpml_alpha_factor`$\cdot\pi f_\mathrm{min}$ and the start of `/results/zrad` | $0 <$ `f_min` $<$ `f_max` (otherwise the run fails at the end, while writing `/results/zrad`, with `invalid band`) |
| `f_max` | float or `null` | Hz | `null` → $\min(20\,\mathrm{kHz},\ 10c/(\pi D_r))$ | Upper band edge. Sets $h$ through `ppw`, the pulse width, and decimation | $ka = \pi f_\mathrm{max} D_r / c > 10$ gives a warning |

### `numerics`

| Field | Type | Unit | Default | Meaning | Valid range / checks |
|---|---|---|---|---|---|
| `ppw` | float | — | 20.0 | Points per wavelength at `f_max`: $h \le \lambda_\mathrm{min}/$`ppw` | Realized ppw < 20 gives a warning |
| `cells_across_radiator` | int | — | 10 | Minimum cells across $D_r$: $h \le D_r/$`cells_across_radiator` | — |
| `h_override` | float or `null` | m | `null` | Fixed grid spacing. Bypasses the `ppw` and radiator rules (used by validation sweeps) | Warning whenever set |
| `cfl_safety` | float | — | 0.99 | $\Delta t = $ `cfl_safety`$\cdot h/(c\sqrt3)$ | ≤ 1 (3-D CFL limit) |
| `gap_cells` | int or `null` | cells | `null` → 10 | Air cells between the box (or radiator bounding box) and the CPML | ≥ 5, because the Huygens surface at 3 cells needs 2 more (error otherwise); < 10 gives a warning |
| `cpml_cells` | int | cells | 16 | CPML thickness | < 16 gives a warning |
| `cpml_alpha_factor` | float | — | 1.0 | CPML frequency shift $\alpha = $ factor$\cdot\pi f_\mathrm{min}$ (1/s) | Keep small; brief: ≥ 100 degrades low-frequency Re z |
| `cpml_m` | float | — | 3.0 | Polynomial grading order of $\sigma$ and $\kappa$ | — |
| `cpml_R` | float | — | 1e-6 | Design normal-incidence reflection; $\sigma_\mathrm{max} = -(m+1)c\ln R/(2L)$ | 0 < R < 1 |
| `cpml_kappa_max` | float | — | 1.0 | CPML $\kappa_\mathrm{max}$ | ≥ 1 |
| `cell_budget` | int | cells | 60,000,000 | Maximum total grid cells (including gap and CPML) at the chosen $h$ | Exceeding it is an error, reported by the dry run too (see [§i](#i-warnings-and-troubleshooting)) |
| `pulse_db` | float | dB | −40.0 | Pulse spectrum level at `pulse_f_factor`·`f_max` | < 0 |
| `pulse_f_factor` | float | — | 1.2 | See `pulse_db` | > 1 |
| `n_transits` | float | — | 3.0 | Run length in box transits (see [§d](#d-choosing-numerics)) | Raise it if the tail warning appears; the shipped validation and demo configs use 14 |
| `min_edge_cells` | int | cells | 3 | Minimum clearance between the radiator outline and any face edge (box mode) | Smaller clearance is an error |

The Huygens surface offset is fixed at 3 cells in 0.1.0 and has no config
key. Its time samples are decimated by
$M = \lfloor 1/(2.5\,f_\mathrm{max}\,\Delta t)\rfloor$. Within the band that
is lossless, because the pulse limits the content.

### `output`

| Field | Type | Unit | Default | Meaning | Valid range |
|---|---|---|---|---|---|
| `path` | string | — | `""` | HDF5 path. `""` → `outputs/<config stem>.h5` under `proj/boxed_piston/`; CLI `-o` overrides | — |
| `probes` | list of `[x, y, z]` or `null` | m | `null` | FDTD pressure probes, snapped to the nearest pressure-cell center and recorded at the decimated rate. `Result.field` uses them for points inside or near the Huygens surface | Inside the grid (indices are clamped) |
| `probes_baffle_z_cells` | list of int or `null` | cells | `null` | Baffle mode only: probes on the radiator axis at $z = (k + \tfrac12)h$ | Baffle mode |

### Example configs

JSON has no comments. The `//` notes below are annotations; strip them before
use. The real files are under `configs/`.

**0. `configs/default.json`**: the quickstart example. It is the C9 geometry
below without `h_override`, plus `"ppw": 20.0` and a top-level `"_doc"` list
of notes (ignored by the loader). $h = \min(343/3000/20,\ 0.06/10) \approx
5.72$ mm, the box snaps to 35×26×44 cells (0.2001×0.1486×0.2515 m), the
radiator spans about 10.5 cells across its short side with edge clearance
about 5.2 cells (x) and 4.4 cells (y), and `validate()` reports no warnings.

**1. `configs/c09_box.json`**: box, offset rectangle (the C9/C10 validation
case).

```jsonc
{
  "geometry": "box",
  "air": {"c": 343.0, "rho": 1.204},
  "box": {"W": 0.2, "H": 0.15, "D": 0.25},          // 40×30×50 cells at h = 5 mm
  "radiator": {
    "shape": "rect", "dims": [0.08, 0.06],          // D_r = 0.06 m → 12 cells across
    "x_c": 0.03, "y_c": 0.02                        // edge clearance 6 cells (x), 5 cells (y)
  },
  "band": {"f_min": 50.0, "f_max": 3000.0},         // ppw_realized = 343/3000/0.005 ≈ 22.9
  "numerics": {
    "h_override": 0.005,                            // fixed h for validation → warning
    "cells_across_radiator": 10, "cfl_safety": 0.99,
    "gap_cells": null,                              // → 10
    "cpml_cells": 16, "cpml_alpha_factor": 1.0, "cpml_m": 3.0,
    "cpml_R": 1e-6, "cpml_kappa_max": 1.0,
    "cell_budget": 60000000, "pulse_db": -40.0, "pulse_f_factor": 1.2,
    "n_transits": 14.0,                             // long record; tail well below −60 dB
    "min_edge_cells": 3
  },
  "output": {"path": ""}                            // → outputs/c09_box.h5
}
```

**2. `configs/c06_baffle.json`**: infinite baffle, centered circle (the C6
validation case).

```jsonc
{
  "geometry": "baffle",
  "box": null,                                      // no box in baffle mode
  "radiator": {"shape": "circle", "dims": [0.1], "x_c": 0.0, "y_c": 0.0},
  "band": {"f_min": 100.0, "f_max": 6000.0},        // ka_max = π·6000·0.1/343 ≈ 5.5
  "numerics": {
    "ppw": 20.0,                                    // h = min(343/6000/20, 0.1/10) ≈ 2.86 mm (~35 cells across)
    "cells_across_radiator": 10,
    "n_transits": 14.0
    // other numerics fields: defaults (as in the file)
  },
  "output": {"path": "", "probes_baffle_z_cells": [4, 6, 8]}   // on-axis probes at z = 4.5h, 6.5h, 8.5h
}
```

**3. `configs/demo/demo_racetrack_box.json`**: box, offset racetrack, $h$
derived from the rules.

```jsonc
{
  "geometry": "box",
  "box": {"W": 0.25, "H": 0.18, "D": 0.2},          // snapped: 42×30×33 cells → 0.252×0.18×0.198 m
  "radiator": {
    "shape": "racetrack", "dims": [0.12, 0.06],     // 0.12 m long (x), semicircular ends of ⌀0.06 m
    "x_c": 0.02, "y_c": -0.03                       // clearance to the −y edge: 0.03 m = 5 cells
  },
  "band": {"f_min": 50.0, "f_max": 2000.0},
  "numerics": {
    "ppw": 20.0,                                    // λ_min/20 = 8.6 mm
    "cells_across_radiator": 10,                    // D_r/10 = 6 mm → h = 6 mm (radiator-limited)
    "n_transits": 14.0
    // other numerics fields: defaults (as in the file)
  },
  "output": {"path": ""}
}
```

Other shipped configs: `configs/demo/demo_circle_box.json` (offset circle on
a 0.3×0.4×0.25 m box) and `configs/demo/demo_coarse_circle.json` (the same
system at 6 cells across, using `h_override`).

## d. Choosing numerics

**Grid spacing.**

$$h = \min\!\left(\frac{\lambda_\mathrm{min}}{\mathrm{ppw}},\ \frac{D_r}{\texttt{cells\_across\_radiator}}\right),\qquad \lambda_\mathrm{min} = c/f_\mathrm{max}$$

`h_override` fixes $h$ directly instead. If the resulting domain exceeds
`cell_budget`, validation fails (there is no automatic coarsening): raise $h$
(lower `cells_across_radiator` or `ppw`, or set `h_override`), lower
`f_max`, or raise `cell_budget` if RAM allows. The time step is
$\Delta t = 0.99\,h/(c\sqrt3)$.

- **ppw (default 20) controls dispersion.** For a baffled circle in C8 at
  ppw 20, $\max|z - z_\mathrm{ref}| = 0.0102$ at $f_\mathrm{max}$ (gate
  0.015). The observed order is 1.96, so the error scales as roughly
  $(f/f_\mathrm{max})^2 (20/\mathrm{ppw})^2$. The C8 box self-convergence
  (Richardson $p = 2.26$) puts far-field directivity within 0.08 dB of the
  extrapolated result at $kL = 2, 5, 10$. Phase lag accumulates with path
  length: about 1°/λ on axis at ppw 20 (brief, C2).
- **cells across the radiator (default 10) controls the near-field and
  edge error.** For a circle in C7 ($ka \le 1.5$),
  $\max|z - z_\mathrm{ref}| = 0.0126$, $0.0065$, and $0.0029$ at 6, 10, and
  16 cells. With 12 cells across the short side of a rectangle in C9, the
  default-resolution $|z_\mathrm{FDTD} - z_\mathrm{BEM}|$ grows to 0.0251 at
  3 kHz. Most of that is excess radiator-edge reactance, which shrinks with
  refinement: the $h$-refined Richardson value agrees with BEM to 0.0038. If
  downstream fits are sensitive to Im z, raise `cells_across_radiator`.

**Band and run length.** `f_max` sets $h$ and therefore the cost. When ppw
sets $h$, cells scale as $f_\mathrm{max}^3$ and steps as $f_\mathrm{max}$,
so the cost scales roughly as $f_\mathrm{max}^4$. `f_min` sets the CPML
$\alpha$ and the lower end of the stored z table. The run length is

$$T = t_0 + 6\sigma + n_\mathrm{transits}\,\frac{L_\mathrm{max} + 2\,g}{c},\qquad \sigma = \frac{\sqrt{2\ln 10^{-\mathrm{pulse\_db}/20}}}{2\pi\,\mathrm{pulse\_f\_factor}\,f_\mathrm{max}},\quad t_0 = 6\sigma$$

Here $L_\mathrm{max}$ is the largest realized box side and $g$ is the gap in
meters. In baffle mode, $L_\mathrm{max} = \max(W_\mathrm{bbox}, H_\mathrm{bbox}, (\mathrm{gap}+\mathrm{cpml})h)$.
The number of steps is $\lceil T/\Delta t\rceil$. After a run, `tail_db` is
the peak $|\langle p\rangle|$ in the last 5 % of the record, relative to the
run peak. If it exceeds −60 dB a warning is printed; increase `n_transits`.

**Dry run.** `boxrad-run CONFIG --dry-run` prints:

| Line | Meaning |
|---|---|
| `grid_shape` | Total grid `nx×ny×nz`, including gap and CPML |
| `h` | Grid spacing [m] |
| `dt` | Time step [s] |
| `n_steps` | Number of time steps |
| `f_min_hz`, `f_max_hz` | Band [Hz] (`f_max` after defaults) |
| `ppw_realized` | $\lambda_\mathrm{min}/h$ at `f_max` |
| `cells_across_radiator` | $D_r/h$ |
| `memory_bytes` | Field + CPML array estimate, float32 [GB]; > 5 GiB is an error |
| `decimation_m` | Huygens/surface/probe decimation factor $M$ |
| `output_bytes_est` | Uncompressed HDF5 payload estimate [MB] |
| `wall_time_s_est` | $n_\mathrm{steps}\cdot$cells / (375 M cell-updates/s), the C1 envelope without recording |
| `zrad_source` | Always `fdtd` |
| `geometry` | `box` or `baffle` |
| `box_dims` | Realized box size (box mode), noting any snapping |

Warnings are printed before the summary (see [§i](#i-warnings-and-troubleshooting)).
Config errors (for example, a domain over `cell_budget`) are reported as
`error: …` instead of the summary, with exit code 1.

**Memory, disk, and time.** The C9 case (`configs/c09_box.json`) runs in
about 6–7 s and produces about 16 MB on an M1 with 8 GB. Larger systems
follow the brief §4 envelope, measured with C1 at about 375 M
cell-updates/s:

| Box | f_max at 20 ppw | Cells | Est. RAM | Projected run time |
|---|---|---|---|---|
| 200 mm | 20 kHz | 1.9×10⁷ | 0.37 GB | ~70 s |
| 300 mm | 17 kHz | 3.7×10⁷ | 0.68 GB | ~3 min |
| 300 mm | 20 kHz | 5.8×10⁷ | 1.04 GB | ~5 min |
| ~500 mm | 20 kHz | ~2.6×10⁸ | ~5 GB (RAM limit) | ~25 min (extrapolated) |

Disk use is dominated by the Huygens and box-surface records: about
$n_\mathrm{steps}/M \times (8 N_\mathrm{huygens} + 4 N_\mathrm{box})$ bytes,
before gzip. Run the dry run first.

## e. Running

### `boxrad-run`

```
boxrad-run CONFIG [-o OUT] [--dry-run] [--no-show] [--plot] [--quiet]
```

| Flag | Effect |
|---|---|
| `CONFIG` | JSON config path |
| `-o, --output OUT` | HDF5 output path (overrides `output.path`) |
| `--dry-run` | Validate, print warnings and the summary, then open the geometry window. Nothing is run or written |
| `--no-show` | With `--dry-run`: do not open the geometry window |
| `--plot` | After a full run, open the geometry window from the result file |
| `--quiet` | Suppress per-step progress lines |

Exit code 0 on success. On a config error or a stability failure it prints
`error: …` and exits with 1. Progress lines show
`step n/N  t=…  elapsed=…  ETA=…  steps/s` about every 2 %. On success it
prints `wrote <path>  (<size> MB)`. The command does not ask for
confirmation.

**caffeinate:** on macOS a full `boxrad-run` (not a dry run) runs
`caffeinate -i -w <pid>` to prevent idle sleep. It prints
`[keep_awake] caffeinate started/stopped (pid …)` and stops `caffeinate` on
normal exit, Ctrl-C, or SIGTERM. On other platforms this is a no-op. The
Python `run()` does not do this. To get the same effect, wrap the call with
`boxrad.runner.keep_awake("label")`.

### `boxrad-info`

```
boxrad-info H5
```

Prints `Result.summary()`; a provenance block with `boxrad_version`,
`zrad_source`, and the `/results/zrad` frequency range and point count; the
`/meta` `conventions` string; the `/meta/accuracy` attributes; every dataset
with its shape and dtype; and the file size.

### `boxrad-plot`

```
boxrad-plot CONFIG_OR_H5 [-o PNG] [--no-show] [--huygens] [--face]
```

| Flag | Effect |
|---|---|
| `CONFIG_OR_H5` | JSON config or result `.h5`/`.hdf5` |
| `-o, --output PNG` | Also save the figure (150 dpi) |
| `--no-show` | Do not open a window (use with `-o`) |
| `--huygens` | Overlay the Huygens patch centers in 3-D |
| `--face` | Only the front-face mesh panel |

### Python

```python
from boxrad import run, Result

summary = run("configs/default.json", dry_run=True)   # dict (see the dry-run table)
path = run("configs/default.json")                    # pathlib.Path to the HDF5 file
res = Result(path)
```

`run(cfg, out_path=None, *, dry_run=False, progress=True, plot=False,
ram_limit_bytes=5 GiB, stability_interval=100, stability_p_max=1e9)`
accepts a `Config` object or a JSON path. It raises `ValueError` for config
errors (also with `dry_run=True`) and `RuntimeError` when the stability guard trips (non-finite $p$ or
$|p| > 10^9$ Pa). Warnings go to stderr.

**Output location:** `-o`/`out_path`, else `output.path`, else
`proj/boxed_piston/outputs/<config stem>.h5` (`run.h5` for a `Config`
object). Parent directories are created.

## f. Outputs (HDF5 layout)

One file per run. Notation: $n_s$ = `n_steps`; $n_d$ = decimated samples
$= \lceil n_s/M\rceil$; $N_h$, $N_b$, and $N_p$ are the numbers of Huygens
patches, box-surface patches, and probes; $n_f$ is the number of stored z
frequencies. Pressures are in Pa and velocities in m/s for the unit-peak
velocity pulse. Time arrays are in s.

| Group / dataset | Shape | dtype | Units | Meaning |
|---|---|---|---|---|
| `/meta` (attrs) | — | — | — | `h` [m], `dt` [s], `n_steps`, `n_samples_decimated`, `decimation_factor` ($M$), `decimation_note`, `huygens_offset_cells` (3), `geometry`, `zrad_source`, `conventions`, `time_after_step_n`, `tail_db` [dB re peak], `boxrad_version`, `wall_time_s`, `numba_num_threads`, `git_commit`; `aborted`, `aborted_at_step` only when the stability guard tripped |
| `/meta/config_json` | scalar | bytes | — | Full config as JSON |
| `/meta/derived_json` | scalar | bytes | SI | All `Derived` fields as JSON (`h`, `dt`, `n_steps`, `nx/ny/nz`, `realized_W/H/D`, `pulse_sigma`, `pulse_t0`, `ppw_realized`, `cell_budget_exceeded`, …); `Result.derived` ignores keys it does not know |
| `/meta/sample_times_decimated_p` | $(n_d,)$ | float64 | s | $(nM+1)\Delta t$ |
| `/meta/sample_times_decimated_v` | $(n_d,)$ | float64 | s | $(nM+\tfrac12)\Delta t$ |
| `/meta/accuracy` (attrs) | — | — | — | `ppw_realized`, `cells_across_radiator`, `h` [m], `notes` (C8/C9 error summary) |
| `/radiator/p_avg` | $(n_s,)$ | float64 | Pa | Area-weighted radiator face pressure $\langle p\rangle$ (with the $(h/2)\rho\,\partial v/\partial t$ face correction) |
| `/radiator/times_p` | $(n_s,)$ | float64 | s | $(n+1)\Delta t$, the time level of `p_avg` |
| `/radiator/v_pulse_integer` | $(n_s,)$ | float64 | m/s | Drive $v(t)$ at `times_p` |
| `/radiator/v_pulse_half` | $(n_s,)$ | float64 | m/s | Drive $v(t)$ at `times_v_half` |
| `/radiator/times_v_half` | $(n_s,)$ | float64 | s | $(n+\tfrac12)\Delta t$ |
| `/huygens/centers` | $(N_h, 3)$ | float64 | m | Patch centers |
| `/huygens/normals` | $(N_h, 3)$ | float64 | — | Unit outward normals |
| `/huygens/areas` | $(N_h,)$ | float64 | m² | Patch areas ($h^2$) |
| `/huygens/p_face` | $(n_d, N_h)$ | float32 | Pa | Face pressure (4-point cubic along the normal) |
| `/huygens/vn` | $(n_d, N_h)$ | float32 | m/s | Outward normal velocity |
| `/huygens/times_p`, `/huygens/times_vn` | $(n_d,)$ | float64 | s | Time levels of `p_face` and `vn` |
| `/box_surface/centers`, `normals`, `areas` | $(N_b,3)$, $(N_b,3)$, $(N_b,)$ | float64 | m, —, m² | Rigid-box patches, all six faces (box mode only) |
| `/box_surface/p` | $(n_d, N_b)$ | float32 | Pa | Surface pressure (adjacent cell center; radiator patches face-corrected) |
| `/box_surface/times` | $(n_d,)$ | float64 | s | Time levels |
| `/probes/requested` | $(N_p, 3)$ | float64 | m | Requested probe positions (only if probes are configured) |
| `/probes/centers` | $(N_p, 3)$ | float64 | m | Actual positions (nearest pressure-cell center) |
| `/probes/p` | $(n_d, N_p)$ | float32 | Pa | Probe pressure |
| `/probes/times_p` | $(n_d,)$ | float64 | s | Time levels |
| `/results/zrad` | $(n_f,)$ | complex128 | — | Dimensionless $z(f)$ on a log grid, 100 points/decade over `[f_min, f_max]` |
| `/results/zrad` (attrs) | — | — | — | `f` ($(n_f,)$ float64, Hz), `f_min_hz`, `f_max_hz`, `points_per_decade` (100), `convention` |

Large datasets are gzip-compressed and written in chunks during the run.
`boxrad-info` lists the actual shapes.

## g. Result API reference

```python
from boxrad import Result
res = Result("outputs/default.h5")       # or: with Result(path) as res: ...
```

`f` accepts a scalar or an array-like in Hz. Outputs always have a leading
frequency axis of length $n_f$. Transfer functions are phasors per **1 m/s
peak** radiator velocity, with phase relative to $v$.

**Band check (every method that takes `f`):** any frequency outside
`res.band` $= [f_\mathrm{min}, f_\mathrm{max}]$ raises `ValueError`. Pass
`allow_out_of_band=True` to evaluate anyway; a `UserWarning` is then issued.
Above $f_\mathrm{max}$ the grid is under-resolved, above about
$1.2 f_\mathrm{max}$ the drive spectrum is below −40 dB (`z` is NaN there),
and below $f_\mathrm{min}$ results are not validated. `z()` with no
argument returns the stored table and is never checked.

| Member | Returns | Units / notes |
|---|---|---|
| `config` | `Config` | Stored config |
| `derived` | `Derived` | Stored derived parameters |
| `meta` | `dict` | `/meta` attributes |
| `accuracy` | `dict` | `/meta/accuracy` attributes (`{}` if missing) |
| `band` | `(f_min, f_max)` | Hz; the supported range |
| `S_grid` | float | m², $h^2\sum w$ |
| `summary()` | str | Multi-line text |
| `close()` | — | Closes the file; also called on `with` exit |

**`z(f=None, *, allow_out_of_band=False) → (f_hz, z)`**: `f_hz` is float64
`(n_f,)`, and `z` is complex128 `(n_f,)`, dimensionless. With `f=None` it
returns the stored `/results/zrad` table. Spectra are exact at each `f`, with
no interpolation. With `allow_out_of_band=True`, `z` is NaN where $|V(f)|$ is
more than 40 dB below its peak (above about $1.2 f_\mathrm{max}$ with the
default pulse) and for $f \le 0$.

**`Z_acoustic(f, *, allow_out_of_band=False) → (n_f,)` complex128**, in
Pa·s/m³: $\rho c\,z/S_\mathrm{grid}$.

**`Z_mechanical(f, *, allow_out_of_band=False) → (n_f,)` complex128**, in
N·s/m: $\rho c\,S_\mathrm{grid}\,z$.

**`power(f, *, allow_out_of_band=False) → (n_f,)` float64**, in W:
$\tfrac12\rho c S_\mathrm{grid}\,\mathrm{Re}\,z$ at 1 m/s peak. Multiply by
$|v|^2$ for other velocities.

**`field(points, f, *, per="velocity", allow_out_of_band=False) → (n_f, n_pts)`
complex128**: the pressure transfer at `points` (`(n_pts, 3)` or `(3,)`, in
m).

- `per="velocity"` (default) returns $H = p/v$ in Pa/(m/s).
  `per="volume_velocity"` returns $H_U = p/U = H/S_\mathrm{grid}$ in
  Pa/(m³/s), with $U = S_\mathrm{grid} v$ (see [§h](#h-conventions)).
- Points outside the Huygens surface use KH on the stored Huygens data. This
  is exact up to discretization at any distance, so the near field is
  included. Box and baffle mode both use a chunked, vectorized KH kernel.
- **Validity:** a point enclosed by the Huygens surface, or within $h/2$ of
  one of its finite $h \times h$ patches, raises `ValueError`, unless an
  `output.probes` probe cell center is within $h/2$ of the point. In that
  case the FDTD probe record is used. The Huygens faces are 3 cells outside
  the box, at $z = 3h$ in front and $x = -x_c - W/2 - 3h$, and so on (in
  baffle mode, an open five-face surface over $z = 0$). Points in the plane
  of a face but beyond its edges are ordinary exterior points.

**`directivity(theta, phi, f, *, normalize=False, allow_out_of_band=False) →
(n_f, n_dir)` complex128**: the complex far-field pattern

$$D(\theta,\varphi,f) = \lim_{r\to\infty} r\,e^{+jkr}\,\frac{p(r,\theta,\varphi)}{v}\quad[\mathrm{Pa\cdot m/(m/s)}],$$

evaluated as the Fraunhofer limit of the KH integral. **Phase reference:**
$r$ is measured from the coordinate origin, the radiator center on the
radiating face, so at large $r$, $p/v \approx D\,e^{-jkr}/r$ (time
convention $e^{+j\omega t}$). `theta` and `phi` are in radians; they are
broadcast together and flattened to $n_\mathrm{dir}$. With `normalize=True`,
each row is divided by the complex on-axis value $D(0, \cdot, f)$, so the
on-axis entry is exactly $1 + 0j$ and the result is dimensionless. Use
`np.abs(D)` for the magnitude and `20*np.log10(np.abs(D))` for a pattern in
dB. In baffle mode only $\theta \le 90^\circ$ is physical (larger $\theta$
returns the mirror image).

**`surface_p(points, f, *, allow_out_of_band=False) → (n_f, n_pts)`
complex128**, in Pa/(m/s), box mode only. Each point maps to the nearest box-surface patch, with no
interpolation. **Validity:** a point more than $h$ from the plane of its
nearest patch raises `ValueError`. Baffle files raise `ValueError` (there is
no `box_surface`). The pressure is singular at box edges and at the radiator
rim, so single-cell values there are resolution dependent.

**`directivity_index(f, *, allow_out_of_band=False) → (n_f,)` float64**, in
dB: on-axis DI from a KH quadrature at $r = 50\max(W, H, D, h)$ (24
Gauss–Legendre points in $\cos\theta$ × 48 in $\varphi$). Box mode integrates
the full sphere. Baffle mode integrates only the physical half space
$z \ge 0$, giving the conventional half-space DI (3.01 dB for a small
baffled piston). The quadrature is the pure function
`boxrad.post.directivity.di_from_pattern(pattern_fn, half_space, n_theta,
n_phi)`, which can also be applied to any pattern of your own.

## h. Conventions

After finalize, `boxrad-info` prints the canonical `/meta/conventions` string
(also from `conventions_text` in `boxrad/post/finalize.py`):

`e^{+jωt}; z = <p>_rad/(ρc v); H = p/v [Pa/(m/s)]; coordinates: origin = radiator center on the front face; +z outward (radiating axis); box occupies z ∈ [−D, 0] (box mode) or baffle plane z = 0 (baffle mode); θ from +z, φ from +x; directivity phase referenced to the origin`

- **Time:** $e^{+j\omega t}$. Phasors are **peak** amplitudes:
  $p(t) = \mathrm{Re}\{P e^{j\omega t}\}$. NumPy's forward FFT sign is used
  as is.
- **Radiator average and area:**
  $\langle p\rangle_\mathrm{rad} = \sum_i w_i p_i / \sum_i w_i$ and
  $S = S_\mathrm{grid} = h^2\sum_i w_i$, where $w_i$ is the area fraction of
  face cell $i$.
- **Dimensionless impedance:**
  $$z = \frac{\langle p\rangle_\mathrm{rad}}{\rho c\, v}$$
  At low frequency, Re z tends to $(ka)^2/4$ for a compact box in free space
  (monopole, $\mathrm{Re}\,z \approx k^2 S/4\pi$) and to $(ka)^2/2$ in the
  baffle ($k^2 S/2\pi$). Im z is positive (mass-like).
- **Acoustic impedance:** $Z_A = \rho c\, z / S$ [Pa·s/m³], pressure per
  volume velocity $U = S v$.
- **Mechanical impedance:** $Z_M = \rho c\, S\, z$ [N·s/m], force on the
  piston per piston velocity (front side only).
- **Pressure transfer:** $H = p/v$ [Pa/(m/s)]. Per volume velocity,
  $H_U = p/U = H/S$ [Pa/(m³/s)].
- **Power:** $W = \tfrac12\rho c\, S\,\mathrm{Re}\{z\}\,|v|^2$ [W].
- **SPL** for a peak velocity $|v|$: $L_p = 20\log_{10}\!\big(|H|\,|v| / (\sqrt2\cdot 20\,\mu\mathrm{Pa})\big)$.
- **Far-field pattern:** $D(\theta,\varphi) = \lim_{r\to\infty} r\,e^{+jkr} p/v$
  [Pa·m/(m/s)], phase referenced to the origin (radiator center).
- **Directivity index:**
  $\mathrm{DI} = 10\log_{10}\!\big(4\pi |D_\mathrm{axis}|^2 / \int |D|^2\,d\Omega\big)$,
  integrated over the full sphere in box mode and over $z \ge 0$ in baffle
  mode. Equivalently $10\log_{10}\!\big(4\pi r^2 |H_\mathrm{axis}|^2 / (2\rho c\,W)\big)$
  with $W$ the power through that sphere or hemisphere.
- **Time levels** (from `/meta/time_after_step_n`): after step $n$, $p$ is
  at $(n+1)\Delta t$ and $v$ at $(n+\tfrac12)\Delta t$. Post-processing reads
  the stored time arrays.

## i. Warnings and troubleshooting

Warnings are printed to stderr as `warning: …` and the run continues.

| Message | Meaning | What to do |
|---|---|---|
| `ka=… > 10 at f_max=… Hz` | $f_\mathrm{max}$ is beyond the rigid-piston regime ($a = D_r/2$) | Lower `f_max`, or accept that a real cone has broken up there |
| `baffle mode: radiator offset (…) clips the radiator outline by … cells (S_grid slightly below S)` | Baffle mode with an offset that clips the outline by at most one cell (sub-cell offsets as in C7). The same message names "rounding of the radiator bounding box" when a centered radiator's rounded bounding box is slightly smaller than the outline | Use `x_c = y_c = 0` unless you are testing sub-cell offsets; $S_\mathrm{grid}$ already accounts for the clipped area |
| `realized … points per wavelength < default 20` | $h$ is coarser than $\lambda_\mathrm{min}/20$ (low `ppw` or a large `h_override`) | Expect larger dispersion error near $f_\mathrm{max}$ ([§d](#d-choosing-numerics)) |
| `box dimensions ignored in baffle geometry mode` | `box` is set while `geometry` is `"baffle"` | Remove `box` or switch to box mode |
| `box W=… mm outside 10–600 mm range` (also H, D) | Outside the intended range | The run works. Check cost (large) or resolution (small) |
| `box aspect ratio … > 4` | Longest/shortest side > 4 | The run works, outside the intended range |
| `gap_cells=… below default 10` | Smaller air gap than the default | Keep ≥ 5 (required); 10 is the validated default |
| `cpml_cells=… below default 16` | Thinner absorber | Expect more reflection, especially at low frequency |
| `h_override=… m active (intended for validation sweeps)` | $h$ was fixed by hand | Expected for the validation configs; otherwise remove it |
| `p_avg tail … dB re peak (> −60 dB); increase numerics.n_transits` | The radiator pressure had not decayed by the end of the record (`tail_db`) | Increase `n_transits` and rerun |

Errors stop the run (`ValueError`; the CLI prints `error: …` and exits 1):

| Message | Fix |
|---|---|
| `geometry=box requires box dimensions in config` / `run requires cfg.box when geometry=box` | Add `box`, or set `geometry: "baffle"` |
| `radiator does not fit on the box front face` | Reduce `dims` or move `(x_c, y_c)` |
| `radiator edge clearance … cells < min_edge_cells=3` | Move the radiator inward, refine $h$, or (with care) lower `min_edge_cells` |
| `domain nx×ny×nz = … cells at h=… mm exceeds numerics.cell_budget=…; …` | Raise $h$ (lower `cells_across_radiator` or `ppw`, or set `h_override`), lower `f_max`, or raise `cell_budget` if RAM allows. Typical trigger: a tiny radiator (e.g. a port) on a large box |
| `baffle mode: radiator offset (…) clips the radiator outline by … cells (> 1 cell); …` | Use `x_c = y_c = 0` in baffle mode |
| `estimated memory … GB exceeds limit … GB` | Lower `f_max`, raise `ppw`/`h`, or pass `ram_limit_bytes` to `run()` |
| `huygens offset 3 leaves … cells to CPML/gap (need ≥ 2 …)` | Set `gap_cells ≥ 5` |
| `invalid band … Hz` (at the end of the run) | Use $0 <$ `f_min` $<$ `f_max` |
| `stability guard failed at step …` (`RuntimeError`) | Usually `cfl_safety > 1`; keep it ≤ 1 |
| `TypeError: …unexpected keyword argument…` | Misspelled or unsupported config key (raised as a traceback, not `error:`) |

`Result` errors:

| Message | Fix |
|---|---|
| `… frequenc(y/ies) outside the supported band [f_min, f_max] Hz …; pass allow_out_of_band=True …` | Clip `f` to `res.band`, or pass `allow_out_of_band=True` and accept unsupported values |
| `field evaluation inside or within h/2 of the Huygens surface; …` | Move the point outward. For points inside, add `output.probes` at those positions and rerun |
| `surface_p point … is … m from the nearest box patch (limit h=…)` | Use points on the box faces |
| `HDF5 has no box_surface group (baffle geometry)` | `surface_p` is box mode only |

## j. Limits

- **Rigid piston only:** uniform normal velocity. No cone modes, surround,
  dust cap, or velocity profile.
- **Single radiator** per run, flush (not recessed or domed), on the +z face.
- **Rigid box:** no wall vibration. The box interior is not modeled as a
  cavity. Because $v_n$ on the radiator is prescribed, the interior air
  cannot affect any output, and no interior quantity is reported. Front-side
  radiation only; enclosure compliance and rear or port radiation must be
  modeled elsewhere.
- **Linear, lossless air.** No viscous or thermal losses, no nonlinearity.
- **Frequencies outside $[f_\mathrm{min}, f_\mathrm{max}]$ are
  unsupported.** `Result` methods raise unless `allow_out_of_band=True`.
- **Edge reactance:** about 1.5–2.6 % excess Im z at 10–12 cells across the
  short side (C9). It decreases with `cells_across_radiator`.
- **Dispersion:** about 1 % $|\Delta z|$ at $f_\mathrm{max}$ for ppw 20
  (C8), with an axis phase lag of about 1°/λ. Over long KH paths at the band
  top the far-field phase error accumulates (brief §8: about 35–42° over
  600 mm at 20 kHz).
- **Fixed offsets:** the Huygens offset is 3 cells. In baffle mode, keep
  the radiator centered: offsets that clip the mask warn (≤ 1 cell) or are
  rejected (> 1 cell).
- **Cell budget:** a domain larger than `cell_budget` is rejected; there is
  no coarser fallback and no analytic $z$ substitute.
