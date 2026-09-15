# Baffled-piston BEM validation

Sanity-check the bempp-cl BEM pipeline against the closed-form infinite-baffle
circular-piston radiation impedance (Blackstock / Kinsler-Frey) before the full
parametric Z_rad study.

## Physics

- Rigid circular piston, radius `a = 0.01 m`, centered in a square rigid baffle.
- Baffle is a thin 3D solid (thickness `a/30`) meshed as a closed exterior surface.
- Piston (top disk): Neumann BC `u = 1 m/s`.
- All other baffle surfaces: rigid, `u = 0`.
- `c = 343 m/s`, `rho = 1.21 kg/m³`.
- Five test points: `ka ∈ {0.05, 0.1, 0.5, 1, 5}`, each on its own mesh.

Baffle half-width (center to edge): margin `d = 10/k` from the piston rim,
`L_baffle = a + d` (square side = `2 L_baffle`).

| ka   | f (Hz) | L_baffle (m) |
|------|--------|--------------|
| 0.05 | 273    | 2.010        |
| 0.1  | 546    | 1.010        |
| 0.5  | 2730   | 0.210        |
| 1    | 5459   | 0.110        |
| 5    | 27296  | 0.030        |

## How to run

```bash
conda activate simenv
cd sim/proj/zrad/validation

python mesh.py          # generate all five meshes → meshes/
python solve.py         # BEM solve all ka → results/result_ka*.json
python analysis.py      # compare + plot → results/validation_plot.png
```

Individual ka values:

```bash
python mesh.py --ka 0.5
python solve.py --ka 0.5
```

Inspect a mesh in the Gmsh GUI:

```bash
python mesh.py --ka 1 --gui
```

## What "pass" looks like

After running `analysis.py`, BEM markers should sit on the analytical curves for
both resistance (Re) and reactance (Im) across all five ka points. A reasonable
target is **< 5 % error** on both real and imaginary parts at every ka once meshes
are converged (~6 elements per wavelength with grading near the piston edge).

Typical failure modes to watch for:

- Wrong Struve function (`scipy.special.struve`, not `kv`) in the reference.
- Piston/baffle surface tags mis-assigned in the mesh.
- Hypersingular sign or projection RHS not matching the hrtf Burton-Miller convention.
- Baffle too small (insufficient `kd` margin) — shows up as error at low ka.

## Files

| File            | Purpose                                      |
|-----------------|----------------------------------------------|
| `mesh.py`       | Gmsh graded surface meshes                   |
| `solve.py`      | Burton-Miller BEM solve, Z_rad extraction    |
| `analytical.py` | Closed-form Z_rad/(rho c) reference        |
| `analysis.py`   | Overlay plot + error table                   |
| `meshes/`       | Generated `.msh` files (not tracked)         |
| `results/`      | JSON results and validation plot (not tracked) |
