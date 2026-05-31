# Baffled Piston POC

2D axisymmetric FEA of a circular baffled piston radiating into a half-space, validated against the analytical solution.

## Physics

- **Domain** — quarter-circle `(r, z)` half-plane, `r² + z² ≤ R_fluid`  
- **Equation** — axisymmetric Helmholtz with cylindrical Jacobian `2πr`  
- **PML** — spherical radial-stretch absorber annulus `(R_fluid, R_PML)` terminating the domain  
- **Piston** — uniform normal velocity `v₀ = 1 m/s` on `z = 0, 0 ≤ r ≤ a`  
- **Baffle** — rigid `(∂p/∂n = 0)` on `z = 0, r > a`

## Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| `a` | 10 mm | Piston radius |
| `R_fluid` | 80 mm | Fluid domain radius (= 8a) |
| `f` | 10–40 kHz | Frequency sweep (40 log-spaced points) |
| `c` | 343 m/s | Speed of sound (air) |
| `ρ` | 1.21 kg/m³ | Air density |
| `ka` | 1.83–7.33 | Dimensionless frequency range |

## Files

| File | Purpose |
|------|---------|
| `mesh.py` | Gmsh mesh: fluid quarter-circle + PML annulus, physical group tags |
| `solve.py` | FEniCSx frequency sweep — spherical PML, MUMPS direct solver, saves `pressure_f*.npy` |
| `validate.py` | Validation plots: PML attenuation curve, near/far-field FEA vs analytical heatmaps, on-axis check |
| `extra_plots.py` | Far-field polar directivity overlay + 2D pressure field heatmaps (FEA vs Rayleigh integral) |

## Usage

```bash
python mesh.py          # generate mesh  (~seconds)
python solve.py         # frequency sweep (~minutes, depends on f resolution)
python validate.py      # validation figures → outputs/
python extra_plots.py   # directivity + heatmap figures → outputs/
```

All outputs are written to `outputs/`.

## Solver details

- **Elements** — P1 Lagrange (real-valued 2×2 block split of the complex system)  
- **PML** — spherical coordinate stretch, quadratic grading `σ ∝ (ρ − R_fluid)^3 / d_pml²`, target reflection `R_ref = 10⁻⁶`  
- **Resolution** — 12 elements/wavelength at `f_max`; PML thickness 0.5 λ at `f_min`  
- **Far-field extraction** — axisymmetric Kirchhoff–Helmholtz integral with rigid-baffle image, radius-averaged over interior arcs `[0.45, 0.92] R_fluid`
