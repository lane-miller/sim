# Expansion Chamber POC

Rectangular expansion chamber muffler solved two ways — 3D FEM (FEniCSx) and
mode-matching — then compared. Validated against the plane-wave analytical formula.

## Environment
- Activate: `conda activate simenv`
- Key imports: `dolfinx`, `gmsh`, `numpy`, `scipy`, `matplotlib`

## Files

| File | Purpose |
|------|---------|
| `config.py` | Shared geometry and fluid constants |
| `mesh.py` | Gmsh 3D mesh → `outputs/chamber.msh` |
| `solve.py` | FEniCSx Helmholtz frequency sweep + on-axis extraction → `outputs/fem_results.npz` |
| `mode_matching.py` | Mode-matching model: overlap integrals, modal system, TL sweep, on-axis pressure |
| `validate.py` | FEM vs mode-matching comparison → `outputs/validate_tl.png`, `outputs/validate_onaxis.png` |

## Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| `INLET_W × INLET_H` | 20 × 30 mm | Inlet/outlet duct cross-section |
| `CHAMBER_W × CHAMBER_H` | 60 × 80 mm | Expansion chamber cross-section |
| `INLET_L` / `CHAMBER_L` / `OUTLET_L` | 40 / 150 / 40 mm | Section lengths |
| `c` | 343 m/s | Speed of sound |
| `ρ` | 1.225 kg/m³ | Air density |
| `f` | 50–2000 Hz | Frequency sweep (25 log-spaced points) |

## Usage

```bash
python mesh.py          # generate mesh
python solve.py         # FEM sweep, saves fem_results.npz
python validate.py      # comparison figures → outputs/
```

`mode_matching.py` can also be run standalone for a self-contained mode-matching
sweep with convergence check.

## Solver details

**FEM (`solve.py`)**
- Elements: P2 Lagrange, real 2×2 block split of the complex Helmholtz system
- BCs: Sommerfeld ABC (`∂p/∂n = jkp`) at inlet and outlet; rigid walls elsewhere
- Incident field: unit plane wave sourced via RHS at inlet face
- TL from power flux ratio: `10·log10(|p_inc|²·S_inlet / ∫_outlet |p|² ds)`
- Solver: MUMPS direct via PETSc `LU`

**Mode-matching (`mode_matching.py`)**
- 8 × 8 modes per section (duct and chamber)
- Overlap integrals `C[p,q,m,n]` computed once via `scipy.integrate.quad`
- Junction matching: pressure continuity projected onto duct modes, velocity
  continuity projected onto chamber modes
- TL from propagating-mode power flux at outlet

## Validation notes

- Max |TL_FEM − TL_MM| across all frequencies: **~0.6 dB**
- FEM, mode-matching, and the plane-wave formula agree closely across the full sweep
- **Junction spikes in the on-axis pressure plots are expected, not errors.**
  Mode-matching expands the field in N=8 cosine modes per direction; the
  finite series cannot represent the true near-field singularity at the area-step
  corners, producing Gibbs-like overshoot within a few mm of each junction.
  The FEM resolves this continuously via its mesh. The spike is a cosmetic
  near-field artifact — TL (the far-field quantity of interest) is already
  converged at N=8, shifting by < 0.001 dB when increased to N=15.
- `outputs/` is not tracked in version control
