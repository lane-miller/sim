# HRTF BEM POC

FABIAN HATS HRTF validation via Bempp BEM.

## Layout

```
poc/hrtf/
  config.py          — paths, constants, band definitions
  mesh.py            — truncate + recenter FABIAN STL (shared preprocessing)
  mesh_bands.py      — generate three frequency-band graded meshes
  inspect_mesh.py    — mesh inspection utility
  common/            — shared SOFA, BEM, plotting helpers
  legacy/            — single-mesh pipeline (superseded)
  banded/            — per-band BEM solve + validation
  interp/            — sparse solve + frequency interpolation
  outputs/           — meshes and npz results
```

## Pipelines

| Pipeline | Steps | Output |
|----------|-------|--------|
| **Shared** | `python mesh.py` → `python mesh_bands.py` | truncated STL + `FABIAN_band_{low,mid,high}.stl` |
| **Legacy** | `legacy/mesh_grade.py` → `legacy/solve.py` → `legacy/validate.py` | `phi_graded.npz` |
| **Banded** | `banded/solve_bands.py` → `banded/validate_bands_polar.py` or `validate_bands_fr.py` | `phi_banded.npz` |
| **Interp** | `interp/solve_interp.py` → `interp/validate_interp_polar.py` or `validate_interp_fr.py` | `hrtf_interp.npz` |

Polar validators accept a `PLOT_FREQS` list at the top of the script to select comparison frequencies.

## Dependencies

bempp-cl, trimesh, gmsh, sofar, numpy, scipy, matplotlib, pymeshlab

## Data

FABIAN HRTF Database v4 (TU Berlin), CC-BY 4.0. Path configured in `config.py` (`FABIAN_ROOT`).
