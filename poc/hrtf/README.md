# HRTF BEM POC

FABIAN HATS HRTF validation via Bempp BEM.

## Files

- `config.py` — paths, physical constants, source positions, mesh params
- `mesh.py` — loads FABIAN 6k STL, truncates torso at Z=-200mm, reseals via Gmsh, recenters to interaural midpoint
- `mesh_grade.py` — adaptive decimation via PyMeshLab, preserves pinna detail, coarsens cranium/torso. Reduces DOFs ~45% with no accuracy loss.
- `solve.py` — Burton-Miller BEM (P1, Galerkin) with reciprocal monopole at left ear, saves surface coefficients
- `validate.py` — evaluates far-field HRTF, compares to FABIAN measured + Mesh2HRTF SOFA references, polar plots
- `inspect_mesh.py` — one-off mesh inspection utility (not part of pipeline)

## Results

Validation RMSE vs FABIAN measured (horizontal plane), compared to Mesh2HRTF reference where noted:

**Graded mesh (4,711 DOFs):**
- 1 kHz: 1.66 dB RMSE vs measured, Mesh2HRTF baseline 0.82 dB
- 4 kHz: 1.71 dB RMSE vs measured, Mesh2HRTF baseline 1.68 dB
- Assembly: ~3 min/freq

**Ungraded mesh (8,564 DOFs):**
- 1 kHz: 1.76 dB RMSE vs measured
- 4 kHz: 2.41 dB RMSE vs measured
- Assembly: ~9 min/freq

Note: 1 kHz shadow-side deviation (min azimuth 224° vs reference 324°) is attributed to torso truncation, not mesh or solver error. 4 kHz pattern matches Mesh2HRTF reference within measurement uncertainty.

## Dependencies

bempp-cl, trimesh, gmsh, sofar, numpy, scipy, matplotlib, pymeshlab

## Data

FABIAN HRTF Database v4 (TU Berlin), CC-BY 4.0. Dataset on external SSD at path configured in `config.py` (`FABIAN_ROOT`).
