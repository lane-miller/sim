# HRTF BEM POC

FABIAN HATS HRTF validation via Bempp BEM.

## Files

- `config.py` — paths, physical constants, source positions, mesh params
- `mesh.py` — loads FABIAN 6k STL, truncates torso at Z=-200mm, reseals via Gmsh, recenters to interaural midpoint
- `solve.py` — Burton-Miller BEM (P1, Galerkin) with reciprocal monopole at left ear, saves surface coefficients
- `validate.py` — evaluates far-field HRTF, compares to FABIAN measured + Mesh2HRTF SOFA references, polar plots
- `inspect_mesh.py` — one-off mesh inspection utility (not part of pipeline)

## Results

Validation RMSE vs FABIAN measured (horizontal plane):

- 1 kHz: ~1.8 dB (error largely attributed to torso truncation)
- 4 kHz: ~2.4 dB

## Dependencies

bempp-cl, trimesh, gmsh, sofar, numpy, scipy, matplotlib

## Data

FABIAN HRTF Database v4 (TU Berlin), CC-BY 4.0. Dataset on external SSD at path configured in `config.py` (`FABIAN_ROOT`).
