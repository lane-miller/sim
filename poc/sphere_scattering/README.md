# Sphere Scattering POC

Plane wave incident on a rigid sphere (r = 0.025 m), solved via BEM over 50
log-spaced frequencies (100–10,000 Hz). Validated against analytical solution.

## Environment
- Activate: `conda activate simenv`
- Key imports: `bempp_cl.api`, `gmsh`, `dolfinx`, `scipy`, `matplotlib`

## Files
- `mesh.py` — Gmsh surface mesh → `outputs/sphere.msh`
- `solve.py` — Burton-Miller BEM frequency sweep → `outputs/results.npz`
- `validate.py` — BEM vs analytical plot → `outputs/validation.png`

## Parameters
- Radius: 0.025 m, mesh size: 0.003 m (~6 el/λ at 10 kHz)
- Incident: plane wave +x, evaluation point: (0.1, 0, 0) m
- Formulation: Burton-Miller (no spurious interior resonances)

## Notes
- Bempp imports as `bempp_cl.api` (not `bempp.api`)
- No MPS/GPU — CPU only, not a bottleneck at this scale
- `outputs/` not tracked in version control