"""
mesh.py — Baffled Piston POC
Generates a 2D axisymmetric mesh in the (r, z) half-plane:
  - Quarter-circle fluid domain (r=0 to R_fluid) — unstructured triangles
  - PML annulus (R_fluid to R_PML) — unstructured triangles
  - Boundary tags for FEniCSx

Coordinate convention: r = horizontal axis, z = vertical axis (axis of symmetry at r=0)

Physical tags (line boundaries):
  1 : axis          r=0, 0 <= z <= R_PML   (symmetry axis)
  2 : piston        z=0, 0 <= r <= a       (velocity source)
  3 : baffle        z=0, a < r <= R_PML    (rigid wall, homogeneous Neumann)
  4 : ff_arc        arc at r=R_fluid       (fluid/PML interface, far-field extraction)
  5 : pml_outer     arc at r=R_PML         (outer PML boundary)

Physical tags (surface):
  10 : fluid        interior quarter-circle (triangles)
  11 : pml          PML annulus (triangles)
"""

import gmsh
import os
import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────
# Element size is set by the HIGHEST frequency (resolution), while the PML
# thickness is set by the LOWEST frequency (longest wavelength). These MUST be
# decoupled: deriving the PML thickness from the (tiny) high-frequency element
# size leaves the PML only a few % of a wavelength thick at the low end, so the
# p=0 outer boundary sits inside the source near-field and inflates it (a ~2×
# near-field error was observed earlier with a very thin, h_fluid-derived PML).
#
# PML performance levers (all addressed below):
#  • THICKNESS in wavelengths — coordinate-stretch attenuation exp(-(1/c)∫σ) needs
#    physical path length. pml_frac=0.5·λ_max keeps σ_max gentle and the path long.
#  • STANDOFF — at R_fluid=6a the field entering the PML still has curved/evanescent
#    near-field content (worst at f_min) that a PML absorbs poorly; R_fluid=8a lets
#    the wave planarize before it reaches the absorber.
#  • RESOLUTION/GRADING — h_pml=h_fluid/10 resolves the σ ramp; m_pml=3 (set in
#    solve.py) softens it near the interface, lowering discretization reflection.
a         = 0.010   # piston radius [m]
R_fluid   = 0.080   # fluid domain outer radius [m]  (= 8a, standoff from near field)
f_max     = 40e3    # highest frequency to resolve [Hz]
f_min     = 10000.0  # lowest frequency in the sweep [Hz]
c         = 343.0   # speed of sound [m/s]
lam_min   = c / f_max          # shortest wavelength = 0.008575 m (sets element size)
lam_max   = c / f_min          # longest wavelength = 0.034300 m (sets PML thickness)
h_fluid   = lam_min / 12       # ~0.00071 m — 12 el/wavelength at f_max
h_pml     = h_fluid             # match fluid spacing (~24 layers across the PML)
pml_frac  = 0.50                # PML thickness in longest wavelengths
PML_thick = pml_frac * lam_max  # ~0.01715 m  (~0.5 λ at f_min)
R_PML     = R_fluid + PML_thick # PML outer radius
h_piston  = a / 10             # at least 10 elements across piston face

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
MESH_FILE  = os.path.join(OUTPUT_DIR, "baffled_piston.msh")

# ── Build geometry ─────────────────────────────────────────────────────────────
gmsh.initialize()
gmsh.model.add("baffled_piston")
geo = gmsh.model.occ

# Points — fluid boundary
p_origin    = geo.addPoint(0,       0,       0, h_piston)
p_piston_r  = geo.addPoint(a,      0,       0, h_piston)
p_baffle_r  = geo.addPoint(R_fluid, 0,       0, h_fluid)
p_axis_top  = geo.addPoint(0,      R_fluid,  0, h_fluid)

# Points — PML outer boundary
p_pml_base  = geo.addPoint(R_PML,  0,       0, h_pml)
p_pml_top   = geo.addPoint(0,     R_PML,    0, h_pml)

# Lines — fluid domain
l_axis_fluid = geo.addLine(p_origin,   p_axis_top)
l_ff_arc     = geo.addCircleArc(p_baffle_r, p_origin, p_axis_top)
l_piston     = geo.addLine(p_origin,   p_piston_r)
l_baffle     = geo.addLine(p_piston_r, p_baffle_r)

# Lines — PML domain
l_axis_pml   = geo.addLine(p_axis_top,  p_pml_top)
l_pml_outer  = geo.addCircleArc(p_pml_base, p_origin, p_pml_top)
l_baffle_pml = geo.addLine(p_baffle_r,  p_pml_base)

# Surface — fluid domain (unstructured triangles)
cl_fluid = geo.addCurveLoop([l_axis_fluid, -l_ff_arc, -l_baffle, -l_piston])
s_fluid  = geo.addPlaneSurface([cl_fluid])

# Surface — PML annulus (unstructured triangles)
cl_pml = geo.addCurveLoop([l_ff_arc, l_axis_pml, -l_pml_outer, -l_baffle_pml])
s_pml  = geo.addPlaneSurface([cl_pml])

geo.synchronize()

# ── Mesh sizing ───────────────────────────────────────────────────────────────
gmsh.model.mesh.setSize([(0, p_origin),   (0, p_piston_r)], h_piston)
gmsh.model.mesh.setSize([(0, p_baffle_r), (0, p_axis_top)], h_fluid)
gmsh.model.mesh.setSize([(0, p_pml_base), (0, p_pml_top)],  h_pml)

# ── Physical groups ────────────────────────────────────────────────────────────
gmsh.model.addPhysicalGroup(1, [l_axis_fluid, l_axis_pml], tag=1, name="axis")
gmsh.model.addPhysicalGroup(1, [l_piston],                 tag=2, name="piston")
gmsh.model.addPhysicalGroup(1, [l_baffle, l_baffle_pml],   tag=3, name="baffle")
gmsh.model.addPhysicalGroup(1, [l_ff_arc],                 tag=4, name="ff_arc")
gmsh.model.addPhysicalGroup(1, [l_pml_outer],              tag=5, name="pml_outer")
gmsh.model.addPhysicalGroup(2, [s_fluid],                  tag=10, name="fluid")
gmsh.model.addPhysicalGroup(2, [s_pml],                    tag=11, name="pml")

# ── Generate and save ──────────────────────────────────────────────────────────
gmsh.model.mesh.generate(2)

os.makedirs(OUTPUT_DIR, exist_ok=True)
gmsh.write(MESH_FILE)
print(f"Mesh written to {MESH_FILE}")

nodes = gmsh.model.mesh.getNodes()
_, elem_tags, _ = gmsh.model.mesh.getElements(2)
n_tri = sum(len(t) for t in elem_tags)

print(f"  Nodes    : {len(nodes[0])}")
print(f"  Triangles: {n_tri}")
print(f"  R_fluid  : {R_fluid:.6f} m")
print(f"  R_PML    : {R_PML:.6f} m")
print(f"  PML thick: {PML_thick:.6f} m  ({PML_thick/h_fluid:.0f} layers, "
      f"{PML_thick/lam_max:.2f} λ at f_min)")

gmsh.finalize()
