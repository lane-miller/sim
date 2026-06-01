"""
mesh.py — Expansion Chamber POC
Generates a 3D quarter-symmetry mesh of a rectangular expansion chamber.

Quarter symmetry: model x >= 0, y >= 0 quadrant only.
Symmetry planes (x=0, y=0) and rigid walls are natural Neumann BCs.

Physical tags (surfaces):
  1 : inlet    z=0 face (Robin source port)
  2 : outlet   z=TOTAL_L face (Robin anechoic termination)

Physical tags (volumes):
  10 : fluid   entire domain (all three sections)
"""

import gmsh
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MESH_FILE  = os.path.join(OUTPUT_DIR, "chamber.msh")


def generate_mesh(gui=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gmsh.initialize()
    gmsh.model.add("expansion_chamber")

    # ── Mesh size ────────────────────────────────────────────────────────────
    lam_min = cfg.C / cfg.F_MAX                    # ~0.1715 m at 2 kHz
    h = min(lam_min / 10, cfg.INLET_W / 2 / 3)    # ~3.3 mm
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    # ── Geometry (quarter section: x≥0, y≥0) ────────────────────────────────
    occ = gmsh.model.occ

    box1 = occ.addBox(0, 0, 0,
                      cfg.INLET_W / 2, cfg.INLET_H / 2, cfg.INLET_L)
    box2 = occ.addBox(0, 0, cfg.INLET_L,
                      cfg.CHAMBER_W / 2, cfg.CHAMBER_H / 2, cfg.CHAMBER_L)
    box3 = occ.addBox(0, 0, cfg.INLET_L + cfg.CHAMBER_L,
                      cfg.INLET_W / 2, cfg.INLET_H / 2, cfg.OUTLET_L)

    # Fragment creates conformal nodes at junction faces
    occ.fragment([(3, box1), (3, box2), (3, box3)], [])
    occ.synchronize()

    # ── Identify surfaces by center-of-mass ──────────────────────────────────
    TOL = 1e-6
    all_surfs = gmsh.model.getEntities(dim=2)
    all_vols  = gmsh.model.getEntities(dim=3)

    inlet_tags  = []
    outlet_tags = []

    for _, tag in all_surfs:
        cx, cy, cz = occ.getCenterOfMass(2, tag)
        if abs(cz) < TOL:
            inlet_tags.append(tag)
        elif abs(cz - cfg.TOTAL_L) < TOL:
            outlet_tags.append(tag)

    vol_tags = [tag for _, tag in all_vols]

    # ── Physical groups ───────────────────────────────────────────────────────
    gmsh.model.addPhysicalGroup(2, inlet_tags,  tag=1, name="inlet")
    gmsh.model.addPhysicalGroup(2, outlet_tags, tag=2, name="outlet")
    gmsh.model.addPhysicalGroup(3, vol_tags,    tag=10, name="fluid")

    # ── Mesh & optimize ───────────────────────────────────────────────────────
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    # ── Save ──────────────────────────────────────────────────────────────────
    gmsh.write(MESH_FILE)

    # ── Summary ───────────────────────────────────────────────────────────────
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)

    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
    n_tets = sum(len(et) for et in elem_tags)

    print(f"\nMesh summary")
    print(f"  Nodes        : {n_nodes}")
    print(f"  Tetrahedra   : {n_tets}")
    print(f"\nPhysical surface groups:")
    print(f"  Tag 1  — inlet   (z = 0 face,          surfaces {inlet_tags})")
    print(f"  Tag 2  — outlet  (z = {cfg.TOTAL_L:.4f} face, surfaces {outlet_tags})")
    print(f"\nPhysical volume group:")
    print(f"  Tag 10 — fluid   (all volumes:          {vol_tags})")
    print(f"\nMesh written to: {MESH_FILE}")

    if gui:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh(gui=False)
