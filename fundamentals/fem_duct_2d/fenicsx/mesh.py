# fenicsx/mesh.py
import gmsh
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import L, H, OUTPUT_DIR_FENICSX

def build_mesh():
    gmsh.initialize()
    gmsh.model.add("duct")

    # --- Geometry ---
    # Four corner points: (x, y, z, mesh_size)
    # Mesh size here is a target element size — we'll aim for ~lambda/10
    lc = min(L, H) / 10

    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)  # bottom-left  (inlet)
    p2 = gmsh.model.geo.addPoint(L, 0, 0, lc)  # bottom-right (outlet)
    p3 = gmsh.model.geo.addPoint(L, H, 0, lc)  # top-right    (outlet)
    p4 = gmsh.model.geo.addPoint(0, H, 0, lc)  # top-left     (inlet)

    # Four edges
    l_bottom = gmsh.model.geo.addLine(p1, p2)  # wall
    l_right  = gmsh.model.geo.addLine(p2, p3)  # outlet
    l_top    = gmsh.model.geo.addLine(p3, p4)  # wall
    l_left   = gmsh.model.geo.addLine(p4, p1)  # inlet

    # Closed loop and surface
    loop    = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    # --- Physical groups ---
    # These tags are how FEniCSx identifies boundary regions after import.
    # Every boundary and the domain itself must be tagged.
    gmsh.model.addPhysicalGroup(1, [l_left],             tag=1)  # inlet
    gmsh.model.addPhysicalGroup(1, [l_right],            tag=2)  # outlet
    gmsh.model.addPhysicalGroup(1, [l_bottom, l_top],    tag=3)  # walls
    gmsh.model.addPhysicalGroup(2, [surface],            tag=10) # domain

    gmsh.model.setPhysicalName(1, 1,  "inlet")
    gmsh.model.setPhysicalName(1, 2,  "outlet")
    gmsh.model.setPhysicalName(1, 3,  "walls")
    gmsh.model.setPhysicalName(2, 10, "domain")

    # --- Mesh and save ---
    gmsh.model.mesh.generate(2)

    os.makedirs(OUTPUT_DIR_FENICSX, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR_FENICSX, "duct.msh")
    gmsh.write(out_path)

    print(f"Nodes:    {len(gmsh.model.mesh.getNodes()[0])}")
    print(f"Saved:    {out_path}")

    gmsh.finalize()

if __name__ == "__main__":
    build_mesh()