# mesh.py
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import L, H, NX, NY, OUTPUT_DIR_SCRATCH

def build_mesh():
    # --- Nodes ---
    # linspace gives NX+1 x-coordinates and NY+1 y-coordinates
    xs = np.linspace(0, L, NX + 1)
    ys = np.linspace(0, H, NY + 1)

    # meshgrid + reshape gives every (x,y) combination as a flat list of nodes
    # node index = iy * (NX+1) + ix
    xv, yv = np.meshgrid(xs, ys)
    nodes = np.column_stack([xv.ravel(), yv.ravel()])  # shape (N, 2)

    # --- Elements ---
    # Loop over each rectangular cell, split into 2 triangles
    elements = []
    for iy in range(NY):
        for ix in range(NX):
            # Four corners of this quad cell (node indices)
            n0 = iy       * (NX + 1) + ix        # bottom-left
            n1 = iy       * (NX + 1) + (ix + 1)  # bottom-right
            n2 = (iy + 1) * (NX + 1) + (ix + 1)  # top-right
            n3 = (iy + 1) * (NX + 1) + ix        # top-left

            # Split quad into 2 triangles along the n0-n2 diagonal
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

    elements = np.array(elements)  # shape (M, 3)

    # --- Boundary node sets ---
    # These are sets of node indices on each boundary edge
    tol = 1e-12
    inlet_nodes   = np.where(np.abs(nodes[:, 0])       < tol)[0]  # x = 0
    outlet_nodes  = np.where(np.abs(nodes[:, 0] - L)   < tol)[0]  # x = L
    wall_nodes    = np.where((np.abs(nodes[:, 1]) < tol) |
                             (np.abs(nodes[:, 1] - H) < tol))[0]  # y = 0 or H

    # --- Save ---
    os.makedirs(OUTPUT_DIR_SCRATCH, exist_ok=True)
    np.savez(os.path.join(OUTPUT_DIR_SCRATCH, "mesh.npz"),
             nodes=nodes,
             elements=elements,
             inlet_nodes=inlet_nodes,
             outlet_nodes=outlet_nodes,
             wall_nodes=wall_nodes)

    print(f"Nodes:    {len(nodes)}")
    print(f"Elements: {len(elements)}")
    print(f"Inlet nodes:  {len(inlet_nodes)}")
    print(f"Outlet nodes: {len(outlet_nodes)}")

if __name__ == "__main__":
    build_mesh()