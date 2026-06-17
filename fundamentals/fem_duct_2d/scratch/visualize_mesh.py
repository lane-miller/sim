# visualize_mesh.py  (run after mesh.py has generated outputs/mesh.npz)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OUTPUT_DIR_SCRATCH

data = np.load(os.path.join(OUTPUT_DIR_SCRATCH, "mesh.npz"))
nodes    = data["nodes"]
elements = data["elements"]
inlet_nodes  = data["inlet_nodes"]
outlet_nodes = data["outlet_nodes"]

fig, ax = plt.subplots(figsize=(10, 3))

# Draw triangle edges
triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
ax.triplot(triangulation, color="steelblue", linewidth=0.6)

# Highlight boundary nodes
ax.scatter(nodes[inlet_nodes,  0], nodes[inlet_nodes,  1],
           color="green",  s=20, zorder=5, label="Inlet (Dirichlet)")
ax.scatter(nodes[outlet_nodes, 0], nodes[outlet_nodes, 1],
           color="red",    s=20, zorder=5, label="Outlet (Robin)")

ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"Mesh — {len(nodes)} nodes, {len(elements)} elements")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR_SCRATCH, "mesh.png"), dpi=150)
plt.show()
print("Saved mesh.png")