"""
HRTF POC — Mesh Grading
=========================
Adaptive decimation: preserve pinna detail, coarsen cranium/torso.
Uses quadric edge collapse with distance-weighted vertex quality.

Usage:
    python mesh_grade.py
"""

import pymeshlab
import trimesh
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
INPUT_STL = OUTPUT_DIR / "FABIAN_6k_HATO0_truncated.stl"
OUTPUT_STL = OUTPUT_DIR / "FABIAN_6k_HATO0_graded.stl"

EAR_LEFT_MM = np.array([-2.22, 66.23, -2.00])
EAR_RIGHT_MM = np.array([2.22, -66.23, 2.00])

DIST_PROTECT = 40.0
DIST_FULL_DECIMATE = 150.0
TARGET_FACE_RATIO = 0.55

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f"Loading {INPUT_STL.name}...")
ms = pymeshlab.MeshSet()
ms.load_new_mesh(str(INPUT_STL))

m = ms.current_mesh()
n_orig_verts = m.vertex_matrix().shape[0]
n_orig_faces = m.face_matrix().shape[0]

# ---------------------------------------------------------------------------
# Set per-vertex quality = protection score based on distance from ears
# ---------------------------------------------------------------------------
print("Computing distance-based quality...")

lx, ly, lz = EAR_LEFT_MM
rx, ry, rz = EAR_RIGHT_MM

expr = (
    f"max(0, min(1, "
    f"({DIST_FULL_DECIMATE} - min("
    f"sqrt((x-({lx}))^2 + (y-({ly}))^2 + (z-({lz}))^2),"
    f"sqrt((x-({rx}))^2 + (y-({ry}))^2 + (z-({rz}))^2)"
    f")) / ({DIST_FULL_DECIMATE} - {DIST_PROTECT})"
    f"))"
)

ms.compute_scalar_by_function_per_vertex(q=expr)

# ---------------------------------------------------------------------------
# Adaptive decimation
# ---------------------------------------------------------------------------
target_faces = int(n_orig_faces * TARGET_FACE_RATIO)
print(f"Decimating: {n_orig_faces} -> ~{target_faces} faces")

ms.meshing_decimation_quadric_edge_collapse(
    targetfacenum=target_faces,
    qualityweight=True,
    preserveboundary=True,
    preservenormal=True,
    preservetopology=True,
    planarquadric=True,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
ms.save_current_mesh(str(OUTPUT_STL))

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
print(f"\nVerifying {OUTPUT_STL.name}...")
result = trimesh.load(str(OUTPUT_STL), force="mesh")

edges_u = result.edges_unique
elen = np.linalg.norm(
    result.vertices[edges_u[:, 0]] - result.vertices[edges_u[:, 1]], axis=1
)

fv = result.vertices[result.faces]
e0 = np.linalg.norm(fv[:, 1] - fv[:, 0], axis=1)
e1 = np.linalg.norm(fv[:, 2] - fv[:, 1], axis=1)
e2 = np.linalg.norm(fv[:, 0] - fv[:, 2], axis=1)
fe = np.stack([e0, e1, e2], axis=1)
aspect = fe.max(axis=1) / fe.min(axis=1)

new_dofs = len(result.vertices)
speedup = (n_orig_verts / new_dofs) ** 2

print(f"Vertices:  {new_dofs:,} (was {n_orig_verts:,})")
print(f"Faces:     {len(result.faces):,} (was {n_orig_faces:,})")
print(f"Watertight: {result.is_watertight}")
print(f"Z range:   [{result.vertices[:, 2].min():.1f}, {result.vertices[:, 2].max():.1f}] mm")
print(f"Edge (mm): min={elen.min():.2f}  max={elen.max():.2f}  "
      f"mean={elen.mean():.2f}  median={np.median(elen):.2f}")
print(f"Aspect:    max={aspect.max():.2f}  mean={aspect.mean():.2f}  "
      f">10: {(aspect > 10).sum()}")
print(f"Estimated assembly speedup: {speedup:.1f}x")
print(f"Saved: {OUTPUT_STL}")