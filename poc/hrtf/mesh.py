"""
HRTF POC — Mesh Processing
===========================
Load FABIAN STL, truncate torso, reseal with coarse cap elements,
recenter to interaural midpoint, and save watertight surface mesh.

Operations:
1. Extract ear canal source coordinates from known element clusters
2. Pre-snap vertices near cut plane to eliminate slivers
3. Truncate at TORSO_CUT_Z, remove trimesh fan cap
4. Remesh cap via Gmsh using max body edge length
5. Recenter to interaural midpoint
6. Export watertight STL
"""

import gmsh
import trimesh
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FABIAN_ROOT = Path(
    "/Volumes/LPM02 storage/Datasets/Audio/HRTF/FABIAN/FABIAN_HRTF_DATABASE_v4"
)
MESH_PATH = FABIAN_ROOT / "2 SurfaceMeshes" / "FABIAN_6k_HATO0.stl"

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_STL = OUTPUT_DIR / "FABIAN_6k_HATO0_truncated.stl"

TORSO_CUT_Z = -200.0  # mm
SNAP_TOL = 4.5         # mm — vertices within this distance of cut plane snap to it

# Ear canal element clusters (Gmsh 1-indexed, from original STL inspection).
EAR_L_ELEMENTS = [1920, 1921, 1922, 1923, 1924]
EAR_R_ELEMENTS = [5392, 5393, 5394, 5395, 5396]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_shared_vertex(mesh, face_indices):
    """Find the vertex common to all faces in a cluster (fan center)."""
    vertex_sets = [set(mesh.faces[fi]) for fi in face_indices]
    shared = vertex_sets[0]
    for vs in vertex_sets[1:]:
        shared &= vs
    if len(shared) == 1:
        return mesh.vertices[shared.pop()]
    elif len(shared) > 1:
        return mesh.vertices[list(shared)].mean(axis=0)
    else:
        raise RuntimeError(
            f"No shared vertex in faces {face_indices}; check element numbers"
        )


def chain_edges_to_loops(edges):
    """Convert unordered degree-2 edges into ordered closed vertex loops."""
    adj = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    loops, visited = [], set()
    for start in sorted(adj):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        curr, prev = start, None
        while True:
            neighbors = [n for n in adj[curr] if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            if nxt == start:
                loops.append(loop)
                break
            if nxt in visited:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt
    return loops


# ---------------------------------------------------------------------------
# 1. Load and extract ear source coordinates (before any modification)
# ---------------------------------------------------------------------------
mesh = trimesh.load(str(MESH_PATH), force="mesh")
orig_verts = len(mesh.vertices)
orig_faces = len(mesh.faces)

ear_l_faces = [e - 1 for e in EAR_L_ELEMENTS]
ear_r_faces = [e - 1 for e in EAR_R_ELEMENTS]

source_left = find_shared_vertex(mesh, ear_l_faces)
source_right = find_shared_vertex(mesh, ear_r_faces)
interaural_midpoint = (source_left + source_right) / 2.0

# ---------------------------------------------------------------------------
# 2. Pre-snap: move vertices near the cut plane onto it exactly
#    This prevents trimesh from clipping triangles into slivers.
# ---------------------------------------------------------------------------
near_cut = np.abs(mesh.vertices[:, 2] - TORSO_CUT_Z) < SNAP_TOL
n_snapped = near_cut.sum()
mesh.vertices[near_cut, 2] = TORSO_CUT_Z

# Remove any faces that collapsed to zero area after snapping
mesh.update_faces(mesh.nondegenerate_faces())
mesh.remove_unreferenced_vertices()

# ---------------------------------------------------------------------------
# 3. Truncate — trimesh slice with fan cap, then remove fan faces
# ---------------------------------------------------------------------------
capped = mesh.slice_plane(
    plane_origin=[0, 0, TORSO_CUT_Z],
    plane_normal=[0, 0, 1],
    cap=True,
)

CAP_TOL = 0.1
v_c = capped.vertices
at_cut = np.abs(v_c[:, 2] - TORSO_CUT_Z) < CAP_TOL
cap_face_mask = np.all(at_cut[capped.faces], axis=1)

body_mesh = trimesh.Trimesh(
    vertices=v_c,
    faces=capped.faces[~cap_face_mask],
    process=False,
)
body_mesh.remove_unreferenced_vertices()

# ---------------------------------------------------------------------------
# 4. Boundary loop + cap element sizing
# ---------------------------------------------------------------------------
v = body_mesh.vertices

edges_s = np.sort(body_mesh.edges, axis=1)
unique_edges, counts = np.unique(edges_s, axis=0, return_counts=True)
boundary_edges = unique_edges[counts == 1]

on_cut = np.abs(v[:, 2] - TORSO_CUT_Z) < CAP_TOL
cut_edges = boundary_edges[on_cut[boundary_edges[:, 0]] & on_cut[boundary_edges[:, 1]]]

if len(cut_edges) == 0:
    raise RuntimeError(f"No boundary edges at Z = {TORSO_CUT_Z} mm")

boundary_loops_idx = chain_edges_to_loops(cut_edges)
if not boundary_loops_idx:
    raise RuntimeError("No closed boundary loops found")

boundary_loops = [v[idx] for idx in boundary_loops_idx]

# Cap lc = max edge length in body mesh (coarsest possible cap)
body_edges = body_mesh.edges_unique
body_edge_lengths = np.linalg.norm(
    v[body_edges[:, 0]] - v[body_edges[:, 1]], axis=1
)
cap_lc = float(body_edge_lengths.max())

# ---------------------------------------------------------------------------
# 5. Gmsh: mesh the cap
# ---------------------------------------------------------------------------
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.model.add("cap")

try:
    for loop_verts in boundary_loops:
        n = len(loop_verts)
        pt_tags = [
            gmsh.model.geo.addPoint(float(p[0]), float(p[1]), float(p[2]), cap_lc)
            for p in loop_verts
        ]
        line_tags = [
            gmsh.model.geo.addLine(pt_tags[i], pt_tags[(i + 1) % n])
            for i in range(n)
        ]
        cl = gmsh.model.geo.addCurveLoop(line_tags)
        gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(coords, dtype=np.float64).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    etypes, _, everts = gmsh.model.mesh.getElements(2)
    cap_tris = []
    for etype, evert_arr in zip(etypes, everts):
        if etype == 2:
            n_tri = len(evert_arr) // 3
            for j in range(n_tri):
                cap_tris.append(
                    [tag_to_idx[int(evert_arr[j * 3 + k])] for k in range(3)]
                )
    cap_tris = np.array(cap_tris, dtype=np.int64)

finally:
    gmsh.finalize()

if len(cap_tris) == 0:
    raise RuntimeError("Gmsh produced no cap triangles")

# ---------------------------------------------------------------------------
# 6. Combine body + cap, recenter to interaural midpoint
# ---------------------------------------------------------------------------
n_body = len(body_mesh.vertices)
all_verts = np.vstack([body_mesh.vertices, coords])
all_faces = np.vstack([body_mesh.faces, cap_tris + n_body])

result = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
result.merge_vertices(digits_vertex=8)
result.update_faces(result.unique_faces())

result.vertices -= interaural_midpoint

source_left_centered = source_left - interaural_midpoint
source_right_centered = source_right - interaural_midpoint

result.fix_normals()
result.export(str(OUTPUT_STL))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
edges_u = result.edges_unique
elen = np.linalg.norm(
    result.vertices[edges_u[:, 0]] - result.vertices[edges_u[:, 1]], axis=1
)

# Face aspect ratio: longest edge / shortest edge per face
fv = result.vertices[result.faces]
e0 = np.linalg.norm(fv[:, 1] - fv[:, 0], axis=1)
e1 = np.linalg.norm(fv[:, 2] - fv[:, 1], axis=1)
e2 = np.linalg.norm(fv[:, 0] - fv[:, 2], axis=1)
fe = np.stack([e0, e1, e2], axis=1)
aspect = fe.max(axis=1) / fe.min(axis=1)

print(f"Original:  {orig_verts:,} verts, {orig_faces:,} faces")
print(f"Truncated: {len(result.vertices):,} verts, {len(result.faces):,} faces")
print(f"Snapped:   {n_snapped} verts moved to cut plane")
print(f"Watertight: {result.is_watertight}")
print(f"Z range:   [{result.vertices[:, 2].min():.1f}, {result.vertices[:, 2].max():.1f}] mm")
print(f"Edge (mm): min={elen.min():.2f}  max={elen.max():.2f}  "
      f"mean={elen.mean():.2f}  median={np.median(elen):.2f}")
print(f"Aspect ratio: min={aspect.min():.2f}  max={aspect.max():.2f}  "
      f"mean={aspect.mean():.2f}  median={np.median(aspect):.2f}  "
      f">10: {(aspect > 10).sum()}  >20: {(aspect > 20).sum()}")
print(f"Cap lc: {cap_lc:.2f} mm")
print(f"Left ear:  ({source_left_centered[0]:.2f}, {source_left_centered[1]:.2f}, {source_left_centered[2]:.2f}) mm")
print(f"Right ear: ({source_right_centered[0]:.2f}, {source_right_centered[1]:.2f}, {source_right_centered[2]:.2f}) mm")
print(f"Saved: {OUTPUT_STL}")