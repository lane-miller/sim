"""
HRTF POC — Mesh Inspection
===========================
Run this first to understand the FABIAN mesh structure before writing
the remeshing pipeline.
 
Usage (from sim/poc/hrtf/):
    python inspect_mesh.py
 
Requires: meshio (pip install meshio)
Optional: trimesh for watertightness check (pip install trimesh)
"""
 
import os
import sys
from pathlib import Path
import numpy as np
 
# ---------------------------------------------------------------------------
# 1. Find mesh files
# ---------------------------------------------------------------------------
FABIAN_ROOT = Path("/Volumes/LPM02 storage/Datasets/Audio/HRTF/FABIAN/FABIAN_HRTF_DATABASE_v4")
MESH_DIR = FABIAN_ROOT / "2 SurfaceMeshes"
 
print("=" * 70)
print("FABIAN MESH INSPECTION")
print("=" * 70)
 
# List everything in the mesh directory
print(f"\n--- Contents of {MESH_DIR} ---")
if not MESH_DIR.exists():
    print(f"ERROR: {MESH_DIR} does not exist. Check the path.")
    sys.exit(1)
 
for root, dirs, files in os.walk(MESH_DIR):
    level = root.replace(str(MESH_DIR), "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = "  " * (level + 1)
    for f in sorted(files):
        fpath = Path(root) / f
        size_mb = fpath.stat().st_size / 1e6
        print(f"{sub_indent}{f}  ({size_mb:.1f} MB)")
 
# ---------------------------------------------------------------------------
# 2. Also check HRIR directory for SOFA file names
# ---------------------------------------------------------------------------
HRIR_DIR = FABIAN_ROOT / "1 HRIRs"
print(f"\n--- SOFA files in {HRIR_DIR} ---")
if HRIR_DIR.exists():
    for root, dirs, files in os.walk(HRIR_DIR):
        for f in sorted(files):
            if f.endswith(".sofa"):
                level = root.replace(str(HRIR_DIR), "").count(os.sep)
                indent = "  " * (level + 1)
                fpath = Path(root) / f
                size_mb = fpath.stat().st_size / 1e6
                print(f"{indent}{f}  ({size_mb:.1f} MB)")
 
# ---------------------------------------------------------------------------
# 3. Load and inspect the HATO 0° low-res mesh
# ---------------------------------------------------------------------------
print("\n--- Searching for HATO 0° mesh files ---")
 
# Find mesh files matching HATO 0 / neutral / low-res patterns
mesh_candidates = []
for root, dirs, files in os.walk(MESH_DIR):
    for f in files:
        flow = f.lower()
        if flow.endswith((".obj", ".stl", ".ply", ".off")):
            mesh_candidates.append(Path(root) / f)
 
if not mesh_candidates:
    print("No mesh files found! Listing all files:")
    for root, dirs, files in os.walk(MESH_DIR):
        for f in files:
            print(f"  {Path(root) / f}")
    sys.exit(1)
 
print(f"Found {len(mesh_candidates)} mesh file(s):")
for mc in mesh_candidates:
    size_mb = mc.stat().st_size / 1e6
    print(f"  {mc.relative_to(MESH_DIR)}  ({size_mb:.1f} MB)")
 
# ---------------------------------------------------------------------------
# 4. Detailed inspection of each mesh
# ---------------------------------------------------------------------------
try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False
    print("\nmeshio not installed. Install with: pip install meshio")
 
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
 
if HAS_MESHIO or HAS_TRIMESH:
    for mesh_path in mesh_candidates:
        print(f"\n{'=' * 70}")
        print(f"MESH: {mesh_path.name}")
        print(f"{'=' * 70}")
 
        if HAS_TRIMESH:
            try:
                tm = trimesh.load(str(mesh_path), force="mesh")
                verts = tm.vertices
                faces = tm.faces
 
                print(f"\n  Vertices:  {len(verts):,}")
                print(f"  Faces:     {len(faces):,}")
                print(f"  Watertight: {tm.is_watertight}")
                print(f"  Volume:    {tm.is_volume}")
 
                # Bounding box
                bb_min = verts.min(axis=0)
                bb_max = verts.max(axis=0)
                bb_size = bb_max - bb_min
                print(f"\n  Bounding box (raw units — likely mm):")
                print(f"    X: [{bb_min[0]:.2f}, {bb_max[0]:.2f}]  size: {bb_size[0]:.2f}")
                print(f"    Y: [{bb_min[1]:.2f}, {bb_max[1]:.2f}]  size: {bb_size[1]:.2f}")
                print(f"    Z: [{bb_min[2]:.2f}, {bb_max[2]:.2f}]  size: {bb_size[2]:.2f}")
 
                # Check if units are mm or m
                if bb_size.max() > 10:
                    print(f"    → Likely MILLIMETERS (head ~200 mm wide)")
                    scale = "mm"
                else:
                    print(f"    → Likely METERS (head ~0.2 m wide)")
                    scale = "m"
 
                # Edge length statistics
                edges = tm.edges_unique
                edge_lengths = np.linalg.norm(
                    verts[edges[:, 0]] - verts[edges[:, 1]], axis=1
                )
                print(f"\n  Edge lengths ({scale}):")
                print(f"    Min:    {edge_lengths.min():.4f}")
                print(f"    Max:    {edge_lengths.max():.4f}")
                print(f"    Mean:   {edge_lengths.mean():.4f}")
                print(f"    Median: {np.median(edge_lengths):.4f}")
                print(f"    Std:    {edge_lengths.std():.4f}")
 
                # Vertex distribution along Z (to plan torso truncation)
                print(f"\n  Vertex Z-distribution (for torso truncation planning):")
                z = verts[:, 2]
                percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
                for p in percentiles:
                    print(f"    {p:3d}th percentile: {np.percentile(z, p):.2f} {scale}")
 
                # Centroid
                centroid = verts.mean(axis=0)
                print(f"\n  Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}) {scale}")
 
                # Try to identify ear regions by looking at extremes in Y (or X)
                # Convention varies — ears are lateral extremes
                print(f"\n  Lateral extremes (potential ear locations):")
                for axis, label in [(0, "X"), (1, "Y"), (2, "Z")]:
                    idx_min = np.argmin(verts[:, axis])
                    idx_max = np.argmax(verts[:, axis])
                    print(f"    {label} min vertex {idx_min}: ({verts[idx_min, 0]:.2f}, {verts[idx_min, 1]:.2f}, {verts[idx_min, 2]:.2f})")
                    print(f"    {label} max vertex {idx_max}: ({verts[idx_max, 0]:.2f}, {verts[idx_max, 1]:.2f}, {verts[idx_max, 2]:.2f})")
 
                # Face area statistics
                areas = tm.area_faces
                print(f"\n  Face areas ({scale}²):")
                print(f"    Min:    {areas.min():.6f}")
                print(f"    Max:    {areas.max():.6f}")
                print(f"    Mean:   {areas.mean():.6f}")
                print(f"    Total:  {areas.sum():.2f}")
 
                # Check for multiple mesh components
                components = tm.split(only_watertight=False)
                print(f"\n  Connected components: {len(components)}")
                if len(components) > 1:
                    for i, comp in enumerate(components):
                        print(f"    Component {i}: {len(comp.vertices)} verts, {len(comp.faces)} faces")
 
            except Exception as e:
                print(f"  trimesh error: {e}")
                # Fall back to meshio
                if HAS_MESHIO:
                    try:
                        mesh = meshio.read(str(mesh_path))
                        print(f"\n  Points: {len(mesh.points):,}")
                        for cell_block in mesh.cells:
                            print(f"  {cell_block.type}: {len(cell_block.data):,}")
                        bb_min = mesh.points.min(axis=0)
                        bb_max = mesh.points.max(axis=0)
                        print(f"\n  Bounding box:")
                        print(f"    X: [{bb_min[0]:.2f}, {bb_max[0]:.2f}]")
                        print(f"    Y: [{bb_min[1]:.2f}, {bb_max[1]:.2f}]")
                        print(f"    Z: [{bb_min[2]:.2f}, {bb_max[2]:.2f}]")
                    except Exception as e2:
                        print(f"  meshio error: {e2}")
 
        elif HAS_MESHIO:
            try:
                mesh = meshio.read(str(mesh_path))
                print(f"\n  Points: {len(mesh.points):,}")
                for cell_block in mesh.cells:
                    print(f"  {cell_block.type}: {len(cell_block.data):,}")
                bb_min = mesh.points.min(axis=0)
                bb_max = mesh.points.max(axis=0)
                bb_size = bb_max - bb_min
                print(f"\n  Bounding box:")
                print(f"    X: [{bb_min[0]:.2f}, {bb_max[0]:.2f}]  size: {bb_size[0]:.2f}")
                print(f"    Y: [{bb_min[1]:.2f}, {bb_max[1]:.2f}]  size: {bb_size[1]:.2f}")
                print(f"    Z: [{bb_min[2]:.2f}, {bb_max[2]:.2f}]  size: {bb_size[2]:.2f}")
            except Exception as e:
                print(f"  meshio error: {e}")
 
# ---------------------------------------------------------------------------
# 5. Quick SOFA file check
# ---------------------------------------------------------------------------
print(f"\n{'=' * 70}")
print("SOFA FILE CHECK")
print(f"{'=' * 70}")
 
try:
    import sofar as sf
    HRIR_MEASURED = HRIR_DIR / "FABIAN_HRIR_measured_HATO_0.sofa"
    if HRIR_MEASURED.exists():
        sofa = sf.read_sofa(str(HRIR_MEASURED))
        pos = sofa.SourcePosition
        print(f"\n  Measured HRTF (HATO 0°):")
        print(f"    Source positions: {pos.shape[0]}")
        print(f"    Ears (receivers): {sofa.DataIR.shape[1]}")
        print(f"    IR length:        {sofa.DataIR.shape[2]} samples")
        print(f"    Sample rate:      {sofa.DataSamplingRate} Hz")
        print(f"    Position type:    {sofa.SourcePositionType}")
        print(f"    Azimuth range:    [{pos[:, 0].min():.1f}, {pos[:, 0].max():.1f}]°")
        print(f"    Elevation range:  [{pos[:, 1].min():.1f}, {pos[:, 1].max():.1f}]°")
        print(f"    Radius:           {np.unique(pos[:, 2])}")
 
        # How many points on the horizontal plane?
        horiz_mask = np.abs(pos[:, 1]) < 0.5  # elevation ~0°
        print(f"    Horizontal plane points (|elev| < 0.5°): {horiz_mask.sum()}")
    else:
        # Try to find the right filename
        print(f"  {HRIR_MEASURED} not found.")
        print(f"  Available .sofa files:")
        for f in sorted(HRIR_DIR.rglob("*.sofa")):
            print(f"    {f.relative_to(HRIR_DIR)}")
 
except ImportError:
    print("  sofar not installed. Install with: pip install sofar")
except Exception as e:
    print(f"  Error reading SOFA: {e}")
 
print(f"\n{'=' * 70}")
print("EXTRA CHECKS")
print(f"{'=' * 70}")

"""Quick inspection of FABIAN_6k_HATO0.stl with trimesh."""
import trimesh
import numpy as np

path = "/Volumes/LPM02 storage/Datasets/Audio/HRTF/FABIAN/FABIAN_HRTF_DATABASE_v4/2 SurfaceMeshes/FABIAN_6k_HATO0.stl"
tm = trimesh.load(path, force="mesh")

print(f"Vertices: {len(tm.vertices):,}")
print(f"Faces:    {len(tm.faces):,}")
print(f"Watertight: {tm.is_watertight}")

verts = tm.vertices  # mm

# Edge lengths
edges = tm.edges_unique
elen = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
print(f"\nEdge lengths (mm):")
print(f"  min={elen.min():.2f}  max={elen.max():.2f}  mean={elen.mean():.2f}  median={np.median(elen):.2f}")

# Z distribution for torso truncation planning
z = verts[:, 2]
for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
    print(f"  Z {p:3d}%ile: {np.percentile(z, p):.1f} mm")

# Find ear region: vertices near lateral extremes, near interaural height
# Ears should be at large |X|, moderate Z (roughly head center height)
z_ear_band = (z > 40) & (z < 120)  # rough ear height guess
x = verts[:, 0]

# Left ear: most negative X in ear band
left_mask = z_ear_band & (x < -60)
if left_mask.any():
    left_ear_verts = verts[left_mask]
    left_extreme_idx = np.argmin(left_ear_verts[:, 0])
    left_ear_tip = left_ear_verts[left_extreme_idx]
    print(f"\nLeft ear region extreme: ({left_ear_tip[0]:.1f}, {left_ear_tip[1]:.1f}, {left_ear_tip[2]:.1f}) mm")
    # Find the conchal depression: vertices near the lateral extreme
    near_left = left_ear_verts[left_ear_verts[:, 0] < (left_ear_tip[0] + 15)]
    print(f"  Vertices within 15mm of lateral extreme: {len(near_left)}")
    print(f"  Centroid of those: ({near_left.mean(0)[0]:.1f}, {near_left.mean(0)[1]:.1f}, {near_left.mean(0)[2]:.1f})")

# Right ear: most positive X in ear band  
right_mask = z_ear_band & (x > 60)
if right_mask.any():
    right_ear_verts = verts[right_mask]
    right_extreme_idx = np.argmax(right_ear_verts[:, 0])
    right_ear_tip = right_ear_verts[right_extreme_idx]
    print(f"\nRight ear region extreme: ({right_ear_tip[0]:.1f}, {right_ear_tip[1]:.1f}, {right_ear_tip[2]:.1f}) mm")
    near_right = right_ear_verts[right_ear_verts[:, 0] > (right_ear_tip[0] - 15)]
    print(f"  Vertices within 15mm of lateral extreme: {len(near_right)}")
    print(f"  Centroid of those: ({near_right.mean(0)[0]:.1f}, {near_right.mean(0)[1]:.1f}, {near_right.mean(0)[2]:.1f})")

# Connected components
components = tm.split(only_watertight=False)
print(f"\nConnected components: {len(components)}")
for i, c in enumerate(components):
    bb = c.vertices.max(0) - c.vertices.min(0)
    print(f"  {i}: {len(c.vertices)} verts, {len(c.faces)} faces, bbox size ({bb[0]:.0f}, {bb[1]:.0f}, {bb[2]:.0f}) mm")