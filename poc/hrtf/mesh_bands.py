"""
HRTF POC — Frequency-Banded Mesh Generation
============================================
Generate three graded meshes from the truncated FABIAN STL.

Usage:
    python mesh_bands.py
"""

import pymeshlab
import trimesh
import numpy as np

from config import C_AIR, MESH_TRUNCATED, OUTPUT_DIR, SOURCE_LEFT_MM, SOURCE_RIGHT_MM

INPUT_STL = MESH_TRUNCATED

EAR_LEFT_MM = SOURCE_LEFT_MM
EAR_RIGHT_MM = SOURCE_RIGHT_MM

DIST_PROTECT = 40.0
DIST_FULL_DECIMATE = 150.0

BANDS = [
    {"name": "low", "f_min": 200.0, "f_max": 2000.0, "lc_pinna": 25.0, "lc_far": 40.0, "target_ratio": 0.20},
    {"name": "mid", "f_min": 2000.0, "f_max": 6000.0, "lc_pinna": 9.0, "lc_far": 25.0, "target_ratio": 0.55},
    {"name": "high", "f_min": 6000.0, "f_max": 12000.0, "lc_pinna": 4.5, "lc_far": 15.0, "target_ratio": 0.85},
]

tm_orig = trimesh.load(str(INPUT_STL), force="mesh")
n_orig_verts = len(tm_orig.vertices)
n_orig_faces = len(tm_orig.faces)
print(f"Source: {n_orig_verts:,} verts, {n_orig_faces:,} faces\n")

lx, ly, lz = EAR_LEFT_MM
rx, ry, rz = EAR_RIGHT_MM


def make_quality_expr(dist_protect, dist_far):
    return (
        f"max(0, min(1, "
        f"({dist_far} - min("
        f"sqrt((x-({lx}))^2 + (y-({ly}))^2 + (z-({lz}))^2),"
        f"sqrt((x-({rx}))^2 + (y-({ry}))^2 + (z-({rz}))^2)"
        f")) / ({dist_far} - {dist_protect})"
        f"))"
    )


results = []

for band in BANDS:
    name = band["name"]
    output_stl = OUTPUT_DIR / f"FABIAN_band_{name}.stl"
    target_faces = int(n_orig_faces * band["target_ratio"])

    print(f"--- Band: {name} ({band['f_min']:.0f}–{band['f_max']:.0f} Hz) ---")
    print(f"  lc_pinna={band['lc_pinna']}mm  lc_far={band['lc_far']}mm  "
          f"target_ratio={band['target_ratio']}")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(INPUT_STL))
    ms.compute_scalar_by_function_per_vertex(q=make_quality_expr(DIST_PROTECT, DIST_FULL_DECIMATE))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        qualityweight=True,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        planarquadric=True,
    )
    ms.save_current_mesh(str(output_stl))

    result = trimesh.load(str(output_stl), force="mesh")
    edges_u = result.edges_unique
    elen = np.linalg.norm(
        result.vertices[edges_u[:, 0]] - result.vertices[edges_u[:, 1]], axis=1
    )
    n_verts = len(result.vertices)
    n_faces = len(result.faces)
    speedup = (n_orig_verts / n_verts) ** 2
    lam_min = C_AIR / band["f_max"] * 1000.0
    epw = lam_min / elen.mean()

    print(f"  Verts: {n_verts:,}  Faces: {n_faces:,}  Watertight: {result.is_watertight}")
    print(f"  Edge (mm): min={elen.min():.2f}  max={elen.max():.2f}  mean={elen.mean():.2f}")
    print(f"  λ_min={lam_min:.1f}mm  mean elements/λ={epw:.1f}")
    print(f"  Assembly speedup vs source: {speedup:.1f}x")
    print(f"  Saved: {output_stl}\n")

    results.append({"name": name, "f_min": band["f_min"], "f_max": band["f_max"],
                    "n_dofs": n_verts, "n_faces": n_faces})

print("=" * 65)
print(f"{'Band':<8} {'Freq range':<16} {'DOFs':>6} {'Faces':>7} {'Assembly speedup':>18}")
print("-" * 65)
total_dofs_sq = 0
for r in results:
    speedup = (n_orig_verts / r["n_dofs"]) ** 2
    print(f"{r['name']:<8} {r['f_min']:>5.0f}–{r['f_max']:>5.0f} Hz   "
          f"{r['n_dofs']:>6,} {r['n_faces']:>7,}   {speedup:>15.1f}x")
    total_dofs_sq += r["n_dofs"] ** 2
print("-" * 65)
print(f"Combined assembly speedup vs 3x source mesh: "
      f"{(3 * n_orig_verts ** 2) / total_dofs_sq:.1f}x")
