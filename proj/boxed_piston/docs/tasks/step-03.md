# boxrad v0.3 — Step 3: geometry model (in-house tools, Geometry object, parametric primitives, radiators)

Spec: design v2 §0.1, §1.5, §1.6, §9 step 3, §9.1. Branch `boxrad-v0.3`. All paths are relative to `sim/proj/boxed_piston` unless noted.

## Rules for this task
- **Read first:** `boxrad/frames.py`, `boxrad/geomconfig.py`, the config loader, the existing box-geometry code in `derive()` and the kernel (how the box extents come from W/H/D, `radiator.x_c/y_c` and the realized dimensions), and the existing radiator weight code (how circle/rect/racetrack outlines are defined).
- **Do not change:** solver, kernel, recorder, KH/post, `Result`, or legacy config behavior. `run()` guards from step 2 stay in place.
- **Dependencies:**
  - numpy, numba and the stdlib are always allowed.
  - `manifold3d` is allowed **only** inside the parametric builder and for triangulating outlines with holes, always as a lazy import (inside a function). If it's missing, raise `ImportError("parametric geometry needs: python -m pip install -e proj/boxed_piston[geom]")`.
  - **The box path must not import manifold3d.**
  - Never import shapely, trimesh, OCP, or anything in the license guard's lists.
- **manifold3d API:** read the installed package's stubs/docstrings (`site-packages/manifold3d*`) to confirm the names. Do not guess.
- **Precision:** geometry is float64 end to end.
  - manifold3d's standard mesh export is float32.
  - **(a)** If the installed binding offers a 64-bit mesh export, use it.
  - **(b)** Otherwise, after any boolean, *restore* each output vertex to the nearest float64 input vertex within 1e-6·L_bbox. Output vertices that match no input vertex are an error for v0.3 primitives, which only touch or overlap along planar faces. Name the method you used in the report.
  - Don't send a single part through manifold3d at all; use its float64 mesh directly.
- **Allowed commands:** `date` at start and end, plus exactly this, run from `sim/` with the `simenv` Python 3.11 (**at most 3 fix cycles**):
  `python -m pytest proj/boxed_piston/tests/test_poly2d.py proj/boxed_piston/tests/test_mesh.py proj/boxed_piston/tests/test_geometry.py -q -x --tb=short`
  No full suite, showcase, scripts, pip, or git.
- **End with a report of ≤ 10 lines:**
  - files created or changed;
  - the final pytest summary line;
  - the number of cycles used;
  - start and end times;
  - the manifold precision method, (a) or (b);
  - assumptions made;
  - anything left open.

## 1. `boxrad/geom/poly2d.py` (new; numpy only)
Rings are (N, 2) float64 arrays, not closed (the last vertex ≠ the first).
- `signed_area(ring)`, `centroid(ring)` (shoelace), `orient(ring, ccw=True)`.
- `Outline(outer, holes=())`:
  - outer is normalized to CCW and holes to CW;
  - `.area` = |outer| − Σ|holes|, and `.centroid` is area-weighted;
  - `.contains(pts)`: winding number, inside the outer and not inside any hole.
- **Shape builders** in local 2-D coordinates, centered at the origin, with arcs sampled so the chord sagitta is ≤ `tol` (default 1e-6 m):
  - segments n = max(16, ⌈π / arccos(1 − tol/r)⌉);
  - `rect(w, h)`;
  - `circle(d)`;
  - `racetrack(w, h)`: straight length w − h, end radius h/2, along x;
  - `ellipse(a, b)` (semi-axes; sample by parameter, with n set from r = max(a, b));
  - `from_shape_spec(ShapeSpec, tol)`, which builds the outer and its hole. Raise `ValueError` if the hole isn't strictly inside the outer: every hole vertex inside the outer, and every outer vertex outside the hole.
- **`clip_rect(ring, xmin, xmax, ymin, ymax) -> float`:** signed area of ring ∩ rectangle, by Sutherland–Hodgman against the 4 half-planes. It must be correct for nonconvex rings.
- **`clip_outline_rect(outline, …) -> float`:** outer clip minus the hole clips.
- **`triangulate_convex_fan(ring)`:** fan from vertex 0, for convex rings without holes.
- **`triangulate(outline)`:**
  - no holes → check convexity and use the fan;
  - otherwise (or nonconvex) → lazy `manifold3d.triangulate`.
  - Returns (vertices (V, 2), triangles (T, 3)), CCW.
  - Assert Σ triangle area = `outline.area` to 1e-12 relative.

## 2. `boxrad/geom/mesh.py` (new; numpy + numba)
- `TriMesh(vertices (N,3) float64, faces (M,3) int64)`, outward-oriented, with:
  - `.volume` (divergence theorem);
  - `.area`, `.face_normals`, `.face_areas`, `.bbox`;
  - `.transformed(R, t)`;
  - `.is_closed()`: every undirected edge used exactly twice, with opposite orientations.
- `weld(vertices, faces, tol)`: hash-grid weld at `tol` (default 1e-9·L_bbox). Drops degenerate faces.
- `winding_number(mesh, pts)`: generalized winding number Σ Ω_tri / 4π (Van Oosterom–Strackee solid angle), numba-parallel over points. Plus `inside(mesh, pts) = winding > 0.5`.
- **Analytic float64 primitives** in local coordinates, outward normals, closed:
  - `box_mesh(size)`: centered, 12 triangles;
  - `cylinder_mesh(d, length, tol)`: centered, axis along z, n from the §1 rule, with fan-triangulated caps;
  - `sphere_mesh(d, tol)`: icosphere subdivided until the max sagitta is ≤ tol, with vertices projected onto the sphere.
- **Exact discrete reference volumes** for tests and the showcase:
  - polygonal cylinder V_n = (n/2) r² sin(2π/n) L;
  - box = product of the sides.

## 3. `boxrad/geom/geometry.py` (new)
- **`RadiatorGeom`:** `name`, `pose` (Pose), `kind="flat"`, `outline` (`Outline`, local 2-D), `vertices_world` (V,3), `triangles` (T,3), `area` (= outline.area), `motion`.
- **`Geometry`:**
  - fields: `solid` (TriMesh, world), `radiators` (list), `planes` (from the spec), `provenance` (dict: builder, tol, part count, precision method);
  - `.inside(pts)`: delegates to the winding number (planes are not included yet).
- **`build_geometry(config, tol=1e-5) -> Geometry | None`:**
  - `"box"` → `from_box`;
  - `"baffle"` → `None`;
  - `"parametric"` → `from_parametric`;
  - `"step"` → `NotImplementedError("STEP ingest lands in v0.3 step 11")`.
- **`from_box(config)`:** numpy only.
  - The box mesh in the legacy/world frame **must reproduce the existing box extents exactly**, using the same realized W/H/D and `x_c/y_c` logic the kernel uses. Box in z ∈ [−D, 0]; the radiator at the origin; +z out.
  - Radiator 0 is the legacy shape at the identity pose. Its outline comes from `poly2d`, matching the legacy outline definition (racetrack orientation etc.), with tol 1e-6.
  - Box configs with extra radiators: build those too, as flat outlines with their poses.
- **`from_parametric(config, tol)`:**
  - Each `PrimitiveSpec` → analytic mesh → `transformed(pose.R, pose.position)`.
  - One part → used directly. More than one → manifold3d union, with float64 handling per the rules.
  - Weld the result. Assert closed, and volume > 0.
- **Radiators (parametric):** outline via `from_shape_spec`, triangulated, mapped to world by the pose.
- **`seating_check(geometry, radiator, eps=1e-4, n=200)`:**
  - Sample points: the outline boundary vertices (subsampled) plus an interior grid that keeps points inside the outline. That's about n points in total.
  - For each local point q, take world p = pose.to_world([q, 0]). It must satisfy `inside(p − eps·n̂)` and `not inside(p + eps·n̂)`.
  - On failure raise `ValueError` naming the radiator, the failure count and the first failing world point: "radiator must sit on a flat face of the solid; it overhangs or is not on the surface".
- `build_geometry` runs `seating_check` on every radiator. Radiator names come from the spec; box radiator 0 is `"radiator"`.

## 4. Config and fixtures
- **Parametric and STEP configs** no longer need the legacy `radiator` section. Make it optional for those tags only; box and baffle configs behave exactly as before. Remove the placeholder `radiator` section from `configs/showcase/s1_cylinder.json` and `s2_bookshelf.json`.
- **New `configs/showcase/s4_tilted.json` (`"parametric"`):** the C9 box tilted. Read W, H, D, the radiator shape and size, and `x_c`/`y_c` from `configs/c09_box.json`. Let R = R_z(37°)·R_x(20°).
  - Box: size [W, H, D], position R·(−x_c, −y_c, −D/2), normal R·ẑ, x_axis R·x̂.
  - Radiator `radiator`: the same shape as C9, position (0, 0, 0), normal R·ẑ, x_axis R·x̂.
  - Write the numbers with ≥ 15 significant digits. Copy the remaining sections from c09_box.json, as in step 2.

## 5. Tests
### `tests/test_poly2d.py`
- Areas and centroids vs analytic or exact polygon formulas for rect, circle (exact n-gon area (n/2) r² sin(2π/n)), racetrack, ellipse and an annulus: ≤ 1e-12 relative to the exact polygon value. Also, |polygon − analytic| ≤ the sagitta bound.
- `clip_rect`:
  - square fully inside, fully outside, partial;
  - a nonconvex L-shape;
  - an annulus over a square;
  - each vs a brute-force check with 4e6 sample points (tolerance 1e-3 relative);
  - and partitions: summing the clips of an outline over a tiling grid = outline area to 1e-12.
- Triangulation area equality, CCW orientation, the hole case (skip if no manifold3d), and the bad-hole `ValueError`.

### `tests/test_mesh.py`
- `box_mesh`, `cylinder_mesh`, `sphere_mesh`: closed, outward (volume > 0), volume vs exact discrete reference (box and cylinder to 1e-12), sphere within the sagitta bound.
- Weld merges duplicates. `is_closed` fails on a mesh with one face removed.
- Winding number:
  - 2000 random points vs the analytic inside test for a box and a sphere (ignoring points within 1e-6 of the surface);
  - ≈ 1 deep inside, ≈ 0 far outside, to 1e-9.
- Transformed mesh volume unchanged to 1e-12.

### `tests/test_geometry.py`
- `from_box` for every existing box config (not baffle):
  - closed;
  - volume = W·H·D exactly (realized dimensions);
  - z-range [−D, 0];
  - x/y extents match the kernel's box logic;
  - radiator 0 area vs the legacy exact area;
  - seating passes.
- **Box path import guard:** in a `subprocess` (`sys.executable -c …`), build `from_box` for `configs/c09_box.json` and assert `manifold3d` is not in `sys.modules`.
- **S1** (skip if no manifold3d): closed; volume = the exact polygonal cylinder; driver area = the exact n-gon; seating passes.
- **S2:** the union is closed; volume = pedestal + box (they touch at z = 0.15) to 1e-12 relative; both radiators seat.
- **S4:** volume = W·H·D to 1e-12; seating passes; the radiator's world normal = R·ẑ.
- **Overlap union:** two boxes overlapping by a known volume → V = V₁ + V₂ − V_overlap, to 1e-12.
- **Seating failures (`ValueError`), one test each:**
  - S1 driver moved to z = +0.01 (off the surface);
  - driver d = 0.25 on S1 (overhangs the 0.20 end);
  - a radiator on the curved side of S1 (normal +x at x = 0.1);
  - S2 woofer moved so it straddles the box's top edge.
- `build_geometry` for baffle → `None`; for a STEP tag → `NotImplementedError`.

## 6. Showcase: `showcase/step03_geometry.py` (write it; do not run it)
- Build S1, S2, S3 (= `configs/c09_box.json` via `from_box`) and S4. Print one table per case:
  - n_vertices, n_triangles, closed;
  - bbox;
  - mesh volume | exact discrete volume | analytic volume | relative errors;
  - per radiator: area | exact polygon area | analytic area (πd²/4, w·h, …) | relative errors;
  - seating result;
  - **PASS** when mesh vs exact discrete ≤ 1e-12 and the seating passes, otherwise **CHECK**.
- Also print:
  - the S4 vs S3 volume difference (expected ≈ 1e-16 relative);
  - the S4 radiator normal vs R·ẑ;
  - the precision method from `provenance`.
- **Plot:** 2×2 3-D subplots, equal aspect:
  - solid as a light translucent surface with thin edges;
  - radiators filled orange;
  - pose triads (length 0.05);
  - for S2, the floor square;
  - each title gives the case name and its volume.
- Save to `showcase/out/step03_geometry.png`, overwritten in place. `--show` opens the window.
- Print the start, the PASS/CHECK summary and the output path. It should run in under 15 s.
