# boxrad v0.3 — Step 4: fractions engine (cut cells)

Spec: design v2 §2.1–2.2, §2.4–2.5 (method revised below), §9 step 4, §9.1. Branch `boxrad-v0.3`. All paths are relative to `sim/proj/boxed_piston` unless noted.

## Rules for this task
- **Read first:** `boxrad/geom/{geometry,mesh,poly2d}.py`, the existing kernel's grid and array layout (Yee staggering, index order, where p and u live), and the existing v0.2 radiator face-weight code. **Match the kernel's index conventions,** and document the mapping in the module docstring.
- **Do not change:** solver, kernel, recorder, KH/post, `Result`, configs, or the step 2/3 behavior.
- **Dependencies:** numpy, numba and the stdlib only. **No manifold3d in this module.** The box path must stay free of it.
- **Allowed commands:** `date` first and last, plus up to **3** runs of this, from `sim/`:
  `python -m pytest proj/boxed_piston/tests/test_cutcell.py -q --tb=short --durations=5`
  and **one** final run of:
  `MPLBACKEND=Agg python proj/boxed_piston/showcase/step04_fractions.py`
- **HARD STOP** after the 3rd pytest run: report even if red. No full suite, other scripts, pip, or git. **Never loosen a tolerance to get green.** Fix the code, or report.
- **End with a report of ≤ 8 lines:**
  - files;
  - the pytest summary;
  - runs used;
  - start and end times;
  - the key numbers (see §6);
  - anything left open.

## Method (this replaces v1 §2.4 slicing): clipping plus sweep
Notation:
- Cell c spans [x₀ + i h, x₀ + (i+1) h] × …
- x-faces sit at x₀ + i h, and likewise for y and z.
- n̂_w is the **solid outward** normal (it points into the fluid).
- e_a is the unit vector along axis a.

1. **Snap.** Mesh and radiator vertices within 1e-9 h of a grid plane are moved onto it.
2. **Wall patches.** Clip every solid triangle into cells (numba Sutherland–Hodgman against the cell's 6 planes, visiting only the cells in the triangle's index bbox). For each cell, accumulate:
   - A_c = Σ A;
   - S_c = Σ A x̄ (for the centroid);
   - N_c = Σ A n̂_w, a vector.

   **On-plane rule:** a triangle lying exactly on grid plane a = const, with n̂ = ±e_a, belongs only to the layer on its n̂ side. Clip it against the other 4 planes of that layer only. **No ε shifts.**
3. **Face fractions by sweep.** The closure of the fluid region in cell c along axis a gives
   $$\alpha_{a}^{+} = \alpha_{a}^{-} + \frac{N_{c,a}}{h^2}.$$
   - Sweep each grid line along a, starting from an **absorbing** (non-plane) domain side where α = 1, and stepping cell to cell.
   - If one side of axis a is a rigid environment plane, start from the other side.
   - **Checks:**
     - every α ∈ [−1e-9, 1 + 1e-9] (then clamp to [0, 1]);
     - on reaching the opposite absorbing boundary, |α − 1| ≤ 1e-9. That's the **line closure residual**; report its max.
     - A failure means a non-watertight mesh, and is a hard `ValueError` that gives the axis and the line.
4. **Volume fractions,** by the divergence theorem about the cell center x_c. For axis-aligned faces, (x̄_f − x_c)·n_f = h/2 whatever the face centroid, so
   $$\beta_c h^3=\frac13\Big[\frac h2\,h^2\sum_{f\in\partial c}\alpha_f\;-\;\sum_{w\in c}(\bar x_w-x_c)\cdot\hat n_w A_w\Big].$$
   - Accumulate the second term per triangle piece as Σ A (x̄ − x_c)·n̂.
   - Clamp to [0, 1] with a 1e-9 tolerance; violations are an error.
   - Cells without patches come out as exactly 0 or 1.
5. **Radiator weights.** Clip each radiator's triangles the same way (on-plane rule included):
   $$a_{r,c}=\sum_k A_k\,(\hat e_r\cdot\hat n_k),$$
   with ê_r = the pose normal. Then S_r = Σ_c a_{r,c}.
6. **Cell state:**
   - `solid`: β = 0 and no patch;
   - `fluid`: β = 1 and no patch;
   - `cut`: otherwise.
7. **Flood fill.** From all cells touching an absorbing boundary, through faces with α > 0. Reached fluid/cut cells stay; unreached ones become solid (β = 0, their faces α = 0). Record the count and volume removed.

The thin-wall detector and the small-cell floor come in step 5, not here.

## 1. `boxrad/cutcell.py` (new)
- **`Grid`:** a frozen dataclass with `x0` (3,) (the corner of cell (0,0,0)), `h`, `shape` (nx, ny, nz), and `rigid_sides` (a dict like `{"z-": True}`). Helpers:
  - `cell_center(i, j, k)` (vectorized);
  - `face_plane(axis, idx)`;
  - `world_to_index`.
- **`grid_for_geometry(geometry, h, margin_cells=3) -> Grid`.** A simplified §2.1, for tests and showcases only (`derive()` takes over in step 6):
  - **Face-plane anchors per axis:**
    - environment plane offsets;
    - radiator 0's plane, if its normal is ±e_a;
    - otherwise world 0.
  - **Two anchors on one axis**, separated by d: shrink to
    $$h' = d/\lceil d/h\rceil,$$
    applied isotropically, and record it.
  - **Extents:** solid bbox ± margin on absorbing sides. A rigid side ends exactly at its plane. It's an error if the solid crosses a plane by more than 1e-9.
- **`compute_fractions(geometry, grid) -> Fractions`:**
  - `alpha` = (αx, αy, αz) as float64 face arrays, shaped to match the kernel's u arrays;
  - `beta` (nx, ny, nz) float64;
  - `state` uint8;
  - patch arrays: cell flat index, A, centroid (3,), normal (3,), area-weighted;
  - per radiator: (cell flat index, a) arrays and S_r;
  - `stats` dict:
    - n_cut_cells, n_cut_faces;
    - min β > 0;
    - Σ(1 − β) h³;
    - mesh volume and the relative volume error;
    - max line closure residual;
    - voids removed (count and volume);
    - per-radiator S_r vs the exact outline area;
    - wall time.
- **`baffle` (geometry `None`):** not supported; raise `ValueError`.

## 2. Tests: `tests/test_cutcell.py`
- **Aligned box** (`from_box(c09)` on its natural grid, and a parametric box on grid planes):
  - all α and β ∈ {0, 1} exactly;
  - zero cut cells;
  - Σ(1 − β) h³ = W·H·D to 1e-12.
- **Shifted box** (offset 0.37 h in x, y and z):
  - α on the boundary faces = the analytic overlap fractions (e.g. 0.63) to 1e-12;
  - β of the corner and edge cells = the analytic products;
  - Σ(1 − β) h³ = V to 1e-12.
- **Rotated cube** (R_z(37°)·R_x(20°)) and **sphere** (d = 0.1, h = 5 mm):
  - Σ(1 − β) h³ = mesh volume to 1e-12 relative;
  - line residual ≤ 1e-9;
  - per-cell β vs a 16³ sub-sample oracle using `mesh.inside`: max |Δβ| ≤ 0.05 and mean |Δβ| ≤ 0.01, over all cut cells.
- **Radiators:**
  - **C9 grid-aligned:** every a_{r,c}/h² equals the existing v0.2 face weight w_f to 1e-14. (Find the v0.2 weight function and call it. If none is callable, compare against `poly2d.clip_outline_rect`.) Also Σa = S_grid of v0.2.
  - **S1:** Σa = the exact polygon area to 1e-12.
  - **S4:** Σa = the area to 1e-12.
  - **S6:** the down-firing woofer has Σa = its area, and the fluid layers between the woofer and the slab number round(0.03/h) along the axis.
- **S2** (floor and wall planes):
  - the grid ends at z = 0 and at y = −0.425;
  - the volume check passes;
  - the sweeps start from the absorbing sides.
- **Non-watertight mesh** (one face removed) → `ValueError`.
- **Flood fill:** a synthetic enclosed fluid pocket gets removed and counted. (Build it from a box with an internal box void, made as two closed shells with the inner one reversed.)
- **Box path import guard:** in a subprocess, `compute_fractions` for C9 leaves `manifold3d` out of `sys.modules`.
- **Performance:** S5 at h = 5 mm takes < 20 s, including numba compile. Report the measured time.

## 3. Showcase: `showcase/step04_fractions.py`
- S1–S6 at h = 5 mm via `grid_for_geometry`. Print a table per case:
  - grid shape, h (and h′ if shrunk), and N cells;
  - n_cut_cells and n_cut_faces;
  - Σ(1 − β) h³ | mesh volume | relative error;
  - max line residual;
  - min β > 0;
  - per radiator: S_r | exact area | relative error;
  - time.
- **PASS** when the volume relative error is ≤ 1e-12, the residual ≤ 1e-9, and every S_r matches to ≤ 1e-12. Otherwise **CHECK**.
- **Plot:** a 2×3 grid of β slices (imshow, 0 → 1) through each case's radiator-0 center plane, perpendicular to the radiator surface:
  - S1: x–z;
  - S2: y–z at x = 0;
  - S3: x–z;
  - S4: the slice nearest the radiator center;
  - S5: y–z;
  - S6: x–z through the axis, showing the gap.
  - Overlay radiator cells as contour or markers colored by a/h², and draw the cell grid lightly. Each title gives the case, cut cells and S_r error.
- Save to `showcase/out/step04_fractions.png`, overwritten in place. `--show` opens the window.
