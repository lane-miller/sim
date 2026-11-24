# Step 12: dry-run report and geometry plot (parametric/STEP)

## Goal (spec v1 §6, v2 §9 step 12)
- **`boxrad-run --dry-run`:** prints a full pre-run report for parametric/STEP configs, without stepping.
- **`boxrad-plot`:** gains a geometry/voxel view, built from the config alone (no run file needed).
- **Box/baffle:** dry-run output and the existing plot modes are unchanged.

## Hard rules
- No changes to numerics, the kernel, the recorder or the Result API.
- Reuse `dry_run_geom` / `prepare_geom_run` (no stepping). Build fractions once.
- matplotlib only, with the Agg backend in tests. No new dependencies.
- Existing CLI names and flags unchanged; additions only.
- One showcase, one PNG, one text/JSON report.

## 1. Dry-run report
- **New function:** `boxrad/report.py: geom_report(cfg) -> dict` (plain JSON-able) plus `format_geom_report(d) -> str`.
- **CLI:** `boxrad-run --dry-run` prints the text for parametric/STEP configs. `--json PATH` also writes the dict. Exit code 0, or 2 if there are errors (budget exceeded, a probe in the CPML, seating failure, and similar). Errors must be listed, not raised as tracebacks.

**Sections and fields** (omit a field only if it does not apply):
1. **Geometry:**
   - tag, part count, n_tri, mesh volume, bbox (world);
   - planes (axis, side, offset);
   - for STEP (after step 11): file, units, sha256.
2. **Radiators:** per radiator, the name, shape, S_exact (polygon/area), S_grid, ratio, n̂, clearance (min, median) and boss margin (where available).
   - **Warn** if clearance < 5h, giving the clearance in cells and the suggested h = clearance/5.
   - **Warn** if margin < 2h.
3. **Grid:**
   - h_nominal, h (and the shrink factor, if any), ppw at f_max, grid shape, total cells;
   - memory estimate vs limit; cell budget;
   - CPML sides, gap/CPML cells, Huygens offset;
   - test-source cell, if any.
4. **Time:**
   - cfl_safety_resolved and its source, dt, n_steps;
   - runs = number of driven radiators;
   - wall-time estimate per run and total (use the existing estimator).
5. **Fractions:**
   - n_cut_cells, n_cut_faces, n_floored, min raw β;
   - ΔV, δ_eq/h with its status: OK ≤ 0.02, CHECK ≤ 0.05, WARN > 0.05;
   - max closure residual, voids removed (count, volume), thin-wall cells.
6. **Recording:**
   - probes (count, each with OK or "in CPML", using the same check as the recorder guard but reported instead of raised);
   - surface_p n_wet and bytes;
   - decimation factor;
   - estimated output size per run.
7. **Warnings / errors:** a deduplicated list at the end.

## 2. Geometry plot
- **CLI:** `boxrad-plot --geometry CONFIG.json [--out PNG] [--slice x,y,z] [--h H]`. The default slice point is the radiator-0 center, or the test-source position, or the bbox center. `--h` overrides `h_override`, for a quick look.
- **Figure (one PNG, 2×2):**
  - **Three orthogonal slices** through the slice point (x = const, y = const, z = const). In each panel:
    - draw the cell state as an image: solid = dark grey; fluid = white; cut cells colored by β̃ (sequential colormap, 0–1);
    - mark floored cells with a red dot;
    - outline radiator cells (a_c > 0) in blue;
    - overlay the triangle mesh cross-section as black line segments (triangle–plane intersection, computed in numpy);
    - draw the CPML boundary as a dashed rectangle and the Huygens surface as a dotted rectangle;
    - axes in mm, with a title giving the plane and its offset.
  - **One 3-D panel:** scatter of the wetted-cell wall points (step 9 x_s), colored by β̃. Floored cells in red, radiator cells in blue. At most 20 000 points, stride-subsampled, with the count stated in the title.
  - **Figure title:** config name, h, cells, n_cut, n_floored, δ_eq/h, cfl.
- **Implementation:** `boxrad/plot/geometry.py`. Pure functions that take the prepared run objects, plus a thin CLI wrapper.

## 3. Tests (`tests/test_report_plot.py`, fast)
1. **`geom_report` on `c13_sphere_ps`** (n = 9 via h_override):
   - n_cut_cells and n_floored match `prepare_geom_run`'s fractions stats;
   - cfl source "auto_cutcell";
   - δ_eq/h status OK.
2. **`geom_report` on `s6_downfire`:** contains a clearance warning only if clearance < 5h at its h, and the suggested h equals clearance/5 (compute the expectation from the geometry clearance and derived h in the test).
3. **A probe placed in the CPML** is reported as "in CPML", with no exception; exit status logic gives 2.
4. **Box config `--dry-run` text** is unchanged: compare against the pre-change function output for one legacy config.
5. **`plot_geometry`** on the sphere config at coarse h (0.01) writes a PNG > 10 kB with the Agg backend; the 3-D panel point count ≤ 20 000.
6. **Triangle–plane slicing helper:** a unit cube cut at z = 0.5 gives 4 segments with total length 4 (≤ 1e-12).

## 4. Showcase: `showcase/step12_report_plot.py`
- Prints the text report for `s1`–`s6` and `c13_sphere_ps`.
- Writes one JSON (`outputs/step12/reports.json`, all configs keyed by name).
- Writes one PNG: the geometry plot of `s6_downfire`, slice through the radiator center (`outputs/step12/s6_geometry.png`).
- No stepping; it must finish in under about a minute.

## Report back
- Files changed.
- CLI usage lines.
- Any fields not available yet (STEP-only ones), and the s6 clearance numbers.
