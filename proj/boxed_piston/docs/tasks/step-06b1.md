# boxrad v0.3 — Step 6b1: derive, grid and operator for geometry; stepping loop

Context: decision 6-0 = B. The v0.2 box and baffle path stays untouched. This step makes parametric configs (no environment planes) **step** through `cutkernel.step_cut`. Recording, finalize and `Result` come in 6b2. Paths are relative to `sim/proj/boxed_piston` unless noted.

## Cursor guardrails (these override everything else)
1. **Commands.** The only commands allowed are these, run from `sim/`, in the foreground, one at a time:
   - `date`
   - `/Users/lane/miniconda3/envs/simenv/bin/python -m pytest proj/boxed_piston/tests/test_geomrun.py -q -x --tb=short --durations=5`
   - `MPLBACKEND=Agg /Users/lane/miniconda3/envs/simenv/bin/python proj/boxed_piston/showcase/step06b1_geomrun.py`
2. **Budget:** at most **3** pytest runs and **1** showcase run. Before each run, write "run k/3: <what changed>". If the same test fails twice in a row, stop and report your diagnosis.
3. **No background processes.** If a command runs longer than 3 minutes, stop it and report.
4. **Speed:** each test < 30 s, and the file < 120 s.
5. **Don't touch:** `kernels.py`, `source.py`, `cpml.py`, `grid.py`, `recorder.py`, `config.py`'s existing functions, `cutkernel.py`, `cutcell.py`, `smallcell.py`, `tests/ref_*.py`, or existing tests (except the one guard test named in §5). Only the files listed below may be created or changed.
6. **Never loosen a tolerance.** Fix the code or report.
7. **Report (≤ 8 lines):**
   - files;
   - each run's summary and wall time;
   - runs used;
   - start and end;
   - the key numbers (§6);
   - anything left open.

## 1. `boxrad/geomderive.py` (new)
- **`derive_geom(cfg, geometry) -> (Derived, GeomGrid)`.** Mirror `config.derive` exactly, with these geometry rules:
  - **Radiator length D_min:** the smallest characteristic dimension over all radiators in `geometry_spec`:
    - circle: d;
    - rect and racetrack: min(w, h);
    - ellipse: 2·min(a, b).
  - **f_max:** `band.f_max`, or else min(20 kHz, 10c/(π·D_min)).
  - **h:** `numerics.h_override` if set, else min(λ_min/ppw, D_min/cells_across_radiator).
  - **Δt** is computed as in `derive`.
  - **Anchors per axis a.** First compute the solid's bbox [lo, hi].
    - **(i)** If radiator 0's pose normal is ±e_a (|n_a| > 1 − 1e-12), anchor = radiator-0 position[a] (a face plane).
    - **(ii)** Otherwise anchor = lo[a].
    - Snapped bbox: lo_s = anchor + floor((lo − anchor)/h + 1e-9)·h and hi_s = anchor + ceil((hi − anchor)/h − 1e-9)·h.
  - **Box cells:** nb_a = round((hi_s − lo_s)/h).
  - **Domain:** n_a = nb_a + 2·gap + 2·cpml. The corner of cell (0,0,0) is x₀ = lo_s − (gap + cpml)·h.
  - **realized_W/H/D** = hi_s − lo_s (documented as "solid bbox, snapped"). box_nx/ny/nz = nb.
  - **radiator_ic/jc** = 0.
  - **pulse, run time, n_steps, decimation:** as in `derive`, with max_dim = max(realized_W, realized_H, realized_D).
  - **cpml_sides** = all True; `geometry` = "parametric".
- **`GeomGrid`:** `nx, ny, nz, h, cpml, x0` (float64, 3), `cpml_slabs` (via `grid.build_cpml_slabs`), and `cfg`. It must be accepted by `grid.allocate_fields` and `cpml.make_cpml_state`. Read both to confirm which attributes they use.
- **`cut_grid(gg) -> cutcell.Grid`** with the same x0, h and shape, and no rigid sides.
- **`validate_geom(cfg, derived, geometry) -> (errors, warnings)`:**
  - memory and `cell_budget` checks, as in `validate`;
  - the realized-ppw warning;
  - `geometry.clearance_warnings(geometry, h)` added to the warnings;
  - an error if environment planes exist ("environment planes land in v0.3 step 7").

## 2. `boxrad/geomop.py` (new): assemble the `CutOperator` for a run
`build_run_operator(cfg, derived, gg, geometry) -> (op, info)`:
1. `fractions = cutcell.compute_fractions(geometry, cut_grid(gg))`.
2. `thin_wall_cells(..., max_thin=cfg.numerics.max_thin_cells or 0)`. That numerics field doesn't exist yet, so use 0.
3. `beta_tilde, floor_stats = smallcell.apply_floor(fractions, lam)` with λ = cΔt/h (`smallcell.courant_lambda`). Assert `stability_ok`.
4. **Radiator 0:**
   - **Fast path** if its pose normal is ±e_a **and** every radiator cell lies in a single layer **and** α = 0 on every face that receives it.
     - Face index along a: the cell's index if n̂_a > 0 (the cell's minus face), or the cell's index + 1 if n̂_a < 0.
     - Weight w = a_{r,c}/h²; sign s = sign(n̂_a).
   - **Otherwise patch flux:** cells and coefficients a_{r,c}/h².
   - Record the path in `info`.
5. **Radiators 1…N−1** stay rigid: no fast path and no patch flux. They're already wall in the fractions. Warn "radiators 1.. held rigid; multi-radiator runs land in step 8".
6. `assert_geometry_clear_of_cpml(op, gg.cpml_slabs)`.
7. **`info`:**
   - S_grid = Σ_c a_{0,c};
   - path;
   - fractions stats, floor stats, thin-wall stats;
   - n_special_cells, n_closed_faces.

## 3. `boxrad/geomrun.py` (new): stepping only
- **`prepare_geom_run(cfg) -> GeomRun`:**
  - `geometry = build_geometry(cfg)`;
  - `derive_geom`, then `validate_geom` (errors raise; warnings print to stderr);
  - fields = `allocate_fields(gg)`;
  - coeffs = `kernels.make_update_coeffs`;
  - cpml = `make_cpml_state(gg, cfg, derived, dt)`;
  - `build_run_operator`;
  - the pulse parameters (σ, t₀, Δt) as in `BoxBoundary`.
- **`step_geom(run, n)`:** `cutkernel.step_cut(fields, coeffs, op, cpml, [v_now])`, where v_now = float32 `source.velocity_at_half_step(n, dt, σ, t₀)` (same as `BoxBoundary.v_at_step`).
- **`simulate_geom(cfg, n_steps=None, stability_interval=100)`:** runs `step_geom` for `derived.n_steps` (or the override), with the same stability guard as `runner`. Returns the GeomRun.
- **`dry_run_geom(cfg) -> dict`:**
  - grid shape, h, Δt, n_steps, f_max, ppw;
  - memory and the wall-time estimate (same envelope as `runner`);
  - S_grid, radiator path;
  - n_cut, n_floored, δ_eq/h;
  - clearance per radiator;
  - warnings.
- **`runner.run`:** add one branch at the top (the box and baffle path stays unchanged). If `config.geometry == "parametric"` and `dry_run`, return `dry_run_geom`. If it's parametric and not a dry run, raise `NotImplementedError("parametric recording lands in v0.3 step 6b2")`.
- **`geomconfig.check_geometry_v03_guards`:** allow "parametric" when it has **no environment planes**. Planes, STEP, and box configs with extra radiators keep raising.

## 4. New fixture: `configs/showcase/s3p_c09_parametric.json`
C9 written as parametric. Read W, H, D, the radiator shape and dims, x_c, y_c, band, air and numerics from `configs/c09_box.json`.
- **Part:** box with size [W, H, D] at position (−x_c, −y_c, −D/2).
- **Radiator** `radiator`: the same shape, at position (0, 0, 0), normal +z, x_axis +x.
- No environment.

## 5. Tests: `tests/test_geomrun.py`
- **Derive parity:** for `s3p_c09_parametric` vs `derive(c09)`, h, Δt, nx/ny/nz, n_steps, decimation_m, pulse σ/t₀ and realized W/H/D are **exactly equal**. Also, x₀ equals the legacy corner (−x_c − W/2 − i_box·h, −y_c − H/2 − j_box·h, −D − k_box·h) to 1e-12.
- **Operator parity:**
  - The s3p radiator takes the fast path.
  - Its face indices equal the legacy `grid.radiator_face_indices`.
  - Its weights match the legacy weights to ≤ 2e-6.
  - S_grid matches the legacy `radiator_area_grid` to a relative 1e-6.
- **Stepping parity:** 400 steps of s3p via `step_geom` vs the legacy C9 via `kernels.step` (+ BoxBoundary + cpml). Exterior p (β > 0) relative difference ≤ 5e-5. Then repeat with the operator's fast-path weights replaced by the legacy weights: bit-identical (`np.array_equal`) on β > 0 cells.
- **S1, S4 and S6** each run 300 steps via `simulate_geom(n_steps=300)` with no instability. Record the radiator path: S1 and S6 fast (S6 with s = −1), S4 patch.
- **dry_run_geom** works for S1, S4, S5 and S6:
  - S5 warns that radiator 1 is held rigid;
  - S6 reports woofer clearance 0.030 m;
  - S2 raises on planes.
- **Guard update:** change the one existing step-2/3 test that expects parametric to raise on `run(dry_run=True)` so it now expects a summary dict (S1). Planes (S2) still raise.

## 6. Showcase: `showcase/step06b1_geomrun.py` (< 60 s)
- Print `dry_run_geom` summaries for S1, S3p, S4, S5 and S6 (one line each).
- Stepping parity: S3p vs C9 over 400 steps, p(t) 3h above the radiator center. Plot both traces and the relative difference.
- Plot p(t) for S1, S4 and S6 at a point 0.05 m in front of radiator 0 along its normal, for 300 steps.
- Save to `showcase/out/step06b1_geomrun.png`.

## Key numbers for the report
- derive parity (True/False);
- radiator path per case;
- S_grid relative difference vs legacy;
- stepping parity relative difference, and bit-identity with legacy weights;
- per-case n_cut, n_floored, δ_eq/h;
- ms/step for S1, S4, S6.
