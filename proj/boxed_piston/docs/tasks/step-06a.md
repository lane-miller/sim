# boxrad v0.3 — Step 6a: cut-cell kernel (float32), gated against v0.2

Decision 6-0 = **B**: the v0.2 box and baffle path is **untouched**. This adds a second kernel path for parametric and STEP geometry, gated against v0.2. Prototype verified by Claude (reference: `tests/ref_cutkernel.py`):
- **T1:** bit-identical to a v0.2 replica outside the box over 300 steps;
- **T2:** patch flux vs face path ≤ 1e-6;
- **T3:** float32 energy drift 2.5e-7 over 4000 steps.

Paths are relative to `sim/proj/boxed_piston` unless noted.

## Cursor guardrails (these override everything else)
1. **Commands.** The only commands allowed are these, run from `sim/`, in the foreground, one at a time:
   - `date`
   - `/Users/lane/miniconda3/envs/simenv/bin/python -m pytest proj/boxed_piston/tests/test_cutkernel.py -q -x --tb=short --durations=5`
   - `MPLBACKEND=Agg /Users/lane/miniconda3/envs/simenv/bin/python proj/boxed_piston/showcase/step06a_kernel.py`
2. **Budget:** at most **3** pytest runs and **1** showcase run. Before each run, write "run k/3: <what changed>". If the same test fails twice in a row, stop and report your diagnosis; no third blind change.
3. **No background processes.** Wait for each command to finish. If one runs longer than 3 minutes, stop it and report.
4. **Speed:** each test < 20 s, and the file < 90 s. Never run the full suite.
5. **Don't touch:** `kernels.py`, `source.py`, `cpml.py`, `grid.py`, `recorder.py`, `runner.py`, `config.py`, existing tests, or any `tests/ref_*.py`. Only the files listed below may be created or changed.
6. **Never loosen a tolerance.** Fix the code or report.
7. **End with a report of ≤ 8 lines:**
   - files;
   - each pytest run's summary and wall time;
   - runs used;
   - start and end (`date`);
   - the key numbers (§4);
   - anything left open.

## 1. Read first
- `kernels.py`: the `step` order, `update_v`, `update_p` and `zero_rigid_face_velocities`.
- `source.py`: `BoxBoundary.apply` and `apply_radiator`.
- `cpml.py`: `apply_v` and `apply_p`, and how the cpml object is built.
- `runner.py`: how Fields, coeffs, cpml and BoxBoundary are assembled for a box run.
- `tests/ref_cutkernel.py`: the verified semantics.

## 2. `boxrad/cutkernel.py` (new; numba, float32 fields; **no float64 inside kernels**)
- **`CutOperator`** (frozen dataclass):
  - `afl_x, afl_y, afl_z`: float32 face weights = α, **with fast-path radiator faces set to 1.0**.
  - `open_x, open_y, open_z`: float32 {0, 1} = (afl > 0).
  - `inv_beta`: float32 = 1/β̃ where β̃ > 0, else 0.
  - **Fast-path radiators** (axis-aligned flat radiators), per radiator:
    - axis (0/1/2);
    - flat face indices into vx, vy or vz;
    - **signed** float32 weights s·w, with s = sign(n̂_axis). A down-firing radiator (n̂ = −ẑ) has s = −1.
  - **Patch-flux radiators** (all others), per radiator: flat cell indices and float32 coefficients a_{r,c}/h².
- **`apply_open_mask(vx, vy, vz, open_x, open_y, open_z)`:** v *= open.
- **`set_fastpath_faces(v_flat, idx, sw, v_now)`:** v[idx] = sw·v_now.
- **`update_p_cut(p, vx, vy, vz, afl_x, afl_y, afl_z, cp, inv_beta)`:** skip cells with inv_beta = 0. Otherwise:
  ```
  div = afl_x[i+1,j,k]*vx[i+1,j,k] - afl_x[i,j,k]*vx[i,j,k] + afl_y[i,j+1,k]*vy[i,j+1,k] - afl_y[i,j,k]*vy[i,j,k] + afl_z[i,j,k+1]*vz[i,j,k+1] - afl_z[i,j,k]*vz[i,j,k]
  p[i,j,k] -= (cp*inv_beta[i,j,k]) * div
  ```
  **Use exactly this left-to-right expression,** the same association as `kernels.update_p`. Use `parallel=True, fastmath=False`, as v0.2 does.
- **`add_patch_flux(p, cells, coef, cp, inv_beta, v_now)`:**
  ```
  p.flat[c] -= (cp*inv_beta.flat[c]) * (-coef[q]*v_now)
  ```
  This is applied after `update_p_cut` for patch-flux radiators. It's algebraically the same as subtracting coef·v from div; T2 tolerance covers the float32 order difference.
- **`step_cut(fields, coeffs, op, cpml, v_now_list)`.** One step, in this order:
  1. `zero_rigid_face_velocities`;
  2. `kernels.update_v` (full domain);
  3. `cpml.apply_v` (if cpml);
  4. `zero_rigid_face_velocities`;
  5. `apply_open_mask`;
  6. for each fast-path radiator, `set_fastpath_faces`;
  7. `update_p_cut`;
  8. for each patch radiator, `add_patch_flux`;
  9. `cpml.apply_p` (if cpml).
- **`build_operator(fractions, beta_tilde, radiators_fast, radiators_patch, h)`:** builds float32 arrays from step-4/5 outputs (snapped α, floored β̃).
- **`assert_geometry_clear_of_cpml(op, slabs)`:** in every CPML slab, open = 1, inv_beta = 1, and afl = 1. Raise if geometry intrudes.

## 3. Tests: `tests/test_cutkernel.py`
- **C12a′ kernel gate (bit-identity vs real v0.2).**
  - Build the C9 config exactly as `runner.py` does: Grid, Fields, coeffs, cpml, BoxBoundary.
  - Build a `cutcell.Grid` aligned to the legacy grid. Its x₀ is the float64 corner of p-cell (0,0,0): `-xc - W/2 - i_box*h`, `-yc - H/2 - j_box*h`, `-D - k_box*h`. Use the same h and shape.
  - Compute fractions of `from_box(cfg)` on it, and β̃ (no floor needed: no cut cells).
  - **Radiator:** fast path, using the **legacy** `grid.radiator_face_indices` and `grid.radiator_weights` (s = +1).
  - Run 400 steps of `kernels.step` (legacy) and of `step_cut` (new) from zero fields, with the same pulse v_now = `BoxBoundary.v_at_step(n)`.
  - **Assert** the pressure equals bitwise on all cells with β > 0, and v equals bitwise on all faces with open = 1. Use `np.array_equal`. The box interior is excluded: v0.2 also drives the sealed interior through the radiator's back face, and that never couples out.
- **C12a′ exact weights:** the same run, but with fast-path weights from `cutcell` (a/h², exact rect clip). Exterior p relative difference ≤ 1e-5 (the float32 centers in v0.2 shift the edge weights by about 1e-6).
- **T2, patch vs fast path:**
  - Closed rigid 20³ domain, h = 1, box cells [6,14)×[7,13)×[4,12), 5×5 rect radiator centered at (10,10) on the top face.
  - Fast path vs patch flux (cells directly above the faces, coef = w).
  - 300 steps, cv = 0.5 and cp = 0.99/1.5.
  - Exterior p max relative difference ≤ 2e-6 (verified 9.9e-7).
- **T2-down, the down-firing sign check:** the same as T2, with the radiator on the box **bottom** face (n̂ = −ẑ, s = −1, patch cells directly below). Fast vs patch ≤ 2e-6, and the pressure in the cell just below the radiator center is **positive** at the pulse peak.
- **T3, energy:**
  - Rotated cube of side 6 (R_z(37°)·R_x(20°)), 14³ grid, h = 1, x₀ = −7, closed rigid domain (α = 0 on the outer faces).
  - β̃ from `smallcell.apply_floor` at λ = 0.99/√3; cv = cp = λ.
  - Random p (seed 0), v = 0, 4000 steps.
  - Energy, accumulated in float64 from the float32 fields:
    $$E^n=\tfrac12\sum\tilde\beta\,(p^n)^2+\tfrac12\sum \mathrm{afl}\;v^{n-1/2}v^{n+1/2}$$
    using p before the step and v before and after `update_v` plus the mask. Require relative drift ≤ 1e-6 (verified 2.5e-7).
- **CPML guard:** shifting a box into the CPML raises in `assert_geometry_clear_of_cpml`.

## 4. Showcase: `showcase/step06a_kernel.py` (< 60 s)
- C9, 400 steps, legacy vs `step_cut` (legacy weights). Plot p(t) at the cell 3h above the radiator center for both, plus |difference| (expect 0) on a log axis with zeros floored at 1e-12.
- The T3 energy drift vs step.
- Print:
  - bit-identity (True/False);
  - the exact-weight relative difference;
  - T2 and T2-down differences;
  - T3 drift;
  - legacy vs `step_cut` time per step (ms).
- Save to `showcase/out/step06a_kernel.png`, overwritten in place.

## Key numbers for the report
- C12a′ bit-identity (True/False);
- exact-weight relative difference;
- T2 and T2-down;
- T3 drift;
- time per step, legacy vs cut (ms).
