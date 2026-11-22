# boxrad v0.3 — Step 5: small-cell floor, thin-wall detector, energy-stable reference stepper

Spec: design v2 §2.3 (v1 §2.3, §2.5), amendments A1–A4. Branch `boxrad-v0.3`. Paths are relative to `sim/proj/boxed_piston` unless noted.

## Cursor guardrails (read first; these override anything else)
1. **Commands.** The only commands allowed are these, run from `sim/`, in the foreground, one at a time:
   - `date`
   - `/Users/lane/miniconda3/envs/simenv/bin/python -m pytest proj/boxed_piston/tests/test_smallcell.py -q -x --tb=short --durations=5`
   - `MPLBACKEND=Agg /Users/lane/miniconda3/envs/simenv/bin/python proj/boxed_piston/showcase/step05_smallcell.py`
2. **Budget:** at most **3** pytest runs and **1** showcase run.
   - Before each run, write one line: "run k/3: <what changed since the last run>".
   - If the same test fails twice in a row, **do not make a third blind change.** Stop and report your diagnosis.
3. **No background processes** (no `&`, no detached terminals). Wait for each command to finish. If a command runs longer than 3 minutes, stop it and report.
4. **Speed is a requirement:** each test < 15 s, and the whole file < 60 s. A test that would be slow must use a smaller case. Never run the full suite.
5. **Don't touch:** `tests/ref_cutcell.py`, existing tests, solver, kernel, recorder, `Result`, configs. Only the files listed below may be created or changed.
6. **Never loosen a tolerance.** If a check fails, fix the code or report.
7. **End with a report of ≤ 8 lines:**
   - files changed;
   - the pytest summary of each run and its wall time;
   - runs used;
   - start and end (`date`);
   - the key numbers in §5;
   - anything left open.

## 1. `boxrad/smallcell.py` (new; numpy + numba only)
- **`courant_lambda(cfg, derived) -> float`:** λ = cΔt/h, taken from the existing `derive()` output. Read the config and derive code to find Δt and h; don't recompute them differently.
- **`apply_floor(fr: Fractions, lam: float, gamma: float = 1.05) -> (beta_tilde, stats)`:**
  - For cells with β > 0:
    $$\tilde\beta_c=\max\!\Big(\beta_c,\;\gamma\,\frac{\lambda^2}{2}\sum_{f\in\partial c}\alpha_f\Big).$$
    Cells with β = 0 are unchanged (0).
  - `stats`:
    - n_floored;
    - min raw β among cut cells;
    - ΔV = Σ(β̃ − β) h³;
    - A_wall = Σ patch_A;
    - δ_eq = ΔV / A_wall, and δ_eq/h;
    - `floor_status`: "ok" if δ_eq/h ≤ 0.02, "check" if ≤ 0.05, "warn" otherwise;
    - the worst floored cell's index and its β → β̃.
- **`stability_ok(beta_tilde, alpha, lam) -> (bool, worst_margin)`:** checks
  $$\tilde\beta_c\ \ge\ \frac{\lambda^2}{2}\sum_f\alpha_f$$
  for every cell with β̃ > 0, with a relative slack of 1e-12.
- **`thin_wall_cells(fr, geometry, grid, max_thin=0) -> (list_of_cells, stats)`:**
  - **Candidates:** cut cells with A_c > 0.5 h² and |N_c| < 0.2 A_c (opposing wall patches inside one cell).
  - **Confirm** each candidate with a 4³ sub-sample of `mesh.inside` (candidates only). A cell is confirmed thin if its fluid sub-samples form ≥ 2 face-connected components that each touch a different cell face.
  - Raise `ValueError` listing the world positions if the confirmed count > `max_thin`.
  - `stats`: n_candidates, n_confirmed.
- Store `beta_tilde` alongside `Fractions`, either as a returned array or as a new optional field. Don't change existing fields.

## 2. `boxrad/refstep.py` (new; numpy only, for tests and the showcase): energy-stable reference stepper
On a small **closed rigid** domain (no CPML; α = 0 on the domain-boundary faces), with ρ = c = h = 1 and Δt = λ:

$$\tilde\beta_c\,\frac{p_c^{n+1/2}-p_c^{n-1/2}}{\Delta t}=-\sum_{f\in\partial c}s_{c,f}\,\alpha_f\,u_f^{\,n},
\qquad
\frac{u_f^{\,n+1}-u_f^{\,n}}{\Delta t}=-\big(p_{c^+}^{n+1/2}-p_{c^-}^{n+1/2}\big)\ \ (\alpha_f>0).$$

Energy, which must be conserved to round-off:

$$H^{n}=\tfrac12\sum_c\tilde\beta_c\big(p_c^{n+1/2}\big)^2+\tfrac12\sum_f\alpha_f\,u_f^{\,n}u_f^{\,n+1}.$$

- `step(p, u, beta_tilde, alpha, lam, nsteps) -> (p, u, H_history)`, vectorized numpy.
- Solid cells (β̃ = 0) are excluded: their p stays 0, and their faces have α = 0.

## 3. Tests: `tests/test_smallcell.py` (each < 15 s)
- **Floor criterion:** after `apply_floor`, `stability_ok` is True for S1, S4 and S6 at h = 5 mm. A full cell (β = 1, Σα = 6) is stable at λ ≤ 1/√3 with no floor needed.
- **Aligned box (C9):** n_floored = 0 and ΔV = 0.
- **Energy, floored:**
  - Domain: a 14³ grid with h = 1 and x₀ = (−7, −7, −7), containing a rotated cube of side 6 (R_z(37°)·R_x(20°)). Shift it so at least one cut cell has raw β < 1e-3. If none, sweep the shift in 0.05 steps until one does.
  - Random initial p (seed 0), u = 0, λ = 0.99/√3.
  - Run 4000 steps: relative drift max|H − H₀|/H₀ ≤ 1e-11.
- **Energy, unfloored:** the same case with β̃ = β must blow up, with max|p| > 1e6·max|p₀| within 2000 steps (or NaN). This documents why the floor exists.
- **Thin wall:**
  - A plate of thickness 0.4 h, positioned so it lies inside one cell layer, gives confirmed thin cells and a `ValueError` at `max_thin=0`.
  - A plate of thickness 2.5 h gives none.
- **δ_eq** for S1 at h = 5 mm ≤ 0.05 h. Print the value.

## 4. Showcase: `showcase/step05_smallcell.py`
- S1–S6 at h = 5 mm, with λ from the C9 config's `derive()`. Print one line per case:
  - n_cut;
  - min raw β;
  - n_floored;
  - ΔV and δ_eq/h with its status;
  - stability_ok;
  - thin candidates and confirmed;
  - **PASS** if stable, δ_eq/h ≤ 0.05, and 0 confirmed thin cells; otherwise **CHECK**.
- **Figure** (3×3):
  - rows 1–2: per case, a histogram of raw β for cut cells, on a log x-axis from 1e-6 to 1, with the floor level γλ²/2·Σα (median) marked;
  - row 3, one wide axis: relative energy drift vs step for the floored and unfloored reference runs (semi-log). Clip the unfloored curve at 1e6.
- Save to `showcase/out/step05_smallcell.png`, overwritten in place. It must run in < 60 s.

## 5. Key numbers for the report
- λ;
- per case: min raw β, n_floored, δ_eq/h;
- floored energy drift;
- the step at which the unfloored run blew up;
- thin-wall candidates and confirmed on S1–S6.
