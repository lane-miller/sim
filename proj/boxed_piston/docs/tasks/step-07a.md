# boxrad v0.3 — step 7a: environment planes — grid, kernel, recording; gate C19(B′)

Scope: rigid infinite planes (`environment.planes`, ≤ 1 per axis) as **domain boundaries** for parametric runs: anchoring, truncation, CPML off on rigid sides, open Huygens surface, recording. **Image KH in `Result` (field / directivity / DI) is step 7b**; in 7a, `Result.field`, `directivity`, `directivity_index` raise `NotImplementedError("image KH lands in v0.3 step 7b")` for files with planes, while `z`, `Z_*`, `power`, `S_grid`, `summary` work.

Do not modify: `kernels.py`, `source.py`, `recorder.py`, `cpml.py`, `grid.py`, `config.py`, `post/*` (except as listed), `cutkernel.py`, `smallcell.py`, `tests/ref_*.py`. Box/baffle paths unchanged.

## 0. Semantics

Plane `{axis, side, offset}`: the plane $x_a = \text{offset}$ is a rigid wall; `side "-"` → fluid at $x_a \ge$ offset (domain starts at the plane); `side "+"` → fluid at $x_a \le$ offset. A solid may touch a plane; crossing by more than $\varepsilon = 10^{-9}\max(1, L_{\text{bbox}})$ is a `ValueError`.

## 1. Guards

Remove the planes `NotImplementedError` from `check_geometry_v03_guards` and the planes error in `validate_geom`. Keep "at most one plane per axis" (add it if the parser lacks it). Update the tests that expected `"step 7"` (`test_geomconfig.py`, `test_geomrun.py::test_dry_run_showcases`): S2 dry run now succeeds and returns a summary.

## 2. `geomderive.derive_geom` — anchors, truncation, CPML sides

Read `cutcell.grid_for_geometry` first; it already implements plane anchoring, the crossing check and the $h'$ shrink for fraction grids. Mirror that logic (do not call it; margins differ).

1. **Anchors per axis**, priority: plane offset → radiator-0 position if its normal is along this axis → bbox lo (current rule).
2. **Shrink.** If one axis has both a plane and the radiator-0 plane, separated by $d > 0$, and $d/h$ is not an integer within $10^{-9}$:
   $$ h' = \frac{d}{\lceil d/h - 10^{-9}\rceil} , $$
   applied isotropically before any other derived quantity. Keep $h_{\text{nominal}}$.
3. **Extents per side:** absorbing → snapped bbox + `(gap + cpml)` cells (current); rigid → the domain ends **exactly** at the plane (no gap, no CPML). $\mathbf x_0$ = domain low corner; $n_a$ = extent / $h$ (integer by construction; assert within 1e-9).
4. **Run time:** $t_0 + 6\sigma + n_{\text{transits}} L_{\text{dom}}/c$ with $L_{\text{dom}}$ = largest non-CPML domain extent. With no planes this must equal the current formula exactly (s3p ↔ legacy `n_steps` parity stays).
5. `Derived.cpml_sides` = `(x−, x+, y−, y+, z−, z+)`, `False` on rigid sides.
6. **CPML slabs:** `build_cpml_slabs(...)`, then `dataclasses.replace` the inner ranges so a rigid low side has inner lo = 0 and a rigid high side has inner hi = n.
7. `GeomGrid` gains (defaulted, at the end) `rigid_sides: dict[str, bool]` (keys as `cutcell._side_key`, e.g. `"z-"`), `planes: tuple` and `h_nominal: float`. `cut_grid(gg)` passes `rigid_sides`.

`make_cpml_state` already honors `derived.cpml_sides`. The kernel already zeroes the outermost faces each step, so a domain edge is a rigid wall; check that `step_geom` does this on every side (read `geomrun.py` / `cutkernel.py`) and report if it doesn't. Do not change the kernel.

## 3. `geomrec.GeomRecorder` — open Huygens surface

Replace the closed box patches with `huygens_patches_planes(lo_s, hi_s, h, offset_cells, rigid_sides, domain_lo, domain_hi)` (new function in `geomrec.py`):

- Absorbing side: face at the snapped bbox ± `offset_cells·h`, as now.
- Rigid side: **no face**; the other faces extend along that axis to the plane.
- Patch centers on the same staggered lattice as now (normal coordinate on a face plane, tangential at $(i+\tfrac12)h$ from the domain-aligned start). With no planes, the output must equal `huygens_box_patches` exactly (test).

Clearance validation: only absorbing sides need `gap − offset ≥ 2`. Interior validation uses the adjusted slabs.

Additional `meta.attrs`: `environment_json` (planes list as JSON), `n_planes`, `h_nominal`. A probe outside the domain or behind a plane → `ValueError` naming the plane.

## 4. `result.py`

Add a `planes` property (from `environment_json`; `[]` if absent) and the 7b `NotImplementedError` guard above in `field`, `directivity`, `directivity_index` when `planes` is non-empty. No other change.

## 5. Fixtures

From `configs/showcase/s3p_c09_parametric.json` (C9 box, rect radiator 0.08 × 0.06 at the origin, normal +z; symmetric about x = 0 and y = 0), create:

- `configs/showcase/c19_half_x.json`: box halved in x and placed at $x \in [0, W/2]$; radiator `w` halved, centered at $x = w/4$; plane `{axis: "x", side: "-", offset: 0}`.
- `configs/showcase/c19_quarter_xy.json`: also halved in y ($y \in [0, H/2]$, radiator `h` halved, centered at $y = h/4$); planes x− and y− at 0.

Everything else (band, numerics, air) identical to s3p. Read the actual box and radiator values from the s3p file; do not hard-code numbers from this text.

## 6. Tests — `tests/test_planes.py` (fast)

1. **Truncation.** S2 derive: `cpml_sides` false on z− and y−; `x0[2] == 0` (floor) and `x0[1] == -0.425` (wall), to 1e-12.
2. **Grid coincidence.** s3p vs `c19_half_x` vs `c19_quarter_xy` at production h: equal h, dt, `n_steps`, and `nz`; the half grid's x-faces coincide with the full grid's faces at $x \ge 0$ (index offset integer, 1e-12 m); $n_x^{\text{half}}$ = number of full-grid cells with center $x > 0$; same for y in the quarter case.
3. **S_grid.** $S^{\text{half}} = S/2$ and $S^{\text{quarter}} = S/4$ to 1e-9 relative.
4. **Crossing error.** A plane cutting through the s3p box raises `ValueError`.
5. **Shrink.** A synthetic config with a floor at $z = 0$ and an axis-aligned radiator facing +z at a non-commensurate height: $h' \le h$, $d/h'$ an integer to 1e-9, and `h_nominal` recorded.
6. **Huygens open surface.** No planes: identical to `huygens_box_patches`. `c19_half_x`: no patch with normal −x; every patch has $x \ge 0$; total area = closed area − (−x face area) + the extension of the four side faces to the plane (compute the expected value analytically from the extents).
7. **Short-run symmetry parity.** s3p vs `c19_half_x`, 150 steps, production h: `p_avg` max abs diff ≤ $10^{-5}$ × max |p_avg| (float32 round-off; the full run is symmetric only to round-off).
8. **Rigid-edge kernel check.** On the half run after 150 steps, the x-velocity on the plane face (index 0) is exactly 0.

## 7. Gate script — `showcase/step07a_planes.py`

Production settings; keep_awake; progress. Writes one `outputs/step07a/gates.json` and one `outputs/step07a/gates.png` (overwrite). Reuses `outputs/step06b2a/s3p.h5` if present.

Runs: s3p (full), `c19_half_x`, `c19_quarter_xy`, S2 (floor + wall), and **S2-free** (S2 with `environment.planes` removed, built in memory). Add the same probe set to s3p and the half and quarter runs, built in memory: (0.03, 0.02, 0.02), (0.12, 0.05, −0.10), (0.06, 0.09, 0.05).

**C19(B′) gate (symmetry; half and quarter vs full):**

$$
\frac{\max_t |p_{\text{avg}} - p_{\text{avg}}^{\text{full}}|}{\max_t |p_{\text{avg}}^{\text{full}}|} \le 10^{-5}, \qquad
\frac{\max_f |z - z^{\text{full}}|}{\max_f |z^{\text{full}}|} \le 10^{-5}, \qquad
\text{probes: same as } p_{\text{avg}} .
$$

**Power consistency (open faces):** for half, quarter and S2, the step 6b2a definition, with the flux summed over the stored (open) patches only: ratio within 2 %.

**S2 report (closed-form low-frequency reference, CHECK not gate):** print $\Re z_{\text{S2}} / \Re z_{\text{S2-free}}$ at 30, 50, 80, 120, 200, 400 Hz, next to the compact-monopole image formula. With the woofer center $\mathbf s$ and its three images $\mathbf s_m$ (floor, wall, both):

$$
\frac{\Re z_{\text{planes}}}{\Re z_{\text{free}}} \approx 1 + \sum_{m=1}^{3} \frac{\sin k d_m}{k d_m}, \qquad d_m = |\mathbf s - \mathbf s_m| ,
$$

which → 4 as $f \to 0$. Expect agreement within a few percent at the lowest frequencies, worsening as $ka$ grows (box scattering, finite driver).

Plots: (a) $p_{\text{avg}}(t)$ for full, half and quarter, plus the difference curves on a log axis; (b) $\Re z$ ratio S2 / S2-free vs f, with the image formula and the line at 4.

## 8. Docs

`docs/tasks/step-07a.md` = this file.
