# boxrad v0.3 — step 8b: multi-radiator Result API; gates C19(B) mirror-and-sum, C20(iii)

Scope: `Result` accessors for schema-2 files (one file per driven radiator, step 8a), superposition, `inside()`. No kernel, recorder or derive changes.

Do not modify: kernel/recorder/derive files, `post/kirchhoff.py`, `post/transfer.py`, `post/zrad.py`, `post/finalize.py`, `tests/ref_*.py`.

## 1. `Result` additions (all additive; schema-1 and box/baffle files behave as before)

- `names -> list[str]`: from `radiator_names_json`; schema-1 files → `["radiator"]`.
- `radiator(key: int | str) -> Result`: the `Result` of the file that drives that radiator, opened with the same `frame` default. Index 0 or radiator 0's name → `self`. Siblings are resolved from `run_files_json`, relative to `self.path.parent`. Radiator not driven (`null`) → `ValueError("radiator <name> was not driven in this run set")`; sibling missing on disk → `FileNotFoundError` naming the expected path. When called on a sibling, resolve through the primary: the primary is the file whose `driven_index == 0`, found as the `run_files_json[0]` basename in the same folder.
- `driven_index -> int` (0 for schema-1).
- `S_list -> np.ndarray` (N,), from `radiator_S_json` (schema-1: `[S_grid]`).
- `Z_matrix(f, *, allow_out_of_band=False) -> np.ndarray` (F, N, N) complex128, Pa·s/m³:
  $$Z_{ij}(f)=\frac{\rho c\, z_{ij}(f)}{S_j},\qquad z_{ij}=\frac{P_i(f)}{\rho c\,V(f)}\ \text{from } \texttt{radiator/p\_avg\_all}[:, i] \text{ of the file driving } j .$$
  Spectra at the requested f, with the same estimator as `post/zrow.py` (add a `zrow_at_frequencies(path, f)` there if it only supports the stored log grid). Columns of undriven radiators are NaN. Docstring: raw mutual data, `mutual_status = "recorded_unvalidated"`.
- `field_sum(points, f, U=None, *, frame="world", allow_out_of_band=False) -> np.ndarray` (F, M), Pa:
  $$p(\mathbf x, f)=\sum_j \frac{U_j(f)}{S_j}\,H_j(\mathbf x, f),$$
  where $H_j$ = `radiator(j).field(points, f, frame=frame)` (Pa per m/s of radiator j). U is the volume velocity, shape (N,) or (F, N); default ones (equal, in phase; for plotting). A nonzero $U_j$ for an undriven radiator → `ValueError`. `frame="local"` means radiator 0's local frame.
- `inside(points, *, frame="world") -> bool array`: True if the point is inside the solid (rebuild `build_geometry(self.config)` once, cached; use the geometry's existing inside test) **or** behind a rigid plane. Box files: inside the box $[-W/2, W/2]\times[-H/2, H/2]\times[-D, 0]$ shifted by the radiator offset (use the box geometry built by `build_geometry`, which already handles this).
- `poses -> list[tuple[R, t]]` for all radiators, from the rebuilt geometry (same R construction as `pose`).

## 2. Tests — `tests/test_result_multirad.py` (fast: coarse h 0.01, ≤ 150-step truncated runs written with the 8a runner pieces into `tmp_path`, both radiators of S2-free driven)

1. `names == ["woofer", "tweeter"]`; `radiator(1)` and `radiator("tweeter")` open the sibling; `radiator(0) is self`; `radiator(1).radiator(0)` resolves back to the primary's path.
2. Missing sibling (delete it) → `FileNotFoundError` with the expected path in the message; an undriven radiator (drive list = woofer only) → `ValueError`.
3. `Z_matrix` at 3 frequencies that lie exactly on the stored log grid equals $\rho c\,z/S_j$ built from `results/zrad_row` of each file to 1e-12 relative.
4. `field_sum` with U = (1, 0) equals `field(...)/S_0`, and with U = (0, 1) equals `radiator(1).field(...)/S_1`, to 1e-12; with U = (1, 1) it is their sum (linearity).
5. `inside`: box center True; a point 1 cm outside a face False; for a file with a floor plane, a point at z = −0.01 True.

## 3. Fixtures (for C19(B))

- `configs/showcase/c19b_floor.json`: the S2 box on a floor plane only (no wall), **woofer only** (tweeter entry removed), `numerics.h_override = 0.005`.
- `configs/showcase/c19b_floor_mirror.json`: free field (no planes); the box and its mirror image through z = 0 as **one** box, size [0.20, 0.25, 0.60] centered at z = 0 (x, y as S2); radiators `woofer` at the S2 woofer pose and `woofer_m` at the same x, y with z = −0.10 and the same normal (+y). `h_override = 0.005`. Both radiators driven.

Read the S2 values from `s2_bookshelf.json`; do not hard-code them from this text. At h = 5 mm the grids coincide on z ≥ 0 (0.30/h is an integer); assert this in the gate script (equal h; z = 0 is a face of the mirror grid).

## 4. Gate script — `showcase/step08b_result.py`

One `outputs/step08b/gates.json` (numeric values) and one `gates.png` (overwrite). keep_awake; progress.

**C19(B) mirror-and-sum (floor), gate.** Run `c19b_floor` (1 run) and `c19b_floor_mirror` (2 runs). Superposition with equal velocities, in the time domain, over the common length $T=\min$ of the two records:
$$p_{\text{avg}}^{\text{floor}}(t) \;\text{vs}\; p_{\text{avg},0}^{(\text{drive }0)}(t)+p_{\text{avg},0}^{(\text{drive }1)}(t),$$
where the second term is `p_avg_all[:, 0]` of the `woofer_m` file. Also probes at (0.05, 0.30, 0.12) and (−0.08, 0.40, 0.25) (add them in memory to both configs; equal velocities, summed the same way). PASS: max abs difference ≤ $10^{-5}\max|p|$ (float32 round-off). Report the far field too: `field_sum` of the mirror pair with $U = (S_0, S_1)$ (equal velocities) at 6 points at r = 3 m in z > 0, vs `field` of the floor file; print the max dB and degrees (report only, because the run lengths differ).

**C20(iii) power, report.** S2-free from `outputs/step08a/` (rerun if missing). For U = (1, 1) and (1, −1) (volume velocities) at 8 log-spaced f in [0.1 f_max, f_max]:
- the Huygens-flux power of the summed field, $W_{\text{huy}}=\tfrac12\Re\sum_q A_q\,p_q v_{n,q}^*$, with $p_q=\sum_j (U_j/S_j)\,P^{(j)}_q/V^{(j)}$ (same for $v_n$), using each file's Huygens spectra (`load_huygens_spectra`; both files share the same surface, so assert the centers are equal);
- the circuit power $W_Z=\tfrac12\Re\,(U^{\mathsf H} Z U)$ from `Z_matrix`.

Print $W_{\text{huy}}/W_Z - 1$. Expectation: within 2 % (as in 6b2a).

Plots: (a) the C19(B) p_avg traces and their difference (log axis); (b) the C20(iii) ratio vs f for both U.

## 5. Docs

`docs/tasks/step-08b.md` = this file.
