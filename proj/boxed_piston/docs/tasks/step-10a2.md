# boxrad v0.3 — step 10a2: C18(ii) BEM tier (pulled forward); remove gmsh from tests

Scope: (1) a BEM reference for the C9 geometry, used to judge both s3p and S4; (2) replace `bempp.shapes.sphere` (which calls gmsh) with an in-house icosphere, and guard against it.

Do not modify: kernel/recorder/derive/result modules, `tests/ref_*.py`. Edit `boxrad/post/bem_ref.py` only additively (§1).

## 1. In-house sphere mesh (licensing)

Add to `boxrad/post/bem_ref.py`:

```python
icosphere(r: float, h: float) -> bempp.Grid
```

Regular icosahedron; repeated midpoint subdivision (each triangle → 4, shared midpoints deduplicated) until the max edge ≤ h; project every vertex to radius r; orient every triangle outward (centroid · normal > 0, assert); `domain_indices` all `RADIATOR_TAG`. Pure numpy.

`tests/test_bem_ref.py`: replace both `bempp.shapes.sphere(r=a, h=h_mesh)` calls with `icosphere(a, h_mesh)`. Keep the tolerances. If a case fails only because of the polyhedral radius (the inscribed mesh is slightly small), report the numbers; do not loosen.

`tests/test_license_guard.py`: add a test, scoped to `proj/boxed_piston` only (gmsh stays allowed elsewhere in the `sim` repo), that scans every `.py` file under `proj/boxed_piston/` (`boxrad/`, `tests/`, `showcase/`, `tools/`), excluding `tests/test_license_guard.py` itself, for:
- `bempp.shapes` and `bempp_cl.api.shapes` (these call gmsh as a subprocess);
- the word `gmsh` in any form (import, subprocess call, string), matched case-insensitively as a whole word.

It fails listing file:line for every hit. Also add a runtime check: after importing `boxrad` and `boxrad.post.bem_ref`, `"gmsh" not in sys.modules`.

## 2. C18(ii) gate script — `showcase/step10a2_c18_bem.py`

**BEM reference (C9 box, unrotated; BEM is rotation-invariant, so one reference serves both s3p and S4):** `box_mesh(0.20, 0.15, 0.25, (0.08, 0.06), origin, h)` with origin = (−0.03 − 0.10, −0.02 − 0.075, −0.25) (C9: radiator at the origin, box center at (−0.03, −0.02)). Read W, H, D, the radiator dims and the offsets from `configs/c09_box.json` and assert they equal these. `solve_neumann_bm(grid, k, 1.0, rho, c)`; `z_bem = sol.zrad()`.

Frequencies: 300, 500, 800, 1000, 1500, 2000, 2500, 3000 Hz. Mesh `h = h_target_from_frequency(f, c, epw=12, h_max=0.02)`, default radiator refinement (rw/8). Cache each solve in `outputs/step10a2/bem_cache/` with the existing `c09_bem_cache_path` / `save_c09_bem_cache` / `load_c09_bem_cache` helpers (reuse on rerun). Print a progress line per frequency (`format_bem_progress_line`).

**BEM self-check (report):** at 1000 and 3000 Hz also solve with epw = 18 and radiator refinement rw/12; print $|z_{12}-z_{18}|/|z_{18}|$. This bounds the BEM's own error.

**Also on-axis far field:** `sol.field_at` at the point (0, 0, 20) m → $H_{\text{bem}} = p/(\rho c v)$.

**FDTD:** the step-10a files `outputs/step10a/{s3p,s4}_h1.h5` (5 mm) and `_h3.h5` (2.5 mm). `Result.z(f)` at the same frequencies; `Result.field([0, 0, 20], f, frame="local")` / (ρc) for the on-axis comparison (both in their local frame, where the radiator normal is +z).

**Table per f (printed and in JSON):** for each of s3p-h1, s3p-h3, S4-h1, S4-h3:
- $e_z = |z_{\text{fd}} - z_{\text{bem}}|/|z_{\text{bem}}|$, plus $\Delta\Re z$ and $\Delta\Im z$ relative to $|z_{\text{bem}}|$ (signed);
- on-axis far field: $20\log_{10}|H_{\text{fd}}/H_{\text{bem}}|$ dB and the phase difference in degrees.

**Gates (C18(ii), replacing C18(i) as the gate; C18(i) becomes report-only):**
- **Tilted no worse than aligned:** at both h, for every f, $e_z(\text{S4}) \le e_z(\text{s3p}) + 0.01$.
- **Far field:** S4 on-axis $|\Delta| \le 0.2$ dB at every f, at h3.
- **Report only:** $e_z$ at h3 for both, against the v0.2 note "edge reactance excess ≈ 1.5–2.6 % at 10–12 cells across".

Output: one `outputs/step10a2/c18_bem.json` and one `c18_bem.png`: (a) Re and Im z vs f for BEM (markers) and the four FDTD files (lines); (b) $e_z$ vs f for the four; (c) on-axis dB difference vs f.

## 3. Docs

`docs/tasks/step-10a2.md` = this file.
