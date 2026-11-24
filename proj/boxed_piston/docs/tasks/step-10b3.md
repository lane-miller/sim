# Step 10b3: automatic cfl_safety = 0.7 for cut-cell grids

## Why (C13a/C16 finding, step 10b2)
- **The floor's added volume acts as a compliant wall.** The small-cell floor (target γλ²Σα/2) adds a volume ΔV that behaves like a compliant wall layer of thickness δ_eq = ΔV/A_wall:
  $$\frac{\partial p}{\partial r} + k^2\delta_{eq}\,p = 0.$$
  At λ = 0.99/√3, δ_eq/h ≈ 0.043, which gave first-order convergence (p = 1.17) and 0.31 dB of C13a error at ppw 20.
- **cfl_safety = 0.7 fixes it.** δ_eq/h drops to ≈ 0.003, and C13a/C16/shift all pass (0.085 dB / 0.57°, p = 1.82, spread 0.12 %), for about 1.41× the steps.
- **Decision (Lane, option A):** default 0.7 whenever the grid has cut cells; everything else is unchanged.

## Rule
- `numerics.cfl_safety` becomes optional: `None` = auto, which is the dataclass default.
- **Resolution order:**
  1. **User value set** (any number in the config, including 0.99): use it, `source = "user"`.
  2. **Box / baffle path** (legacy derive): 0.99, `source = "default"`. **Must be bit-identical to today.**
  3. **Parametric / STEP path:** build fractions first (they do not depend on dt).
     - If `n_cut_cells > 0`: 0.7, `source = "auto_cutcell"`.
     - Otherwise: 0.99, `source = "default"`.
- The constants live in one place: `CFL_DEFAULT = 0.99`, `CFL_CUTCELL = 0.7`.

## Implementation notes
- **Where to resolve:** in `prepare_geom_run` (and `dry_run_geom`), after fractions and before the floor/operator.
  - If the resolved value differs from the provisional one used by `derive_geom`, call `derive_geom` again with the resolved value. dt, n_steps, decimation and the pulse-dependent fields must all be consistent.
  - Then compute the floor with λ = c·dt/h from the final dt (already the case, per 10b2).
  - Do not build fractions twice if avoidable. If the cleanest route is to pass the resolved cfl into `derive_geom` as an argument, do that.
- **`Derived` / meta:** add `cfl_safety_resolved` (float) and `cfl_safety_source` (str). Write both to the HDF5 meta (accuracy section or meta attrs, wherever `ppw_realized` lives). `dry_run_geom` prints them.
- **`Config.to_json`:** `None` must round-trip as `None` (absent or null), never as 0.99. Explicit values round-trip unchanged.
- **Every consumer of `cfl_safety`** (legacy derive, geom derive, pulse params, decimation, tests, showcases): treat `None` via the resolution rule. A plain `None` must never reach arithmetic.
- **Fixtures:** do not edit any existing config or fixture. Fixtures that pin 0.99 keep 0.99.

## Hard rules
- No kernel, floor-formula, CPML, KH or recorder changes.
- Box/baffle results are bit-identical (C12 and C1–C11 unchanged).
- Additive meta only.

## Tests: `tests/test_cfl_auto.py`
1. **Legacy box config without `cfl_safety`:** derive gives the same dt as with an explicit 0.99 (exact equality); `source == "default"`.
2. **Parametric sphere** (`configs/c13_sphere_ps.json` geometry with the key removed, h = 0.05/9): `cfl_safety_resolved == 0.7`, `source == "auto_cutcell"`, dt = 0.7·h/(c√3) exactly, and the floor stats give δ_eq/h < 0.01.
3. **Same sphere with explicit `cfl_safety` 0.99:** resolved 0.99, `source == "user"`, δ_eq/h > 0.03.
4. **Grid-aligned parametric** (`c19_sym_full` with the `cfl_safety` key removed): `n_cut_cells == 0`, resolved 0.99, `source == "default"`. If `c19_sym_full` does have cut cells, report that and use a plain aligned box part instead.
5. **`to_json` round trip** with the key absent stays absent/None; with 0.7 stays 0.7.
6. **`dry_run_geom`** on the auto sphere config reports resolved 0.7.

## Report back
- Where the resolution happens (file:function).
- Whether `derive_geom` is called once or twice.
- Files changed.
- Which existing fixtures pin `cfl_safety` (list them; no edits).
