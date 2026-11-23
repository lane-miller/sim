# boxrad v0.3 — step 8a: multi-radiator runs (engine + files); gate C20(i)

Scope: parametric configs with N radiators run **once per driven radiator**. Each run drives radiator j (the others are rigid surface), records the pickups of **all** radiators, and writes one complete single-radiator HDF5 file. The `Result` API for these files (`names`, `radiator()`, `Z_matrix`, `field_sum`, `inside`) is step 8b. Box/baffle paths unchanged.

Do not modify: `kernels.py`, `source.py`, `recorder.py`, `cpml.py`, `grid.py`, `config.py`, `post/kirchhoff.py`, `post/transfer.py`, `post/zrad.py`, `post/finalize.py`, `cutkernel.py`, `cutcell.py`, `smallcell.py`, `tests/ref_*.py`.

## 0. Files

For output path `out.h5` and drive list J (radiator indices in config order):

- j = 0 → `out.h5` (the legacy view; layout exactly as today plus the attrs below);
- j > 0 → `out.r{j}_{name}.h5` (`name` = radiator name, sanitized to `[A-Za-z0-9_-]`).

Every file is a complete single-radiator file: `radiator/p_avg`, `huygens/…`, `probes/…`, `results/zrad` refer to **that file's driven radiator**. The `radiator0_*` pose attrs and `S_grid` in that file describe its driven radiator (as `Result.pose` already assumes).

New `meta.attrs` in every file:

| attr | value |
|---|---|
| `schema_version` | 2 |
| `n_radiators` | N |
| `radiator_names_json` | JSON list of names, config order |
| `radiator_S_json` | JSON list of $S_i = \sum_c a_{i,c}$ (m²) |
| `driven_index`, `driven_name` | j, name |
| `mutual_status` | `"recorded_unvalidated"` |

In `out.h5` only: `run_files_json` = JSON list, length N, of sibling **basenames** (`"out.h5"` for j = 0, `null` for radiators not driven).

## 1. Drive list

Optional config key `numerics.drive_radiators`: list of radiator names; default = all. Radiator 0 is always included (added if missing, with a stderr note). Unknown name → `ValueError`. Parse it so `Config.to_json` round-trips it (box/baffle serialization unchanged).

## 2. `geomop.build_run_operator(..., driven: int = 0, cache=None)`

- Generalize `_build_radiator_0` to `_build_radiator(j)`: the same fast/patch logic for radiator j. Radiators ≠ j are simply rigid surface (nothing to build).
- `driven=0` must produce exactly today's operator (test: all `CutOperator` arrays equal).
- `cache`: when given, reuse its `fractions`, `beta_tilde`, floor/thin-wall stats (computed once per config), so runs j > 0 skip `compute_fractions` and the floor.
- Remove the "radiators 1.. held rigid" warning.

`geomrun.prepare_geom_run(cfg, driven=0, cache=None)` passes these through and returns a run object whose `info["driven"] == j`, plus the cache for reuse.

## 3. `GeomRecorder(..., driven: int = 0)`

- Driven-radiator pickup `radiator/p_avg`: the existing formula with radiator j's cells, weights and centroids (unchanged for j = 0).
- **All-radiator pickups** (new): `radiator/p_avg_all`, float64, shape (T, N), same time levels as `p_avg`. Column i:
  $$\langle p\rangle_i=\frac{\sum_c w_{i,c}\,\big(p_c+\delta_{ij}\,\rho\,d_{i,c}\,w_{i,c}\,\dot v\big)}{\sum_c w_{i,c}},\qquad w_{i,c}=a_{i,c}/h^2 .$$
  The $\dot v$ correction applies only to the driven radiator (the others do not move). Column j must equal `p_avg` exactly.
- **Mutual row** (new): after the existing finalize, write `results/zrad_row`, complex128 shape (F, N) on the same log grid as `results/zrad`:
  $$z_{ij}(f)=\frac{P_i(f)}{\rho c\,V(f)} ,$$
  using the **same** discrete spectral estimate as `post/zrad.py` (read it and mirror it in a new `boxrad/post/zrow.py`; do not edit `zrad.py`). Column j must equal `results/zrad` to 1e-12 relative. Attr: `convention = "z_ij = <p>_i / (rho c v_j), driven j = meta.driven_index; Z_ij = rho c z_ij / S_j"`.
- The pose attrs and `S_grid` are those of radiator j.

## 4. `runner.run` (parametric)

- Loop over the drive list: build the cache once, then for each j: `prepare_geom_run(cfg, driven=j, cache)`, `GeomRecorder(..., driven=j)`, step loop, close. Progress lines are prefixed `run k/K (name)`. The stability guard and abort attrs are per file.
- After all runs, write `run_files_json` into `out.h5` (append mode).
- Returns the path of `out.h5`.
- Dry run: adds `n_runs`, the per-run and total wall-time and output-size estimates, and the list of planned files.

## 5. Tests — `tests/test_multirad.py` (fast: coarse `h_override` 0.01, ≤ 150 steps, truncated loops as in `test_planes.py`)

1. **Operator parity.** S2-free (S2 with planes removed, in memory): `build_run_operator(driven=0)` arrays equal today's (call the old path via `driven=0` with no cache vs with cache: all arrays equal).
2. **Tweeter operator.** S2-free `driven=1`: fast path; the set of face indices with $w > 10^{-5}$ is exactly the tweeter's cells; $\sum w\,h^2 = S_1$ to 1e-12 relative.
3. **C20(i) bit identity.** S2-free with both radiators vs S2-free with the tweeter entry removed from `radiators`: run 0 for 150 steps; `radiator/p_avg`, `huygens/p_face`, `huygens/vn` are **bit-identical** (`np.array_equal`).
4. `p_avg_all[:, j] == p_avg` exactly, for j = 0 and j = 1.
5. `results/zrad_row[:, j]` equals `results/zrad` to 1e-12.
6. **Runner files.** `run()` on coarse S2-free with `n_transits` small writes `out.h5` and `out.r1_tweeter.h5`; `run_files_json == ["out.h5", "out.r1_tweeter.h5"]`; `driven_index` 0 and 1; `radiator0_position` of the sibling = the tweeter position.
7. **Drive subset.** `drive_radiators: ["tweeter"]` → runs radiators 0 and 1 (0 auto-added); unknown name → `ValueError`.

## 6. Gate script — `showcase/step08a_multirad.py`

Production settings, S2-free (both radiators driven, 2 runs), keep_awake, progress. Writes one `outputs/step08a/gates.json` (numeric values) and one `outputs/step08a/gates.png` (overwrite).

- **C20(i):** top-level `radiator/p_avg` and `huygens/p_face` of the new `out.h5` vs `outputs/step07a/s2_free.h5` (same config, run 0): max abs difference; PASS if exactly 0. (If the 7a file is missing, skip with a note.)
- **Reciprocity (report):** with $Z_{ij}=\rho c\,z_{ij}/S_j$, $\max_f |Z_{01}-Z_{10}|/\max_f|Z_{01}|$, and the same split into Re and Im.
- **Low-frequency mutual check (report, closed form):** at 50, 80, 120, 200, 300 Hz, $\Re z_{ij}/\Re z_{jj}$ for (i, j) = (1, 0) and (0, 1), next to $\sin kd/kd$ with $d = |\mathbf s_0-\mathbf s_1|$ from the config (S2: 0.15 m).
- Plots: (a) Re and Im of $Z_{00}, Z_{11}, Z_{01}, Z_{10}$ vs f (log f); (b) the low-f ratio vs sinc(kd).

## 7. Docs

`docs/tasks/step-08a.md` = this file.
