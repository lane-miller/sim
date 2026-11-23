# boxrad v0.3 — step 6b2a: parametric recording, finalize, Result S_grid, end-to-end gates

Scope: record a parametric (cut-kernel) run into the **same HDF5 layout** as v0.2, reuse finalize, make `Result` load it in the **world frame**, and wire `runner.run` for parametric configs. Frame transforms / legacy local view are **out of scope** (step 6b2b).

Do not modify: `kernels.py`, `source.py`, `recorder.py`, `cpml.py`, `grid.py`, `config.py`, `post/kirchhoff.py`, `post/transfer.py`, `post/zrad.py`, `cutkernel.py`, `geomop.py`, `geomderive.py`, any `tests/ref_*.py`. Box/baffle code paths must stay byte-for-byte behaviorally identical.

## 1. `cutcell.py` — radiator centroids (additive)

Add a field at the **end** of `Fractions`:

```python
radiator_centroid: list[np.ndarray] = field(default_factory=list)   # per radiator, (n_cells, 3) float64, world m
```

(`from dataclasses import field`.)

In `_accumulate_radiator`, add an output array `m_cell` of shape `(ncell, 3)` float64. At every site where `a_cell[idx] += area * edot`, also add `m_cell[idx, :] += area * edot * (cx, cy, cz)`:

- on-plane branches (three axis branches): take `cx, cy, cz` from `_poly_area_centroid(poly_a, nv)` (currently discarded as `_`);
- general branch: centroid of the sub-triangle = mean of its three vertices.

In `compute_fractions`: `radiator_centroid.append(m_c[mask] / a_c[mask, None])`. Nothing else in `cutcell.py` changes.

## 2. `boxrad/geomrec.py` — `GeomRecorder` (new)

Same public interface as `Recorder` (`sample(fields, n)`, `close()`, `decimation`, `_file`, `_tail_db`). Constructor:

```python
GeomRecorder(cfg, derived, gg: GeomGrid, geometry, op_info: dict, path, *, offset_cells=None, buffer_samples=256)
```

Reuse from `boxrad.recorder` (import, do not copy): `_ChunkWriter`, `_STENCIL_W4`, `_CONVENTIONS`, `resolve_decimation_factor`, `resolve_huygens_offset`, `validate_huygens_clearance`. Reuse `velocity_pulse`, `velocity_at_half_step`, `velocity_pulse_derivative` from `boxrad.source`, and `compute_finalize_results`, `write_finalize_meta` from `boxrad.post.finalize`. Read `geomrun.py` first and use exactly the pulse `step_geom` drives with.

### Index mapping (world → grid)

With $\mathbf{x}_0$ = `gg.x0`, cell centers at $\mathbf{x}_0 + (i+\tfrac12)h$:

$$
i_p = \left\lfloor \frac{x - x_{0}}{h} \right\rfloor, \qquad
i_v^{(a)} = \operatorname{round}\!\left(\frac{x_a - x_{0,a}}{h}\right)\ \text{(normal axis of a velocity face)} .
$$

Out-of-bounds or CPML-slab indices raise `ValueError` (same rule as `validate_huygens_patches_interior`, written against `gg.cpml_slabs`).

### Huygens surface

Snapped solid bbox: $\mathbf{lo}_s = \mathbf{x}_0 + (\text{gap}+\text{cpml})h$, extents `(realized_W, realized_H, realized_D)`. Patches = `huygens_box_patches(W, H, D, h, origin=lo_s, offset_cells=offset)`. Validate with `validate_huygens_clearance(derived, offset, geometry="box")`. Gathers (`idx4`, `iv`, `sgn`, `vaxis`) built with the mapping above, same semantics as `_build_huygens_gathers`.

### Radiator pressure pickup (one formula, both kernel paths)

Radiator 0 cells $c$ from `fractions.radiator_cell[0]`, $a_c$ = `radiator_a[0]`, centroids $\mathbf{g}_c$ = `radiator_centroid[0]`, unit normal $\hat n$ = `pose.R[:, 2]`, cell centers $\mathbf{x}_c$. Precompute

$$
w_c = \frac{a_c}{h^2}, \qquad d_c = (\mathbf{x}_c - \mathbf{g}_c)\cdot\hat n ,
$$

and at each sample ($t_p = (n+1)\Delta t$)

$$
p_{\text{avg}} = \frac{\sum_c w_c\left(p_c + \rho\, d_c\, w_c\, \dot v(t_p)\right)}{\sum_c w_c} .
$$

For an axis-aligned radiator on a grid face, $d_c = h/2$ and this is exactly the v0.2 formula. For tilted radiators the correction is an $O(h)$ reactance approximation (Re z unaffected); document that in the docstring.

### Probes

`cfg.output.probes` in world m → nearest cell center via $i_p$. Raise `ValueError` if a probe cell has `beta_tilde == 0` (inside solid). Same datasets as v0.2 (`requested`, `centers`, `p`, `times_p`).

### HDF5 layout

Identical dataset names, dtypes, sampling and decimation to `Recorder`: `meta/config_json`, `meta/derived_json`, `radiator/{v_pulse_integer, v_pulse_half, p_avg, times_p, times_v_half}`, `huygens/{centers, normals, areas, p_face, vn, times_p, times_vn}`, optional `probes/…`, `meta/sample_times_decimated_{p,v}`, `meta.attrs n_steps, n_samples_decimated`. **No `box_surface` group.** Additional `meta.attrs`:

| attr | value |
|---|---|
| `geometry` | `"parametric"` |
| `S_grid` | `op_info["S_grid"]` (m²) |
| `frame` | `"world"` |
| `radiator0_position`, `radiator0_normal`, `radiator0_x_axis` | float64 (3,), world |
| `kernel_path` | `op_info["path"]` (`"fast"` / `"patch"`) |
| `geom_stats_json` | JSON of fractions/floor/thin-wall stats (drop arrays) |

`close()` mirrors `Recorder.close()` (finalize on the closed file), then overwrites `meta.attrs["conventions"]` with:
`"e^{+jωt}; z = <p>_rad/(ρc v); H = p/v [Pa/(m/s)]; coordinates: world frame of the config; θ from +z, φ from +x; directivity phase referenced to the world origin"`.

If `post/finalize.cells_across_radiator` fails for parametric configs, make it additive: when `cfg.geometry_spec is not None`, return `geomderive._radiator_D_min(spec) / derived.h`. No other finalize change.

## 3. `runner.py`

Replace the parametric `NotImplementedError` with a full run mirroring the box loop: `prepare_geom_run` → `GeomRecorder` → for each step `step_geom` + `rec.sample` + progress + stability guard (same abort attrs) → `finally` wall time / thread count / git commit attrs + `close()` → tail warning. Same `out_path` resolution. `plot=True` for parametric: print `"plot not available for parametric geometry yet"` to stderr and skip.

## 4. `result.py`

`S_grid`: if `"S_grid"` in `meta.attrs`, return it; else the v0.2 path (unchanged). Check that `Result.config` round-trips a parametric `config_json`, and that `kh_huygens_pressure_at_points(..., geometry="parametric")`, `_image_plane_z`, `_too_close_huygens`, `directivity_index` treat parametric as a closed surface with no image plane (fix only by treating `"parametric"` like `"box"`; no other behavior change). `surface_p` keeps raising for files without `box_surface`.

## 5. Tests — `tests/test_geomrec.py` (fast; total ≤ 30 s after numba warm-up)

Use coarse settings in memory (`numerics.h_override`, small `n_transits`) so every grid is ≤ ~50³ cells and every run ≤ ~400 steps. Pass `n_steps` by truncating the loop in the test (call `prepare_geom_run`/`step_geom`/`GeomRecorder` directly), not by changing `derive_geom`.

1. **Centroids on plane.** S1, s3p, S4: $|(\mathbf{g}_c - \mathbf{p}_0)\cdot\hat n| \le 10^{-9}$ m. S1 and s3p: $|d_c - h/2| \le 10^{-9}h$ for all cells.
2. **Huygens parity with legacy.** s3p vs legacy C9 at the same `h_override`: patch `centers`, `normals`, `areas` equal within 1e-12; `idx4`, `iv`, `sgn`, `vaxis` identical.
3. **Pickup parity with legacy.** Same pair: the set of p indices with $w > 10^{-5}$ equals legacy `_rad_p_idx` (restricted to legacy $w > 10^{-5}$); weights agree within 1e-6.
4. **Short-run parity.** Same pair, 300 steps, legacy loop = `kernels.step` + `Recorder` as in `runner.run`: $\max|p_{\text{avg}}^{\text{geom}} - p_{\text{avg}}^{\text{legacy}}| \le 10^{-5}\max|p_{\text{avg}}^{\text{legacy}}|$; same bound for `huygens/p_face` and `huygens/vn`.
5. **Layout.** Short s3p run: required groups/datasets/attrs present, no `box_surface`, `results/zrad` present, `Result(path).S_grid == attrs["S_grid"]`, `Result(path).summary()` runs.
6. **Probe in solid** raises `ValueError`.
Box-path regression is covered by the existing suite; add no box tests here.

## 6. Showcase / gate script — `showcase/step06b2a_gates.py`

Production settings (no overrides). Prints each config's dry-run estimate first, wraps the runs in `runner.keep_awake`, prints progress, writes **one** `outputs/step06b2a/gates.json` and **one** `outputs/step06b2a/gates.png` (overwrite). Runs: legacy C9, `s3p_c09_parametric`, `s1_cylinder`, `s4_tilted`, `s6_downfire`.

**C12a″ (s3p vs legacy C9)**, over the stored `zrad` log grid:

$$
\frac{\max_f |z_p - z_\ell|}{\max_f |z_\ell|} \le 10^{-4}, \qquad
\max_f \frac{|W_p - W_\ell|}{W_\ell} \le 10^{-4} \quad (W = \tfrac12\rho c S_{\text{grid}}\,\Re z),
$$

directivity at 5 log-spaced $f \in [f_{\min}, f_{\max}]$, $\theta \in [0, \pi]$ step 5°, $\varphi \in \{0, \pi/2\}$, where $|D_\ell| > -30$ dB re its max: $\max |20\log_{10}|D_p/D_\ell|| \le 0.001$ dB.

**Power consistency (S1, S4, S6)** at 8 log-spaced $f \in [\max(f_{\min}, 0.1 f_{\max}), f_{\max}]$:

$$
W_{\text{huy}}(f) = \frac{1}{2|V(f)|^2}\,\Re\sum_q A_q\, P_q(f)\, V_{n,q}^*(f), \qquad
\left|\frac{W_{\text{huy}}}{\tfrac12\rho c S_{\text{grid}}\Re z} - 1\right| \le 0.02 .
$$

Reuse the Huygens-flux helper used by the C10/C11 tests if one exists (search `tests/` and `post/`); otherwise build it from `load_huygens_spectra`.

**C18(i) preview (report only):** S4 vs s3p, $\max_f |z_{S4} - z_{s3p}| / \max_f |z_{s3p}|$, printed separately for Re and Im.

Printout: one table per gate (f, values, ratio, PASS/FAIL), then a summary line per gate. Plot: (a) Re/Im z for legacy C9, s3p, S4; (b) power ratio vs f for S1, S4, S6 with ±2% band.

## 7. Docs

`docs/tasks/step-06b2a.md` = this file. Add `GeomRecorder` to the module list in the package README if one exists. No spec edits.
