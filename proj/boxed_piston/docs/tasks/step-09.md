# Step 9: surface_p from wall patches (parametric/STEP path)

## Goal (spec v1 §5, v2 §5/§6.2)
- **Box files:** keep the v0.2 surface points, order and values, bit-identical.
- **Parametric/STEP files:** record one surface point per **wetted cell**, i.e. every fluid cell that carries wall-patch area. Each point has:
  - its pressure transfer;
  - its patch centroid x̄_w, wetted area A_w and area vector N_w;
  - its unit normal n̂_w.
- Stored per run file, so each driven radiator gets its own.

## Hard rules
- No kernel, floor, CPML, KH or pickup changes.
- The fractions engine change is additive accumulation only: α, β, N_c, M_c and the radiator arrays must stay bit-identical. The oracle comparison stays at 1e-12.
- Box/baffle bit-identical; C12 and C1–C11 unchanged.
- One showcase, one JSON, one PNG.

## 1. Fractions (`cutcell.py` numba engine + `tests/ref_cutcell.py` oracle)
- **New per-cell accumulators**, over the same clipped wall polygons that already build N_c:
  - scalar area $A_c=\sum A$;
  - first moment $\sum A\,\bar x$ (world coordinates).
  - On-plane rule unchanged: the patch goes to the n̂-side cell.
- **New `Fractions` fields** (cells with A_c > 1e-12 h² only, in compact form like the existing patch arrays):
  - `wall_cell` (flat index);
  - `wall_area` A_w;
  - `wall_centroid` x̄_w = ΣA x̄ / A_w;
  - `wall_area_vec` N_w = N_c, the existing sum A n̂, outward from the solid into the fluid. Use whichever sign convention N_c already has, and document it.
- **Wetted cells** = `wall_cell` entries with β̃ > 0, after flood fill (cells removed as voids are excluded).
- **Oracle:** extend `ref_cutcell.py` the same way. Numba vs oracle ≤ 1e-12 on the existing oracle cases.
- Radiator triangles are part of the solid mesh, so the radiator area is included in the wall patches. That is intended: surface_p covers the radiator face.

## 2. Recording (`geomrec.py`) and finalize
- **Record** decimated p at the wetted cells with the same decimation, buffering and time base as the probes.
- **Transfer:** finalize converts to a frequency transfer exactly as for probes, with the same normalization as v0.2 `surface_p` (per unit drive). For test-source runs this gives Pa/(m³/s).
- **HDF5:**
  - `surface/p` (F, n_wet) complex;
  - `surface/points` (n_wet, 3) = x̄_w;
  - `surface/normals` (n_wet, 3) = N_w/|N_w|;
  - `surface/areas` (n_wet,) = A_w;
  - `surface/area_vectors` (n_wet, 3) = N_w;
  - all in the world frame.
  - Attr `surface_kind = "wall_patches"`. Box files: `surface_kind = "box_faces"`, and nothing else changes.
- **`output.surface_p`:** optional bool, default true. When false, skip recording. The dry run reports n_wet and the extra bytes.

## 3. Result
- **`.surface_p`:** same signature and rank as v0.2. For wall-patch files it returns (F, n_wet).
- **Points, normals, areas, area vectors:** keep the existing points accessor name, whatever v0.2 calls it. Add `.surface_normals`, `.surface_areas` and `.surface_area_vectors`.
  - Points, normals and area vectors follow `frame` (local/world) exactly like `field`: rotate the vectors, transform the points.
  - Areas are frame-invariant.
- **Box files:** `.surface_normals` / `.surface_areas` return the v0.2 face-mesh normals and areas if v0.2 stores enough to compute them. Otherwise raise `NotImplementedError("box files: surface normals/areas not stored")`. Report which.
- **Docstring:** net force on the body $\mathbf F = -\sum_w p_w \mathbf N_w$, with $\mathbf N_w$ pointing into the fluid. If the stored convention is the opposite, state the sign correctly.

## 4. Analytic reference (`validation/analytic.py`, additive)
Surface-pressure moments for a monopole Q at distance r₀ from the center of a rigid sphere of radius a, with e^{+jωt}. On the surface,
$$p(a,\theta)=\sum_n c_n P_n(\cos\theta),$$
$$\frac{c_n}{Q}=j\omega\rho\,(2n+1)\,\frac{-k\,h_n^{(2)}(kr_0)}{4\pi\,(ka)^2\,h_n^{(2)\prime}(ka)}$$
(by the Wronskian of $j_n$ and $h_n^{(2)}$).
- **Mean surface pressure:** $\langle p\rangle = c_0$.
- **Force on the sphere along the axis from the center toward the source:** $F = -\tfrac{4\pi a^2}{3}\,c_1$.
- Add `sphere_surface_moments(f, a, r0, rho, c) -> (c0, F)`.
- Claude verified both against direct quadrature of `sphere_point_source` at r = a (rel ≤ 3e-7, 0.3–3 kHz).

## 5. Tests
**`tests/test_surface_patches.py`** (fast):
1. **Oracle match** for `wall_area`, `wall_centroid` and `wall_area_vec` ≤ 1e-12. α, β, N_c, M_c unchanged (bit-identical vs the pre-change arrays on one fixture via the oracle).
2. **Sphere** (`c13_sphere_ps` geometry, n = 9, no stepping):
   - ΣA_w vs the triangle-mesh area ≤ 1e-9 relative;
   - |ΣN_w| ≤ 1e-9·ΣA_w (closure);
   - every |N_w| ≤ A_w(1 + 1e-12);
   - every x̄_w within h of the sphere surface.
3. **Aligned box** (`c19_sym_full`, no stepping):
   - ΣA_w = box surface area ≤ 1e-12 relative;
   - every normal is ±x̂/±ŷ/±ẑ, or a corner/edge combination with |N_w| < A_w;
   - wetted cells all have β̃ = 1.
4. **`sphere_surface_moments`** vs quadrature of `sphere_point_source` at r = a (4001 θ points): ≤ 1e-6 at 300 Hz and 3 kHz.
5. **Box legacy file:** `surface_p` dataset and points unchanged (existing tests); `surface_kind == "box_faces"`.
6. **`output.surface_p = false`:** no `surface/` group; the dry run reports 0 extra bytes.

## 6. Showcase: `showcase/step09_surface_p.py`
- **Runs:** sphere + test source from `configs/c13_sphere_ps.json` at n = 9 and n = 18 (h = 0.05/n, cfl auto). The 10b2 files lack surface data, so these are new runs. Caffeinate, progress lines.
- **Per f** (300, 500, 800, 1000, 1500, 2000, 2500, 3000 Hz):
  - **Mean pressure:** ΣA_w p_w / ΣA_w vs c₀ (dB, deg).
  - **Force:** F_fd = −Σ p_w N_w · ẑ vs F (dB, deg), where ẑ points from the sphere center toward the source (+z here). Fix the sign to match §3 before comparing; if the sign is ambiguous, print both.
  - **Pointwise:** p_w vs `sphere_point_source` at x̄_w, with radius clipped to ≥ a. Print the max |dB| and max |deg| over wetted cells, and the 95th percentile.
- **Gates (n = 9, initial):**
  - mean pressure ≤ 0.15 dB / 1°;
  - force ≤ 0.3 dB / 2°;
  - pointwise 95th percentile ≤ 0.5 dB / 3°.
  - n = 18 is report-only, along with the ratio of n = 9 to n = 18 errors.
- **Outputs:** `outputs/step09/surface_p.json` and one PNG with 3 panels: c₀ error vs f, force error vs f, and |p_w| vs polar angle θ at 3 kHz (FDTD dots against the series line).

## Report back
- Files changed.
- The N_c sign convention.
- What box files return for normals/areas.
- n_wet at n = 9 and n = 18.
- Any deviation.
