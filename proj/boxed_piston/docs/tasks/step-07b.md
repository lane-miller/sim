# boxrad v0.3 — step 7b: image KH in Result; gate C19(A); S2 showcase

Scope: `Result.field`, `directivity`, `directivity_index` for files with planes, using the Neumann image Green's function implemented as **patch expansion**. Remove the 7a `NotImplementedError` guards. Box, baffle and plane-free parametric behavior unchanged.

Do not modify: `post/kirchhoff.py`, `post/transfer.py`, `post/directivity.py`, recorder/kernel/derive files, `tests/ref_*.py`.

## 1. Method (prototyped and verified; see table at the end)

For $n_p$ perpendicular planes, the reflections $M_m$, $m = 0 \dots 2^{n_p}-1$, are the products over every subset of the planes (the empty subset is the identity). Each plane reflection maps $y_a \to 2\,\text{offset} - y_a$ and $n_a \to -n_a$. The KH integral with
$$
G_N(\mathbf x,\mathbf y)=\sum_m G(\mathbf x, M_m\mathbf y)
$$
over the open surface S equals ordinary KH over the expanded surface $\bigcup_m M_m S$, with **unchanged** scalar data $p$ and $v_n$ on each reflected copy (normals reflected). The plane itself contributes nothing.

## 2. `boxrad/post/images.py` (new)

```python
plane_reflections(planes) -> list[tuple[(axis:int, offset:float), ...]]   # 2**n_p subsets, identity first
expand_patches(patches, planes, *arrays) -> (SurfacePatches, *arrays_expanded)
    # centers/normals reflected per subset; areas, h copied; each array (..., N) tiled along the last axis
in_fluid(points, planes, tol=1e-12) -> bool array    # side "-": x_a >= offset - tol; side "+": x_a <= offset + tol
direction_open(dirs, planes, tol=1e-12) -> bool array # side "-": d_a >= -tol; side "+": d_a <= tol
open_solid_angle(planes) -> float                    # 4π / 2**n_p
```

`planes` items are the dicts stored in `meta.attrs["environment_json"]` (`axis` "x"/"y"/"z", `side`, `offset`).

## 3. `result.py`

- Constructor: `Result(path, *, frame: Literal["local","world"] = "local")` sets `frame_default` (spec §6.2). Existing per-call `frame=` keywords still override.
- Remove the 7a guards. When `self.planes` is non-empty:
  - **`field`**: points (after the frame map) not `in_fluid` → `ValueError` naming the plane. `_too_close_huygens`: the enclosed box extends to the plane on rigid sides (as the baffle case does for z). KH: expand `patches`, `p_face`, `vn_face` with `expand_patches`, then call the existing KH routine with the closed-surface geometry setting (no `image_plane_z`).
  - **`directivity(..., allow_masked: bool = False)`**: world directions not `direction_open` → `ValueError`, or NaN when `allow_masked=True`. Far field from expanded data (no `image_plane_z`). `normalize=True` uses local θ = 0 (or world +z); if that direction is not open → `ValueError`.
  - **`directivity_index`**: the pattern from expanded data on the **full** sphere (existing quadrature, `half_space=False`), then add $10\log_{10} 2^{n_p}$. This is exact: the expanded pattern is the mirror-symmetric extension, so $\int_{4\pi}|p|^2 = 2^{n_p}\int_\Omega |p|^2$. The on-axis direction must be open, else `ValueError`.
- Files without planes: no behavior change; `allow_masked` is accepted and ignored.
- Docstrings: state the planes behavior and the DI convention ($4\pi|p_{ax}|^2/\int_\Omega|p|^2$; a monopole in a quarter space gives 6.02 dB).

## 4. Tests — `tests/test_images.py` (fast)

1. **C19(A) synthetic, production KH.** Monopole $p=e^{-jkr}/r$ at $\mathbf s=(0.05, 0.08, 0.10)$ plus its images, for planes {z−0}, {z−0, y−0}, {z−0, y−0, x−0}. Open surface from `geomrec.huygens_patches_planes` on a box (lo = plane on rigid sides, else −0.15; hi = (0.20, 0.23, 0.25)), h = 5 mm, offset 0. Exact $p$ and $\partial p/\partial n$ at the patch centers. Expand, then call `kirchhoff.kh_far_field` and `kirchhoff.kh_pressure` directly. At f ∈ {300, 1500, 3000} Hz:
   - far field at θ = 0…90° step 10°, φ = 0.7 rad, vs $\sum_m e^{jk\hat d\cdot\mathbf s_m}$;
   - near field at (0.5, 0.6, 0.4), (1.0, 0.3, 0.05), (0.3, 1.2, 0.8) vs the exact sum.

   Gate: ≤ 0.1 dB and ≤ 1°.
2. **DI convention.** The same synthetic setup at 100 Hz with $\mathbf s=(0.01, 0.01, 0.01)$: DI = $3.01\,n_p$ dB ± 0.05 for $n_p$ = 1, 2, 3, through the same helper `directivity_index` uses.
3. **Result-level B′ consistency.** Short (150-step) runs of `c19_sym_full`, `c19_half_x`, `c19_quarter_xy` (truncated loops, as in `test_planes.py`): the world directivity at 3 frequencies and 8 open directions, and `field` at 2 exterior points with $x, y > 0$, agree between half/quarter and full to $10^{-5}$ relative. The expanded patch set must equal the full closed set (assert centers equal as sets to 1e-12).
4. `field` behind a plane → `ValueError`; masked `directivity` → `ValueError`, and with `allow_masked=True` → NaN in exactly the masked entries.
5. `Result(path, frame="world")` sets `frame_default`.

## 5. Gate script — `showcase/step07b_images.py`

Reuses `outputs/step07a/*.h5` (runs missing ones). One `outputs/step07b/gates.json` with **numeric values, not only booleans**, and one `gates.png` (overwrite).

- **C19(A)** table (test 1 setup), per $n_p$ and f: max dB and max degrees, near and far. PASS ≤ 0.1 dB / 1°.
- **C19(B′) far field:** half and quarter vs full, from the step 7a files: max relative difference of the world directivity (θ 0–90°, φ 0–90°, 5° steps, 5 frequencies), of `field` at the 7a probe points that lie outside the Huygens surface, and of DI. PASS ≤ 1e-5 (DI ≤ 1e-4 dB).
- **S2 showcase (report, with closed-form references):** local frame (woofer on-axis = world +y, horizontal, on the floor boundary).
  - On-axis $|D_{S2}/D_{\text{free}}|$ in dB at 50, 80, 120, 200, 400, 1000 Hz, next to the image reference for that direction, $20\log_{10}|2(1+e^{-jk\,2d_w})|$ with $d_w = 0.55$ m. The floor image is exactly in phase horizontally; the limit is +12.04 dB.
  - $DI_{S2}$ and $DI_{\text{free}}$ at the same frequencies; expect $DI_{S2} - DI_{\text{free}} \to 6.02$ dB at low f.
- Plot: (a) C19(A) errors vs f; (b) S2 on-axis ratio vs the image reference; (c) DI S2 and free vs f with a 6.02 dB offset line.

## 6. Docs

`docs/tasks/step-07b.md` = this file.

### Prototype results (Claude, numpy; open surface, exact data, h = 5 mm)

| planes | far field, max dB at 0.3 / 1.5 / 3 kHz | far-field phase |
|---|---|---|
| z | 0.0004 / 0.002 / 0.011 | ≤ 0.36° |
| z, y | 0.0003 / 0.004 / 0.008 | ≤ 0.12° |
| z, y, x | 0.0003 / 0.003 / 0.056 | ≤ 0.17° |
