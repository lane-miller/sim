# boxrad v0.3 — step 6b2b: radiator-frame view in Result, S6 gap probe

Scope: `Result` serves parametric files in the **radiator-0 local frame** (legacy single-radiator view, the default) or the **world frame**, for `field`, `directivity`, `directivity_index`. Box/baffle files are unchanged (their local frame = world frame). No recorder, kernel or config changes.

Do not modify anything except `boxrad/result.py`, plus the new test, showcase and doc files below.

## 1. Pose

From `meta.attrs` (`radiator0_position` = $\mathbf t$, `radiator0_normal` = $\hat n$, `radiator0_x_axis` = $\hat x$): $\mathbf R = [\hat x,\ \hat n\times\hat x,\ \hat n]$ (columns; local → world). Files without these attrs (box/baffle): $\mathbf R = \mathbf I$, $\mathbf t = \mathbf 0$. Expose read-only `Result.pose -> (R, t)` (float64 (3,3), (3,)) and `Result.frame_default`, which is `"local"`.

Check on load: $\|\mathbf R^\top\mathbf R - \mathbf I\|_\infty \le 10^{-9}$ and $\det\mathbf R > 0$, else `ValueError`.

## 2. Transforms (verified numerically, rel. err 4e-7)

Local point → world: $\mathbf x_w = \mathbf R\,\mathbf x_\ell + \mathbf t$.

Local direction $\hat r_\ell$ → world $\hat r_w = \mathbf R\,\hat r_\ell$. With $p \approx D\,e^{-jkr}/r$ ($e^{+j\omega t}$), the local pattern (phase referenced to the radiator center) is

$$
D_\ell(\hat r_\ell) = D_w(\mathbf R\,\hat r_\ell)\; e^{-jk\,(\mathbf R\hat r_\ell)\cdot\mathbf t} .
$$

## 3. API (additive keyword `frame: Literal["local", "world"] | None = None`; `None` → `frame_default`)

- `field(points, f, *, frame=None, ...)`: points in the chosen frame; map to world, then the existing path (Huygens-proximity check, probe fallback, KH) runs unchanged. Probe coordinates stay in world (as stored).
- `directivity(theta, phi, f, *, frame=None, ...)`: θ, φ in the chosen frame. Build $\hat r_\ell$ from θ, φ as now; for `"local"` evaluate the existing world KH far field at $\mathbf R\hat r_\ell$ and multiply by $e^{-jk(\mathbf R\hat r_\ell)\cdot\mathbf t}$. `normalize=True` divides by the value at local θ = 0 (i.e. world direction $\hat n$) in local, by world +z in world.
- `directivity_index(f, *, frame=None, ...)`: on-axis = local +z ($\hat n$ in world) for `"local"`, world +z for `"world"`. Sphere center = $\mathbf t$ for local, origin for world; radius rule unchanged.
- Invalid `frame` → `ValueError`. For box/baffle files both frames give identical results by construction (R = I, t = 0); the code path must not branch on geometry for this.

Update the docstrings: coordinates and phase reference depend on `frame`; parametric default = radiator-0 local frame (origin at radiator center, +z outward = $\hat n$, +x = pose x-axis).

## 4. Tests — `tests/test_result_frames.py` (fast: coarse `h_override`, ≤ 250 steps; truncated loops as in `test_geomrec.py`)

These are consistency identities; they hold for any recorded data, so short runs suffice.

1. **Far-field phase (local).** Short coarse S4 file. At $f$ = 2 frequencies ≤ 1 kHz, 6 local directions, $r = 2000$ m: $\mathbf x_\ell = r\hat r_\ell$,
   $$
   \frac{|\,\texttt{field}(\mathbf x_\ell,\ \text{local}) - D_\ell\,e^{-jkr}/r\,|}{|D_\ell|/r} \le 2\times10^{-3}.
   $$
2. **Far-field phase (world).** Same file, same check in world frame with $r$ from the world origin.
3. **Frame consistency.** `field(x_l, frame="local") == field(R x_l + t, frame="world")` to 1e-12 relative, at 4 exterior points.
4. **Box file unchanged.** Short coarse C9 legacy file (`kernels.step` + `Recorder`): `directivity` with `frame="local"`, `frame="world"` and no keyword are identical arrays; `Result.pose` is (I, 0).
5. **Bad pose.** Writing a non-orthonormal `radiator0_x_axis` into a copy of the S4 file → `ValueError` on access to `pose`.
6. **Invalid frame string** → `ValueError`.

## 5. Showcase — `showcase/step06b2b_frames.py` (production settings; report only, no pass/fail)

Reuses `outputs/step06b2a/{s3p,s4,s6}.h5` if present; otherwise runs them (keep_awake, progress). Writes **one** `outputs/step06b2b/frames.json` and **one** `outputs/step06b2b/frames.png` (overwrite).

**(a) C18(i) directivity preview.** S4 (local) vs s3p, `normalize=True`, f ∈ {500, 1000, 2000, 3000} Hz, θ = 0…180° step 5°, φ ∈ {0, 90°}. Print, per f and φ, the max $|20\log_{10}|D_{S4}/D_{s3p}||$ over angles where s3p is above −20 dB re its max. Plot the polar curves (S4 dashed over s3p solid), both φ cuts at 1 and 3 kHz.

**(b) Frame sanity printout.** For S4, print `pose` (R, t), and the world direction of local θ = 0 (must equal $\hat n$ to 1e-12).

**(c) S6 gap probe.** S6 needs a probe at world (0, 0, 0.015) m: 15 mm below the down-firing woofer, in the 30 mm gap above the slab. If `s6_downfire.json` has no probe there, add it to the config file under `output.probes` (this is the only allowed config edit) and rerun S6. Plot $|p/v|$ in dB re 1 Pa/(m/s) vs f for: the gap probe (via `field` → probe fallback); a free-field point 0.5 m out sideways at the same height, world (0.5, 0, 0.015); and a point 0.5 m above the top, world (0, 0, 0.73). Print those three at 100, 300, 1000, 3000 Hz.

## 6. Docs

`docs/tasks/step-06b2b.md` = this file.
